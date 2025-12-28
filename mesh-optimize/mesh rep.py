import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torchvision.models import vgg19
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import pytorch_lightning as pl
from einops import rearrange, repeat
from src.utils.train_util import instantiate_from_config

class MVRecon(pl.LightningModule):
    def __init__(self, lrm_generator_config, input_size=256, render_size=512, init_ckpt=None, learning_rate=0.001):
        super(MVRecon, self).__init__()
        self.input_size = input_size
        self.render_size = render_size
        self.learning_rate = learning_rate

        self.lrm_generator = instantiate_from_config(lrm_generator_config)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        self.style_loss = StyleLoss()
        self.augmentation = advanced_augmentation()

        if init_ckpt:
            self.load_pretrained_weights(init_ckpt)

        self.validation_step_outputs = []

    def load_pretrained_weights(self, init_ckpt):
        try:
            sd = torch.load(init_ckpt, map_location='cpu')['state_dict']
            sd = {k: v for k, v in sd.items() if k.startswith('lrm_generator')}
            sd_fc = self.extract_weights(sd)
            self.lrm_generator.load_state_dict(sd_fc, strict=False)
            print(f'Loaded weights from {init_ckpt}')
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    def extract_weights(self, sd):
        sd_fc = {}
        for k, v in sd.items():
            if k.startswith('lrm_generator.synthesizer.decoder.net.'):
                if k.startswith('lrm_generator.synthesizer.decoder.net.6.'):  # last layer
                    sd_fc[k.replace('net.', 'net_sdf.')] = -v[0:1] if 'weight' in k else 10.0 - v[0:1]
                    sd_fc[k.replace('net.', 'net_rgb.')] = v[1:4]
                else:
                    sd_fc[k.replace('net.', 'net_sdf.')] = v
                    sd_fc[k.replace('net.', 'net_rgb.')] = v
            else:
                sd_fc[k] = v
        return {k.replace('lrm_generator.', ''): v for k, v in sd_fc.items()}

    def on_fit_start(self):
        device = torch.device(f'cuda:{self.global_rank}')
        self.lrm_generator.init_flexicubes_geometry(device)
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'images_val'), exist_ok=True)

    def prepare_batch_data(self, batch):
        lrm_generator_input = self.prepare_lrm_generator_input(batch)
        render_gt = self.prepare_render_gt(batch)
        return lrm_generator_input, render_gt

    def prepare_lrm_generator_input(self, batch):
        lrm_generator_input = {'images': self.process_images(batch['input_images'])}
        cameras, render_cameras = self.process_cameras(batch)
        lrm_generator_input.update({'cameras': cameras, 'render_cameras': render_cameras})
        lrm_generator_input['render_size'] = self.render_size
        return lrm_generator_input

    def process_images(self, images):
        return self.augmentation(images).to(self.device)

    def process_cameras(self, batch):
        input_c2ws = batch['input_c2ws']
        input_Ks = batch['input_Ks']
        target_c2ws = batch['target_c2ws']
        render_c2ws = torch.cat([input_c2ws, target_c2ws], dim=1)
        render_w2cs = torch.linalg.inv(render_c2ws)
        input_extrinsics = input_c2ws.flatten(-2)[:, :, :12]
        input_intrinsics = torch.stack([input_Ks.flatten(-2)[:, :, i] for i in [0, 4, 2, 5]], dim=-1)
        cameras = torch.cat([input_extrinsics, input_intrinsics], dim=-1)
        cameras += torch.rand_like(cameras) * 0.04 - 0.02
        return cameras.to(self.device), render_w2cs.to(self.device)

    def prepare_render_gt(self, batch):
        render_gt = {}
        target_images = self.concatenate_and_resize(batch, 'input_images', 'target_images', v2.functional.resize)
        render_gt['target_images'] = target_images
        render_gt['target_depths'] = self.concatenate_and_resize(batch, 'input_depths', 'target_depths', v2.functional.resize, interpolation=0)
        render_gt['target_alphas'] = self.concatenate_and_resize(batch, 'input_alphas', 'target_alphas', v2.functional.resize, interpolation=0)
        render_gt['target_normals'] = self.concatenate_and_resize(batch, 'input_normals', 'target_normals', v2.functional.resize)
        return render_gt

    def concatenate_and_resize(self, batch, key1, key2, resize_fn, interpolation=3):
        concatenated = torch.cat([batch[key1], batch[key2]], dim=1)
        return resize_fn(concatenated, self.render_size, interpolation=interpolation, antialias=True).clamp(0, 1).to(self.device)

    def forward_lrm_generator(self, images, cameras, render_cameras, render_size=512):
        planes = torch.utils.checkpoint.checkpoint(self.lrm_generator.forward_planes, images, cameras, use_reentrant=False)
        return self.lrm_generator.forward_geometry(planes, render_cameras, render_size)

    def forward(self, lrm_generator_input):
        return self.forward_lrm_generator(
            lrm_generator_input['images'], 
            lrm_generator_input['cameras'], 
            lrm_generator_input['render_cameras'], 
            lrm_generator_input['render_size']
        )

    def training_step(self, batch, batch_idx):
        lrm_generator_input, render_gt = self.prepare_batch_data(batch)
        render_out = self.forward(lrm_generator_input)
        loss, loss_dict = self.compute_loss(render_out, render_gt)
        self.log_dict(loss_dict, rank_zero_only=True)
        return loss

    def compute_loss(self, render_out, render_gt):
        render_images = self.scale_images(render_out['img'])
        target_images = self.scale_images(render_gt['target_images'])
        loss_mse = F.mse_loss(render_images, target_images)
        loss_lpips = 2.0 * self.lpips(render_images, target_images)
        loss_style = self.style_loss(render_images, target_images)

        loss_mask = F.mse_loss(render_out['mask'], render_gt['target_alphas'])
        loss_depth = 0.5 * F.l1_loss(render_out['depth'][render_gt['target_alphas']>0], render_gt['target_depths'][render_gt['target_alphas']>0])

        render_normals = self.scale_images(render_out['normal'])
        target_normals = self.scale_images(render_gt['target_normals'])
        similarity = (render_normals * target_normals).sum(dim=-3).abs()
        loss_normal = 0.2 * (1 - similarity[render_gt['target_alphas'].squeeze(-3)>0].mean())

        sdf = render_out['sdf']
        sdf_reg_loss = render_out['sdf_reg_loss']
        sdf_reg_loss_entropy = sdf_reg_loss_batch(sdf, self.lrm_generator.geometry.all_edges).mean() * 0.01
        flexicubes_surface_reg, flexicubes_weights_reg = sdf_reg_loss[1:]
        flexicubes_surface_reg = flexicubes_surface_reg.mean() * 0.5
        flexicubes_weights_reg = flexicubes_weights_reg.mean() * 0.1

        loss_reg = sdf_reg_loss_entropy + flexicubes_surface_reg + flexicubes_weights_reg

        loss = loss_mse + loss_lpips + loss_style + loss_mask + loss_depth + loss_normal + loss_reg

        loss_dict = {
            'train/loss_mse': loss_mse,
            'train/loss_lpips': loss_lpips,
            'train/loss_style': loss_style,
            'train/loss_mask': loss_mask,
            'train/loss_normal': loss_normal,
            'train/loss_depth': loss_depth,
            'train/loss_reg_sdf': sdf_reg_loss_entropy,
            'train/loss_reg_surface': flexicubes_surface_reg,
            'train/loss_reg_weights': flexicubes_weights_reg,
            'train/loss': loss,
        }

        return loss, loss_dict

    def scale_images(self, images):
        return rearrange(images, 'b n ... -> (b n) ...') * 2.0 - 1.0

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        lrm_generator_input = self.prepare_validation_batch_data(batch)
        render_out = self.forward(lrm_generator_input)
        render_images = rearrange(render_out['img'], 'b n c h w -> b c h (n w)')
        self.validation_step_outputs.append(render_images)

    def on_validation_epoch_end(self):
        images = torch.cat(self.validation_step_outputs, dim=-1)
        all_images = self.all_gather(images)
        all_images = rearrange(all_images, 'r b c h w -> (r b) c h w')

        if self.global_rank == 0:
            image_path = os.path.join(self.logdir, 'images_val', f'val_{self.global_step:07d}.png')
            grid = make_grid(all_images, nrow=1, normalize=True, value_range=(0, 1))
            save_image(grid, image_path)
            print(f"Saved image to {image_path}")

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.lrm_generator.parameters(), lr=self.learning_rate, betas=(0.90, 0.95), weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100000, eta_min=0)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss', 'gradient_clip_val': 1.0}
