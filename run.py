# -----------------------------------------------------------------------------
# Background Removal with AI
# Author: Mohammad Reza Niknam
# Institution: Shahrood University of Technology
# Program: Master's in Telecommunications Systems
# Description: This code implements a personalized background removal tool
#              using artificial intelligence. All rights reserved.
# -----------------------------------------------------------------------------

# Import required libraries
import os
import argparse
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

# Import utility functions from local source files
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, # Converts Field of View to camera intrinsic matrix
    get_zero123plus_input_cameras, # Retrieves camera parameters for input views
    get_circular_camera_poses, # Generates camera poses for circular rendering
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, save_video


def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters for a circular path around the object.

    Args:
        batch_size (int): Batch size (typically 1 for inference).
        M (int): Number of camera poses (frames) in the circle.
        radius (float): Distance from the camera to the object center.
        elevation (float): Vertical angle of the camera path.
        is_flexicubes (bool): Flag indicating if the model uses the FlexiCubes representation (affects camera format).
        
    Returns:
        torch.Tensor: Camera parameters, shape (batch_size, M, N) where N depends on is_flexicubes.
    """
    # Get camera poses (world-to-camera or camera-to-world)
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        # FlexiCubes uses the camera-to-world matrix (or its inverse)
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1) # (B, M, 4, 4)
    else:
        # Standard format: [Extrinsics, Intrinsics] flattened
        extrinsics = c2ws.flatten(-2) # (M, 16) - c2w or w2c depending on implementation
        
        # Calculate intrinsics matrix from a fixed FOV (30.0 degrees)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2) # (M, 9)
        
        # Concatenate extrinsics and intrinsics
        cameras = torch.cat([extrinsics, intrinsics], dim=-1) # (M, 25)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1) # (B, M, 25)
    return cameras


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes by iterating through camera poses.

    Args:
        model (nn.Module): The 3D reconstruction/rendering model.
        planes (torch.Tensor): The generated triplane representation of the object.
        render_cameras (torch.Tensor): Camera parameters for rendering views.
        render_size (int): The resolution of the rendered image (e.g., 512x512).
        chunk_size (int): Number of frames to render in one forward pass (to manage VRAM).
        is_flexicubes (bool): Flag indicating the rendering path to use.

    Returns:
        torch.Tensor: Stacked rendered frames, shape (M, H, W, 3).
    """
    frames = []
    # Iterate over camera poses in chunks
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            # Rendering path for FlexiCubes-based models
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            # Rendering path for other triplane-based models (e.g., LRM)
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
    
    # Concatenate all rendered frames. Assuming batch size (dim 0) is 1.
    frames = torch.cat(frames, dim=1)[0] # (M, C, H, W) -> (1, M, C, H, W) -> (M, C, H, W)
    return frames


###############################################################################
# Arguments.
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('input_path', type=str, help='Path to input image or directory.')
parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')
args = parser.parse_args()
# Set the random seed for reproducibility
seed_everything(args.seed)

###############################################################################
# Stage 0: Configuration.
###############################################################################

# Load configuration from YAML file
config = OmegaConf.load(args.config)
# Extract config name to determine model type (e.g., 'instant-mesh')
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

# Check if the model uses the FlexiCubes representation for geometry
IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

device = torch.device('cuda')

# load diffusion model (Zero123Plus for multi-view generation)
print('Loading diffusion model ...')
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
)
# Configure the scheduler for the diffusion model
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# load custom white-background UNet checkpoint (for better generation)
print('Loading custom white-background unet ...')
if os.path.exists(infer_config.unet_path):
    unet_ckpt_path = infer_config.unet_path
else:
    # Download UNet checkpoint from Hugging Face if local path not found
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)

pipeline = pipeline.to(device)

# load reconstruction model (e.g., LRM or InstantMesh generator)
print('Loading reconstruction model ...')
model = instantiate_from_config(model_config)
if os.path.exists(infer_config.model_path):
    model_ckpt_path = infer_config.model_path
else:
    # Download LRM/InstantMesh checkpoint
    model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
# Strip the 'lrm_generator.' prefix from keys in the state dictionary
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
model.load_state_dict(state_dict, strict=True)

model = model.to(device)
if IS_FLEXICUBES:
    # Initialize necessary FlexiCubes components/tensors if using that method
    model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval() # Set model to evaluation mode

# make output directories for images, meshes, and videos
image_path = os.path.join(args.output_path, config_name, 'images')
mesh_path = os.path.join(args.output_path, config_name, 'meshes')
video_path = os.path.join(args.output_path, config_name, 'videos')
os.makedirs(image_path, exist_ok=True)
os.makedirs(mesh_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)

# process input files (can be a single file or a directory)
if os.path.isdir(args.input_path):
    # If directory, list all supported image files
    input_files = [
        os.path.join(args.input_path, file) 
        for file in os.listdir(args.input_path) 
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp')
    ]
else:
    # If single file
    input_files = [args.input_path]
print(f'Total number of input images: {len(input_files)}')


###############################################################################
# Stage 1: Multiview generation. (Zero123Plus)
###############################################################################
# Initialize rembg session only if background removal is requested
rembg_session = None if args.no_rembg else rembg.new_session()

outputs = []
for idx, image_file in enumerate(input_files):
    name = os.path.basename(image_file).split('.')[0]
    print(f'[{idx+1}/{len(input_files)}] Imagining {name} ...')

    # remove background optionally
    input_image = Image.open(image_file)
    if not args.no_rembg:
        # Remove background using rembg
        input_image = remove_background(input_image, rembg_session)
        # Resize foreground to fit within a canonical frame
        input_image = resize_foreground(input_image, 0.85)
    
    # sampling: Generate multi-view consistent images using the diffusion pipeline
    output_image = pipeline(
        input_image, 
        num_inference_steps=args.diffusion_steps, 
    ).images[0]
    

    # Save the generated multi-view image grid
    output_image.save(os.path.join(image_path, f'{name}.png'))
    print(f"Image saved to {os.path.join(image_path, f'{name}.png')}")

    # Convert the image grid to a batch of individual images
    images = np.asarray(output_image, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float() # (C, H_total, W_total) e.g., (3, 960, 640)
    # Rearrange the image grid (e.g., 3x2 grid) into individual views (6, C, H, W)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2) # (N_views, 3, 320, 320)

    outputs.append({'name': name, 'images': images})

# Free up VRAM used by the diffusion model pipeline
del pipeline
# NOTE: The subsequent reconstruction stage needs the LRM/InstantMesh model, not the diffusion model.

###############################################################################
# Stage 2: Reconstruction. (Triplane generation and Mesh/Video extraction)
###############################################################################

# Get camera parameters for the input views used by the reconstruction model
input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale).to(device)
# Set chunk size for rendering: FlexiCubes models can handle larger chunks
chunk_size = 20 if IS_FLEXICUBES else 1

for idx, sample in enumerate(outputs):
    name = sample['name']
    print(f'[{idx+1}/{len(outputs)}] Creating {name} ...')

    images = sample['images'].unsqueeze(0).to(device) # (1, N_views, C, H, W)
    # Resize input images to the expected resolution (e.g., 320x320) with high-quality interpolation
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

    # Optional: Select only 4 out of 6 views if specified by argument
    if args.view == 4:
        indices = torch.tensor([0, 2, 4, 5]).long().to(device)
        images = images[:, indices]
        input_cameras = input_cameras[:, indices]

    with torch.no_grad():
        # get triplane: Feed multi-view images and cameras to the generator to produce the 3D representation
        # 
        planes = model.forward_planes(images, input_cameras)

        # get mesh
        mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')

        # Extract the mesh from the triplane representation
        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=args.export_texmap,
            **infer_config,
        )
        if args.export_texmap:
            # Save mesh with texture map (MTL file and texture image)
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_path_idx,
            )
        else:
            # Save mesh with per-vertex colors
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, mesh_path_idx)
        print(f"Mesh saved to {mesh_path_idx}")

        # get video: Render a circular view video
        if args.save_video:
            video_path_idx = os.path.join(video_path, f'{name}.mp4')
            render_size = infer_config.render_resolution
            # Get camera parameters for a circular rendering path
            render_cameras = get_render_cameras(
                batch_size=1, 
                M=120, # 120 frames for the video
                radius=args.distance, 
                elevation=20.0,
                is_flexicubes=IS_FLEXICUBES,
            ).to(device)
            
            # Render the frames from the generated triplane
            frames = render_frames(
                model, 
                planes, 
                render_cameras=render_cameras, 
                render_size=render_size, 
                chunk_size=chunk_size, 
                is_flexicubes=IS_FLEXICUBES,
            )

            # Save the sequence of frames as a video file
            save_video(
                frames,
                video_path_idx,
                fps=30,
            )
            print(f"Video saved to {video_path_idx}")