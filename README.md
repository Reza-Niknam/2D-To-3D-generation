# Converting 2D images to 3D based on LRM architecture

This repo is a feed-forward framework for converting 2D images into 3D models is presented, which utilizes a hybrid method including an off-the-shelf multiview diffusion model and a sparse-view reconstruction model based on the LRM architecture, enabling the rapid creation of diverse 3D models.

- providing a complete personalized 3D rendering framework.
- Mesh representation optimization has been improved by integrating a differentiable iso-surface extraction module aiming to increase training efficiency and exploit more geometric observations.
- modifying and improving the camera view angles by defining new and optimal conditions.
- increasing the resolution of the generated model and reducing the loss of areas due to noise or other contributing factors
applying the texture baking process to the generated mesh in the post-processing process.


# ⚙️ Dependencies and Installation

We recommend using `Python>=3.10`, `PyTorch>=2.1.0`, and `CUDA>=12.1`.
```bash
conda create --name instantmesh python=3.10
conda activate instantmesh
pip install -U pip

# Ensure Ninja is installed
conda install Ninja

# Install the correct version of CUDA
conda install cuda -c nvidia/label/cuda-12.1.0

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different PyTorch version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7

# For Linux users: Install Triton 
pip install triton

# For Windows users: Use the prebuilt version of Triton provided here:
pip install https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl

# Install other requirements
pip install -r requirements.txt
```

# 💫 How to Use

## Running with command line

To generate 3D meshes from images via command line, simply run:
```bash
python run.py configs/instant-mesh-large.yaml examples/hatsune_miku.png --save_video
```

We use [rembg](https://github.com/danielgatis/rembg) to segment the foreground object. If the input image already has an alpha mask, please specify the `no_rembg` flag:
```bash
python run.py configs/instant-mesh-large.yaml examples/hatsune_miku.png --save_video --no_rembg
```

By default, our script exports a `.obj` mesh with vertex colors, please specify the `--export_texmap` flag if you hope to export a mesh with a texture map instead (this will cost longer time):
```bash
python run.py configs/instant-mesh-large.yaml examples/hatsune_miku.png --save_video --export_texmap
```

Please use a different `.yaml` config file in the [configs](./configs) directory if you hope to use other reconstruction model variants. For example, using the `instant-nerf-large` model for generation:
```bash
python run.py configs/instant-nerf-large.yaml examples/hatsune_miku.png --save_video
```

# 💻 Training

We provide our training code to facilitate future research. But we cannot provide the training dataset due to its size. Please refer to our [dataloader](src/data/objaverse.py) for more details.

To train the sparse-view reconstruction models, please run:
```bash
# Training on NeRF representation
python train.py --base configs/instant-nerf-large-train.yaml --gpus 0,1,2,3,4,5,6,7 --num_nodes 1

# Training on Mesh representation
python train.py --base configs/instant-mesh-large-train.yaml --gpus 0,1,2,3,4,5,6,7 --num_nodes 1
```

We also provide our Zero123++ fine-tuning code since it is frequently requested. The running command is:
```bash
python train.py --base configs/zero123plus-finetune.yaml --gpus 0,1,2,3,4,5,6,7 --num_nodes 1
```
