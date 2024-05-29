## OccSora: 4D Occupancy Generation Models as World Simulators for Autonomous Driving

## Installation
1. Create conda environment with python version 3.8.0

2. Install all the packages in environment.yaml

3. Anything about the installation of mmdetection3d, please refer to [mmdetection3d](https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation)

## Preparing
1. Create soft link from data/nuscenes to your_nuscenes_path

2. Prepare the gts semantic occupancy introduced in [Occ3d]

3. Download (From: https://github.com/wzzheng/TPVFormer/tree/main ) generated train/val pickle files and put them in data/

    [nuscenes_infos_train_temporal_v3_scene.pkl]

    [nuscenes_infos_val_temporal_v3_scene.pkl]

  The dataset should be organized as follows:

```
OccSora/data
    nuscenes                 -    downloaded from www.nuscenes.org
        lidarseg
        maps
        samples
        sweeps
        v1.0-trainval
        gts                  -    download from Occ3d
    nuscenes_infos_train_temporal_v3_scene.pkl
    nuscenes_infos_val_temporal_v3_scene.pkl
```

## Getting Started

### Training
Train the VQVAE on A100 with 80G GPU memory.
```
python train_1.py --py-config config/train_vqvae.py --work-dir out/vqvae
```
Generate training Token data using the vqvae results
```
python step02.py --py-config config/train_vqvae.py --work-dir out/vqvae
```
Train the OccSora on A100 with 80G GPU memory. 
```
torchrun --nnodes=1 --nproc_per_node=8 train_2.py --model DiT-XL/2 --data-path /path
```
### Evaluation
Eval the model on A100 with 80G GPU memory.  

The token is obtained by denoising the noise samples_array.npy
```
python sample.py --model DiT-XL/2 --image-size 256 --ckpt "/results/001-DiT-XL-2/checkpoints/1200000.pt"
```
### Visualize
python sample.py --model DiT-XL/2 --image-size 256 --ckpt "/results/001-DiT-XL-2/checkpoints/1200000.pt"
```
python visualize_demo.py --py-config config/train_vqvae.py --work-dir out/vqvae
