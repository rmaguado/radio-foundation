# Radiology Foundation Model using DINOv2

This repository is an adaptation of the DINOv2 framework tailored specifically for training a foundation model in the radiology domain. 

### TODO

- remove use of PIL image object (limited to 1,3 or 4 channels)
- implement more datasets and benchmarks
- implement use multiple slices at once
- training ViT-l and ViT-g

## Overview

### 1. **PyTorch 2.4 Compatibility**
The original DINOv2 codebase has been patched to be compatible with PyTorch 2.4. See this repo for source (https://github.com/zinccat/dinov2-patch).

### 2. **Gradient Accumulation**
Implemented gradient accumulation to enable training with larger batch sizes on limited GPU memory. 

### 3. **Flexible Transforms Module**
Modified the transforms module to allow for easy customization of the transforms used and implemented some transforms more suited for radiological images. 

## Configuring transforms

Here is an example of how the config file is parsed to select which transforms are used and in what order:

```
augmentations:
    crops:
        local_crops_size: 98
    norm:
        mean: 0.124
        std: 0.121
    global_1:
    - rotation_0.8_90
    - crop
    - contrast_0.8_0.4
    - brightness_0.8_0.4
    - blur_1
    global_2:
    - crop
    - contrast_0.8_0.4
    - brightness_0.8_0.4
    - solarize_0.2_0.5_1
    - noise_0.5_0.02_1
    - blur_0.1
    local:
    - crop
    - contrast_0.8_0.4
    - brightness_0.8_0.4
    - blur_0.5
```
Check dinov2/data/transforms.py to check the parameter usage.

## Data Preparation

The project is currently only geared to processing CT scans. To prepare a dataset for training, follow these steps:

1. **Resample scans and generate h5 files**

2. **Generate entry files**

3. **Create a dataset child object**

## Running the Training Script

Below is an example command for training without a Slurm system:

```bash
./scripts/train.sh --devices 0,1,2,3 --config "lidc_idri/vits14_reg4.yaml" --output lidc_test
```