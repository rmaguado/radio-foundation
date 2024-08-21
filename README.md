# Radiology Foundation Model using DINOv2

This repository is an adaptation of the DINOv2 framework tailored specifically for training a foundation model in the radiology domain. 

### TODO

- implement use multiple slices at once
- redesign entries for possible multiple files (entry file only contains series ids, generate indexes for slices during runtime)
- config validation
- unit tests for image transforms and data loaders

- write userguide for data processing
- add docstrings and type hints
- add unit tests
- implement more datasets and benchmarks
- use various PatchEmbed modules (from dinov2.layers) for different modalities / num slices

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
  global_1:
  - name: rotation
    p: 0.8
    degrees: 90
  - name: globalcrop
  - name: flip
    p: 0.5
  - name: contrast
    p: 0.8
    contrast: 0.4
  - name: brightness
    p: 0.8
    brightness: 0.4
  - name: blur
    p: 1.0
  global_2:
  - name: globalcrop
  - name: flip
    p: 0.5
  - name: contrast
    p: 0.8
    contrast: 0.4
  - name: brightness
    p: 0.8
    brightness: 0.4
  - name: solarize
    p: 0.2
    threshold: 0.5
    max_value: 1.0
  - name: noise
    p: 0.5
    noise_level: 0.02
    max_value: 1.0
  - name: blur
    p: 0.1
  local:
  - name: localcrop
  - name: flip
    p: 0.5
  - name: contrast
    p: 0.8
    contrast: 0.4
  - name: brightness
    p: 0.8
    brightness: 0.4
  - name: blur
    p: 0.5
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
