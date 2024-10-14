# Radiology Foundation Model using DINOv2

This repository is an adaptation of the DINOv2 framework tailored specifically for training a foundation model in the radiology domain. 

### TODO

- Remove patches with only black pixels
- FFT for sharpening
- Use sagitan and coronal projections
- Test and merge with main

- Dataloader to eval using entire volume
- script for running benchmark tests (after I have many datasets)
- scripts to test transforms (generate samples)

- write userguide for data processing

## Overview

### 1. **PyTorch 2.4 Compatibility**
The original DINOv2 codebase has been patched to be compatible with PyTorch 2.4. See this repo for source (https://github.com/zinccat/dinov2-patch).

### 2. **Gradient Accumulation**
Implemented gradient accumulation to enable training with larger batch sizes on limited GPU memory. 

### 3. **Flexible Transforms Module**
Modified the transforms module to allow for easy customization of the transforms used and implemented some transforms more suited for radiological images. 

## Configuring transforms

Here is an example of how the config file is parsed to select which transforms are used and in what order. Multiple groups of augmentations can be passed and used for different datasets.

```
augmentations:
  default_ct:
    global_1:
    - name: rotate
      p: 0.8
    - name: globalcrop
    - name: flip
      p: 0.5
    - name: contrast
      p: 0.8
    - name: brightness
      p: 0.8
    - name: gaussian_blur
      p: 1.0
    global_2:
    - name: globalcrop
    - name: flip
      p: 0.5
    - name: contrast
      p: 0.8
    - name: brightness
      p: 0.8
    - name: noise
      p: 0.5
      mean: 0.0
      std: 0.1
    - name: gaussian_blur
      p: 0.1
    local:
    - name: localcrop
    - name: flip
      p: 0.5
    - name: contrast
      p: 0.8
    - name: brightness
      p: 0.8
    - name: gaussian_blur
      p: 0.5
```
Check dinov2/data/transforms.py to check the parameter usage.

## Data Preparation

Suppose you have a main folder containing multiple untouched CT datasets.

```
datasets
 +- LIDC-IDRI
 |   +- patient1
 |       +- 0-001.dcm
 |       +- 0-002.dcm
 |       ...
 |   ...
 +- NSCLC-Radiomics
 ...
```

The following script will search through all DICOM files in the LIDC-IDRI folder, quality-check the scans, and index the valid ones in an SQLite database. 

```
python3 -m data.dataprep.CT.dicoms --root_path <path to /datasets> --dataset_name <LIDC-IDRI> --db_name <name of dataset group e.g ct_datasets>
```

This will create the directory data/index/<db_name>/index.db which contains the paths to each DICOM file and their order in their respective scans. 

Assuming that <db_name> is ct_datasets, the following is an example config section to use that dataset for training:
```
datasets:
  - name: ct_datasets
    weight: 1.0
    root_path: .
    type: ct
    storage: dicom
    augmentation: default_ct
    channels: 1
    pixel_range:
      lower: -1000.0
      upper: 1900.0
    norm:
      mean: -573.8
      std: 461.3
```

The argument ```weight``` is optional and used for adjusting the sampling ratio of multiple datasets.


## Running the Training Script

Below is an example command for training without a Slurm system:

```bash
./train.sh --devices 0,1,2,3 --config "configs/quicktest.yaml" --output "runs/quicktest"
```
