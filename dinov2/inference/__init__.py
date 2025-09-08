import os
import numpy as np
from functools import partial
import pydicom
import nibabel as nib
import SimpleITK as sitk

import torch
import torch.nn.functional as F

from einops import rearrange

from dinov2.models import vits
from .visualize import view_volume


def get_autocast_dtype(config):
    teacher_dtype_str = (
        config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    )
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def build_model(path_to_checkpoint, config, img_size, device):
    args = config.student
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=args.patch_size,
        in_chans=args.channels,
        init_values=args.layerscale,
        ffn_layer=args.ffn_layer,
        block_chunks=args.block_chunks,
        qkv_bias=args.qkv_bias,
        proj_bias=args.proj_bias,
        ffn_bias=args.ffn_bias,
        embed_layer=args.embed_layer,
        conv_channels=args.conv_channels,
        num_register_tokens=args.num_register_tokens,
        interpolate_offset=args.interpolate_offset,
        interpolate_antialias=args.interpolate_antialias,
    )
    model = vits.__dict__[args.arch](**vit_kwargs)

    state_dict = torch.load(path_to_checkpoint)["teacher"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(k.startswith(p) for p in ["dino_head", "ibot_head"])
    }
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(device)

    autocast_dtype = get_autocast_dtype(config)
    autocast_ctx = partial(
        torch.autocast, enabled=True, dtype=autocast_dtype, device_type="cuda"
    )

    return model, autocast_ctx


def is_HU(img):
    img = torch.clip(img, -1000, 1900)
    return img.min() < 800 and img.max() > -100


def load_dicom(folder_path: str):

    dicom_files = [
        os.path.join(folder_path, x)
        for x in os.listdir(folder_path)
        if x.endswith(".dcm")
    ]

    slices = []
    for filepath in dicom_files:
        dataset = pydicom.dcmread(filepath)
        slices.append(dataset)

    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    z_spacing = float(
        slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]
    )
    x_spacing = float(slices[0].PixelSpacing[0])
    y_spacing = float(slices[0].PixelSpacing[1])

    spacing = (z_spacing, x_spacing, y_spacing)

    rescale_slope = slices[0].RescaleSlope
    rescale_intercept = slices[0].RescaleIntercept

    data_stack = []
    for i, s in enumerate(slices):
        data_stack.append(torch.from_numpy(s.pixel_array).float())

    image = torch.stack(data_stack)
    image = rescale_slope * image + rescale_intercept
    image = torch.clip(image, -1000, 1900)

    assert is_HU(image)

    return image, spacing


def load_mhd(path):
    image_obj = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image_obj)
    spacing = image_obj.GetSpacing()
    spacing = np.array(spacing)[::-1]
    assert abs(spacing[2] - spacing[1]) < 0.001

    image = torch.from_numpy(image).float()
    image = image.clip(-1000, 1900)

    assert is_HU(image)

    return image, spacing


def load_nifti(path):
    nifti = nib.loadsave.load(path)
    image = nifti.get_fdata()  # type: ignore
    affine = nifti.affine  # type: ignore

    s = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    spacing = (float(s[2]), float(s[0]), float(s[1]))
    assert abs(spacing[2] - spacing[1]) < 0.001

    image = torch.from_numpy(image).float()
    image = rearrange(image, "w h d -> d w h")
    image = image.clip(-1000, 1900)

    assert is_HU(image)

    return image, spacing


def crop_volume(img, k=13, use_cuda=True):
    img_device = img.device

    if not torch.cuda.is_available():
        use_cuda = False
    device = "cuda" if use_cuda else "cpu"
    img = img.to(device)

    x_mask = img > -800
    vol_f = x_mask.float().unsqueeze(0).unsqueeze(0)

    kernel = torch.ones((1, 1, k, k, k), dtype=torch.float32).to(device)

    conv_sum = F.conv3d(vol_f, kernel, padding="same")

    out = conv_sum == k * k * k

    mask = out[0, 0].bool()

    coords = mask.nonzero(as_tuple=False)

    if coords.numel() == 0:
        raise ValueError("Mask is empty — no bounding box to extract.")

    z_min, y_min, x_min = coords.min(dim=0).values
    z_max, y_max, x_max = coords.max(dim=0).values

    z_min = max(z_min.item() - k, 0)
    y_min = max(y_min.item() - k, 0)
    x_min = max(x_min.item() - k, 0)

    z_max = min(z_max.item() + k, mask.shape[0] - 1)
    y_max = min(y_max.item() + k, mask.shape[1] - 1)
    x_max = min(x_max.item() + k, mask.shape[2] - 1)

    bbox = (slice(z_min, z_max + 1), slice(y_min, y_max + 1), slice(x_min, x_max + 1))

    cropped_img = img[bbox]
    return cropped_img.to(img_device)


def prepare_image(img, img_size, channels, fmean, fstd, vmin=-1000):
    D, W, H = img.shape

    w_pad2 = max(W, H) - W
    h_pad2 = max(W, H) - H

    w_pad = w_pad2 // 2
    h_pad = h_pad2 // 2

    img = torch.nn.functional.pad(
        img, (h_pad, h_pad2 - h_pad, w_pad, w_pad2 - w_pad), value=vmin
    )

    d_mod2 = D % channels
    d_mod = d_mod2 // 2

    img = img[d_mod : D - d_mod2 + d_mod, :, :]

    img = torch.nn.functional.interpolate(
        img.unsqueeze(0), size=(img_size, img_size), mode="bilinear"
    ).squeeze(0)

    img = rearrange(img, "(g c) w h -> g c w h", c=channels)

    img = (img - fmean) / fstd

    return img


def generate_embeddings(
    img,
    *,
    model,
    img_size,
    patch_size,
    channels,
    fmean,
    fstd,
    device,
    block_size=64,
    no_crop=False,
    autocast_ctx
):
    pdim = img_size // patch_size

    if not no_crop:
        img = crop_volume(img)

    x_prep = prepare_image(img, img_size, channels, fmean, fstd)
    n_groups = x_prep.shape[0]

    batches = []
    for idx in range(0, n_groups, block_size):
        if idx + block_size > n_groups:
            batches.append(x_prep[idx:])
        else:
            batches.append(x_prep[idx : idx + block_size])

    batch_features = []
    for data in batches:
        data = data.to(device=device)
        with torch.inference_mode():
            with autocast_ctx():
                features = model.forward_features(data)
        batch_features.append(
            {k: v.cpu() for k, v in features.items() if isinstance(v, torch.Tensor)}
        )

    cls_tokens = torch.cat([f["x_norm_clstoken"] for f in batch_features], dim=0)
    patch_tokens = torch.cat([f["x_norm_patchtokens"] for f in batch_features], dim=0)
    patch_tokens = rearrange(patch_tokens, "a (h w) e -> a h w e", h=pdim, w=pdim)

    return {
        "cls": cls_tokens,
        "patch": patch_tokens,
    }
