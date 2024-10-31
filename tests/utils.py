from IPython.display import display
import PIL.Image
import numpy as np
import torch


def show_image(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    assert isinstance(img, np.ndarray)

    norm_img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0).astype(
        np.uint8
    )
    display(PIL.Image.fromarray(norm_img))
