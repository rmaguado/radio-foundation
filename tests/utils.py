from IPython.display import display
import PIL.Image
import numpy as np

def show_image(image_tensor):
    image_np = (image_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
    display(PIL.Image.fromarray(image_np))