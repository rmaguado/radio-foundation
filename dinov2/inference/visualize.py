import matplotlib.pyplot as plt


def view_volume(img, spacing=None):

    D, W, H = img.shape

    slices = (D // 2, W // 2, H // 2)
    d, w, h = slices

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    im_axial = img[d, :, :].rot90(1)
    im_sagittal = img[:, w, :].rot90(2)
    im_coronal = img[:, :, h].rot90(2)

    ax[0].imshow(im_axial, cmap="gray", aspect="equal", vmin=-1000, vmax=1000)
    ax[0].set_title("Axial")
    ax[0].axis("off")

    ax[1].imshow(im_sagittal, cmap="gray", aspect="equal", vmin=-1000, vmax=1000)
    ax[1].set_title("Sagittal")
    ax[1].axis("off")

    ax[2].imshow(im_coronal, cmap="gray", aspect="equal", vmin=-1000, vmax=1000)
    ax[2].set_title("Coronal")
    ax[2].axis("off")

    if spacing is not None:
        as_axial = spacing[2] / spacing[1]
        as_sagittal = spacing[0] / spacing[2]
        as_coronal = spacing[0] / spacing[1]

        ax[0].set_aspect(as_axial)
        ax[1].set_aspect(as_sagittal)
        ax[2].set_aspect(as_coronal)

    plt.tight_layout()
    plt.show()
