import torch
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from einops import rearrange


def plot_train_curves(train_items, val_values, ylabel, hline_type="min"):
    val_vmin = min(val_values)
    val_vmax = max(val_values)

    if hline_type == "min":
        plt.axhline(y=val_vmin, color="k", linestyle=":", alpha=0.5)
    elif hline_type == "max":
        plt.axhline(y=val_vmax, color="k", linestyle=":", alpha=0.5)
    plt.plot(train_items, label="Train")
    plt.plot(val_values, label="Validation")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def plot_confusion_matrix(all_labels, all_predictions, normalize=True):
    normalize_arg = "true" if normalize else None
    cm = confusion_matrix(all_labels, all_predictions, normalize=normalize_arg)
    fig, ax = plt.subplots(figsize=(6, 5))
    if normalize:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    else:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    plt.colorbar(im, ax=ax)
    ax.set(xlabel="Predicted label", ylabel="True label", title="Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.02f}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    ax.set_xticks(range(cm.shape[0]))
    ax.set_yticks(range(cm.shape[0]))

    plt.tight_layout()
    plt.show()


def plot_patch_similarity(patch_features, ref_image, ref_x=0, ref_y=0, thresh=0.0):
    ref_patch = patch_features[ref_y, ref_x].unsqueeze(0)
    img_size = ref_image.shape[-1]
    patch_dim = patch_features.shape[0]

    flat_features = rearrange(patch_features, "x y d -> (x y) d")

    cos_similarity = F.cosine_similarity(flat_features, ref_patch, dim=1)
    cos_similarity = rearrange(cos_similarity, "(x y) -> x y", x=patch_dim, y=patch_dim)
    cos_similarity = torch.clip(cos_similarity, 0, 1)

    cos_similarity = torch.where(cos_similarity > thresh, cos_similarity, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    im1 = ax1.imshow(ref_image[0, 5], cmap="gray")
    ax1.plot(
        [
            (ref_x + 0.5) / patch_dim * (img_size - 1),
            (ref_x + 0.5) / patch_dim * (img_size - 1),
        ],
        [ref_y / patch_dim * (img_size - 1), (ref_y + 1) / patch_dim * (img_size - 1)],
        c="r",
    )
    ax1.plot(
        [
            (ref_x) / patch_dim * (img_size - 1),
            (ref_x + 1) / patch_dim * (img_size - 1),
        ],
        [
            (ref_y + 0.5) / patch_dim * (img_size - 1),
            (ref_y + 0.5) / patch_dim * (img_size - 1),
        ],
        c="r",
    )
    fig.colorbar(im1, ax=ax1)
    ax1.set_title("CT Image")

    im2 = ax2.imshow(cos_similarity, cmap="plasma", vmin=0.0, vmax=1.0)
    fig.colorbar(im2, ax=ax2)
    ax2.set_title("Cosine Similarity")

    plt.tight_layout()
    plt.show()


def plot_feature_map(patch_features, ref_image, components=(0, 1, 2), channels=10):
    H, W, D = patch_features.shape
    X = patch_features.view(H * W, D).numpy()

    max_comp = max(components) + 1

    pca = PCA(n_components=max_comp)
    X_pca = pca.fit_transform(X)

    X_sel = X_pca[:, components]

    X_min, X_max = X_sel.min(0), X_sel.max(0)
    X_sel = (X_sel - X_min) / (X_max - X_min + 1e-6)

    feature_map = X_sel.reshape(H, W, 3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    im1 = ax1.imshow(ref_image[0, channels // 2], cmap="gray")
    fig.colorbar(im1, ax=ax1)
    ax1.set_title("CT Image")

    im2 = ax2.imshow(feature_map)
    # fig.colorbar(im2, ax=ax2)
    ax2.set_title("Colorized Feature Map")

    plt.tight_layout()
    plt.show()


def view_object(img, relx, rely):
    D, W, H = img.shape
    posx = int(relx * W)
    posy = int(rely * H)

    im_axial = img[D // 2, :, :]

    fig, ax = plt.subplots()

    ax.imshow(im_axial, cmap="gray", aspect="equal", vmin=-1000, vmax=1000)

    ax.plot([posx - 5, posx + 5], [posy, posy], color="red", linewidth=0.5)
    ax.plot([posx, posx], [posy - 5, posy + 5], color="red", linewidth=0.5)

    ax.axis("off")
    fig.tight_layout()
    plt.show()
