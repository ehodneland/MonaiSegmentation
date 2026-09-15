import os
import numpy as np
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def scroll_through_fused_channels(img, mask, transpose_order=None, alpha_init=0.4):
    """
    Interactive visualization of a 3D/4D volume with a fused mask.
    The base image is always sharp (one imshow), and the mask is blended into the RGB array.
    """

    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()

    if transpose_order:
        img = np.transpose(img, transpose_order)
        mask = np.transpose(mask, transpose_order)

    mask = mask.astype(float)
    if mask.max() > mask.min():
        mask = (mask - mask.min()) / (mask.max() - mask.min())

    # legg til kanal-dim hvis 3D
    if img.ndim == 3:
        img = img[np.newaxis, ...]

    C, D, H, W = img.shape
    slice_idx = D // 2
    chan_idx = 0

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.subplots_adjust(bottom=0.35 if C > 1 else 0.3)

    def make_fused(c, d, a):
        img_slice = img[c, d]
        img_min, img_max = img_slice.min(), img_slice.max()
        if img_max - img_min < 1e-8:
            norm_img = np.zeros_like(img_slice)
        else:
            norm_img = (img_slice - img_min) / (img_max - img_min)

        rgb = np.stack([norm_img]*3, axis=-1)
        # legg til rødt i R-kanalen, men behold G/B
        rgb[...,0] = (1-a)*rgb[...,0] + a*mask[d]
        return rgb

    fused = make_fused(chan_idx, slice_idx, alpha_init)
    img_plot = ax.imshow(fused)
    ax.set_title(f"Channel {chan_idx}/{C-1}, Slice {slice_idx}/{D-1}, alpha={alpha_init:.2f}")
    ax.axis("off")

    # sliders
    ax_slider_slice = plt.axes([0.2, 0.2, 0.6, 0.03], facecolor="lightgray")
    slider_slice = Slider(ax_slider_slice, "Slice", 0, D-1, valinit=slice_idx, valstep=1)

    if C > 1:
        ax_slider_chan = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor="lightgray")
        slider_chan = Slider(ax_slider_chan, "Channel", 0, C-1, valinit=chan_idx, valstep=1)
    else:
        slider_chan = None

    ax_slider_alpha = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor="lightgray")
    slider_alpha = Slider(ax_slider_alpha, "Alpha", 0.0, 1.0, valinit=alpha_init, valstep=0.05)

    def update(val):
        d = int(slider_slice.val)
        c = int(slider_chan.val) if slider_chan else 0
        a = slider_alpha.val
        fused = make_fused(c, d, a)
        img_plot.set_data(fused)
        ax.set_title(f"Channel {c}/{C-1}, Slice {d}/{D-1}, alpha={a:.2f}")
        fig.canvas.draw_idle()

    slider_slice.on_changed(update)
    if slider_chan:
        slider_chan.on_changed(update)
    slider_alpha.on_changed(update)

    plt.show()



def scroll_through_slices(img, mask, channel=0, num=0, transpose_order=None):
    """
    Interaktiv visualisering av 3D-bildevolumer.
    
    Parametere:
    - img: Bildevolum (H, W, D), etc.
    - mask: Mask volume with the same shape as the image
    - num: Indeks for batch
    - transpose_order: liste for ombytting, f.eks. [2, 0, 1] for (H, W, D) → (D, H, W)
    """

    if hasattr(img, "cpu"):
        img = img.cpu().numpy()
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()

#    if img.ndim == 5:
#        img = img[num, channel]  # (D, H, W)
#        mask = mask[num, 0]
#    elif img.ndim == 4:
#        img = img[channel]

    print("Image shape before transpose:", img.shape)
    print("Mask shape before transpose:", mask.shape)
    if transpose_order:
        img = np.transpose(img, transpose_order)
        mask = np.transpose(mask, transpose_order)

    mask = mask.astype('float')
    mask = (mask - mask.min()) / (mask.max() - mask.min())

    print("Mask sum:", mask.sum())
    print("Mask unique values:", np.unique(mask))
    print("Mask shape:", mask.shape)

    num_slices = img.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.2)

    slice_idx = num_slices // 2
    img_plot = axes[0].imshow(img[slice_idx], cmap="gray")
    mask_plot = axes[1].imshow(mask[slice_idx], cmap="gray", vmin=0, vmax=1)

    axes[0].set_title("Image Slice")
    axes[1].set_title("Mask Slice")
    axes[0].axis("off")
    axes[1].axis("off")

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor="lightgray")
    slider = Slider(ax_slider, "Slice", 0, num_slices - 1, valinit=slice_idx, valstep=1)

    def update(val):
        idx = int(slider.val)
        img_plot.set_data(img[idx])
        mask_plot.set_data(mask[idx])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def show_mask_over_time(im, timeline, mask, slice_index, alpha=0.5, red_cmap='Reds', savefig=None):
    """
    Show one specific slice of a 4D image series across all time points.

    Parametere:
    - im (numpy.ndarray): 4D-bilde (T, D, H, W) der T = tid.
    - timeline (array-like): List of time points (same length as T).
    - mask (numpy.ndarray): 3D mask (D, H, W).
    - slice_index (int): Index of the slice to display (axis 1) or 'largest_mask_slice'
    - alpha (float): Transparency for the mask overlay.
    - red_cmap (str): Name of the colormap for the mask (default 'Reds').
    - savefig (str or None): If provided, save to this path.
    """
    if slice_index == "largest_mask":
        slice_index = largest_mask_slice(mask)

    assert im.ndim == 4, f"The image volume must be 4D (T, D, H, W), but got {im.shape}"
    assert mask.ndim == 3, f"The mask must be 3D (D, H, W), but got {mask.shape}"
    assert 0 <= slice_index < im.shape[1], f"slice_index {slice_index} is out of range"

        
    num_time_points = im.shape[0]
    num_columns = 3
    num_rows = (num_time_points + num_columns - 1) // num_columns

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(num_columns * 5, num_rows * 5))
    axes = axes.flatten()

    mask_slice = mask[slice_index, :, :]
    vmin, vmax = np.min(im[:, slice_index, :, :]), np.max(im[:, slice_index, :, :])

    for t in range(num_time_points):
        img_slice = im[t, slice_index, :, :]
        time = timeline[t]
    
        # Normalize and convert to RGB
#        img_min, img_max = np.min(img_slice), np.max(img_slice)
        if vmax - vmin < 1e-8:
            norm_image = np.zeros_like(img_slice)
        else:
            norm_image = (img_slice - vmin) / (vmax - vmin)
    
        rgb_image = np.stack([norm_image] * 3, axis=-1)
    
        # Create red overlay
        red_overlay = np.zeros_like(rgb_image)
        red_overlay[..., 0] = mask_slice
    
        axes[t].imshow(rgb_image)
        axes[t].imshow(red_overlay, cmap=red_cmap, alpha=alpha, vmin=0, vmax=1)
        axes[t].set_title(f'Time: {time:.0f} s')
        axes[t].axis('off')

    # Skjul evt. tomme plasser i grid
    for t in range(num_time_points, len(axes)):
        axes[t].axis('off')

    plt.tight_layout()
    if savefig:
        os.makedirs(os.path.dirname(savefig), exist_ok=True)
        plt.savefig(savefig, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def largest_mask_slice(mask):
    slice_sums = np.sum(mask, axis=(1, 2))  # Anta (D, H, W)
    return np.argmax(slice_sums)

def plot_images_and_mask(
    imagelist, 
    imagenames, 
    mask,
    maskname,
    pathsave=None, 
    slices_to_plot_mode="all", 
    transpose_order=None,
    alpha=0.5
):
    """
    Plot images with a red mask overlay plus one separate black-and-white mask.

    Parametere:
    - imagelist: Liste med 3D-bilder [(D, H, W), ...] eller 4D-array (N, D, H, W)
    - names: Navn på bildene
    - mask: 3D mask (D, H, W)
    - subj: Subjekt-ID
    - pathsave: Sti for lagring (PDF/png), None for visning
    - slices_to_plot_mode: 'all', 'largest_mask', or 'mask_slices'
    - transpose_order: f.eks. [2, 0, 1]
    """

    if isinstance(imagelist, np.ndarray) and imagelist.ndim == 4:
        imagelist = [imagelist[i] for i in range(imagelist.shape[0])]

    if isinstance(imagelist, np.ndarray) and imagelist.ndim == 3:
        imagelist = [imagelist]

    if transpose_order:
        imagelist = [np.transpose(img, transpose_order) for img in imagelist]
        mask = np.transpose(mask, transpose_order)
    
    num_slices = imagelist[0].shape[0]
    nim = len(imagelist)

    if slices_to_plot_mode == "all":
        slices_to_plot = range(num_slices)
    elif slices_to_plot_mode == "largest_mask":
        slices_to_plot = [largest_mask_slice(mask)]
    elif slices_to_plot_mode == "mask_slices":
        slices_to_plot = [i for i in range(num_slices) if np.any(mask[i])]
    else:
        raise ValueError(f"Invalid slices_to_plot_mode: {slices_to_plot_mode}")

    if len(slices_to_plot) == 0:
        print("⚠️ No slices to show - the mask is empty.")
        return

    num_rows = len(slices_to_plot)
    num_cols = nim + 1  # images with overlay + one mask

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    if num_rows == 1:
        axs = np.expand_dims(axs, axis=0)
    if num_cols == 1:
        axs = np.expand_dims(axs, axis=1)

    for row, i in enumerate(slices_to_plot):
        for j, image in enumerate(imagelist):
            print(image.shape)
            img_slice = image[i]
            img_min, img_max = np.min(img_slice), np.max(img_slice)
            norm_image = (img_slice - img_min) / (img_max - img_min + 1e-8)
            rgb_image = np.stack([norm_image] * 3, axis=-1)

            # Create red overlay
            red_overlay = np.zeros_like(rgb_image)
            red_overlay[..., 0] = mask[i]

            axs[row, j].imshow(rgb_image)
            axs[row, j].imshow(red_overlay, cmap='Reds', alpha=alpha, vmin=0, vmax=1)
            axs[row, j].set_title(f"{imagenames[j]} (slice {i})")
            axs[row, j].axis("off")

        axs[row, -1].imshow(mask[i], cmap="gray", vmin=0, vmax=1)
        axs[row, -1].set_title(f"{maskname} slice {i}")
        axs[row, -1].axis("off")

    plt.tight_layout()
    if pathsave:
        os.makedirs(os.path.dirname(pathsave), exist_ok=True)
        plt.savefig(pathsave, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_images_and_masks(
    imagelist, 
    masklist,
    names, 
    subj, 
    pathsave=None, 
    slices_to_plot_mode="all", 
    transpose_order=None,
    alpha=0.5
):
    """
    Plot images with a red mask overlay plus a separate mask (for each combination).

    Parametere:
    - imagelist: Liste med 3D-bilder [(D, H, W), ...] eller 4D-array (N, D, H, W)
    - masklist: List of 3D masks [(D, H, W), ...] or one 3D mask
    - names: Navn på bildene
    - subj: Subjekt-ID
    - pathsave: Sti for lagring (PDF/png), None for visning
    - slices_to_plot_mode: 'all', 'largest_mask', or 'mask_slices'
    - transpose_order: f.eks. [2, 0, 1]
    """

    # Håndter arrays -> liste
    if isinstance(imagelist, np.ndarray) and imagelist.ndim == 4:
        imagelist = [imagelist[i] for i in range(imagelist.shape[0])]
    if isinstance(imagelist, np.ndarray) and imagelist.ndim == 3:
        imagelist = [imagelist]

    if isinstance(masklist, np.ndarray) and masklist.ndim == 4:
        masklist = [masklist[i] for i in range(masklist.shape[0])]
    if isinstance(masklist, np.ndarray) and masklist.ndim == 3:
        masklist = [masklist]

    if len(masklist) == 1 and len(imagelist) > 1:
        masklist = masklist * len(imagelist)  # same mask for all

    assert len(imagelist) == len(masklist), "imagelist and masklist must have the same length."

    if transpose_order:
        imagelist = [np.transpose(img, transpose_order) for img in imagelist]
        masklist = [np.transpose(msk, transpose_order) for msk in masklist]

    num_slices = imagelist[0].shape[0]
    nim = len(imagelist)

    # Bestem slices å vise
    if slices_to_plot_mode == "all":
        slices_to_plot = range(num_slices)
    elif slices_to_plot_mode == "largest_mask":
        # Find the slice with the largest mask (use the first mask)
        mask_vols = [np.sum(msk, axis=(1, 2)) for msk in masklist]
        max_idx = [np.argmax(vol) for vol in mask_vols]
        slices_to_plot = [max_idx[0]]
    elif slices_to_plot_mode == "mask_slices":
        slices_to_plot = sorted(set(
            i for msk in masklist for i in range(num_slices) if np.any(msk[i])
        ))
    else:
        raise ValueError(f"Invalid slices_to_plot_mode: {slices_to_plot_mode}")

    if len(slices_to_plot) == 0:
        print("⚠️ No slices to show - the masks are empty.")
        return

    num_rows = len(slices_to_plot)
    num_cols = nim * 2  # each image+mask pair gives 2 columns

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    if num_rows == 1 and num_cols == 1:
        axs = np.array([[axs]])
    elif num_rows == 1:
        axs = axs[np.newaxis, :]
    elif num_cols == 1:
        axs = axs[:, np.newaxis]

    for row, i in enumerate(slices_to_plot):
        for j, (image, mask) in enumerate(zip(imagelist, masklist)):
            img_slice = image[i]
            mask_slice = mask[i]

            img_min, img_max = np.min(img_slice), np.max(img_slice)
            if img_max - img_min < 1e-8:
                norm_image = np.zeros_like(img_slice)
            else:
                norm_image = (img_slice - img_min) / (img_max - img_min)

            rgb_image = np.stack([norm_image] * 3, axis=-1)

            # Column for image+mask
            ax_img = axs[row, j * 2]
            ax_img.imshow(rgb_image)
            ax_img.imshow(mask_slice, cmap='Reds', alpha=alpha, vmin=0, vmax=1)
            ax_img.set_title(f"{names[j]} (slice {i})")
            ax_img.axis("off")

            # Column for mask only
            ax_mask = axs[row, j * 2 + 1]
            ax_mask.imshow(mask_slice, cmap="gray", vmin=0, vmax=1)
            ax_mask.set_title(f"{names[j]} mask (slice {i})")
            ax_mask.axis("off")

    fig.suptitle(f'Subject {subj}')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if pathsave:
        os.makedirs(os.path.dirname(pathsave), exist_ok=True)
        plt.savefig(pathsave, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



