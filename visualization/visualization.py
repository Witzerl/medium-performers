import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_inputs_grid_with_optional_segmentation(case_data, slice_idx=None, overlay_segmentation=False):
    """
    Plot available input modalities from case_data in a 1-row grid,
    optionally overlaying filled segmentation labels using fixed colors.
    """
    modalities = ['t1n', 't1c', 't2w', 't2f']
    available = [mod for mod in modalities if case_data.get(mod) is not None]

    n = len(available)
    if n == 0:
        print("No available modalities to plot!")
        return

    fig, axs = plt.subplots(1, n, figsize=(5 * n, 5))

    if n == 1:
        axs = [axs]  # make iterable

    # Fixed colors per label
    label_colors = {
        1: (1.0, 0.0, 0.0, 0.4),  # red - NCR
        2: (0.0, 1.0, 0.0, 0.3),  # green - SNFH
        3: (1.0, 0.5, 0.0, 0.3),  # orange - ET
        4: (0.0, 1.0, 1.0, 0.3),  # cyan - RC
    }

    for idx, mod in enumerate(available):
        volume = case_data[mod]
        if slice_idx is None:
            slice_idx = volume.shape[2] // 2

        slice_img = volume[:, :, slice_idx]

        axs[idx].imshow(slice_img, cmap='gray')
        axs[idx].set_title(mod.upper())
        axs[idx].axis('off')

        # Overlay filled segmentation
        if overlay_segmentation and case_data.get('seg') is not None:
            seg = case_data['seg']
            seg_slice = seg[:, :, slice_idx]

            for label, rgba in label_colors.items():
                mask = (seg_slice == label)
                if np.any(mask):
                    axs[idx].imshow(np.ma.masked_where(~mask, mask), cmap=mcolors.ListedColormap([rgba]), interpolation='none')

    plt.tight_layout()
    plt.show()
