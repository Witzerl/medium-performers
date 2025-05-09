import numpy as np


def z_score_normalization(layer, eps=1e-8):
    """ Applies z-score normalization on a layer.

    Args:
        layer: Input layer
        eps: Small value to avoid division by zero

    Returns:
        normalized_layer: Normalized layer
    """

    mean = np.mean(layer)
    std  = np.std(layer)

    normalized_layer = (layer - mean) / (std + eps)
    return normalized_layer

# TODO: Could crop over a labeled segment -> always some segmentation label in the patch
def random_crop_3d(volume, segmentation, crop_size, img_pad_value=0, seg_pad_value=0):
    """ Crops a random patch of the given size from the given volume and segmentation and pads if necessary.

     Args:
         volume: 4D numpy array representing the input volume
         segmentation: 4D numpy array representing the segmentation
         crop_size: size of the cropped patch
         img_pad_value: value to pad the image
         seg_pad_value: value to pad the segmentation

    Returns:
        cropped_volume: 4D numpy array representing the cropped volume
        cropped_segmentation: 4D numpy array representing the cropped segmentation
     """

    assert len(volume.shape) == 4, f'Volume must be 4D but is {len(volume.shape)}'
    assert len(segmentation.shape) == 4, f'Segmentation must be 4D but is {len(segmentation.shape)}'

    C, H, W, D = volume.shape
    ch, cw, cd = crop_size

    # Compute padding sizes
    pad_h = max(ch - H, 0)
    pad_w = max(cw - W, 0)
    pad_d = max(cd - D, 0)

    pad_volume = (
        (0,0),   # No padding for the channel dimension
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2),
        (pad_d // 2, pad_d - pad_d // 2))

    # Pad image and mask
    padded_volume = np.pad(volume, pad_volume, mode='constant', constant_values=img_pad_value)
    padded_segmentation = np.pad(segmentation, pad_volume, mode='constant', constant_values=seg_pad_value)

    # Random crop coordinates
    _, H2, W2, D2 = padded_volume.shape
    assert H2 >= ch and W2 >= cw and D2 >= cd, f'Crop size exceeds padded volume dimensions | {(H2, W2, D2)} < {(ch, cw, cd)}'

    start_h = np.random.randint(0, H2 - ch + 1)
    start_w = np.random.randint(0, W2 - cw + 1)
    start_d = np.random.randint(0, D2 - cd + 1)

    # Crop into volume and segmentation
    cropped_volume = padded_volume[:, start_h:start_h + ch, start_w:start_w + cw, start_d:start_d + cd]
    cropped_segmentation = padded_segmentation[:, start_h:start_h + ch, start_w:start_w + cw, start_d:start_d + cd]
    return cropped_volume, cropped_segmentation