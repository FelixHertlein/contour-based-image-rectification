import numpy as np


def tight_crop_image(image: np.ndarray, mask: np.ndarray, return_mask: bool = False):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    image_crop = image[row_min:row_max, col_min:col_max]
    mask_crop = mask[row_min:row_max, col_min:col_max]

    return (image_crop, mask_crop) if return_mask else image_crop
