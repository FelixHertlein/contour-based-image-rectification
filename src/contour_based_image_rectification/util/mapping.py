from typing import *

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.interpolate import (
    RegularGridInterpolator,
    griddata,
)

from .misc import check_tensor


def create_identity_map(resolution: Union[int, Tuple], with_margin: bool = False):
    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    margin_0 = 0.5 / resolution[0] if with_margin else 0
    margin_1 = 0.5 / resolution[1] if with_margin else 0
    return np.mgrid[
        margin_0 : 1 - margin_0 : complex(0, resolution[0]),
        margin_1 : 1 - margin_1 : complex(0, resolution[1]),
    ].transpose(1, 2, 0)


def apply_map(
    image: np.ndarray,
    bm: np.ndarray,
    resolution: Union[None, int, Tuple[int, int]] = None,
):
    check_tensor(image, "h w c")
    check_tensor(bm, "h w c", c=2)

    if resolution is not None:
        bm = scale_map(bm, resolution)

    input_dtype = image.dtype
    img = rearrange(image, "h w c -> 1 c h w")
    img = torch.from_numpy(img).double()

    bm = torch.from_numpy(bm).unsqueeze(0).double()
    bm = (bm * 2) - 1
    bm = torch.roll(bm, shifts=1, dims=-1)

    res = F.grid_sample(input=img, grid=bm, align_corners=True)
    res = rearrange(res[0], "c h w -> h w c")
    res = res.numpy().astype(input_dtype)
    return res


def scale_map(input_map: np.ndarray, resolution: Union[int, Tuple[int, int]]):
    try:
        resolution = (int(resolution), int(resolution))
    except (ValueError, TypeError):
        resolution = tuple(int(v) for v in resolution)

    H, W, C = input_map.shape
    if H == resolution[0] and W == resolution[1]:
        return input_map.copy()

    if np.any(np.isnan(input_map)):
        print(
            "WARNING: scaling maps containing nan values will result in unsteady borders!"
        )

    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    xi = create_identity_map(resolution, with_margin=False).reshape(-1, 2)

    interp = RegularGridInterpolator((y, x), input_map, method="linear")
    return interp(xi).reshape(*resolution, C)
