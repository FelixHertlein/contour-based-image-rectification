from pathlib import Path
from typing import *
import numpy as np

import cv2


def load_image(file: Path):
    return cv2.imread(str(file), cv2.IMREAD_COLOR)


def save_image(file: Path, data: np.ndarray, override: bool = False):
    exist = None if override else False
    assert data.dtype == np.uint8
    assert len(data.shape) in [2, 3]
    if len(data.shape) == 3:
        assert data.shape[2] == 3

    cv2.imwrite(str(file), data)
