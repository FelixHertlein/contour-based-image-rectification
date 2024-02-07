from typing import *

import numpy as np
import pandas as pd
import cv2
from .util.mapping import apply_map, scale_map
from scipy.optimize import least_squares
from shapely.affinity import scale
from shapely.geometry import (
    Polygon,
    Point,
    LineString,
    MultiLineString,
    MultiPoint,
    box,
)
from shapely.ops import nearest_points, split, linemerge, snap
from shapely.validation import make_valid


def rectify(
    *,
    image: np.ndarray,
    mask: np.ndarray,
    output_shape: Tuple[int, int],
    corners: Optional[List[Tuple[int, int]]] = None,
    resolution_map: int = 512,
) -> np.ndarray:
    """
    Rectifies an image based on the given mask. Corners can be provided to speed up the rectification.

    Args:
        image (np.ndarray): The input image as a NumPy array. Shape: (height, width, channels).
        mask (np.ndarray): The mask indicating the region of interest as a NumPy array. Shape: (height, width).
        output_shape (Tuple[int, int]): The desired output shape of the rectified image. First element: height, second element: width.
        corners (Optional[List[Tuple[int, int]]], optional): The corners of the region of interest. List of points with first x, then y coordinate. Has to contain exactly 4 points. Defaults to None.
        resolution_map (int, optional): The resolution of the mapping grid. Higher grid size increases computation time but generates better results.  Defaults to 512.

    Returns:
        np.ndarray: The rectified image as a NumPy array.
    """

    h, w, c = image.shape
    mh, mw = mask.shape

    assert c == 3, "Only RGB images are supported!"
    assert image.dtype == np.uint8, "Only uint8 images are supported!"
    assert mask.dtype == np.bool_, "Only boolean masks are supported!"
    assert corners is None or len(corners) == 4, "Invalid number of corners!"

    segments = _find_segments(mask, corners)

    backward_map = _create_backward_map(segments, resolution_map)

    new_image = apply_map(image, scale_map(backward_map, output_shape))

    return new_image


def _find_segments(
    mask: np.ndarray, corners: Optional[List[Tuple[int, int]]]
) -> Dict[str, LineString]:
    polygon = _find_polygon(mask)

    if corners is None:
        corners = _find_corners(polygon)
    else:
        h, w = mask.shape
        corners = [Point(x / w, y / h) for x, y in corners]
        corners = _snap_corners(polygon, corners)

    segments = _split_polygon(polygon, corners)
    segments = _classify_segments(segments, corners)
    return segments


def _find_polygon(mask: np.ndarray) -> Polygon:
    height, width = mask.shape
    contours, _ = cv2.findContours(
        mask.astype("uint8") * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=lambda x: x.shape[0])
    contour = np.squeeze(contour).astype("float")
    contour[:, 0] /= width
    contour[:, 1] /= height

    polygon = Polygon(contour)
    polygon = make_valid(polygon)

    if not isinstance(polygon, Polygon):
        polygon = max(polygon.geoms, key=lambda x: x.area)
        print(f"WARNING: Multipolygon found! Largest polygon will be used!")
        assert isinstance(polygon, Polygon)

    polygon = scale(geom=polygon, xfact=0.998, yfact=0.998, origin="center")
    return polygon


def _find_corners(polygon: Polygon) -> List[Point]:
    def optimize(x):
        x = Polygon(x.reshape(-1, 2))
        if not x.is_valid:
            return float("inf")

        iou = x.intersection(polygon).area / x.union(polygon).area
        return 1 - iou

    x0 = np.array(polygon.minimum_rotated_rectangle.boundary.coords[:-1]).reshape(-1)
    res = least_squares(optimize, x0, ftol=1e-5)

    estimate = Polygon(res["x"].reshape(-1, 2))

    while not estimate.contains(polygon):
        estimate = scale(estimate, 1.01, 1.01, origin=polygon.centroid)

    corners = [Point(*point) for point in list(estimate.exterior.coords)[:-1]]

    return _snap_corners(polygon, corners)


def _snap_corners(polygon: Polygon, corners: List[Point]) -> List[Point]:
    corners = [nearest_points(polygon, point)[0] for point in corners]
    assert len(corners) == 4
    return corners


def _split_polygon(polygon: Polygon, corners: List[Point]) -> List[LineString]:
    boundary = snap(polygon.boundary, MultiPoint(corners), 0.0000001)
    segments = split(boundary, MultiPoint(corners)).geoms
    assert len(segments) in [4, 5]

    if len(segments) == 4:
        return segments

    return [
        linemerge([segments[0], segments[4]]),
        segments[1],
        segments[2],
        segments[3],
    ]


def _classify_segments(
    segments: Dict[str, LineString], corners: List[Point]
) -> Dict[str, LineString]:
    bbox = box(*MultiLineString(segments).bounds)
    bbox_points = np.array(bbox.exterior.coords[:-1])

    # classify bbox nodes
    df = pd.DataFrame(data=bbox_points, columns=["bbox_x", "bbox_y"])
    name_x = (df.bbox_x == df.bbox_x.min()).replace({True: "left", False: "right"})
    name_y = (df.bbox_y == df.bbox_y.min()).replace({True: "top", False: "bottom"})
    df["name"] = name_y + "-" + name_x
    assert len(df.name.unique()) == 4

    # find bbox node to corner node association
    approx_points = np.array([c.xy for c in corners]).squeeze()
    assert approx_points.shape == (4, 2)

    assignments = [
        np.roll(np.array(range(4))[::step], shift=i)
        for i in range(4)
        for step in [1, -1]
    ]

    costs = [
        np.linalg.norm(bbox_points - approx_points[assignment], axis=-1).sum()
        for assignment in assignments
    ]

    min_assignment = min(zip(costs, assignments))[1]
    df["corner_x"] = approx_points[min_assignment][:, 0]
    df["corner_y"] = approx_points[min_assignment][:, 1]

    # retrieve correct segment and fix direction if necessary
    segment_endpoints = {frozenset([s.coords[0], s.coords[-1]]): s for s in segments}

    def get_directed_segment(start_name, end_name):
        start = df[df.name == start_name]
        start = (float(start.corner_x), float(start.corner_y))

        end = df[df.name == end_name]
        end = (float(end.corner_x), float(end.corner_y))

        segment = segment_endpoints[frozenset([start, end])]
        if start != segment.coords[0]:
            segment = LineString(reversed(segment.coords))

        return segment

    return {
        "top": get_directed_segment("top-left", "top-right"),
        "bottom": get_directed_segment("bottom-left", "bottom-right"),
        "left": get_directed_segment("top-left", "bottom-left"),
        "right": get_directed_segment("top-right", "bottom-right"),
    }


def _create_backward_map(
    segments: Dict[str, LineString], resolution: Tuple[int, int]
) -> np.ndarray:
    # map geometric objects to normalized space
    coord_map = {
        (0, 0): segments["top"].coords[0],
        (1, 0): segments["top"].coords[-1],
        (0, 1): segments["bottom"].coords[0],
        (1, 1): segments["bottom"].coords[-1],
    }
    dst = np.array(list(coord_map.keys())).astype("float32")
    src = np.array(list(coord_map.values())).astype("float32")
    M = cv2.getPerspectiveTransform(src=src, dst=dst)

    polygon = Polygon(linemerge(segments.values()))
    pts = np.array(polygon.exterior.coords).reshape(-1, 1, 2)
    pts = cv2.perspectiveTransform(pts, M).squeeze()
    p_norm = Polygon(pts)

    def transform_line(line):
        line_points = np.array(line.coords).reshape(-1, 1, 2)
        line_points = cv2.perspectiveTransform(line_points, M).squeeze()
        return LineString(line_points)

    h_norm = [transform_line(segments[name]) for name in ["top", "bottom"]]
    v_norm = [transform_line(segments[name]) for name in ["left", "right"]]

    (minx, miny, maxx, maxy) = p_norm.buffer(1).bounds

    res = np.zeros((resolution, resolution, 2))
    key_points = np.linspace(0, 1, resolution)
    edge_cases = {0: 0, resolution - 1: -1}

    # calculate y - values by interpolating x values
    for idx, keypoint in enumerate(key_points):
        if idx in edge_cases:
            pos = edge_cases[idx]
            points = [Point(v_norm[0].coords[pos]), Point(v_norm[1].coords[pos])]
        else:
            divider = LineString([(minx, keypoint), (maxx, keypoint)])
            points = list(MultiLineString(v_norm).intersection(divider).geoms)

        if len(points) > 2:
            points = [min(points, key=lambda p: p.x), max(points, key=lambda p: p.x)]

        assert len(points) == 2, "Unexpected number of points!"

        res[idx, :, 0] = np.linspace(points[0].x, points[1].x, resolution)

    # calculate x - values by interpolating y values
    for idx, keypoint in enumerate(key_points):
        if idx in edge_cases:
            pos = edge_cases[idx]
            points = [Point(h_norm[0].coords[pos]), Point(h_norm[1].coords[pos])]
        else:
            divider = LineString([(keypoint, miny), (keypoint, maxy)])
            points = list(MultiLineString(h_norm).intersection(divider).geoms)

        if len(points) > 2:
            points = [min(points, key=lambda p: p.y), max(points, key=lambda p: p.y)]

            assert len(points) == 2, "Unexpected number of points!"

        res[:, idx, 1] = np.linspace(points[0].y, points[1].y, resolution)

    # transform grid back to original space
    pts = res.reshape(-1, 1, 2)
    pts = cv2.perspectiveTransform(pts, np.linalg.inv(M)).squeeze()
    res = pts.reshape(resolution, resolution, 2)
    res = np.roll(res, shift=1, axis=-1)  # flip x and y coordinate: first y, then x
    return res
