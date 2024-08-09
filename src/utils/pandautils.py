import pandas as pd

from utils.normalization import Rect


def intersection_mask(df: pd.DataFrame, bbox: Rect):
    x0, y0, x1, y1 = bbox

    is_not_left_of_boundary = df["x1"] > x0
    is_not_right_of_boundary = df["x0"] < x1
    is_not_above_boundary = df["y0"] < y1
    is_not_below_boundary = df["y1"] > y0

    return (
        is_not_left_of_boundary
        & is_not_right_of_boundary
        & is_not_above_boundary
        & is_not_below_boundary
    )
