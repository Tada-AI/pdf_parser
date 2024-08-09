from typing import Iterable, Sequence
import pandas as pd

from utils.normalization import X0, X1, Y0, Y1, Rect
from utils.pandautils import intersection_mask

Range = tuple[float, float]


def avg(*numbers: float):
    return sum(numbers) / len(numbers)


def subdivide_bbox(h_lines: list[float], v_lines: list[float], bounding_box: Rect):
    # Sort the lines
    h_lines = sorted(h_lines)
    v_lines = sorted(v_lines)

    # Add the bounding box edges as lines
    h_lines.insert(0, bounding_box[Y0])
    h_lines.append(bounding_box[Y1])
    v_lines.insert(0, bounding_box[X0])
    v_lines.append(bounding_box[X1])

    rectangles: list[Rect] = []
    # Iterate through the lines to create rectangles
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            top = h_lines[i]
            bottom = h_lines[i + 1]
            left = v_lines[j]
            right = v_lines[j + 1]
            rectangles.append((left, top, right, bottom))

    return rectangles


def merge_ranges(ranges: Iterable[Range]) -> list[Range]:
    """
    Sorts and merges the provided ranges
    """
    ranges = sorted(ranges, key=lambda r: r[0])
    if len(ranges) == 0:
        return []

    merged_ranges = []
    current_start, current_end = ranges[0]

    for start, end in ranges[1:]:
        if start <= current_end:  # Overlapping or adjacent ranges
            current_end = max(current_end, end)
        else:  # Non-overlapping range
            merged_ranges.append((current_start, current_end))
            current_start, current_end = start, end

    merged_ranges.append((current_start, current_end))

    return merged_ranges


def find_cuts(ranges: Sequence[Range], min_line_thickness=0.0):
    if len(ranges) == 0:
        return []
    merged_ranges = merge_ranges(ranges)

    # Find gaps between the merged ranges
    gaps = []
    for i in range(1, len(merged_ranges)):
        gaps.append((merged_ranges[i - 1][1], merged_ranges[i][0]))

    return [avg(start, end) for start, end in gaps if end - start > min_line_thickness]


def recursive_lines(
    bbox_df: pd.DataFrame,
    min_horizontal_line_thickness: float = 0,
    min_vertical_line_thickness: float = 0,
    depth=0,
    max_depth=10,
    bbox: Rect | None = None,
) -> list[Rect]:
    if len(bbox_df) == 0:
        return [] if bbox is None else [bbox]

    if bbox is None:
        x0: float = bbox_df["x0"].min()
        y0: float = bbox_df["y0"].min()
        x1: float = bbox_df["x1"].max()
        y1: float = bbox_df["y1"].max()
        bbox = (x0, y0, x1, y1)

    # Find horizontal and vertical cuts
    horizontal_cuts = find_cuts(
        bbox_df[["y0", "y1"]].to_numpy(),  # type: ignore
        min_line_thickness=min_horizontal_line_thickness,
    )
    # print(0,f"Depth {depth} - Horizontal Cuts:", horizontal_cuts)
    vertical_cuts = find_cuts(
        bbox_df[["x0", "x1"]].to_numpy(),  # type: ignore
        min_line_thickness=min_vertical_line_thickness,
    )
    # print(1,f"Depth {depth} - Vertical Cuts:", vertical_cuts)

    # print(3,f"Depth {depth} - Bbox:", bbox)

    if len(horizontal_cuts) == 0 and len(vertical_cuts) == 0:
        return [bbox]

    child_bboxes = subdivide_bbox(horizontal_cuts, vertical_cuts, bbox)

    if depth < max_depth:
        new_bboxes: list[Rect] = []
        for sub_region in child_bboxes:
            sub_df = bbox_df[intersection_mask(bbox_df, sub_region)]
            new_bboxes.extend(
                recursive_lines(
                    sub_df,
                    min_horizontal_line_thickness=min_horizontal_line_thickness,
                    min_vertical_line_thickness=min_vertical_line_thickness,
                    depth=depth + 1,
                    bbox=sub_region,
                )
            )

        return new_bboxes
    else:
        return child_bboxes


def invert_ranges(ranges: Sequence[Range], contained_within: Range):
    left, right = contained_within

    if left >= right:
        raise ValueError("left >= right")
    if len(ranges) == 0:
        return []

    ranges = [
        (max(start, left), min(stop, right))
        for start, stop in ranges
        if stop > left and start < right
    ]
    ranges = merge_ranges(ranges)
    ranges = [
        (start, end) for start, end in ranges if end - start > 0
    ]  # remove 0 width

    current_start, current_end = ranges[0]

    inverted_ranges: list[Range] = []
    if current_start != left:
        inverted_ranges.append((left, current_start))

    for start, end in ranges[1:]:
        inverted_ranges.append((current_end, start))
        current_end = end

    if current_end != right:
        inverted_ranges.append((current_end, right))

    return inverted_ranges


def recursive_xy_cut(
    bbox_df: pd.DataFrame,
    min_vertical_cut_thickness=0.1,
    min_horizontal_cut_thickness=0.1,
    depth: int = 0,
    max_depth=10,
    bbox: Rect | None = None,
) -> list[Rect]:
    if len(bbox_df) == 0:
        return [] if bbox is None else [bbox]

    if bbox is None:
        x0: float = bbox_df["x0"].min()
        y0: float = bbox_df["y0"].min()
        x1: float = bbox_df["x1"].max()
        y1: float = bbox_df["y1"].max()
        bbox = (x0, y0, x1, y1)

    # Find horizontal and vertical cuts
    horizontal_cuts = invert_ranges(
        bbox_df[["y0", "y1"]].to_numpy(),  # type: ignore
        (bbox[Y0], bbox[Y1]),
    )
    horizontal_cuts = [
        (start, end)
        for start, end in horizontal_cuts
        if abs(end - start) >= min_horizontal_cut_thickness
    ]
    # print(0,f"Depth {depth} - Horizontal Cuts:", horizontal_cuts)
    vertical_cuts = invert_ranges(
        bbox_df[["x0", "x1"]].to_numpy(),  # type: ignore
        (bbox[X0], bbox[X1]),
    )
    vertical_cuts = [
        (start, end)
        for start, end in vertical_cuts
        if abs(end - start) >= min_vertical_cut_thickness
    ]

    # Initialize regions list with the current region
    regions: list[Rect] = [bbox]

    # Perform horizontal cuts
    for start, end in horizontal_cuts:
        new_regions: list[Rect] = []
        for x1, y1, x2, y2 in regions:
            if y1 <= start <= y2 and y1 <= end <= y2:
                new_regions.append((x1, y1, x2, start))
                new_regions.append((x1, end, x2, y2))
            else:
                new_regions.append((x1, y1, x2, y2))
        regions = new_regions

    # Perform vertical cuts
    for start, end in vertical_cuts:
        new_regions: list[Rect] = []
        for x1, y1, x2, y2 in regions:
            if x1 <= start <= x2 and x1 <= end <= x2:
                new_regions.append((x1, y1, start, y2))
                new_regions.append((end, y1, x2, y2))
            else:
                new_regions.append((x1, y1, x2, y2))
        regions = new_regions

    # Filter out small regions
    regions = [(x0, y0, x1, y1) for x0, y0, x1, y1 in regions]

    # Debugging: Print regions

    # Recursively apply XY cut to each region if the depth is not too large
    if depth < max_depth:  # Adjust the depth limit as needed
        final_regions: list[Rect] = []
        for x0, y0, x1, y1 in regions:
            sub_df = bbox_df[
                (bbox_df["x0"] > x0)
                & (bbox_df["y0"] > y0)
                & (bbox_df["x1"] < x1)
                & (bbox_df["y1"] < y1)
            ]
            final_regions.extend(
                recursive_xy_cut(
                    sub_df,
                    min_vertical_cut_thickness=min_vertical_cut_thickness,
                    min_horizontal_cut_thickness=min_horizontal_cut_thickness,
                    depth=depth + 1,
                    max_depth=max_depth,
                    bbox=(x0, y0, x1, y1),
                )
            )
        return final_regions
    else:
        return regions
