from dataclasses import dataclass
from typing import Sequence

from layout_analysis.recursive_lines import recursive_lines
from utils.pandautils import intersection_mask
from utils.normalization import Rect, X0, X1, Y0, Y1
from serializable_content_block import Location
import pandas as pd
import math


Range = tuple[float, float]


@dataclass
class CalculatedTableCell:
    location: Location
    cols: tuple[int, ...]
    rows: tuple[int, ...]
    text: str


@dataclass
class CalculatedTable:
    location: Location
    children: list["CalculatedTableCell"]


def closest_line_index(line: float, grid_lines: list[float]):
    closest = math.inf
    closest_index = -1

    for i, grid_line in enumerate(grid_lines):
        distance = abs(line - grid_line)
        if distance < closest:
            closest = distance
            closest_index = i

    return closest_index


@dataclass
class TableCellPosition:
    rows: tuple[int, ...]
    cols: tuple[int, ...]


from sklearn.cluster import DBSCAN


def snap_to_grid(rects: Sequence[Rect], row_tolerance: float, column_tolerance: float):
    dbscan = DBSCAN(row_tolerance * 1.1, min_samples=1)
    horizontal_lines = pd.DataFrame(
        [line for _, y0, _, y1 in rects for line in (y0, y1)], columns=["y"]
    )
    horizontal_lines["cluster"] = dbscan.fit_predict(horizontal_lines)
    horizontal_lines = horizontal_lines.groupby("cluster").agg({"y": "mean"})
    horizontal_lines = sorted(horizontal_lines["y"])

    dbscan = DBSCAN(column_tolerance * 1.1, min_samples=1)
    vertical_lines = pd.DataFrame(
        [line for x0, _, x1, _ in rects for line in (x0, x1)], columns=["x"]
    )
    vertical_lines["cluster"] = dbscan.fit_predict(vertical_lines)
    vertical_lines = vertical_lines.groupby("cluster").agg({"x": "mean"})
    vertical_lines = sorted(vertical_lines["x"])

    rows: list[tuple[int, ...]] = []
    cols: list[tuple[int, ...]] = []

    for rect in rects:
        col_start = closest_line_index(rect[X0], vertical_lines[:-1])
        col_end = closest_line_index(rect[X1], vertical_lines[1:]) + 1
        cols.append(tuple(range(col_start, col_end)))

        row_start = closest_line_index(rect[Y0], horizontal_lines[:-1])
        row_end = closest_line_index(rect[Y1], horizontal_lines[1:]) + 1
        rows.append(tuple(range(row_start, row_end)))

    return [TableCellPosition(rows=rows[i], cols=cols[i]) for i in range(len(rects))]


def calculate_table(
    table_df: pd.DataFrame, column_tolerance: float, row_tolerance: float
):
    table_page_num = table_df["page_num"].unique()
    rows_to_deleted: list[int] = []
    for index in table_df[table_df["text"] == "$"].index:
        if index + 1 < len(table_df):
            table_df.loc[index, "text"] += table_df.loc[index + 1, "text"]  # type: ignore
            table_df.loc[index, "x1"] = table_df.loc[index + 1, "x1"]
            table_df.loc[index, "y1"] = table_df.loc[index + 1, "y1"]
            rows_to_deleted.append(index + 1)

    for index in table_df[table_df["text"] == "%"].index:
        if index - 1 >= 0:
            table_df.loc[index - 1, "text"] += table_df.loc[index, "text"]  # type: ignore
            table_df.loc[index - 1, "x1"] = table_df.loc[index, "x1"]
            table_df.loc[index - 1, "y1"] = table_df.loc[index, "y1"]
            rows_to_deleted.append(index)

    table_df.drop(rows_to_deleted, inplace=True)

    table_page_num = table_df["page_num"].unique()

    cell_rects = recursive_lines(
        table_df,
        min_vertical_line_thickness=column_tolerance,
        min_horizontal_line_thickness=row_tolerance,
    )
    positions = snap_to_grid(cell_rects, row_tolerance, column_tolerance)
    children: list[CalculatedTableCell] = []
    for i, cell_rect in enumerate(cell_rects):
        text = " ".join(table_df[intersection_mask(table_df, cell_rect)]["text"])
        position = positions[i]
        children.append(
            CalculatedTableCell(
                cols=position.cols,
                rows=position.rows,
                location=Location(
                    page_num=table_page_num,  # type: ignore
                    x0=cell_rect[X0],
                    y0=cell_rect[Y0],
                    x1=cell_rect[X1],
                    y1=cell_rect[Y1],
                ),
                text=text,
            )
        )

    location = Location(
        x0=table_df["x0"].min(),
        y0=table_df["y0"].min(),
        x1=table_df["x1"].max(),
        y1=table_df["y0"].max(),
        page_num=table_page_num,  # type: ignore
    )
    return CalculatedTable(children=children, location=location)
