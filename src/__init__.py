import asyncio
from dataclasses import dataclass
from io import BytesIO
import typing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import math

from rtree.index import Index as RTreeIndex


from pdfminer.layout import LTPage, LTChar, LTTextContainer, LTTextLine
from pdfminer.high_level import extract_pages

from layout_analysis.recursive_lines import recursive_lines
from layout_analysis.table_cell_detection import (
    CalculatedTable,
    calculate_table,
)
from layout_analysis.layout_detection import LayoutDetectionResult
from utils.pandautils import intersection_mask
from state_machine import (
    ReconstructionStateMachine,
)
from content_block import (
    RootContentBlock,
)
from base_parser import BaseParser
from .characters import unordered_list_prefix, ordered_list_prefix
from utils.normalization import X0, X1, Y0, Y1, Rect, normalize_bbox, scale_bbox
from utils.colorutils import HSL, color_to_hsl, color_to_rbga_int
from utils.stringutils import (
    is_whitespace,
    normalize_newlines,
    normalize_text,
    normalize_whitespace,
)
from sklearn.feature_extraction.text import CountVectorizer

from pdf2image import convert_from_path

# pd.set_option("future.no_silent_downcasting", True)


@dataclass(frozen=True)
class CharMetadata:
    is_bold: bool
    rgba: int
    fontname: str
    hsl: HSL


def normalize_size(original_size: float):
    return round(original_size)


def char_is_bold(char: LTChar):
    fontname: str = char.fontname
    return "bold" in fontname.lower()


def char_is_empty(char: LTChar):
    return char_text(char).strip() == ""


def char_text(char: LTChar):
    return normalize_text(char.get_text())


def char_metadata(char: LTChar):
    return CharMetadata(
        rgba=color_to_rbga_int(char.graphicstate.ncolor or 0),
        is_bold=char_is_bold(char),
        fontname=char.fontname,
        hsl=color_to_hsl(char.graphicstate.ncolor or 0),
    )


def character_df_features(page: LTPage, char: LTChar):
    # PDFs use a y-axis that is inverted from Pillow and images
    x0, y0, x1, y1 = invert_y(normalize_bbox(page.bbox, char.bbox))

    metadata = char_metadata(char)
    h, s, l = metadata.hsl

    return {
        "char_id": id(char),
        "text": normalize_newlines(normalize_whitespace(char.get_text())),
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "is_bold": 1 if metadata.is_bold else 0,
        "h": h,
        "s": s,
        "l": l,
    }


def tl_characters(text_line: LTTextLine):
    for elem in text_line:
        if isinstance(elem, LTChar):
            yield elem


def page_text_containers(page: LTPage):
    for elem in page:
        if isinstance(elem, LTTextContainer):
            yield elem


def tc_elements(text_container: LTTextContainer):
    for elem in text_container:
        if isinstance(elem, LTChar):
            yield elem
        elif isinstance(elem, LTTextLine):
            yield elem


def tc_characters(text_container: LTTextContainer):
    for elem in tc_elements(text_container):
        if isinstance(elem, LTChar):
            yield elem
        elif isinstance(elem, LTTextLine):
            for v in tl_characters(elem):
                yield v


def page_characters(page: LTPage):
    for tc in page_text_containers(page):
        for char in tc_characters(tc):
            yield char


def invert_y(bbox: Rect) -> Rect:
    x0, y0, x1, y1 = bbox
    return (x0, 1 - y1, x1, 1 - y0)


def _generate_pdf_df(file: typing.BinaryIO) -> pd.DataFrame:
    pdf = extract_pages(BytesIO(file.read()))

    scaler = MinMaxScaler()
    pdf_df = pd.DataFrame(
        {**character_df_features(page, c), "page_num": i}
        for i, page in enumerate(pdf)
        for c in page_characters(page)
        if not is_whitespace(normalize_whitespace(c.get_text()))
    )
    if len(pdf_df) == 0:
        raise Exception("Empty PDF")
    pdf_df[["h", "s", "l"]] = scaler.fit_transform(pdf_df[["h", "s", "l"]])
    pdf_df["is_bold"] = pdf_df["is_bold"] / 3
    pdf_df["x_center"] = (pdf_df["x0"] + pdf_df["x1"]) / 2
    pdf_df["y_center"] = (pdf_df["y0"] + pdf_df["y1"]) / 2

    pdf_df["width"] = pdf_df["x1"] - pdf_df["x0"]
    pdf_df["height"] = pdf_df["y1"] - pdf_df["y0"]

    pdf_df.sort_values(by=["page_num", "y_center", "x0"], inplace=True)
    pdf_df.reset_index(inplace=True, drop=True)
    return pdf_df


@dataclass
class FontMetrics:
    kerning: float
    char_width: float
    features_class: pd.DataFrame


def _calculate_font_class_metrics(pdf_df: pd.DataFrame) -> FontMetrics:
    """
    Warning: Mutates the dataframe
    """
    SIG_FIGS_SIZE = 5
    SIG_FIGS_MULTIPLE = 10**SIG_FIGS_SIZE
    pdf_df["size"] = (pdf_df["height"] * SIG_FIGS_MULTIPLE).round() / SIG_FIGS_MULTIPLE

    ## Kerning
    def index_offset(index: int) -> int:
        return (-1) ** index * (math.floor((index - 1) / 2) + 1)

    features = pdf_df[["size", "is_bold", "h", "s", "l"]]
    features_class = pd.DataFrame(features.value_counts().reset_index(name="counts"))

    middle_index = round(len(features_class) / 2)

    # Ensure that the classes follow a somewhat normal
    locations = [middle_index + index_offset(i) for i in range(len(features_class))]
    features_class["font_class_id"] = locations
    # Reset to 0 since the locations adjustment causes a start index of 1 for some conditions
    features_class["font_class_id"] -= features_class["font_class_id"].min()

    merge_df = pdf_df[["text", "size", "is_bold", "h", "s", "l"]].merge(
        features_class, on=["size", "is_bold", "h", "s", "l"], how="left"
    )
    merge_df.index = pdf_df.index
    pdf_df["font_class_id"] = -1
    pdf_df["font_class_id"] = merge_df["font_class_id"].fillna(-1).astype(int)

    pdf_df["font_class_kerning"] = math.inf

    data_df = pdf_df[pdf_df["font_class_id"] != -1]
    for font_class_id, font_class_df in data_df.groupby("font_class_id"):
        font_class_id = int(font_class_id)  # type: ignore
        font_class_df = font_class_df.copy()

        font_class_df["delta_next_char"] = (
            font_class_df.groupby(["page_num"])["x0"].shift(-1) - font_class_df["x1"]
        )
        font_class_df["delta_prev_char"] = (
            font_class_df["x0"] - font_class_df.groupby(["page_num"])["x1"].shift()
        )

        # Overlapping chars have negative disatance. Set to 0
        font_class_df["delta_next_char"] = font_class_df["delta_next_char"].apply(
            lambda d: max(d, 0)
        )
        font_class_df["delta_prev_char"] = font_class_df["delta_prev_char"].apply(
            lambda d: max(d, 0)
        )

        font_class_df["font_class_kerning"] = font_class_df[
            ["delta_next_char", "delta_prev_char"]
        ].mean(axis=1, skipna=True)

        pdf_df.loc[font_class_df.index, "font_class_kerning"] = font_class_df[
            "font_class_kerning"
        ].quantile(0.8)
        pdf_df.loc[font_class_df.index, "font_size"] = font_class_df["size"].median()

    # Choose the 80th quantile. Majority of characters are near-overlapping. Trimming the tail gets rid of extremes
    kerning = pdf_df[pdf_df["font_class_kerning"] != math.inf][
        "font_class_kerning"
    ].quantile(0.8)
    char_width = pdf_df["width"].quantile(0.8)

    return FontMetrics(
        kerning=kerning, char_width=char_width, features_class=features_class
    )


@dataclass
class OutlierMetrics:
    outlier_lower_bound: float
    outlier_upper_bound: float
    min_font_size: float
    max_font_size: float


def _calculate_outlier_metrics(pdf_df: pd.DataFrame) -> OutlierMetrics:
    """
    Warning: mutates the dataframe
    """
    data_df = pdf_df[pdf_df["font_class_id"] != -1]
    format_id_std_dev = data_df["font_class_id"].std(ddof=0)
    format_id_mean = data_df["font_class_id"].mean()

    # print(format_id_mean, format_id_std_dev)
    pdf_df[pdf_df["font_class_id"] != -1]["font_class_id"].value_counts()
    outlier_lower_bound = format_id_mean - format_id_std_dev
    outlier_upper_bound = format_id_mean + format_id_std_dev

    pdf_df["is_outlier"] = (pdf_df["font_class_id"] < outlier_lower_bound) | (
        pdf_df["font_class_id"] > outlier_upper_bound
    )
    pdf_df[pdf_df["is_outlier"] == True]

    min_font_size = pdf_df[pdf_df["is_outlier"] == False]["size"].min()
    max_font_size = pdf_df[pdf_df["is_outlier"] == False]["size"].max()

    pdf_df["font_is_larger"] = pdf_df["size"] > max_font_size
    pdf_df["font_is_smaller"] = pdf_df["size"] < min_font_size

    return OutlierMetrics(
        outlier_lower_bound=outlier_lower_bound,
        outlier_upper_bound=outlier_upper_bound,
        min_font_size=min_font_size,
        max_font_size=max_font_size,
    )


async def _generate_word_df(
    pdf_df: pd.DataFrame, font_metrics: FontMetrics, outlier_metrics: OutlierMetrics
) -> pd.DataFrame:
    from rtree import index

    page_trees: dict[int, index.Index] = dict()

    latest_word_id = -1
    for current_page_num in pdf_df["page_num"].unique():
        page_df = pdf_df[pdf_df["page_num"] == current_page_num]
        word_bboxes = recursive_lines(
            page_df, min_vertical_line_thickness=font_metrics.kerning * 1.2
        )

        r_index = index.Index()
        for word_id, word_bbox in enumerate(word_bboxes):
            word_id = word_id + latest_word_id + 1
            r_index.insert(word_id + latest_word_id + 1, word_bbox)

        latest_word_id = word_id
        page_trees[current_page_num] = r_index
        await asyncio.sleep(0)

    # TODO: Split this up per-page
    pdf_df["word_id"] = [
        list(page_trees[int(page_num)].intersection((x, y, x, y)))[0]
        for x, y, page_num in pdf_df[["x_center", "y_center", "page_num"]].to_numpy()
    ]
    pdf_df.sort_values(by=["word_id", "x0"], inplace=True)

    word_df = pdf_df.groupby("word_id").agg(
        {
            "page_num": "max",
            "text": "".join,
            "x0": "min",
            "y0": "min",
            "x1": "max",
            "y1": "max",
            "font_class_id": lambda x: x.mode()[0],
        }
    )
    word_df = word_df.merge(font_metrics.features_class, on=["font_class_id"])
    word_df["x_center"] = (word_df["x0"] + word_df["x1"]) / 2
    word_df["y_center"] = (word_df["y0"] + word_df["y1"]) / 2
    word_df["font_is_larger"] = word_df["size"] > outlier_metrics.max_font_size
    word_df["font_is_smaller"] = word_df["size"] < outlier_metrics.min_font_size
    word_df["is_outlier"] = (
        word_df["font_class_id"] < outlier_metrics.outlier_lower_bound
    ) | (word_df["font_class_id"] > outlier_metrics.outlier_upper_bound)
    word_df["width"] = word_df["x1"] - word_df["x0"]
    word_df["height"] = word_df["y1"] - word_df["y0"]
    word_df["x_center"] = (word_df["x0"] + word_df["x1"]) / 2
    word_df["y_center"] = (word_df["y0"] + word_df["y1"]) / 2

    return word_df


async def _pipeline_pdf_to_word(pdf_df: pd.DataFrame):
    font_metrics = _calculate_font_class_metrics(pdf_df)
    outlier_metrics = _calculate_outlier_metrics(pdf_df)
    await asyncio.sleep(0)
    word_df = await _generate_word_df(pdf_df, font_metrics, outlier_metrics)
    return (word_df, font_metrics)


async def _pipeline_file_to_word(file: typing.BinaryIO):
    pdf_df = _generate_pdf_df(file)
    await asyncio.sleep(0)
    return await _pipeline_pdf_to_word(pdf_df)


async def _detect_tables(file: typing.BinaryIO, num_pages: int):
    tables: list[tuple[int, LayoutDetectionResult]] = []
    # convert_from_path is 1-indexed
    for page_num in range(1, num_pages + 1):
        pdf_image = convert_from_path(
            file.name, first_page=page_num, last_page=page_num
        )
        for table in detector.detect(pdf_image[0]):
            # TODO: Figure out why there are two imports
            tables.append((page_num, table))  # type: ignore
            await asyncio.sleep(0)

    return tables


def _calculate_space_width(word_df: pd.DataFrame):
    word_df["delta_next_word"] = (
        word_df.groupby(["page_num"])["x0"].shift(-1) - word_df["x1"]
    )
    word_df["delta_prev_word"] = (
        word_df["x0"] - word_df.groupby(["page_num"])["x1"].shift()
    )
    word_df["space_width"] = word_df[["delta_next_word", "delta_prev_word"]].mean(
        axis=1, skipna=True
    )
    return word_df["space_width"].quantile(0.9)


def _MUTATES_calculate_line_id(word_df: pd.DataFrame, space_width: float):
    page_trees: dict[int, RTreeIndex] = dict()

    latest_line_id = -1
    for current_page_num in word_df["page_num"].unique():
        page_df = word_df[word_df["page_num"] == current_page_num]
        line_bboxes = recursive_lines(
            page_df, min_vertical_line_thickness=space_width * 1.5
        )

        r_index = RTreeIndex()
        for line_id, line_bbox in enumerate(line_bboxes):
            line_id = line_id + latest_line_id + 1
            r_index.insert(line_id + latest_line_id + 1, line_bbox)

        latest_line_id = line_id
        page_trees[current_page_num] = r_index

    # We resort before generating the line_df so that text is in a consistent order for junk detection. We could wait to generate the line_df as well
    word_df["line_id"] = [
        list(page_trees[int(page_num)].intersection((x, y, x, y)))[0]
        for x, y, page_num in word_df[["x_center", "y_center", "page_num"]].to_numpy()
    ]
    word_df.sort_values(by=["line_id", "x0"], inplace=True)


def _generate_line_df(word_df: pd.DataFrame):
    line_df = (
        word_df.groupby("line_id")
        .agg(
            {
                "page_num": "max",
                "text": " ".join,
                "x0": "min",
                "y0": "min",
                "x1": "max",
                "y1": "max",
                "font_class_id": lambda x: x.mode()[0],
            }
        )
        .reset_index()
    )
    line_df["x_center"] = (line_df["x0"] + line_df["x1"]) / 2
    line_df["y_center"] = (line_df["y0"] + line_df["y1"]) / 2
    line_df["width"] = line_df["x1"] - line_df["x0"]
    line_df["height"] = line_df["y1"] - line_df["y0"]

    return line_df


def _pipeline_word_to_line(word_df: pd.DataFrame):
    space_width = _calculate_space_width(word_df)
    _MUTATES_calculate_line_id(word_df, space_width)
    line_df = _generate_line_df(word_df)
    return (line_df, space_width)


async def _calculate_median_line_spacing(line_df: pd.DataFrame):
    all_line_mean_spaces = []
    for current_page_num in line_df["page_num"].unique():
        page_df = line_df[line_df["page_num"] == current_page_num].copy()
        # line_bboxes = recursive_lines(page_df, min_horizontal_line_thickness=kerning * 1.2)

        line_bboxes = page_df[["x0", "y0", "x1", "y1"]].to_numpy()

        r_index = RTreeIndex()
        for i, bbox in enumerate(line_bboxes):
            r_index.insert(i, bbox)

        page_line_spacing = [float("NaN")] * len(line_bboxes)
        for i, (x0, y0, x1, y1) in enumerate(line_bboxes):
            # Coordinates are normalized
            neighbor_candidates = r_index.intersection((x0, y1, x1, 1))
            top_neighbors = sorted(
                (
                    line_bboxes[neighbor_i]
                    for neighbor_i in neighbor_candidates
                    if neighbor_i != i
                ),
                key=lambda bbox: bbox[Y0],
            )
            top_neighbor = None if len(top_neighbors) == 0 else top_neighbors[0]

            neighbor_candidates = r_index.intersection((x0, 0, x1, y0))
            bottom_neighbors = sorted(
                (
                    line_bboxes[neighbor_i]
                    for neighbor_i in neighbor_candidates
                    if neighbor_i != i
                ),
                key=lambda bbox: -bbox[Y1],
            )
            bottom_neighbor = (
                None if len(bottom_neighbors) == 0 else bottom_neighbors[0]
            )

            top_dist = None if top_neighbor is None else top_neighbor[Y0] - y1
            bottom_dist = None if bottom_neighbor is None else y0 - bottom_neighbor[Y1]

            dist = [top_dist, bottom_dist]
            dist = [v for v in dist if v is not None]
            dist = float("NaN") if len(dist) == 0 else min(v for v in dist)
            page_line_spacing[i] = dist

        all_line_mean_spaces.extend(page_line_spacing)

        page_df["line_spacing"] = page_line_spacing
        await asyncio.sleep(0)

    line_df["line_spacing"] = all_line_mean_spaces

    return line_df["line_spacing"].quantile(0.5)


async def _MUTATES_calculate_cluster_id(
    word_df: pd.DataFrame, line_df: pd.DataFrame, median_line_spacing: float
):
    page_trees: dict[int, RTreeIndex] = dict()

    latest_cluster_id = -1
    for current_page_num in line_df["page_num"].unique():
        page_df = line_df[line_df["page_num"] == current_page_num]
        cluster_bboxes = recursive_lines(
            page_df, min_horizontal_line_thickness=median_line_spacing * 1.1
        )

        # display(draw_on_image(pdf_images[current_page_num], cluster_bboxes))

        r_index = RTreeIndex()
        for cluster_id, line_bbox in enumerate(cluster_bboxes):
            cluster_id = cluster_id + latest_cluster_id + 1
            r_index.insert(cluster_id + latest_cluster_id + 1, line_bbox)

        latest_cluster_id = cluster_id
        page_trees[current_page_num] = r_index
        await asyncio.sleep(0)

    word_df["cluster_id"] = [
        list(page_trees[int(page_num)].intersection((x, y, x, y)))[0]
        for x, y, page_num in word_df[["x_center", "y_center", "page_num"]].to_numpy()
    ]
    line_df["cluster_id"] = [
        list(page_trees[int(page_num)].intersection((x, y, x, y)))[0]
        for x, y, page_num in line_df[["x_center", "y_center", "page_num"]].to_numpy()
    ]


async def _pipeline_word_to_cluster(word_df: pd.DataFrame):
    line_df, space_width = _pipeline_word_to_line(word_df)
    median_line_spacing = await _calculate_median_line_spacing(line_df)
    await _MUTATES_calculate_cluster_id(
        word_df=word_df, line_df=line_df, median_line_spacing=median_line_spacing
    )
    return line_df, space_width


async def _pipeline_calculate_tables(
    word_df: pd.DataFrame,
    tables: list[tuple[int, LayoutDetectionResult]],
    space_width: float,
    kerning: float,
):
    _MUTATES_generate_table_id(word_df, tables)
    return _calculate_table_cells(word_df, space_width=space_width, kerning=kerning)


def _MUTATES_generate_table_id(
    word_df: pd.DataFrame, tables: list[tuple[int, LayoutDetectionResult]]
):
    word_df["table_id"] = -1

    for table_id, (table_page_num, table) in enumerate(tables):
        mask = intersection_mask(word_df, table.bbox) & (
            word_df["page_num"] == table_page_num
        )
        word_df.loc[mask, "table_id"] = table_id


def _calculate_table_cells(
    word_df: pd.DataFrame, space_width: float, kerning: float
) -> dict[int, CalculatedTable]:
    column_tolerance = space_width * 1.2
    # TODO: Do smarter thing in future
    row_tolerance = kerning * 0.7

    tables: dict[int, CalculatedTable] = dict()
    table_ids = [
        table_id for table_id in word_df["table_id"].unique() if table_id != -1
    ]
    for table_id in table_ids:
        table_df = word_df[word_df["table_id"] == table_id][
            ["text", "x0", "y0", "x1", "y1", "page_num", "line_id"]
        ].copy()
        table_df.sort_values(by=["line_id", "x0"], inplace=True)
        table_df.reset_index(inplace=True)

        table = calculate_table(table_df, column_tolerance, row_tolerance)
        tables[table_id] = table

    return tables


def _MUTATES_resort_word_and_line_df(word_df: pd.DataFrame, line_df: pd.DataFrame):
    word_df.sort_values(by=["cluster_id", "line_id", "x0"], inplace=True)
    line_df.sort_values(by=["cluster_id", "line_id", "x0"], inplace=True)


def _MUTATES_identify_junk(word_df: pd.DataFrame, line_df: pd.DataFrame):
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 2))
    ngram_vectors = vectorizer.fit_transform(line_df["text"])

    feature_names = vectorizer.get_feature_names_out()
    ngram_df = pd.DataFrame(ngram_vectors.toarray(), columns=feature_names)  # type: ignore
    ngram_df.index = (
        line_df.index
    )  # Ensure the ngram vector can be mapped back to the line_df vector

    num_pages = line_df["page_num"].max()
    if num_pages < 5:
        min_clusters = 2
    elif num_pages < 12:
        min_clusters = round(num_pages * 0.4)
    else:
        min_clusters = round(num_pages * 0.35)
    # Cluster based on non-normalized L1 distance to take length of document into account
    dbscan = DBSCAN(eps=16, min_samples=min_clusters, metric="manhattan")
    ngram_df["text_similarity_id"] = dbscan.fit_predict(ngram_df)
    ngram_df["text"] = line_df["text"]

    features = line_df[["x_center", "y_center", "width", "height"]]
    dbscan = DBSCAN(eps=0.005, min_samples=min_clusters)
    ngram_df["location_similarity_id"] = dbscan.fit_predict(features)

    ngram_df["text"] = line_df["text"]
    ngram_df["page_num"] = line_df["page_num"]
    ngram_df["is_junk"] = False

    # A little jenky since it doesn't check the overlap of IDs, but good enough for now
    junk_mask = (ngram_df["text_similarity_id"] != -1) & (
        ngram_df["location_similarity_id"] != -1
    )
    ngram_df.loc[junk_mask, "is_junk"] = True
    line_df["is_junk"] = ngram_df["is_junk"]

    # Check for page numbers
    left_margin = word_df["x0"].min()
    bottom_margin = word_df["y0"].min()
    right_margin = word_df["x1"].max()
    top_margin = word_df["y1"].max()

    page_margin_bbox: Rect = (left_margin, bottom_margin, right_margin, top_margin)

    margin_with_buffer = scale_bbox(page_margin_bbox, 0.97)
    intersects_with_margin = (
        (line_df["x0"] < margin_with_buffer[X0])
        | (line_df["y0"] < margin_with_buffer[Y0])
        | (line_df["x1"] > margin_with_buffer[X1])
        | (line_df["y1"] > margin_with_buffer[Y1])
    )
    page_num_re = r"^\d+$|^page ?\d+(of|/)?\d+?$"
    line_df.loc[
        (intersects_with_margin & line_df["text"].str.match(page_num_re, case=False)),
        "is_junk",
    ] = True

    # Merge back into word_df
    merge_df = word_df[["line_id"]].merge(
        line_df[["is_junk", "line_id"]], on="line_id", how="left"
    )
    merge_df.index = word_df.index
    word_df["is_junk"] = merge_df["is_junk"].fillna(False)


def _MUTATES_generate_header_order(word_df: pd.DataFrame, features_class: pd.DataFrame):
    header_df = (
        word_df[
            (word_df["is_outlier"] == True)
            & (word_df["is_junk"] == False)
            & (word_df["table_id"] == -1)
        ]
        .groupby("line_id")
        .agg(
            {
                "page_num": "max",
                "text": " ".join,
                "x0": "min",
                "y0": "min",
                "x1": "max",
                "y1": "max",
                "font_class_id": lambda x: x.mode()[0],
            }
        )
    )
    header_df = header_df.merge(
        features_class,
        on=["font_class_id"],
        how="left",
    )

    header_class_df = (
        pd.DataFrame(header_df[["font_class_id", "size"]].value_counts())
        .sort_index()
        .reset_index()
    )

    header_class_df["num_appearances_score"] = 1 - (
        header_class_df["count"] / header_class_df["count"].max()
    )
    header_class_df["size_score"] = (
        header_class_df["size"] / header_class_df["size"].max()
    )

    header_class_df["order"] = -1
    header_appearance_order = dict(
        (font_class_id, idx)
        for idx, font_class_id in enumerate(header_df["font_class_id"].unique())
    )
    header_class_df["order"] = header_class_df["font_class_id"].apply(
        header_appearance_order.get
    )

    num_headers = len(header_class_df)
    header_class_df["order_score"] = (
        num_headers - header_class_df["order"]
    ) / num_headers
    header_class_df["score"] = (
        header_class_df["order_score"]
        + header_class_df["size_score"]
        + header_class_df["num_appearances_score"]
    )

    header_class_df.sort_values(by=["score"], ascending=False, inplace=True)
    header_class_df.reset_index(inplace=True, drop=True)

    word_df["header_order"] = -1

    for header_order, font_class_id in header_class_df[["font_class_id"]].itertuples():
        word_df.loc[word_df["font_class_id"] == font_class_id, "header_order"] = (
            header_order
        )


def _MUTATES_identify_list_items(word_df: pd.DataFrame, char_width: float):
    word_df["is_first_word_in_line"] = word_df["line_id"].diff() != 0

    word_df["is_unordered_list_item_label"] = word_df["text"].str.match(
        unordered_list_prefix
    )
    word_df["is_ordered_list_item_label"] = word_df["text"].str.match(
        ordered_list_prefix
    )
    word_df["is_list_item_label"] = (
        word_df["is_unordered_list_item_label"] | word_df["is_ordered_list_item_label"]
    ) & word_df["is_first_word_in_line"]

    word_df["left_margin_minus_label"] = (
        word_df[word_df["is_list_item_label"] == False]
        .groupby("line_id")["x0"]
        .transform("min")
    )

    line_df = word_df.groupby("line_id").agg(
        {
            "page_num": "max",
            "text": " ".join,
            "is_list_item_label": any,
            "left_margin_minus_label": "min",
            "x0": "min",
            "y0": "min",
            "x1": "max",
            "y1": "max",
        }
    )
    line_df["aligned_with_prev_line"] = line_df["left_margin_minus_label"].diff().abs() < char_width  # type: ignore
    line_df["aligned_with_prev_line"].fillna(False)

    list_items_df = line_df[line_df["is_list_item_label"] == True]

    line_df["list_item_id"] = -1
    for list_id, idx in enumerate(list_items_df.index):
        list_item_idxes = [idx]
        for line_idx, aligned_with_prev_line in line_df.loc[idx + 1 :][
            ["aligned_with_prev_line"]
        ].itertuples():
            if not aligned_with_prev_line:
                break
            list_item_idxes.append(line_idx)

        line_df.loc[list_item_idxes, "list_item_id"] = list_id

    word_df["list_item_id"] = word_df[["line_id"]].merge(
        line_df[["list_item_id"]], left_on="line_id", right_index=True, how="left"
    )["list_item_id"]


def _MUTATES_identify_list_id_and_indentation(word_df: pd.DataFrame, char_width: float):
    word_df["last_in_list"] = (word_df["list_item_id"].shift(-1) == -1) & (
        word_df["list_item_id"] != -1
    )
    # Deal with the final row having NaN when shifted
    if word_df.iloc[-1]["list_item_id"] != -1:
        word_df.at[word_df.index[-1], "last_in_list"] = True

    list_item_df = (
        word_df[word_df["list_item_id"] != -1]
        .groupby("list_item_id")
        .agg(
            {
                "page_num": "max",
                "text": lambda x: " ".join(x),
                "x0": "min",
                "y0": "min",
                "x1": "max",
                "y1": "max",
                "last_in_list": any,
            }
        )
    )

    # TODO: Replace with space_width
    line_indentation_tolerance = char_width * 2

    list_item_df["list_id"] = (
        (list_item_df["last_in_list"].shift(1).fillna(False)).cumsum().astype(int)
    )

    dbscan = DBSCAN(eps=line_indentation_tolerance, min_samples=1)
    list_item_df["list_indentation_level"] = 0
    for _, list_df in list_item_df.groupby("list_id"):
        features = list_df[["x0"]]
        cluster_ids = dbscan.fit_predict(features)
        list_item_df.loc[list_df.index, "list_indentation_level"] = cluster_ids

    merge_df = word_df[["list_item_id"]].merge(
        list_item_df[["list_id", "list_indentation_level"]],
        left_on="list_item_id",
        right_index=True,
        how="left",
    )
    merge_df.index = word_df.index
    word_df["list_id"] = merge_df["list_id"].fillna(-1).astype(int)
    word_df["list_indentation_level"] = (
        merge_df["list_indentation_level"].fillna(-1).astype(int)
    )


def _pipeline_list_calculation(word_df: pd.DataFrame, char_width: float):
    _MUTATES_identify_list_items(word_df=word_df, char_width=char_width)
    _MUTATES_identify_list_id_and_indentation(word_df=word_df, char_width=char_width)


"""
file_to_word_pipeline
    file_to_pdf_pipeline
        file -> pdf_df

    pdf_to_word_pipeline
        pdf_df -> font_metrics
        pdf_df, font_metrics(implicit) -> outlier_metrics
        pdf_df, font_metrics, outlier_metrics -> word_df

word_to_line_pipeline
    word_df -> space_width
    word_df, space_width -> line_df

file_to_table_pipeline
    file -> page_images
    page_image -> list[(page_num,table_bbox)]

word_to_tables_pipeline
    word_df, bboxes -> 
"""


class PdfParser(BaseParser):
    mime_to_suffix = {"application/pdf": "pdf"}

    async def parse(self, file: typing.BinaryIO) -> RootContentBlock:
        # Can technically run together, but they're both CPU heavy operations right now.
        word_df, font_metrics = await _pipeline_file_to_word(file)
        num_pages = word_df["page_num"].max() + 1
        layout_detection_results = await _detect_tables(file, num_pages)
        line_df, space_width = await _pipeline_word_to_cluster(word_df)
        tables = await _pipeline_calculate_tables(
            word_df,
            layout_detection_results,
            space_width=space_width,
            kerning=font_metrics.kerning,
        )
        _MUTATES_resort_word_and_line_df(word_df, line_df)
        _MUTATES_identify_junk(word_df=word_df, line_df=line_df)
        _MUTATES_generate_header_order(
            word_df=word_df, features_class=font_metrics.features_class
        )
        _pipeline_list_calculation(word_df=word_df, char_width=font_metrics.char_width)

        state_machine = ReconstructionStateMachine(word_df, tables)
        return state_machine.process()


from layout_analysis.layout_detection import (
    LayoutDetector,
    ModelReference,
)


detector = LayoutDetector(
    ModelReference(
        "../models/picodet_lcnet_x1_0_fgd_layout_table_infer/",
        {0: "Table"},
    )
)
