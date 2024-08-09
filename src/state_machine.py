from dataclasses import dataclass
from functools import reduce
from itertools import groupby
from typing import Union
import pandas as pd
import re

from file_parsing.parsers.layout_analysis.table_cell_detection import (
    CalculatedTable,
)
from shared.models.serializable_content_block import Location
from file_parsing.content_block import (
    ContentBlock,
    HeaderContentBlock,
    ListContentBlock,
    ListItemContentBlock,
    ParagraphContentBlock,
    RootContentBlock,
    TableCellContentBlock,
    TableContentBlock,
)

from .characters import unordered_list_prefix


def merge_locations(*locations: Location):
    location_by_page_num = groupby(locations, key=lambda l: l.page_num)
    return [
        reduce(lambda l1, l2: l1.merge(l2), page_locations)
        for _, page_locations in location_by_page_num
    ]


@dataclass
class RootState:
    node: RootContentBlock

    def act(self, action: "Action") -> "State":
        if isinstance(action, TableTokenAction):
            return TableState.create(action, self.node.parent)

        if isinstance(action, ListTokenAction):
            return ListState.create(action, self.node)

        if isinstance(action, ParagraphTokenAction):
            return ParagraphState.create(action, self.node)

        if isinstance(action, OutlierTokenAction):
            return OutlierState.create(action, self.node)


@dataclass
class HeaderState:
    font_class_id: int
    header_order: int
    page_num: int
    node: HeaderContentBlock

    @staticmethod
    def create(
        action: "OutlierTokenAction", parent: HeaderContentBlock | RootContentBlock
    ):
        while action.header_order <= parent.header_id:
            parent = parent.parent

        node = HeaderContentBlock(
            text=action.token,
            parent=parent,
            header_id=action.header_order,
            children=[],
            locations=[action.location],
        )

        parent.children.append(node)
        return HeaderState(
            font_class_id=action.font_class_id,
            header_order=action.header_order,
            page_num=action.location.page_num,
            node=node,
        )

    def act(self, action: "Action") -> "State":
        new_state = self._act(action)
        if new_state is not self:
            self.post_transition()
        return new_state

    def _act(self, action: "Action") -> "State":
        if isinstance(action, TableTokenAction):
            return TableState.create(action, self.node.parent)

        if isinstance(action, ParagraphTokenAction):
            return ParagraphState.create(action, self.node)
        if isinstance(action, ListTokenAction):
            return ListState.create(action, self.node)

        if isinstance(action, OutlierTokenAction):
            is_new_page = action.location.page_num != self.page_num
            is_new_font_class = action.font_class_id != self.font_class_id

            if is_new_page or is_new_font_class:
                return OutlierState.create(action, self.node)

            self.node.text += f" {action.token}"
            self.node.locations = merge_locations(*self.node.locations, action.location)
            return self

    def is_table_of_contents(self):
        return self.node.text.lower().replace(" ", "") == "tableofcontents"

    def post_transition(self):
        pass
        # if self.is_table_of_contents():
        #     self.node.delete_me = True


@dataclass
class ParagraphState:
    line_id: int
    cluster_id: int
    node: ParagraphContentBlock
    font_class_id: int

    @staticmethod
    def create(
        action: Union["ParagraphTokenAction", "OutlierTokenAction"],
        parent: HeaderContentBlock | RootContentBlock,
        prepend: str | None = None,
    ):
        text = action.token if prepend is None else f"{prepend} {action.token}"
        paragraph = ParagraphContentBlock(
            text, parent=parent, locations=[action.location]
        )
        parent.children.append(paragraph)
        return ParagraphState(
            node=paragraph,
            line_id=action.line_id,
            cluster_id=action.cluster_id,
            font_class_id=action.font_class_id,
        )

    def act(self, action: "Action") -> "State":
        if isinstance(action, TableTokenAction):
            return TableState.create(action, self.node.parent)

        if isinstance(action, ListTokenAction):
            if action.line_id != self.line_id:
                return ListState.create(action, self.node.parent)

        if isinstance(action, ParagraphTokenAction):
            if self.cluster_id != action.cluster_id:
                return ParagraphState.create(action, self.node.parent)

        if isinstance(action, OutlierTokenAction):
            if self.cluster_id != action.cluster_id:
                return OutlierState.create(action, parent=self.node.parent)

        self.node.text += f" {action.token}"
        self.line_id = action.line_id
        self.node.locations = merge_locations(*self.node.locations, action.location)
        return self


@dataclass
class OutlierState:
    page_num: int
    cluster_id: int
    font_class_id: int
    header_order: int
    parent: (
        HeaderContentBlock | RootContentBlock
    )  # The parent header for whatever is next
    locations: list[Location]

    text: str

    @staticmethod
    def create(
        action: "OutlierTokenAction", parent: HeaderContentBlock | RootContentBlock
    ):
        if re.match(unordered_list_prefix, action.token):
            return ListState.create(action, parent)
        if action.font_is_larger:
            return HeaderState.create(action, parent)
        elif action.font_is_smaller:
            return ParagraphState.create(action, parent)

        while action.header_order <= parent.header_id:
            parent = parent.parent

        return OutlierState(
            action.location.page_num,
            font_class_id=action.font_class_id,
            header_order=action.header_order,
            parent=parent,
            cluster_id=action.cluster_id,
            text=action.token,
            locations=[action.location],
        )

    def _create_header(self):
        header = HeaderContentBlock(
            self.text,
            self.parent,
            header_id=self.header_order,
            children=[],
            locations=self.locations,
        )
        self.parent.children.append(header)
        return header

    def act(self, action: "Action") -> "State":
        if isinstance(action, TableTokenAction):
            return TableState.create(action, self._create_header())

        if isinstance(action, ListTokenAction):
            return ListState.create(action, self._create_header())

        if isinstance(action, ParagraphTokenAction):
            if action.cluster_id == self.cluster_id:
                return ParagraphState.create(
                    action, self.parent, prepend=f"*{self.text}*"
                )
            return ParagraphState.create(action, self._create_header())

        if isinstance(action, OutlierTokenAction):
            is_new_font_class = action.font_class_id != self.font_class_id
            is_new_paragraph = action.cluster_id != self.cluster_id
            is_new_page = action.location.page_num != self.page_num

            if is_new_font_class or is_new_paragraph or is_new_page:
                header = self._create_header()
                return OutlierState.create(action, header)

            self.text += f" {action.token}"
            self.locations = merge_locations(*self.locations, action.location)
            return self


@dataclass
class ListState:
    list_item_id: int
    list_id: int
    node: ListItemContentBlock

    @staticmethod
    def create(
        action: Union["ListTokenAction", "OutlierTokenAction"],
        parent: HeaderContentBlock | RootContentBlock | ListContentBlock,
    ):
        list_block = ListContentBlock(
            parent=parent,
            indentation_level=action.list_indentation_level,
            children=[],
            locations=[action.location],
        )
        parent.children.append(list_block)

        list_item_block = ListItemContentBlock(
            action.token,
            parent=list_block,
            list_item_id=action.list_item_id,
            locations=[action.location],
        )
        list_block.children.append(list_item_block)

        return ListState(
            list_item_id=action.list_item_id,
            list_id=action.list_id,
            node=list_item_block,
        )

    def act(self, action: "Action") -> "State":
        if isinstance(action, TableTokenAction):
            return TableState.create(action, self.node.parent)

        if isinstance(action, ListTokenAction) or (
            isinstance(action, OutlierTokenAction) and action.list_id == self.list_id
        ):
            if self.list_item_id == action.list_item_id:
                self.node.text += f" {action.token}"
                self.node.locations = merge_locations(
                    *self.node.locations, action.location
                )
                self.node.parent.locations = merge_locations(
                    *self.node.parent.locations, action.location
                )
                return self

            parent = self.node.parent
            """
            3 Cases:
                - More deeply nested: Create a new list
                - Same level, create a new item
                - Deeper, move up the list to create a new item
            """
            # Case 1
            if action.list_indentation_level > parent.indentation_level:
                return ListState.create(action, self.node.parent)

            # Case 2/3
            # NOTE: There is an edge case where 2 lists back to back without any content between them are merged.
            #       Choosing this over splitting lists across pages for now
            while (
                parent.indentation_level > action.list_indentation_level
                and isinstance(parent.parent, ListContentBlock)
            ):
                parent = parent.parent

            list_item_block = ListItemContentBlock(
                action.token,
                parent=parent,
                list_item_id=action.list_item_id,
                locations=[action.location],
            )
            parent.children.append(list_item_block)
            self.node = list_item_block
            self.list_item_id = action.list_item_id
            self.node.locations = merge_locations(*self.node.locations, action.location)
            self.node.parent.locations = merge_locations(
                *self.node.parent.locations, action.location
            )

            return self

        parent = self.node.parent
        while isinstance(parent, ListContentBlock):
            parent = parent.parent

        if isinstance(action, ParagraphTokenAction):
            return ParagraphState.create(action, parent)

        if isinstance(action, OutlierTokenAction):
            return OutlierState.create(action, parent)


@dataclass
class TableState:
    parent: Union["HeaderContentBlock", "RootContentBlock"]

    @staticmethod
    def create(
        action: "TableTokenAction",
        current_node: Union["ContentBlock", "ListItemContentBlock", "RootContentBlock"],
    ):
        parent = current_node
        while not isinstance(parent, (HeaderContentBlock, RootContentBlock)):
            parent = parent.parent

        content_block = TableContentBlock(
            parent=parent,
            locations=[action.table.location],
            children=[
                TableCellContentBlock(
                    cell.location, cell.text, rows=cell.rows, cols=cell.cols
                )
                for cell in action.table.children
            ],
        )
        parent.children.append(content_block)
        return TableState(parent)

    def act(self, action: "Action") -> "State":
        if isinstance(action, TableTokenAction):
            return TableState.create(action, self.parent)

        if isinstance(action, ListTokenAction):
            return ListState.create(action, self.parent)

        if isinstance(action, ParagraphTokenAction):
            return ParagraphState.create(action, self.parent)

        if isinstance(action, OutlierTokenAction):
            return OutlierState.create(action, self.parent)


State = ParagraphState | HeaderState | OutlierState | ListState | TableState


## Actions
@dataclass
class ListTokenAction:
    token: str
    cluster_id: str
    line_id: int
    list_item_id: int
    list_id: int
    list_indentation_level: int
    location: Location


@dataclass
class OutlierTokenAction:
    token: str
    line_id: int
    font_class_id: int
    font_is_larger: int
    font_is_smaller: int
    cluster_id: int
    header_order: int
    list_item_id: int
    list_id: int
    list_indentation_level: int
    location: Location


@dataclass
class ParagraphTokenAction:
    token: str
    cluster_id: int
    line_id: int
    font_class_id: int
    location: Location


@dataclass
class TableTokenAction:
    table_id: int
    table: CalculatedTable


Action = ListTokenAction | OutlierTokenAction | ParagraphTokenAction | TableTokenAction


class ReconstructionStateMachine:
    def __init__(
        self, word_df: pd.DataFrame, tables: dict[int, CalculatedTable]
    ) -> None:
        self.word_df = word_df
        self.tables = tables

    def process(self):
        word_df = self.word_df
        root = RootContentBlock()

        ## State machine
        data_df = word_df[word_df["is_junk"] == False]
        data_df = data_df[
            [
                "text",
                "page_num",
                "list_item_id",
                "list_indentation_level",
                "list_id",
                "line_id",
                "font_class_id",
                "font_is_larger",
                "font_is_smaller",
                "header_order",
                "is_outlier",
                "cluster_id",
                "x0",
                "y0",
                "x1",
                "y1",
                "table_id",
            ]
        ]

        root = RootContentBlock()
        state = RootState(root)
        seen_tables: set[int] = set()

        for (
            token,
            page_num,
            list_item_id,
            list_id,
            list_indentation_level,
            line_id,
            font_class_id,
            font_is_larger,
            font_is_smaller,
            header_order,
            is_outlier,
            cluster_id,
            x0,
            y0,
            x1,
            y1,
            table_id,
        ) in data_df.itertuples(index=False):
            if table_id != -1 and table_id in seen_tables:
                continue

            location = Location(x0=x0, y0=y0, x1=x1, y1=y1, page_num=page_num)

            if table_id != -1:
                action = TableTokenAction(table_id, self.tables[table_id])
                seen_tables.add(table_id)
            elif is_outlier:
                action = OutlierTokenAction(
                    token,
                    line_id=line_id,
                    font_class_id=font_class_id,
                    header_order=header_order,
                    font_is_larger=font_is_larger,
                    font_is_smaller=font_is_smaller,
                    cluster_id=cluster_id,
                    list_item_id=list_item_id,
                    list_id=list_id,
                    list_indentation_level=list_indentation_level,
                    location=location,
                )
            elif list_id != -1:
                action = ListTokenAction(
                    token,
                    cluster_id=cluster_id,
                    list_item_id=list_item_id,
                    list_id=list_id,
                    line_id=line_id,
                    list_indentation_level=list_indentation_level,
                    location=location,
                )

            else:
                action = ParagraphTokenAction(
                    token,
                    line_id=line_id,
                    cluster_id=cluster_id,
                    font_class_id=font_class_id,
                    location=location,
                )

            try:
                state = state.act(action)
            except Exception as e:
                print(state)
                print(action)
                raise e

        return root
