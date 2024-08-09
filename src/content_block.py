from functools import cached_property
from typing import Union
import re
from cuid2 import cuid_wrapper
import math

from shared.models.serializable_content_block import (
    Location,
    SerializableHeaderContentBlock,
    SerializableListContentBlock,
    SerializableListItemContentBlock,
    SerializableParagraphContentBlock,
    SerializableRootContentBlock,
    SerializableTableCellContentBlock,
    SerializableTableContentBlock,
)

cuid = cuid_wrapper()


class RootContentBlock:
    header_id = -math.inf
    children: list["ContentBlock"]

    def __init__(self) -> None:
        self.children = []

    @property
    def parent(self):
        raise Exception("Attempted to access parent of RootContentBlock")

    @property
    def text(self):
        raise Exception("Attempted to access text of RootContentBlock")

    @text.setter
    def text(self, val):
        raise Exception("Attempted to set text of RootContentBlock")

    @cached_property
    def word_count(self) -> int:
        return sum(c.word_count for c in self.children)

    @cached_property
    def id(self):
        return cuid()

    def __str__(self) -> str:
        return f'RootContentBlock(id="{self.id}")'

    def to_serializable(self):
        return SerializableRootContentBlock(
            id=self.id,
            children=[child.to_serializable() for child in self.children],
            full_word_count=self.word_count,
        )


class HeaderContentBlock:
    def __init__(
        self,
        text: str,
        parent: Union["HeaderContentBlock", "RootContentBlock"],
        header_id: int,
        children: list["ContentBlock"],
        locations: list[Location],
    ):
        self.header_id = header_id
        self.text = text
        self.parent = parent
        self.children = children
        self.locations = locations

    @cached_property
    def word_count(self) -> int:
        return len(word_delimeter_re.split(self.text)) + sum(
            c.word_count for c in self.children
        )

    @cached_property
    def id(self):
        return cuid()

    def to_serializable(self):
        return SerializableHeaderContentBlock(
            id=self.id,
            text=self.text,
            children=[child.to_serializable() for child in self.children],
            full_word_count=self.word_count,
            locations=self.locations,
        )


word_delimeter_re = re.compile(r"\s+")


class ParagraphContentBlock:
    def __init__(
        self,
        text: str,
        parent: Union["HeaderContentBlock", "RootContentBlock"],
        locations: list[Location],
    ):
        self.text = text
        self.parent = parent
        self.locations = locations

    @cached_property
    def id(self):
        return cuid()

    @cached_property
    def word_count(self):
        return len(word_delimeter_re.split(self.text))

    def __str__(self) -> str:
        return f'ParagraphContentBlock(id="{self.id}", text="{self.text}")'

    def to_serializable(self):
        return SerializableParagraphContentBlock(
            id=self.id,
            text=self.text,
            full_word_count=self.word_count,
            locations=self.locations,
        )


class ListContentBlock:
    def __init__(
        self,
        parent: Union["HeaderContentBlock", "ListContentBlock", "RootContentBlock"],
        indentation_level: int,
        children: list[Union["ListItemContentBlock", "ListContentBlock"]],
        locations: list[Location],
    ):
        self.parent = parent
        self.indentation_level = indentation_level
        self.children = children
        self.locations = locations

    @cached_property
    def word_count(self) -> int:
        return sum(c.word_count for c in self.children)

    @cached_property
    def id(self):
        return cuid()

    def __str__(self) -> str:
        return f'ListContentBlock(id="{self.id}")'

    def to_serializable(self):
        return SerializableListContentBlock(
            id=self.id,
            children=[child.to_serializable() for child in self.children],
            full_word_count=self.word_count,
            locations=self.locations,
        )


class ListItemContentBlock:
    def __init__(
        self,
        text: str,
        parent: "ListContentBlock",
        list_item_id: int,
        locations: list[Location],
    ):
        self.text = text
        self.list_item_id = list_item_id
        self.parent = parent
        self.locations = locations

    @property
    def header_id(self):
        raise Exception("Attempting to access ListItemContentBlock.header_id")

    @cached_property
    def word_count(self):
        return len(word_delimeter_re.split(self.text))

    @cached_property
    def id(self):
        return cuid()

    def __str__(self) -> str:
        return f'ListItemContentBlock(id="{self.id}", text="{self.text}")'

    def to_serializable(self):
        return SerializableListItemContentBlock(
            id=self.id,
            text=self.text,
            full_word_count=self.word_count,
            locations=self.locations,
        )


class TableCellContentBlock:
    def __init__(
        self,
        location: Location,
        text: str,
        cols: tuple[int, ...],
        rows: tuple[int, ...],
    ):
        self.location = location
        self.text = text
        self.cols = cols
        self.rows = rows

    @cached_property
    def id(self):
        return cuid()

    @cached_property
    def word_count(self) -> int:
        return len(self.text.split(" "))

    def to_serializable(self):
        return SerializableTableCellContentBlock(
            id=self.id,
            full_word_count=self.word_count,
            location=self.location,
            text=self.text,
            cols=self.cols,
            rows=self.rows,
        )


class TableContentBlock:
    def __init__(
        self,
        parent: Union["HeaderContentBlock", "RootContentBlock"],
        locations: list[Location],
        children: list[TableCellContentBlock],
    ):
        self.parent = parent
        self.locations = locations
        self.children = children

    @cached_property
    def id(self):
        return cuid()

    @cached_property
    def word_count(self) -> int:
        return sum(c.word_count for c in self.children)

    def to_serializable(self):
        return SerializableTableContentBlock(
            id=self.id,
            full_word_count=self.word_count,
            locations=self.locations,
            children=[c.to_serializable() for c in self.children],
        )


ContentBlock = (
    HeaderContentBlock | ParagraphContentBlock | ListContentBlock | TableContentBlock
)
