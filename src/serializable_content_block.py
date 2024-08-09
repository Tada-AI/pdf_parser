import re
from typing import Any, Generator, Literal, Union
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

_cell_separators = re.compile(r"[\.\;\:] ")


class Location(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    page_num: int

    def merge(self, location: "Location"):
        if self.page_num != location.page_num:
            raise ValueError("Locations had different page_nums")
        return Location(
            x0=min(self.x0, location.x0),
            y0=min(self.y0, location.y0),
            x1=max(self.x1, location.x1),
            y1=max(self.y1, location.y1),
            page_num=self.page_num,
        )


class BaseSerializableContentBlock(BaseModel):
    model_config = model_config = ConfigDict(
        alias_generator=to_camel, populate_by_name=True
    )
    id: str
    full_word_count: int


class SerializableTableCellContentBlock(BaseSerializableContentBlock):
    type: Literal["TableCell"] = "TableCell"
    location: Location
    text: str
    rows: tuple[int, ...]
    cols: tuple[int, ...]

    def itercells(self):
        yield self.text


class SerializableTableContentBlock(BaseSerializableContentBlock):
    type: Literal["Table"] = "Table"
    locations: list[Location]
    children: list[SerializableTableCellContentBlock]

    def to_text(self):
        for cell in self.children:
            yield cell.text

    def itercells(self):
        for child in self.children:
            yield from child.itercells()

    def to_markdown(self, first_line_as_header=False) -> str:
        col_row_value: dict[tuple[int, int], str] = dict()
        for cell in self.children:
            for col in cell.cols:
                for row in cell.rows:
                    col_row_value[(col, row)] = cell.text.replace("|", r"\|").replace(
                        "\n", "<br>"
                    )

        max_row = max(max(cell.rows) for cell in self.children)
        max_col = max(max(cell.cols) for cell in self.children)

        rows = []
        start = 0
        if first_line_as_header:
            row = " | ".join(
                col_row_value.get((col_num, 0), "") for col_num in range(max_col + 1)
            )
            rows.append(f"| {row} |")
            row = " | ".join(["-------"] * (max_col + 1))
            rows.append(f"| {row} |")
            start = 1

        for row_num in range(start, max_row + 1):
            row = " | ".join(
                col_row_value.get((col_num, row_num), "")
                for col_num in range(max_col + 1)
            )
            rows.append(f"| {row} |")

        return "\n".join(rows)


class SerializableRootContentBlock(BaseSerializableContentBlock):
    type: Literal["Root"] = "Root"
    children: list[
        Union[
            "SerializableHeaderContentBlock",
            "SerializableListContentBlock",
            "SerializableParagraphContentBlock",
            "SerializableTableContentBlock",
        ]
    ] = Field(discriminator="type")

    def to_text(self):
        return "\n".join(line for c in self.children for line in c.to_text())

    def itercells(self):
        return iter([])


class SerializableHeaderContentBlock(BaseSerializableContentBlock):
    type: Literal["Header"] = "Header"
    locations: list[Location]
    text: str
    children: list[
        Union[
            "SerializableHeaderContentBlock",
            "SerializableListContentBlock",
            "SerializableParagraphContentBlock",
            "SerializableTableContentBlock",
        ]
    ] = Field(discriminator="type")

    def to_text(self) -> Generator[str, Any, None]:
        yield self.text
        for c in self.children:
            yield from c.to_text()

    def itercells(self):
        for c in _cell_separators.split(self.text):
            if isinstance(c, str) and c.strip() != "":
                yield c


class SerializableParagraphContentBlock(BaseSerializableContentBlock):
    type: Literal["Paragraph"] = "Paragraph"
    locations: list[Location]
    text: str

    def to_text(self) -> Generator[str, Any, None]:
        yield self.text

    def itercells(self):
        for c in _cell_separators.split(self.text):
            if isinstance(c, str) and c.strip() != "":
                yield c


class SerializableListContentBlock(BaseSerializableContentBlock):
    type: Literal["List"] = "List"
    locations: list[Location]
    children: list[
        Union["SerializableListItemContentBlock", "SerializableListContentBlock"]
    ] = Field(discriminator="type")

    def to_text(self) -> Generator[str, Any, None]:
        for c in self.children:
            yield from c.to_text()

    def itercells(self):
        return iter([])


class SerializableListItemContentBlock(BaseSerializableContentBlock):
    type: Literal["ListItem"] = "ListItem"
    locations: list[Location]
    text: str

    def to_text(self) -> Generator[str, Any, None]:
        yield self.text

    def itercells(self):
        for c in _cell_separators.split(self.text):
            if isinstance(c, str):
                yield c


SerializableContentBlock = (
    SerializableHeaderContentBlock
    | SerializableListContentBlock
    | SerializableListItemContentBlock
    | SerializableParagraphContentBlock
    | SerializableTableContentBlock
    | SerializableTableCellContentBlock
)


def iter_block(
    block: SerializableContentBlock, parent_id: str, depth: int
) -> Generator[tuple[str, int, SerializableContentBlock], Any, None]:
    yield (parent_id, depth, block)

    if isinstance(
        block,
        (
            SerializableHeaderContentBlock,
            SerializableRootContentBlock,
            SerializableListContentBlock,
        ),
    ):
        for child in block.children:
            yield from iter_block(child, block.id, depth + 1)
