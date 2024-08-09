from abc import abstractmethod

import asyncio
from content_block import RootContentBlock


import typing


class BaseParser:
    mime_to_suffix: dict[str, str]

    def __init__(self, mime_type: str) -> None:
        self.mime_type = mime_type

    @classmethod
    def supported_mime_types(cls):
        return list(cls.mime_to_suffix.keys())

    def get_suffix(self):
        return "." + self.mime_to_suffix[self.mime_type]

    def run_in_loop(self, storage_path):
        return asyncio.run(self.parse_from_storage(storage_path))

    @abstractmethod
    async def parse(self, file: typing.BinaryIO) -> RootContentBlock:
        raise NotImplemented()
