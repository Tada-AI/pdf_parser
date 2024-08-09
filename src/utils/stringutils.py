import re
from typing import Any, Callable, Dict, Iterable, TypeVar

space_characters = r"\p{Z}"
disallowed_characters = (
    r"""[^A-Za-z0-9`~!@#$%^&*()\-_+=\[\]\{\}\\\|;:'",\./?<>\n\t ©]"""
)

bullet_and_hyphen_characters = (
    r"[••◦◯▪◾■▫◆◇➤➲⁃–—\u2010\u2011\u2013\u2014\u2212\u2012]"
)


# TODO: Handle the case where people use 'o's for the bullet
def normalize_bullets_and_hyphens(text: str):
    # TODO: Roman numerals
    # TODO: Weird number characters
    return re.sub(bullet_and_hyphen_characters, "-", text)


l_parenthesis = r"（"
r_parenthesis = r"）"


def normalize_parens(text: str):
    text = re.sub(l_parenthesis, "(", text)
    return re.sub(r_parenthesis, ")", text)


single_quote = r"[\u2019\u2018\u02BC\u02B9\u02BB\u00B4\u0060\u2032]"
double_quote = r"[\u201C\u201D\u00AB\u00BB\u201E\u201F\u301D\u301E\u301F]"


def normalize_quotes(text: str):
    text = re.sub(single_quote, "'", text)
    return re.sub(double_quote, '"', text)


whitespace = r"[\u00A0\u2002\u2003\u2009\u200A\u200B\u2007\u2008\u00AD]"


def normalize_whitespace(text: str):
    return re.sub(whitespace, " ", text)


def is_whitespace(text: str):
    c = normalize_whitespace(normalize_newlines(text)).strip()
    return c == ""


def normalize_newlines(text: str):
    text = re.sub("\r\n|\n\r", "\n", text)
    return re.sub("\r", "\n", text)


def strip_invalid_characters(text: str):
    text = re.sub(disallowed_characters, "", text)
    return text


def normalize_text(text: str):
    text = normalize_whitespace(text)
    text = normalize_quotes(text)
    text = normalize_newlines(text)
    text = normalize_bullets_and_hyphens(text)
    text = strip_invalid_characters(text)
    return text.strip("\n")


def get_title(metadata) -> str | None:
    title = metadata.get("Title", None)
    if title is None:
        title = metadata.get("/Title", None)
    return title


def word_count(text: str):
    return len([word.strip() for word in text.split(" ") if word.strip() != ""])


page_num_re = r"^\d+$|^page ?\d+$"


def is_likely_page_num(text: str):
    return bool(re.match(page_num_re, text.strip(), re.IGNORECASE))


import re

extension_re = re.compile(r"(.*)\.[a-zA-Z0-9]{1,8}$")


def strip_file_extensions(title: str):
    result = extension_re.match(title.strip())
    if result:
        return result.group(1)
    else:
        return title


def get_title_from_path(path: str):
    last_item = path.split("/")[-1]
    return strip_file_extensions(last_item)
