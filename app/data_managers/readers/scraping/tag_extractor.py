import re
from typing import Any, Optional, Union

from bs4 import BeautifulSoup as Bs
from bs4.element import ResultSet, Tag

from ...namespaces import scraper_ns


class TagExtractor:
    """Base Interface for extracting elements from markup Tags"""

    def _to_bs(self, raw_html: str, parser: str = scraper_ns.DEFAULT_PARSER) -> Tag:
        """Creates BeautifulSoup object from raw html"""
        parsed_markup = Bs(raw_html, parser)
        return parsed_markup

    def save_markup(self, markup: Tag, file_path: str) -> None:
        """Saves markup Tag or BeautifulSoup object to the file"""
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(str(markup))

    def read_bs_from_file(
        self, path: str, parser: str = scraper_ns.DEFAULT_PARSER
    ) -> Tag:
        """Reads markup from the file and returns BeautifulSoup object"""
        with open(path, "r", encoding="utf8") as file:
            parsed_markup = Bs(file, parser)
        return parsed_markup

    def _find(
        self,
        markup: Tag,
        tag_name: Optional[str] = None,
        all: bool = False,
        **attrs: dict,
    ) -> Any:
        """
        Uses BeautifulSoup find or find_all methods to search for
        specific tags. Returns Tag or ResultSet

         Parameters:
            markup : Tag
                BeautifulSoup object
            tag_name : str, optional
                Filter by tag name
            all : bool, default = True
                if False uses find method instead of find_all and
                returns Tag, else ResultSet
        """
        if "class_" in attrs:
            attrs["class"] = attrs.pop("class_")
        if all:
            tags = markup.find_all(name=tag_name, attrs=attrs)
            return tags
        tag = markup.find(name=tag_name, attrs=attrs)
        return tag

    def _get_body_tag(self, markup: Tag) -> Any:
        """Extracts body tag from markup"""
        body = markup.find("body")
        return body

    def _concat_resultset(self, resultset: ResultSet) -> Tag:
        """Concatenate ResultSet and returns one BeautifulSoup object"""
        new_tag = "".join(str(tag) for tag in resultset)
        new_tag = Bs(new_tag, scraper_ns.DEFAULT_PARSER)
        return new_tag

    def _find_tag(
        self,
        markup: Tag,
        tag_dict: dict,
        all: bool = False,
    ) -> Union[Tag, ResultSet]:
        """
        Uses BeautifulSoup find or find_all methods to search for
        specific tags. Returns Tag or ResultSet

         Parameters:
            markup : Tag
                BeautifulSoup object
            tag_dict : dict
                description of the tag, containing information about its
                attributes, check out tags namespaces
            all : bool, default = True
                if False uses find method instead of find_all and
                returns Tag, else ResultSet
        """
        tag_name = tag_dict.pop("tag_name") if "tag_name" in tag_dict else None
        tag_attrs = {}
        for key, value in tag_dict.items():
            use_re = value["re"]
            if use_re:
                attr_value = re.compile(value["value"])
            else:
                attr_value = value["value"]
            tag_attrs[key] = attr_value
        tags = self._find(markup=markup, tag_name=tag_name, all=all, **tag_attrs)
        return tags

    def _tag_to_selector(self, tag_dict: dict) -> str:
        """
        Transform tag_dict to CSS selector

         Parameters:
            tag_dict : dict
                description of the tag, containing information about its
                attributes, check out tags namespaces
        """
        tag_name = tag_dict.pop("tag_name") if "tag_name" in tag_dict else ""
        selector = tag_name
        for key, value in tag_dict.items():
            operator = "*=" if value["re"] else "="
            selector += f"[{key}{operator}{value['value']}]"
        return selector
