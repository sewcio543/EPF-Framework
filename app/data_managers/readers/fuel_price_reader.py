from typing import Union

import pandas as pd

from ..namespaces import data_ns
from .base_reader import WebScraper

SOURCE = "https://www.lotos.pl/145/type,oil_95/dla_biznesu/hurtowe_ceny_paliw/archiwum_cen_paliw"
REPLACE = {" ": "", ",": "."}


class FuelPricesReader(WebScraper):
    def read(self, source: Union[str, list[str]] = SOURCE) -> pd.DataFrame:
        return super().read(source)

    def _read_source(self, source: str) -> pd.DataFrame:
        html = self._scraper.get_source(source)
        df = pd.read_html(html.prettify())[0]
        return df

    def _format(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        vals = df.loc[:, data_ns.VALUE].replace(REPLACE, regex=True)
        vals = vals.apply(pd.to_numeric)
        df[data_ns.VALUE] = vals
        return df
