import pandas as pd

from .base_reader import SOURCE, WebScraper

URL = "https://www.lotos.pl/145/type,oil_95/dla_biznesu/hurtowe_ceny_paliw/archiwum_cen_paliw"


class FuelPricesReader(WebScraper):
    _READ_KWARGS = {"decimal": ",", "thousands": " "}

    def read(self, source: SOURCE = URL) -> pd.DataFrame:
        return super().read(source)

    def _read_source(self, source: str) -> pd.DataFrame:
        html = self._scraper.get_source(source)
        df = pd.read_html(html.prettify(), **self._READ_KWARGS)[0]  # type: ignore
        return df
