from dataclasses import dataclass

from ...exceptions import SourceNamespaceException, UnknownSource
from ..namespaces import data_ns, sources_ns


@dataclass
class SourceMetaData:
    source: str

    def __post_init__(self) -> None:
        source = self._get_source_ns()
        self.renames = source.get(data_ns.RENAMES, {})
        self.numeric_cols = source.get(data_ns.NUMERIC_COLUMNS, [])
        self.freq = self._get_frequency(ns=source)

    def _get_source_ns(self) -> dict:
        ns: dict = vars(sources_ns)

        if self.source not in ns:
            raise UnknownSource(f"{self.source} is unknown")

        return ns[self.source]

    def _get_frequency(self, ns: dict) -> str:
        if data_ns.FREQ not in ns:
            raise SourceNamespaceException(
                f"{self.source} is missing {data_ns.FREQ} key in the namespace"
            )

        return ns[data_ns.FREQ]
