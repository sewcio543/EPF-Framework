class ReaderException(Exception):
    ...


class SourceNamespaceException(ReaderException):
    ...


class UnknownSource(ReaderException):
    ...
