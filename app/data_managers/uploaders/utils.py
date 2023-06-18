import os
from pathlib import Path

PATH = Path | str


def create_directory(path: PATH) -> None:
    """
    Creates directory if it doesn't already exist

    Params:
        path: Path | str
            path to the file or directory

    Makes sure that the target directory exists
    """
    path = Path(path)

    # if path exists, it means there is a directory
    if path.exists():
        return
    # if path is non-existing directory, create directory
    if path.is_dir():
        os.makedirs(path)
    # if path is non-existing file, check if there is a directory, if not - create it
    if path.is_file():
        if not path.parent.exists():
            os.makedirs(path.parent)
