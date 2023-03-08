import os


def create_directory(path: str) -> None:
    """
    Creates directory if it doesn't already exist

    Params:
        path: str
            path to the file

    Makes sure that the directory we will try to access exists
    """
    # if path exists
    if os.path.exists(path):
        return

    split = path.split("/")
    # if non-existing directory was passed
    if len(split) > 1:
        # seperate folder from file
        folder_path = "/".join(split[:-1])
        # if folder doesn't exist, create directory
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
