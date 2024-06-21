import glob
from os.path import abspath


def filepath_resolution(filepath_or_pathlike: (str, list), absolute: bool = None):
    filepaths = []
    if filepath_or_pathlike is None:
        return None
    elif isinstance(filepath_or_pathlike, str):
        filepaths.extend(glob.glob(filepath_or_pathlike))
    elif isinstance(filepath_or_pathlike, list):
        for f in filepath_or_pathlike:
            filepaths.extend(glob.glob(f))
    if absolute:
        return [abspath(p) for p in filepaths]
    return filepaths
  
