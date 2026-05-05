import pickle
import pathlib
from loguru import logger


def load_dataset(funcs, dir_name, args):
    for func in funcs:
        try:
            file_name = func(args)
            path = pathlib.Path(args.dataset_dir, dir_name, file_name)

            with open(path, "rb") as f:
                dataset = pickle.load(f)
            return dataset
        except Exception as exc:
            logger.warning(
                f"Could not load dataset file {path}: {exc}. "
                "Trying legacy file name fallback."
            )
    raise FileNotFoundError("Could not find any dataset file.")
