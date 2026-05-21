import pickle
import pathlib
from loguru import logger


def load_dataset(funcs, dir_name, args):
    last_path = None
    for func in funcs:
        file_name = func(args)
        path = pathlib.Path(args.dataset_dir, dir_name, file_name)
        last_path = path

        try:
            with open(path, "rb") as f:
                dataset = pickle.load(f)
            return dataset
        except FileNotFoundError as exc:
            logger.warning(
                f"Could not load dataset file {path}: {exc}. "
                "Trying legacy file name fallback."
            )
    raise FileNotFoundError(f"Could not find any dataset file. Last path: {last_path}")
