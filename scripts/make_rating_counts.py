import argparse
import re
from pathlib import Path
import tarfile
from loguru import logger
import pandas as pd

# import waac.config as config
from waac.config import (DATA_DIR, ROOT_DIR)

logger.info(ROOT_DIR)

DOWNLOAD_DIR = DATA_DIR / "download"

tar_fp = DOWNLOAD_DIR / "training_set.tar"

parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--local",
#     action="store_true",
#     help="If set, use local data."
# )
# parser.add_argument(
#     "--remote",
#     action="store_true",
#     help="If set, use remote data.",
# )
parser.add_argument(
    "-o",
    "--output-fp",
    type=Path,
    help="Path to save the output.",
    default=DATA_DIR / "intermediate" / "rating_counts.csv",
)
parser.add_argument(
    "-f", "--force", action="store_true", help="If set, overwrite the output file."
)

args = parser.parse_args()
if args.output_fp.exists() and not args.force:
    raise FileExistsError(f"{args.output_fp} already exists. Use -f to overwrite.")

logger.info(f"Running with args: {args}")

tar_fp.parent.mkdir(exist_ok=True, parents=True)
with tarfile.open(tar_fp, "r") as t:
    names = [s for s in t.getnames() if s.endswith(".txt")]
    logger.info(
        f"Found {len(names)} items to extract."
    )
    results = []
    for name in names:
        # Quality checks
        match_filename = re.search(r"[0-9]+(?=\.txt)", name)
        if not match_filename:
            logger.warning(f"Skipping file based on filename match: {name}")
            results.append((name, None, None))
            continue

        data_stream = t.extractfile(member=name)
        movie_ID = next(data_stream).decode("utf-8")
        match_file_header = re.search(r"[0-9]+(?=:\n)", movie_ID)
        if not match_file_header:
            logger.warning(f"Skipping file based on file header match: {movie_ID}")
            results.append((name, None, None))
            continue

        if int(match_filename[0]) != int(match_file_header[0]):
            logger.warning(
                f"File name ({name}) and file header ({movie_ID}) do not correspond to "
                "the same movie_ID"
            )
            results.append((name, match_file_header[0], None))
            continue

        # Count the lines
        n_reviews = len(list(data_stream))
        results.append((name, match_file_header[0], n_reviews))

df = pd.DataFrame.from_records(results, columns=["filename", "movie_ID", "n_reviews"])

args.output_fp.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.output_fp, index=False)

logger.info(f"Saved to {args.output_fp}")

print("Done")
