"""Map Netflix movie titles to IMDB movie titles.

Takes approx 15 minutes to run on my machine.
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import re
import subprocess as sp
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

import waac
import waac.config as config

# import csv


# Set some constants
ROOT_DIR = config.ROOT_DIR
DATA_DIR = config.DATA_DIR
DOWNLOAD_DIR = DATA_DIR / "download"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--local",
    action="store_true",
    help="If set, use local data."
)
parser.add_argument(
    "--remote",
    action="store_true",
    help="If set, use remote data.",
)
parser.add_argument(
    "-o",
    "--output-fp",
    type=Path,
    help="Path to save the output.",
    default=DATA_DIR / "intermediate" / "netflix_to_imdb.csv",
)
parser.add_argument(
    "-f", "--force", action="store_true", help="If set, overwrite the output file."
)

args = parser.parse_args()
if args.output_fp.exists() and not args.force:
    raise FileExistsError(f"{args.output_fp} already exists. Use -f to overwrite.")

logger.info(f"Running with args: {args}")

# Set more constants
IMDB_REMOTE_URI_PREFIX = "https://datasets.imdbws.com/"
IMDB_LOCAL_URI_PREFIX = (
    str(DATA_DIR / "imdb") + "/"
)

rng = np.random.default_rng(seed=16042024)
if hasattr(args, "remote"):
    LOCAL = not args.remote
elif hasattr(args, "local"):
    LOCAL = args.local
else:
    raise ValueError("Must set either --remote or --local.")

if LOCAL:
    IMDB_BASE_URI_PREFIX = IMDB_LOCAL_URI_PREFIX
else:
    IMDB_BASE_URI_PREFIX = IMDB_REMOTE_URI_PREFIX


# Set up netflix df
fp = DOWNLOAD_DIR / "movie_titles.txt"
movie_titles_df = waac.txt_to_df(
    fp,
    config.raw_data_column_names[fp.name],
    encoding="latin-1",
)
movie_titles_df = movie_titles_df.astype(object)
movie_titles_df["movie_ID"] = movie_titles_df["movie_ID"].astype(int)
# Set to float because of missing values
movie_titles_df["year_of_release"] = movie_titles_df["year_of_release"].replace(
    "NULL", None
)
movie_titles_df.dropna(subset=["year_of_release"], inplace=True)
movie_titles_df.sort_values("year_of_release", ascending=False, inplace=True)


# Set up IMDB dfs
SUB_URLS = [
    "name.basics.tsv.gz",
    "title.akas.tsv.gz",
    "title.basics.tsv.gz",
    "title.crew.tsv.gz",
    "title.episode.tsv.gz",
    "title.principals.tsv.gz",
    "title.ratings.tsv.gz",
]

# imdb_metadata = {}
# CHUNK_SIZE = 1_000
imdb_data = {}
# for sub_url in SUB_URLS
for sub_url in SUB_URLS[1:3]:
    url = IMDB_BASE_URI_PREFIX + sub_url
    logger.debug(url)
    # Load without parsing
    df_temp = pd.read_table(url, compression="gzip", na_values=r"\N", dtype=object)
    # Convert to appropriate dtypes
    for col, _dtype in config.imdb_metadata[sub_url]["dtypes"].items():
        if _dtype == "object":
            continue
        try:
            df_temp[col] = df_temp[col].astype(_dtype)
        except ValueError:
            if _dtype.startswith("int"):
                df_temp[col] = df_temp[col].astype(float)
            else:
                raise

    if "startYear" in df_temp.columns:
        df_temp.sort_values("startYear", ascending=False, inplace=True)
        imdb_data[sub_url] = {k: v for k, v in df_temp.groupby("startYear", sort=False)}
    else:
        imdb_data[sub_url] = df_temp

# clear memory
# del df_temp

movie_titles_by_year = {
    k: v for k, v in movie_titles_df.groupby("year_of_release", sort=False)
}
# del movie_titles_df

logger.debug(f"# years in movie_titles_df: {len(movie_titles_by_year.keys())}")
logger.debug(f"# years in imdb_data: {len(imdb_data['title.basics.tsv.gz'].keys())}")
# This also removes any "NaN" years because they were removed from the
# movie_titles_df
common_years = sorted(
    set(movie_titles_by_year.keys()) & set(imdb_data["title.basics.tsv.gz"].keys())
)
logger.info(f"# common years: {len(common_years)}")

for k in list(movie_titles_by_year.keys()):
    if k not in common_years:
        _ = movie_titles_by_year.pop(k)

for k in list(imdb_data["title.basics.tsv.gz"].keys()):
    if k not in common_years:
        _ = imdb_data["title.basics.tsv.gz"].pop(k)


# def iterable_to_map():
#     for year in common_years:
#         # pass only the relevant aka data (it is not split by year)
#         tconsts = imdb_data["title.basics.tsv.gz"][year]["tconst"]
#         aka_data = imdb_data["title.akas.tsv.gz"]
#         aka_data = aka_data[aka_data["titleId"].isin(tconsts)]
#         yield (
#             movie_titles_by_year[year],
#             imdb_data["title.basics.tsv.gz"][year],
#             aka_data,
#             1,  # verbosity level
#         )


# # set spawn context
# ctx = mp.get_context("fork")
# n_processes = mp.cpu_count() - 1
# with ctx.Pool(n_processes) as pool:
#     results = pool.starmap(
#         waac.map_netflix_to_imdb,
#         iterable_to_map()
#     )

n = len(common_years)
# We do this because we know the length of the iterable
results = [None for _ in range(n)]
i = 0
iterable = iterable_to_map()
while True:
    if i % 100 == 0:
        logger.info(f"{i+1}/{n}")
    try:
        results[i] = waac.map_netflix_to_imdb(*next(iterable))
        i += 1
    except StopIteration:
        break

merged_df = pd.concat(results, ignore_index=True)

# Save the merged_df
args.output_fp.parent.mkdir(parents=True, exist_ok=True)
merged_df.to_csv(args.output_fp, index=False)

logger.info(f"Saved to {args.output_fp}")
