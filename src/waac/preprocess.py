import re
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from loguru import logger


def txt_to_df(
    fp: Union[Path, str],
    schema: Union[Dict, List],
    encoding: str = "latin-1",
    dtype=None,
) -> pd.DataFrame:
    """Read the movie_titles.txt file.

    dtype is passed to pd.read_csv.

    """

    if isinstance(fp, str):
        fp = Path(fp)
    if not isinstance(fp, Path):
        raise TypeError(f"`fp` must be a path-like object or a string. Got {type(fp)}.")
    if isinstance(schema, dict):
        assert len(schema) == 1, "schema must have exactly one key."
        header_name = list(schema.keys())[0]
        col_names = schema[header_name]
    elif isinstance(schema, list):
        header_name = None
        col_names = schema
    else:
        raise TypeError(f"`schema` must be a dict or a list. Got {type(schema)}.")

    lines = fp.open(encoding=encoding).readlines()

    if header_name is None:
        df = _df_from_lines(lines, col_names)
    else:
        n = len(lines)
        pattern = re.compile("^\d+(?=:\n)")
        # Get starting lines
        starts = [(line_no, pattern.match(line)) for line_no, line in enumerate(lines)]
        # Keep only the starting lines
        starts = [(line_no, int(match[0])) for line_no, match in starts if match]

        # Create slices per movieID
        slices = [slice(starts[i][0], starts[i + 1][0]) for i in range(len(starts) - 1)]
        # Create the last slice
        slices.append(slice(starts[-1][0], None))

        # Make list of dataframes
        df_list = [None for s in slices]
        n = len(slices)
        for i, s in enumerate(slices):
            print(f"{i+1}/{n}") if i % 1000 == 0 else None
            subset = lines[s]
            # Remove the first row in the list
            header_value = subset.pop(0).split(":")[0]
            header_value = int(header_value)
            df_temp = _df_from_lines(subset, col_names)
            # header is always movie_ID
            df_temp = df_temp.assign(**{header_name: header_value})
            df_list[i] = df_temp
        df = pd.concat(df_list, ignore_index=True)
    return df


def _df_from_lines(
    lines: List[str],
    col_names: List[str],
    sep: str = ",",
) -> pd.DataFrame:
    """Create a dataframe from a list of lines.

    Arguments
    ---------
    lines: List[str]
        A list of strings.
    col_names: List[str]
        A list of column names.
    column_containing_sep: int
        One column (if any) that might contain the separator character in the field.
    sep: str
        The separator character.
    """
    if len(sep) > 1:
        raise ValueError(f"sep must be a single character. Got {sep}.")
    values = [line.strip().split(sep) for line in lines]
    # Sometimes (e.g. in movie_titles.txt) the last column contains the separator character so we need to join it back together
    values = [[v[0], v[1], sep.join(v[2:])] for v in values]

    df = pd.DataFrame.from_records(
        values,
        columns=col_names,
    )
    return df


def map_netflix_to_imdb(
    netflix_df: pd.DataFrame,
    imdb_basics_df: pd.DataFrame,
    imdb_akas_df: pd.DataFrame,
    verbose: int = 0,
) -> pd.DataFrame:
    """Map Netflix movie titles to IMDB movie titles.

    The year of release for `netflix_df` and `imdb_basics_df` should be the same.


    Returns
    -------
    pd.DataFrame
        A dataframe with the following columns:
        - movie_ID
        - title
        - tconst
        - match_type
            - 1: Matched on `primaryTitle`
            - 2: Matched on `originalTitle`
            - 3: Matched on `title` in `imdb_akas_df`
            - None: No match found
    """
    # logger.info(f"{verbose=}")
    year = netflix_df["year_of_release"].iloc[0]
    year1 = imdb_basics_df["startYear"].iloc[0]
    assert (
        year == year1
    ), f"Year of release for `netflix_df` and `imdb_basics_df` should be the same. Got {year} and {year1}."
    if verbose:
        logger.info(f"Running map_netflix_to_imdb for year {year}.")

    FINAL_COLS = ["movie_ID", "title", "tconst", "match_type"]

    ## Prep dataframes
    netflix_df = netflix_df.loc[:, ["movie_ID", "title"]]
    # Drop duplicate rows in case two movies have the same title in the same year
    rows_to_drop = netflix_df.duplicated(subset="title", keep="first")
    if rows_to_drop.any():
        logger.warning(
            f"Dropping {rows_to_drop.sum()} duplicate rows from `netflix_df`."
        )
        netflix_df = netflix_df.loc[~rows_to_drop]

    imdb_basics_df = imdb_basics_df.loc[:, ["tconst", "primaryTitle", "originalTitle"]]
    # Drop duplicate rows in case two movies have the same title in the same year
    rows_to_drop = imdb_basics_df.duplicated(subset="primaryTitle", keep="first")
    if rows_to_drop.any():
        logger.warning(
            f"Dropping {rows_to_drop.sum()} duplicate rows from `imdb_basics_df`."
        )
        imdb_basics_df = imdb_basics_df.loc[~rows_to_drop]

    imdb_akas_df = imdb_akas_df.loc[:, ["titleId", "title"]]
    # We expect some duplicates here so we reduce rows where the foreign title are the same as each other
    imdb_akas_df.drop_duplicates(inplace=True)
    # Rename titleId to tconst
    imdb_akas_df.rename(columns={"titleId": "tconst"}, inplace=True)

    ## Merge dataframes
    first_match = pd.merge(
        netflix_df,
        imdb_basics_df,
        left_on="title",
        right_on="primaryTitle",
        how="left",
        # validate="1:1",  # not needed because we drop duplicates above
    )

    # split based on success of first match
    unmatched = first_match.loc[first_match["tconst"].isna(), netflix_df.columns]
    first_match.dropna(subset=["tconst"], inplace=True)
    first_match["match_type"] = 1
    first_match = first_match.loc[:, FINAL_COLS]

    # Try to match based on the `originalTitle` column in IMDB basics
    second_match = pd.merge(
        unmatched, imdb_basics_df, left_on="title", right_on="originalTitle", how="left"
    )
    unmatched = second_match.loc[second_match["tconst"].isna(), :]
    unmatched.drop(columns="tconst", inplace=True)
    second_match.dropna(subset=["tconst"], inplace=True)
    second_match["match_type"] = 2
    second_match = second_match.loc[:, FINAL_COLS]

    # Merge unmatched with IMDB akas
    third_match = pd.merge(unmatched, imdb_akas_df, on="title", how="left")
    unmatched = third_match.loc[third_match["tconst"].isna(), :]
    # Don't drop the column
    unmatched.loc[:, "match_type"] = None
    unmatched = unmatched.loc[:, FINAL_COLS]

    third_match.dropna(subset=["tconst"], inplace=True)
    third_match["match_type"] = 3
    third_match = third_match.loc[:, FINAL_COLS]

    # If movie_ID is duplicated, there are two movies with the same `title` but different `titleId`. In this case we take the first one.
    third_match.drop_duplicates(subset="movie_ID", keep="first", inplace=True)

    # Merge all the dataframes
    merged_df = pd.concat(
        [first_match, second_match, third_match, unmatched], ignore_index=True
    )

    return merged_df
