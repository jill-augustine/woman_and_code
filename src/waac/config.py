import json
from pathlib import Path

# Path to the root of the project
ROOT_DIR = Path(__file__).parents[2]

# Path to the data directory
DATA_DIR = ROOT_DIR / "data"

raw_data_column_names = {
    # Test data: The ratings for these are not in the training_set dir
    "qualifying.txt": {"movie_ID": ["customer_ID", "date"]},
    # Expected format of predictions (rows correspond to the rows in qualifying.txt)
    "predictions.txt": {"movie_ID": ["rating"]},
    # Validation data: The ratings for these are in the training_set dir
    "probe.txt": {"movie_ID": ["customer_ID"]},
    # Metadata
    "movie_titles.txt": ["movie_ID", "year_of_release", "title"],
    # This is a dir name not a file name
    "training_set": {"movie_ID": ["customer_ID", "rating", "date"]},
}

with open(DATA_DIR / "imdb" / "imdb_metadata.json", "r") as f:
    imdb_metadata = json.load(f)
# /Users/jillianaugustine/Documents/GitHub/women_and_code/data/imdb/imdb_metadata.json
