import json
from pathlib import Path
import pandas as pd
import numpy as np
import tarfile
import io
import re
import subprocess as sp

from typing import Dict, List, Tuple, Union, Optional

from loguru import logger

# import plotly.express as px
# import plotly.graph_objects as go

import pandas as pd
import numpy as np

import torch

import waac
from waac.config import (DATA_DIR, raw_data_column_names)
from waac.modeling import (
    scale,
    unscale,
    MatrixFactorization,
    Loss,
    ModelTrainer,
)
# from sklearn.decomposition import NMF

INTERMEDIATE_DIR = DATA_DIR / "intermediate"
DOWNLOAD_DIR = DATA_DIR / "download"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RATING_PROP = 0.1
N_EPOCHS = 200

# Load intermediate metadata

fp = INTERMEDIATE_DIR / "netflix_to_imdb.csv"
id_mapping = pd.read_csv(fp, sep=";").dropna(subset="tconst")

fp = INTERMEDIATE_DIR / "rating_counts.csv"
rating_counts_import = pd.read_csv(fp)

rating_counts_mapped = (
    rating_counts_import.sort_values("n_reviews", ascending = False)
                        .merge(id_mapping, how="inner", on="movie_ID")
)

end_idx = int(rating_counts_mapped.shape[0]*RATING_PROP)
rating_counts_subset = rating_counts_mapped.iloc[:end_idx]


# Load and process training data
tar_fp = DOWNLOAD_DIR / "training_set.tar"
if not tar_fp.exists():
    raise FileNotFoundError

n = rating_counts_subset.shape[0]
df_list = [None for _ in range(n)]

with tarfile.open(tar_fp, "r") as t:
    for i, name in enumerate(rating_counts_subset.filename):
        logger.info(f"{i+1}/{n}: {name=}") if i % 100 == 0 else None
        data_stream = t.extractfile(member=name)
        file_header = next(data_stream).decode("utf-8")
        match_file_header = re.search(r"[0-9]+(?=:\n)", file_header)
        if not match_file_header:
            logger.warning(f"Skipping file based on file header match: {file_header}")
            continue
        df_temp = pd.read_csv(
            data_stream, encoding="utf-8", header=None,
            names=raw_data_column_names["training_set"]["movie_ID"]
            )
        df_temp.insert(0, "movie_ID", match_file_header[0])
        df_list[i] = df_temp
       
assert all(x is not None for x in df_list), "Msg"

df = pd.concat(df_list)
               
logger.debug(f"Training data long shape: {df.shape}")

assert not df.duplicated(subset=["movie_ID","customer_ID"]).any()

df_pivoted = df.pivot(columns="movie_ID", index="customer_ID", values="rating")
logger.debug(f"Training data wide shape: {df_pivoted.shape}")

loss_class = Loss(lam_u=0.3, lam_v=0.3)

model_trainer = ModelTrainer(
    n_features=5,
    loss_class=loss_class,
    # These are anyway the defaults
    model_class_type=MatrixFactorization,
    optimizer_class_type=torch.optim.Adam,
)

training_data = torch.from_numpy(df_pivoted.values).to(device=DEVICE)

model_trainer.train(training_data, n_epochs=N_EPOCHS)

viewers_to_predict = torch.randint(0, training_data.size(0), (3,))

y_pred, y_true = model_trainer.predict(user_idx = viewers_to_predict)

logger.info("Done.")