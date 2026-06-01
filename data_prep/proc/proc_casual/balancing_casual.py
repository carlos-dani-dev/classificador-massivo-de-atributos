import numpy as np
import pandas as pd
import os

from imblearn.over_sampling import SMOTE
from cuml.neighbors import NearestNeighbors

DF_CASUAL = pd.read_pickle("embeddings_casual.pkl")
ATRIBUTO_PIVO = "fitz_type"


if __name__ == "__main__":

    X = np.vstack(DF_CASUAL["Face_embedding"].values)
    y = DF_CASUAL[ATRIBUTO_PIVO].values

    nn = NearestNeighbors(n_neighbors = 6)

    X_resampled, y_resampled = SMOTE(k_neighbors=nn).fit_resample(X, y)

    df_resampled = pd.DataFrame({"face embedding": list(X_resampled), "fitz_type": y_resampled})
    df_resampled.to_pickle(f"{ATRIBUTO_PIVO}_balanced_embeddings_casual.pkl")