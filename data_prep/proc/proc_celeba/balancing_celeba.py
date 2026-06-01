import pandas as pd
import os

DF_CELEBA = pd.read_pickle("embeddings_celeba.pkl")
ATRIBUTO_PIVO = "Male"


if __name__ == "__main__":
    df_imbalanced = DF_CELEBA[ ( DF_CELEBA[ATRIBUTO_PIVO] == 1 ) | ( ( DF_CELEBA[ATRIBUTO_PIVO] == -1).cumsum() <= 84259 )]

    df_imbalanced.to_pickle(f"{ATRIBUTO_PIVO}_balanced_embeddings_celeba.pkl")