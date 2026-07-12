import pandas as pd

from pathlib import Path


def load_dataset(path: Path):

    """
    Loads CSV dataset

    Parameters
    ----------
    path : Path

    Returns
    -------
    pandas.DataFrame
    """

    try:

        df = pd.read_csv(path)

        print("Dataset Loaded Successfully")

        print(f"Shape : {df.shape}")

        return df

    except Exception as e:

        print(e)

        raise