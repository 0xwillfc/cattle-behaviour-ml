from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent.parent / "data"


def load_data(data_dir=DATA_DIR):
    dfs = []
    for i in range(1, 7):
        path = data_dir / f"cow{i}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["cow_id"] = i
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    data = pd.concat(dfs, ignore_index=True)
    data = data.dropna(subset=["Label"])
    data["Label"] = data["Label"].astype(str)
    return data


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df["Label"].value_counts())
