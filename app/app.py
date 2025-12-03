from pathlib import Path

import numpy as np
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


def extract_features(df, window=125, step=62):
    rows = []
    for cow_id, cow_df in df.groupby("cow_id"):
        cow_df = cow_df.reset_index(drop=True)
        for start in range(0, len(cow_df) - window, step):
            chunk = cow_df.iloc[start:start + window]
            label = chunk["Label"].mode()[0]
            row = {"label": label, "cow_id": int(cow_id)}
            for axis in ["AccX", "AccY", "AccZ"]:
                values = chunk[axis].to_numpy()
                row[f"{axis}_mean"] = values.mean()
                row[f"{axis}_std"] = values.std()
                row[f"{axis}_min"] = values.min()
                row[f"{axis}_max"] = values.max()
                row[f"{axis}_range"] = values.max() - values.min()
                spectrum = np.abs(np.fft.rfft(values))
                row[f"{axis}_fft_mean"] = spectrum.mean()
                row[f"{axis}_fft_std"] = spectrum.std()
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = load_data()
    features = extract_features(data)
    print(data.shape)
    print(features.shape)
