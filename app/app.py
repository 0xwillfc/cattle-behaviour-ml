from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


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


def reduce_features(features, method="pca", sample_size=1500):
    features = features.sample(min(sample_size, len(features)), random_state=42)
    cols = [c for c in features.columns if c not in ["label", "cow_id"]]
    x = StandardScaler().fit_transform(features[cols])
    if method == "tsne":
        coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(x)
    else:
        coords = PCA(n_components=2, random_state=42).fit_transform(x)
    return pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": features["label"].values})


def plot_embedding(coords, title):
    sns.scatterplot(data=coords, x="x", y="y", hue="label", s=15, alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = load_data()
    features = extract_features(data)
    plot_embedding(reduce_features(features, "pca"), "PCA")
    plot_embedding(reduce_features(features, "tsne"), "t-SNE")
