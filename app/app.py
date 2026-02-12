from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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


def filter_rare_classes(features, min_samples=2):
    counts = features["label"].value_counts()
    return features[features["label"].isin(counts[counts >= min_samples].index)]


def train_random_forest(features):
    features = filter_rare_classes(features)
    feature_cols = [c for c in features.columns if c not in ["label", "cow_id"]]
    x = features[feature_cols]
    y = LabelEncoder().fit_transform(features["label"])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print("Holdout accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    return model


def leave_one_cow_out(features):
    feature_cols = [c for c in features.columns if c not in ["label", "cow_id"]]
    results = []
    for cow_id in sorted(features["cow_id"].unique()):
        train_df = filter_rare_classes(features[features["cow_id"] != cow_id])
        test_df = features[features["cow_id"] == cow_id]
        labels = sorted(set(train_df["label"]) & set(test_df["label"]))
        train_df = train_df[train_df["label"].isin(labels)]
        test_df = test_df[test_df["label"].isin(labels)]
        if train_df.empty or test_df.empty:
            continue
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(train_df["label"])
        y_test = encoder.transform(test_df["label"])
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(train_df[feature_cols], y_train)
        predictions = model.predict(test_df[feature_cols])
        results.append({"cow_id": cow_id, "accuracy": accuracy_score(y_test, predictions)})
    return pd.DataFrame(results)


if __name__ == "__main__":
    data = load_data()
    features = extract_features(data)
    print(leave_one_cow_out(features))
