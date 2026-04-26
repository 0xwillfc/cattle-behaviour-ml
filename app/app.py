import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.cm as mcm
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"

st.set_page_config(page_title="Bovine Behavior Classification", layout="wide")
st.title("Bovine Behavior Classification")
st.markdown("Tri-axial accelerometer data from Japanese Black Beef Cows — [Dataset: Zenodo 5849025](https://zenodo.org/records/5849025)")

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    dfs = []
    for i in range(1, 7):
        path = DATA_DIR / f"cow{i}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["cow_id"] = i
            dfs.append(df)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["Label"])
    df["Label"] = df["Label"].astype(str)
    return df

@st.cache_data
def extract_features(df, window=125, step=62):
    features = []
    sort_cols = [col for col in ["cow_id", "TimeStamp_UNIX", "TimeStamp_JST"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    for cow_id, cow_df in df.groupby("cow_id", sort=True):
        cow_df = cow_df.reset_index(drop=True)
        for start in range(0, len(cow_df) - window + 1, step):
            w = cow_df.iloc[start:start + window]
            label = w["Label"].mode()[0]
            row = {"label": label, "cow_id": int(cow_id)}
            for ax in ["AccX", "AccY", "AccZ"]:
                vals = w[ax].values
                row[f"{ax}_mean"] = vals.mean()
                row[f"{ax}_std"] = vals.std()
                row[f"{ax}_min"] = vals.min()
                row[f"{ax}_max"] = vals.max()
                row[f"{ax}_range"] = vals.max() - vals.min()
                fft = np.abs(np.fft.rfft(vals))
                row[f"{ax}_fft_mean"] = fft.mean()
                row[f"{ax}_fft_std"] = fft.std()
            features.append(row)
    return pd.DataFrame(features)

def filter_rare_classes(df, min_samples=2):
    counts = df["label"].value_counts()
    return df[df["label"].isin(counts[counts >= min_samples].index)]

data = load_data()

if data is None:
    st.error("No data found. Please add the CSV files (cow1.csv ... cow6.csv) to the `data/` folder.")
    st.info("Download from: https://zenodo.org/records/5849025")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.header("Filters")
cow_options = sorted(data["cow_id"].unique())
selected_cows = st.sidebar.multiselect("Select cows", cow_options, default=cow_options)
all_behaviors = sorted(data["Label"].unique())
selected_behaviors = st.sidebar.multiselect("Select behaviors", all_behaviors, default=all_behaviors)

st.sidebar.divider()
st.sidebar.header("Chart Settings")
pub_mode = st.sidebar.toggle("Publication Mode", value=False,
                              help="White background, 300 DPI export — ready for articles and presentations")
if pub_mode:
    dpi_export = st.sidebar.select_slider("Export DPI", options=[150, 200, 300], value=300)
    st.sidebar.info("All charts will render in publication style with download buttons.")

filtered = data[data["cow_id"].isin(selected_cows) & data["Label"].isin(selected_behaviors)]

# ── Matplotlib helpers ─────────────────────────────────────────────────────────

QUAL_COLORS = plt.get_cmap("tab20").colors

def mpl_fig(figsize=(9, 5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.tick_params(labelsize=9, colors="#333333")
    return fig, ax

def fig_to_png(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.getvalue()

def show_fig(container, fig, filename, pub, dpi=300):
    container.pyplot(fig, use_container_width=True)
    if pub:
        container.download_button(f"Download PNG — {filename}",
                                   fig_to_png(fig, dpi), file_name=filename, mime="image/png")
    plt.close(fig)

def plotly_download(container, plotly_fig, filename):
    buf = io.BytesIO()
    plotly_fig.write_image(buf, format="png", scale=3)
    container.download_button(f"Download PNG — {filename}", buf.getvalue(),
                               file_name=filename, mime="image/png")

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Signal Viewer", "Dimensionality Reduction", "Classification", "LOCO Validation"
])

# ── Tab 1: Overview ────────────────────────────────────────────────────────────

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total samples", f"{len(filtered):,}")
    col2.metric("Cows selected", len(selected_cows))
    col3.metric("Behaviors selected", len(selected_behaviors))

    counts = filtered["Label"].value_counts().reset_index()
    counts.columns = ["Behavior", "Count"]
    cow_beh = filtered.groupby(["cow_id", "Label"]).size().reset_index(name="count")

    st.subheader("Behavior distribution")
    if pub_mode:
        fig, ax = mpl_fig((9, 4))
        colors = [QUAL_COLORS[i % 20] for i in range(len(counts))]
        ax.bar(counts["Behavior"], counts["Count"], color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Behavior", fontsize=10)
        ax.set_ylabel("Samples", fontsize=10)
        ax.set_title("Behavior Distribution", fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=40)
        plt.tight_layout()
        show_fig(st, fig, "behavior_distribution.png", pub_mode, dpi_export)
    else:
        fig = px.bar(counts, x="Behavior", y="Count", color="Behavior",
                     color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(showlegend=False, xaxis_tickangle=-40)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution per cow")
    if pub_mode:
        pivoted = cow_beh.pivot(index="cow_id", columns="Label", values="count").fillna(0)
        fig, ax = mpl_fig((9, 4))
        bottom = np.zeros(len(pivoted))
        for i, col in enumerate(pivoted.columns):
            ax.bar(pivoted.index, pivoted[col], bottom=bottom,
                   label=col, color=QUAL_COLORS[i % 20], edgecolor="white", linewidth=0.3)
            bottom += pivoted[col].values
        ax.set_xlabel("Cow", fontsize=10)
        ax.set_ylabel("Samples", fontsize=10)
        ax.set_title("Sample Distribution per Cow", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.8)
        plt.tight_layout()
        show_fig(st, fig, "distribution_per_cow.png", pub_mode, dpi_export)
    else:
        fig2 = px.bar(cow_beh, x="cow_id", y="count", color="Label", barmode="stack",
                      labels={"cow_id": "Cow", "count": "Samples"},
                      color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig2, use_container_width=True)

# ── Tab 2: Signal Viewer ───────────────────────────────────────────────────────

with tab2:
    st.subheader("Accelerometer signal")
    col1, col2 = st.columns(2)
    sel_cow = col1.selectbox("Cow", sorted(filtered["cow_id"].unique()))
    sel_behavior = col2.selectbox("Behavior", sorted(filtered[filtered["cow_id"] == sel_cow]["Label"].unique()))
    n_samples = st.slider("Number of samples to display", 100, 1000, 500, step=50)

    segment = filtered[(filtered["cow_id"] == sel_cow) & (filtered["Label"] == sel_behavior)].head(n_samples)

    if segment.empty:
        st.warning("No data for this selection.")
    else:
        if pub_mode:
            axis_colors = {"AccX": "#D62728", "AccY": "#2CA02C", "AccZ": "#1F77B4"}
            fig, ax = mpl_fig((9, 4))
            for axis, color in axis_colors.items():
                ax.plot(segment[axis].values, label=axis, color=color, linewidth=0.9)
            ax.set_title(f"Cow {sel_cow} — {sel_behavior}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Samples (25 Hz)", fontsize=10)
            ax.set_ylabel("Acceleration (g)", fontsize=10)
            ax.legend(fontsize=9, framealpha=0.8)
            plt.tight_layout()
            show_fig(st, fig, f"signal_cow{sel_cow}_{sel_behavior}.png", pub_mode, dpi_export)
        else:
            fig = go.Figure()
            colors = {"AccX": "#EF553B", "AccY": "#00CC96", "AccZ": "#636EFA"}
            for axis, color in colors.items():
                fig.add_trace(go.Scatter(y=segment[axis].values, name=axis,
                                         line=dict(color=color, width=1)))
            fig.update_layout(title=f"Cow {sel_cow} — {sel_behavior}",
                              xaxis_title="Samples (25 Hz)", yaxis_title="Acceleration (g)",
                              legend_title="Axis")
            st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        for col, axis in zip([col1, col2, col3], ["AccX", "AccY", "AccZ"]):
            col.metric(f"{axis} mean ± std",
                       f"{segment[axis].mean():.3f} ± {segment[axis].std():.3f} g")

# ── Tab 3: Dimensionality Reduction ───────────────────────────────────────────

def plot_scatter_matplotlib(coords, color_by, title, subtitle="", dpi=150):
    unique_labels = sorted(set(color_by))
    cmap = plt.get_cmap("tab20", len(unique_labels))
    color_map = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for lbl in unique_labels:
        mask = np.array(color_by) == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color_map[lbl]], label=str(lbl), alpha=0.6, s=18, linewidths=0)
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        ax.text(cx, cy, str(lbl), fontsize=7, fontweight="bold", color="black",
                ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    method_prefix = title.split()[0]
    ax.set_title(f"{title}" + (f"\n{subtitle}" if subtitle else ""),
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(f"{method_prefix} 1", fontsize=10)
    ax.set_ylabel(f"{method_prefix} 2", fontsize=10)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8,
              markerscale=1.5, ncol=2 if len(unique_labels) > 10 else 1, edgecolor="#cccccc")
    plt.tight_layout()
    return fig

with tab3:
    st.subheader("Dimensionality Reduction")
    col1, col2 = st.columns(2)
    method = col1.selectbox("Method", ["PCA", "t-SNE"])
    window_size = col2.selectbox("Window size (samples)", [62, 125, 250], index=1)
    max_windows = st.slider("Max windows to use", 500, 5000, 2000, step=500)

    if st.button("Run reduction"):
        with st.spinner("Extracting features..."):
            feat_df = extract_features(filtered, window=window_size)
            feat_df = feat_df.sample(min(max_windows, len(feat_df)), random_state=42)

        if len(feat_df) < 3:
            st.warning("Not enough windows to run dimensionality reduction. Try selecting more cows or behaviors.")
            st.stop()

        feature_cols = [c for c in feat_df.columns if c not in ["label", "cow_id"]]
        X = StandardScaler().fit_transform(feat_df[feature_cols])

        with st.spinner(f"Running {method}..."):
            if method == "PCA":
                reducer = PCA(n_components=2)
                coords = reducer.fit_transform(X)
                subtitle = f"Explained variance: {reducer.explained_variance_ratio_.sum():.1%}"
            else:
                perplexity = min(30, max(2, (len(feat_df) - 1) // 3))
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                coords = reducer.fit_transform(X)
                subtitle = f"perplexity={perplexity}"

        col1, col2 = st.columns(2)
        export_dpi = dpi_export if pub_mode else 150

        fig1 = plot_scatter_matplotlib(coords, feat_df["label"].values,
                                        f"{method} — by Behavior", subtitle)
        col1.pyplot(fig1, use_container_width=True)
        col1.download_button("Download PNG (Behavior)", fig_to_png(fig1, export_dpi),
                              file_name=f"{method}_behavior.png", mime="image/png")
        plt.close(fig1)

        fig2 = plot_scatter_matplotlib(coords, [f"Cow {c}" for c in feat_df["cow_id"].values],
                                        f"{method} — by Cow", subtitle)
        col2.pyplot(fig2, use_container_width=True)
        col2.download_button("Download PNG (Cow)", fig_to_png(fig2, export_dpi),
                              file_name=f"{method}_cow.png", mime="image/png")
        plt.close(fig2)

# ── Tab 4: Classification ──────────────────────────────────────────────────────

with tab4:
    st.subheader("Model comparison")
    col1, col2 = st.columns(2)
    n_estimators = col1.slider("Number of trees (Random Forest)", 50, 300, 100, step=50)
    test_size = col2.slider("Test size", 0.1, 0.4, 0.2, step=0.05)

    if st.button("Train models"):
        with st.spinner("Extracting features and training..."):
            feat_df = extract_features(filtered)

        if len(feat_df["label"].unique()) < 2:
            st.warning("Select at least 2 behavior classes to train.")
        else:
            feat_df = filter_rare_classes(feat_df)
            if len(feat_df["label"].unique()) < 2:
                st.warning("Not enough samples per class. Try selecting more cows or behaviors.")
                st.stop()

            feature_cols = [c for c in feat_df.columns if c not in ["label", "cow_id"]]
            X = StandardScaler().fit_transform(feat_df[feature_cols])
            le = LabelEncoder()
            y = le.fit_transform(feat_df["label"])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y)

            models = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=n_estimators, random_state=42, n_jobs=-1
                ),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "SVM": SVC(kernel="rbf", random_state=42),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            }

            comparison_rows = []
            predictions = {}
            trained_models = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                model_pred = model.predict(X_test)
                predictions[name] = model_pred
                trained_models[name] = model
                comparison_rows.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, model_pred),
                    "Balanced accuracy": balanced_accuracy_score(y_test, model_pred),
                    "Macro F1": f1_score(y_test, model_pred, average="macro"),
                })

            comparison_df = pd.DataFrame(comparison_rows).sort_values(
                "Macro F1", ascending=False
            )
            metric_cols = ["Accuracy", "Balanced accuracy", "Macro F1"]

            st.subheader("Model comparison")
            st.dataframe(
                comparison_df.assign(**{
                    col: comparison_df[col].map(lambda value: f"{value:.1%}")
                    for col in metric_cols
                }),
                use_container_width=True,
                hide_index=True,
            )

            if pub_mode:
                fig, ax = mpl_fig((8, 4))
                plot_df = comparison_df.set_index("Model")[metric_cols] * 100
                plot_df.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518", "#54A24B"])
                ax.set_ylabel("Score (%)", fontsize=10)
                ax.set_xlabel("Model", fontsize=10)
                ax.set_title("Model comparison", fontsize=11, fontweight="bold")
                ax.set_ylim(0, 100)
                ax.tick_params(axis="x", rotation=30)
                ax.legend(fontsize=8)
                plt.tight_layout()
                show_fig(st, fig, "model_comparison.png", pub_mode, dpi_export)
            else:
                plot_df = comparison_df.melt(
                    id_vars="Model", value_vars=metric_cols,
                    var_name="Metric", value_name="Score"
                )
                plot_df["Score"] = plot_df["Score"] * 100
                fig = px.bar(
                    plot_df, x="Model", y="Score", color="Metric",
                    barmode="group", range_y=[0, 100],
                    title="Model comparison"
                )
                fig.update_layout(yaxis_title="Score (%)", xaxis_tickangle=-25)
                st.plotly_chart(fig, use_container_width=True)

            selected_model = comparison_df.iloc[0]["Model"]
            st.caption(f"Detailed report for the best model by Macro F1: {selected_model}")

            clf = trained_models[selected_model]
            y_pred = predictions[selected_model]

            present_labels = sorted(set(y_test) | set(y_pred))
            report = classification_report(y_test, y_pred, labels=present_labels,
                                           target_names=le.classes_[present_labels], output_dict=True)
            report_df = pd.DataFrame(report).T.round(3)

            st.success(f"{selected_model} accuracy: **{report['accuracy']:.1%}**")
            st.dataframe(report_df, use_container_width=True)

            present_names = le.classes_[present_labels].tolist()
            cm = confusion_matrix(y_test, y_pred, labels=present_labels)
            export_dpi = dpi_export if pub_mode else 150

            # Confusion matrix
            if pub_mode:
                fig, ax = mpl_fig((max(6, len(present_names) * 0.7), max(5, len(present_names) * 0.6)))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=present_names, yticklabels=present_names,
                            ax=ax, linewidths=0.5, linecolor="#eeeeee")
                ax.set_xlabel("Predicted", fontsize=10)
                ax.set_ylabel("True", fontsize=10)
                ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold")
                ax.tick_params(axis="x", rotation=40, labelsize=8)
                ax.tick_params(axis="y", rotation=0, labelsize=8)
                plt.tight_layout()
                show_fig(st, fig, "confusion_matrix.png", pub_mode, export_dpi)
            else:
                fig = px.imshow(cm, x=present_names, y=present_names,
                                labels=dict(x="Predicted", y="True", color="Count"),
                                title="Confusion Matrix", color_continuous_scale="Blues",
                                text_auto=True)
                fig.update_layout(xaxis_tickangle=-40)
                st.plotly_chart(fig, use_container_width=True)

            # Feature importance
            if hasattr(clf, "feature_importances_"):
                importances = pd.Series(clf.feature_importances_, index=feature_cols)
                top10 = importances.nlargest(10).reset_index()
                top10.columns = ["Feature", "Importance"]

                if pub_mode:
                    fig, ax = mpl_fig((7, 4))
                    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top10)))
                    ax.barh(top10["Feature"], top10["Importance"], color=colors, edgecolor="white")
                    ax.set_xlabel("Importance", fontsize=10)
                    ax.set_title("Top 10 Feature Importances", fontsize=11, fontweight="bold")
                    ax.invert_yaxis()
                    plt.tight_layout()
                    show_fig(st, fig, "feature_importance.png", pub_mode, export_dpi)
                else:
                    fig2 = px.bar(top10, x="Importance", y="Feature", orientation="h",
                                  title="Top 10 Feature Importances", color="Importance",
                                  color_continuous_scale="Blues")
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info(f"{selected_model} does not provide feature importance in this configuration.")

# ── Tab 5: LOCO Validation ─────────────────────────────────────────────────────

with tab5:
    st.subheader("Leave-One-Cow-Out (LOCO) Validation")
    st.markdown("""
    **What is LOCO?** The model is trained on data from 5 cows and tested on the remaining cow — repeated for each cow.
    This validates whether the model generalizes to **unseen animals**, not just unseen data from the same animals.
    It is the gold standard validation method for wearable sensor studies in livestock.
    """)

    n_est_loco = st.slider("Number of trees (LOCO)", 50, 200, 100, step=50)

    if st.button("Run LOCO Validation"):
        all_cows = sorted(filtered["cow_id"].unique())

        if len(all_cows) < 2:
            st.warning("Select at least 2 cows to run LOCO.")
            st.stop()

        with st.spinner("Extracting features for all cows..."):
            feat_df = extract_features(filtered)
            feat_df = filter_rare_classes(feat_df, min_samples=2)

        feature_cols = [c for c in feat_df.columns if c not in ["label", "cow_id"]]
        results = []
        per_class_results = []
        progress = st.progress(0)

        for i, test_cow in enumerate(all_cows):
            train_df = feat_df[feat_df["cow_id"] != test_cow]
            test_df  = feat_df[feat_df["cow_id"] == test_cow]

            train_df = filter_rare_classes(train_df, min_samples=2)
            common_classes = set(train_df["label"].unique()) & set(test_df["label"].unique())

            if len(common_classes) < 2:
                progress.progress((i + 1) / len(all_cows))
                continue

            train_df = train_df[train_df["label"].isin(common_classes)]
            test_df  = test_df[test_df["label"].isin(common_classes)]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_df[feature_cols])
            X_test  = scaler.transform(test_df[feature_cols])

            le_fold = LabelEncoder()
            y_train = le_fold.fit_transform(train_df["label"])
            y_test  = le_fold.transform(test_df["label"])

            clf = RandomForestClassifier(n_estimators=n_est_loco, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            results.append({"Test Cow": f"Cow {test_cow}", "Accuracy": acc,
                             "Train samples": len(train_df), "Test samples": len(test_df)})

            report = classification_report(y_test, y_pred, target_names=le_fold.classes_, output_dict=True)
            for cls, metrics in report.items():
                if cls in le_fold.classes_:
                    per_class_results.append({"Cow": f"Cow {test_cow}", "Behavior": cls,
                                               "F1-score": round(metrics["f1-score"], 3)})
            progress.progress((i + 1) / len(all_cows))

        if not results:
            st.warning("Not enough data to run LOCO. Try selecting more cows and behaviors.")
            st.stop()

        results_df = pd.DataFrame(results)
        mean_acc = results_df["Accuracy"].mean()
        std_acc  = results_df["Accuracy"].std()
        export_dpi = dpi_export if pub_mode else 150

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean LOCO Accuracy", f"{mean_acc:.1%}")
        col2.metric("Std Deviation", f"{std_acc:.1%}")
        col3.metric("Cows evaluated", len(results_df))

        st.subheader("Accuracy per test cow")
        results_df["Accuracy %"] = (results_df["Accuracy"] * 100).round(1)

        if pub_mode:
            fig, ax = mpl_fig((7, 4))
            cmap_vals = plt.cm.Blues(results_df["Accuracy %"] / 100)
            bars = ax.bar(results_df["Test Cow"], results_df["Accuracy %"],
                          color=cmap_vals, edgecolor="white")
            ax.axhline(mean_acc * 100, color="darkorange", linestyle="--",
                       linewidth=1.5, label=f"Mean: {mean_acc:.1%}")
            for bar, val in zip(bars, results_df["Accuracy %"]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
            ax.set_ylim(0, 110)
            ax.set_xlabel("Test Cow", fontsize=10)
            ax.set_ylabel("Accuracy (%)", fontsize=10)
            ax.set_title("LOCO Accuracy per Test Cow", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            plt.tight_layout()
            show_fig(st, fig, "loco_accuracy.png", pub_mode, export_dpi)
        else:
            fig = px.bar(results_df, x="Test Cow", y="Accuracy %", color="Accuracy %",
                         color_continuous_scale="Blues", range_y=[0, 100], text="Accuracy %")
            fig.add_hline(y=mean_acc * 100, line_dash="dash", line_color="orange",
                          annotation_text=f"Mean: {mean_acc:.1%}", annotation_position="top right")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Details per cow")
        st.dataframe(results_df.set_index("Test Cow"), use_container_width=True)

        if per_class_results:
            st.subheader("F1-score per behavior across folds")
            per_class_df = pd.DataFrame(per_class_results)
            pivot = per_class_df.pivot(index="Behavior", columns="Cow", values="F1-score").round(3)
            pivot["Mean F1"] = pivot.mean(axis=1).round(3)
            pivot = pivot.sort_values("Mean F1", ascending=False)
            st.dataframe(pivot, use_container_width=True)

            if pub_mode:
                fig, ax = mpl_fig((9, 5))
                behaviors = per_class_df["Behavior"].unique()
                cmap = plt.get_cmap("tab20", len(behaviors))
                for i, beh in enumerate(sorted(behaviors)):
                    vals = per_class_df[per_class_df["Behavior"] == beh]["F1-score"]
                    ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                               boxprops=dict(facecolor=(*cmap(i)[:3], 0.6)),
                               medianprops=dict(color="black", linewidth=1.5),
                               whiskerprops=dict(linewidth=1),
                               capprops=dict(linewidth=1),
                               flierprops=dict(marker="o", markersize=4, alpha=0.5))
                ax.set_xticks(range(len(sorted(behaviors))))
                ax.set_xticklabels(sorted(behaviors), rotation=40, ha="right", fontsize=8)
                ax.set_ylabel("F1-score", fontsize=10)
                ax.set_title("F1-score Distribution per Behavior (across cows)",
                             fontsize=11, fontweight="bold")
                ax.set_ylim(-0.05, 1.05)
                plt.tight_layout()
                show_fig(st, fig, "loco_f1_boxplot.png", pub_mode, export_dpi)
            else:
                fig2 = px.box(per_class_df, x="Behavior", y="F1-score", color="Behavior",
                              title="F1-score distribution per behavior (across cows)",
                              color_discrete_sequence=px.colors.qualitative.Safe)
                fig2.update_layout(showlegend=False, xaxis_tickangle=-40)
                st.plotly_chart(fig2, use_container_width=True)
