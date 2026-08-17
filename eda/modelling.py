from __future__ import annotations
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Dark theme (nhất quán với toàn bộ eda/) ───────────────────────────────────
_BG      = "#0d1117"
_AXES_BG = "#161b22"
_EDGE    = "#30363d"
_TEXT    = "#c9d1d9"
_MUTED   = "#8b949e"
_GRID    = "#21262d"
_UP      = "#3fb950"
_DOWN    = "#f85149"
_BLUE    = "#58a6ff"
_ORANGE  = "#ff9500"
_PURPLE  = "#a371f7"
_YELLOW  = "#f0e68c"
_CYAN    = "#39d353"

# Màu cho tối đa 8 clusters
_CLUSTER_COLORS = [
    _BLUE, _ORANGE, _UP, _DOWN, _PURPLE, _YELLOW, _CYAN, _MUTED,
]

# Features để phân cụm (scale-independent, normalized internally)
_CLUSTER_FEATURES = [
    "rsi_14", "bb_pct", "bb_width",
    "macd_hist", "volume_ratio",
    "rolling_vol_30", "price_zscore",
    "atr_14", "return_pct",
]


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  _BG,
        "axes.facecolor":    _AXES_BG,
        "axes.edgecolor":    _EDGE,
        "axes.labelcolor":   _TEXT,
        "axes.grid":         True,
        "grid.color":        _GRID,
        "grid.alpha":        0.5,
        "grid.linewidth":    0.5,
        "text.color":        _TEXT,
        "xtick.color":       _MUTED,
        "ytick.color":       _MUTED,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.facecolor":  _AXES_BG,
        "legend.edgecolor":  _EDGE,
        "legend.labelcolor": _TEXT,
        "legend.fontsize":   8,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
        "savefig.facecolor": _BG,
        "font.size":         9,
        "axes.titlesize":    11,
        "axes.titlecolor":   _TEXT,
        "axes.titleweight":  "bold",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=_BG, edgecolor="none")
    plt.close(fig)
    logger.info(f"  ✔ Saved: {path.name}")


def _title_block(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, fontsize=14, fontweight="bold", color=_TEXT, y=0.98)
    fig.text(0.5, 0.965, subtitle, ha="center", fontsize=9, color=_MUTED)


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Trả về (X_scaled, feat_cols) — chuẩn hóa StandardScaler, dropna.
    """
    from sklearn.preprocessing import StandardScaler

    avail = [c for c in _CLUSTER_FEATURES if c in df.columns]
    X_raw = df[avail].dropna()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_raw),
        index=X_raw.index,
        columns=avail,
    )
    return X_scaled, avail


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class BaselineModels:
    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        output_dir: Path,
        k_range: tuple[int, int] = (2, 9),
        best_k: Optional[int] = None,
    ) -> dict:
        _apply_style()
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
            from sklearn.metrics import silhouette_score
            from sklearn.decomposition import PCA
        except ImportError:
            logger.error("scikit-learn not installed — skipping clustering.")
            return {}

        X, feat_cols = _prepare_features(df)

        if len(X) < 50:
            logger.warning(f"Not enough data for clustering ({len(X)} rows).")
            return {}

        logger.info(f"[Clustering] {symbol} {timeframe} — {len(X)} rows, {len(feat_cols)} features")

        results: dict = {"symbol": symbol, "timeframe": timeframe}

        # ── 09a: Elbow + Silhouette ───────────────────────────────────────────
        best_k, sil_scores, inertias = self._plot_elbow_silhouette(
            X, symbol, timeframe, output_dir, k_range, best_k
        )
        results["best_k"]       = best_k
        results["sil_scores"]   = sil_scores

        # ── 09b: K-Means ──────────────────────────────────────────────────────
        km_labels = self._plot_kmeans(
            X, df, symbol, timeframe, output_dir, best_k, feat_cols
        )
        results["kmeans_labels"] = km_labels

        # ── 09c: Hierarchical ─────────────────────────────────────────────────
        self._plot_hierarchical(X, df, symbol, timeframe, output_dir, best_k)

        # ── 09d: DBSCAN ───────────────────────────────────────────────────────
        db_labels = self._plot_dbscan(X, df, symbol, timeframe, output_dir)
        results["dbscan_labels"] = db_labels

        # ── 09e: PCA + t-SNE ──────────────────────────────────────────────────
        self._plot_projection(X, km_labels, symbol, timeframe, output_dir, best_k)

        # ── Text report ───────────────────────────────────────────────────────
        self._save_report(results, X, km_labels, db_labels, feat_cols, output_dir)

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 09a: Elbow + Silhouette
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_elbow_silhouette(
        self,
        X: pd.DataFrame,
        symbol: str, tf: str,
        out: Path,
        k_range: tuple,
        forced_k: Optional[int],
    ) -> tuple[int, dict, list]:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        ks        = list(range(k_range[0], k_range[1] + 1))
        inertias  = []
        sil_scores = {}

        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            if k >= 2:
                sil_scores[k] = round(silhouette_score(X, labels, sample_size=min(2000, len(X))), 4)

        best_k = forced_k or max(sil_scores, key=sil_scores.get)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)

        # Elbow
        ax0.plot(ks, inertias, color=_BLUE, linewidth=2, marker="o",
                 markerfacecolor=_ORANGE, markersize=8)
        ax0.axvline(best_k, color=_DOWN, linewidth=1.5, linestyle="--",
                    label=f"Best K = {best_k}")
        ax0.fill_between(ks, inertias, alpha=0.08, color=_BLUE)
        ax0.set_xlabel("Number of Clusters (K)")
        ax0.set_ylabel("Inertia (Within-cluster SSE)")
        ax0.set_title("Elbow Method — Optimal K Selection", loc="left")
        ax0.legend(framealpha=0.3)
        ax0.set_xticks(ks)

        # Silhouette
        sil_ks  = list(sil_scores.keys())
        sil_vals = list(sil_scores.values())
        bar_colors = [_UP if k == best_k else _BLUE for k in sil_ks]
        bars = ax1.bar(sil_ks, sil_vals, color=bar_colors, alpha=0.85, width=0.6)
        ax1.set_xlabel("Number of Clusters (K)")
        ax1.set_ylabel("Silhouette Score (higher = better)")
        ax1.set_title("Silhouette Score — Cluster Cohesion", loc="left")
        ax1.set_xticks(sil_ks)

        for bar, val in zip(bars, sil_vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8, color=_TEXT
            )

        ax1.text(
            0.97, 0.97,
            f"Best K = {best_k}\nSilhouette = {sil_scores[best_k]:.4f}",
            transform=ax1.transAxes, va="top", ha="right",
            fontsize=9, color=_TEXT,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=_BG, edgecolor=_EDGE)
        )

        _title_block(
            fig,
            f"{symbol} — Optimal K Selection (Elbow + Silhouette)",
            f"Timeframe: {tf}  |  K range: {k_range[0]}–{k_range[1]}  |  Best K = {best_k}"
        )
        _save(fig, out / "09a_elbow_silhouette.png")

        return best_k, sil_scores, inertias

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 09b: K-Means
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_kmeans(
        self,
        X: pd.DataFrame, df: pd.DataFrame,
        symbol: str, tf: str,
        out: Path, k: int, feat_cols: list,
    ) -> pd.Series:
        from sklearn.cluster import KMeans

        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        labels_s = pd.Series(labels, index=X.index, name="cluster")

        cluster_colors = [_CLUSTER_COLORS[i % len(_CLUSTER_COLORS)] for i in range(k)]

        fig = plt.figure(figsize=(24, 16))
        gs  = fig.add_gridspec(
            3, k,
            height_ratios=[2.5, 1.2, 1.5],
            hspace=0.35, wspace=0.3
        )

        # ── Row 0: Price timeline colored by cluster ──────────────────────────
        ax_price = fig.add_subplot(gs[0, :])
        close_aligned = df["close"].reindex(X.index)
        ax_price.plot(close_aligned.index, close_aligned.values,
                      color=_MUTED, linewidth=0.6, alpha=0.4, zorder=1)
        for cl in range(k):
            mask = labels_s == cl
            ax_price.scatter(
                labels_s[mask].index,
                close_aligned[mask],
                color=cluster_colors[cl],
                s=10, alpha=0.7, zorder=3,
                label=f"Cluster {cl} ({mask.sum():,})"
            )
        ax_price.set_title(
            f"Price Timeline — K-Means Clusters (K={k})", loc="left"
        )
        ax_price.legend(framealpha=0.3, ncol=k, loc="upper left", fontsize=8,
                        markerscale=2)
        ax_price.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
        )
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax_price.set_xticklabels([])

        # ── Row 1: Cluster size bars ──────────────────────────────────────────
        ax_size = fig.add_subplot(gs[1, :])
        cluster_counts = labels_s.value_counts().sort_index()
        ax_size.bar(
            [f"Cluster {i}" for i in cluster_counts.index],
            cluster_counts.values,
            color=[cluster_colors[i] for i in cluster_counts.index],
            alpha=0.85, width=0.5
        )
        for i, (ci, cnt) in enumerate(cluster_counts.items()):
            pct = cnt / len(labels_s) * 100
            ax_size.text(i, cnt + len(labels_s) * 0.01, f"{pct:.1f}%",
                         ha="center", va="bottom", fontsize=9, color=_TEXT)
        ax_size.set_title("Cluster Size Distribution", loc="left")
        ax_size.set_ylabel("Number of Candles")

        # ── Row 2: Feature profile per cluster (normalized bar) ───────────────
        cluster_means = X.copy()
        cluster_means["cluster"] = labels
        profile = cluster_means.groupby("cluster")[feat_cols].mean()

        for cl in range(k):
            ax_feat = fig.add_subplot(gs[2, cl])
            vals    = profile.loc[cl] if cl in profile.index else pd.Series(0, index=feat_cols)
            bar_c   = [_UP if v >= 0 else _DOWN for v in vals]
            ax_feat.barh(feat_cols, vals, color=bar_c, alpha=0.8, height=0.6)
            ax_feat.axvline(0, color=_MUTED, linewidth=0.7)
            ax_feat.set_title(f"Cluster {cl}", loc="left",
                              fontsize=9, color=cluster_colors[cl])
            ax_feat.tick_params(axis="y", labelsize=7)
            if cl > 0:
                ax_feat.set_yticklabels([])

        _title_block(
            fig,
            f"{symbol} — K-Means Clustering (K={k})",
            f"Timeframe: {tf}  |  Features: {', '.join(feat_cols[:4])}..."
        )
        _save(fig, out / "09b_kmeans_clusters.png")

        return labels_s

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 09c: Hierarchical Clustering
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_hierarchical(
        self,
        X: pd.DataFrame, df: pd.DataFrame,
        symbol: str, tf: str,
        out: Path, k: int,
    ) -> None:
        from sklearn.cluster import AgglomerativeClustering
        from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

        # Sample for dendrogram (max 300 points for readability)
        sample_size = min(300, len(X))
        X_sample = X.iloc[::max(1, len(X) // sample_size)]

        fig, (ax0, ax1) = plt.subplots(
            1, 2, figsize=(24, 10), constrained_layout=True
        )

        # ── Dendrogram ────────────────────────────────────────────────────────
        linked = linkage(X_sample.values, method="ward")
        dend = dendrogram(
            linked,
            ax=ax0,
            truncate_mode="level",
            p=5,
            color_threshold=linked[-k, 2],
            above_threshold_color=_MUTED,
            no_labels=True,
        )
        ax0.set_title(
            f"Hierarchical Dendrogram (Ward, n={len(X_sample)} sample)",
            loc="left"
        )
        ax0.set_xlabel("Samples")
        ax0.set_ylabel("Distance")
        ax0.axhline(
            linked[-k, 2], color=_DOWN, linewidth=1.2, linestyle="--",
            label=f"Cut for K={k} clusters"
        )
        ax0.legend(framealpha=0.3)

        # ── Timeline with hierarchical labels ─────────────────────────────────
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        hier_labels = agg.fit_predict(X.values)
        hier_s = pd.Series(hier_labels, index=X.index)

        close_aligned = df["close"].reindex(X.index)
        ax1.plot(close_aligned.index, close_aligned.values,
                 color=_MUTED, linewidth=0.5, alpha=0.35, zorder=1)
        for cl in range(k):
            mask  = hier_s == cl
            color = _CLUSTER_COLORS[cl % len(_CLUSTER_COLORS)]
            ax1.scatter(
                hier_s[mask].index, close_aligned[mask],
                color=color, s=9, alpha=0.7, zorder=3,
                label=f"Cluster {cl} ({mask.sum():,})"
            )
        ax1.set_title("Price Timeline — Hierarchical Clusters", loc="left")
        ax1.legend(framealpha=0.3, ncol=k, fontsize=8, markerscale=2,
                   loc="upper left")
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
        )
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        _title_block(
            fig,
            f"{symbol} — Hierarchical Agglomerative Clustering (K={k})",
            f"Timeframe: {tf}  |  Linkage: Ward  |  {len(X):,} samples"
        )
        _save(fig, out / "09c_hierarchical.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 09d: DBSCAN
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_dbscan(
        self,
        X: pd.DataFrame, df: pd.DataFrame,
        symbol: str, tf: str,
        out: Path,
    ) -> pd.Series:
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors

        # ── Auto-tune eps via k-NN distance ───────────────────────────────────
        nbrs   = NearestNeighbors(n_neighbors=5).fit(X)
        dists, _ = nbrs.kneighbors(X)
        knn_dist = np.sort(dists[:, -1])

        # Find "elbow" of kNN distance curve
        diff2 = np.diff(knn_dist, 2)
        eps_auto = float(knn_dist[np.argmax(diff2) + 2])
        eps_auto = round(max(0.3, min(eps_auto, 3.0)), 2)

        db = DBSCAN(eps=eps_auto, min_samples=5)
        db_labels_arr = db.fit_predict(X)
        db_labels = pd.Series(db_labels_arr, index=X.index)

        n_clusters = len(set(db_labels_arr)) - (1 if -1 in db_labels_arr else 0)
        n_noise    = (db_labels_arr == -1).sum()

        fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
        ax0, ax1, ax2 = axes

        # ── k-NN distance (eps tuning) ────────────────────────────────────────
        ax0.plot(knn_dist, color=_BLUE, linewidth=1.0)
        ax0.axhline(eps_auto, color=_DOWN, linewidth=1.2, linestyle="--",
                    label=f"Auto eps = {eps_auto}")
        ax0.set_title("k-NN Distance (5th neighbor) — eps Selection", loc="left")
        ax0.set_xlabel("Points (sorted)")
        ax0.set_ylabel("5-NN Distance")
        ax0.legend(framealpha=0.3)

        # ── Cluster timeline ──────────────────────────────────────────────────
        close_aligned = df["close"].reindex(X.index)
        ax1.plot(close_aligned.index, close_aligned.values,
                 color=_MUTED, linewidth=0.5, alpha=0.35, zorder=1)

        unique_cls = sorted(set(db_labels_arr))
        for cl in unique_cls:
            mask  = db_labels == cl
            if cl == -1:
                color, lbl, sz = _DOWN, f"Noise/Outlier ({n_noise})", 18
            else:
                color = _CLUSTER_COLORS[cl % len(_CLUSTER_COLORS)]
                lbl   = f"Cluster {cl} ({mask.sum():,})"
                sz    = 8
            ax1.scatter(
                db_labels[mask].index, close_aligned[mask],
                color=color, s=sz, alpha=0.75, zorder=3, label=lbl
            )

        ax1.set_title(
            f"Price Timeline — DBSCAN (eps={eps_auto}, {n_clusters} clusters)",
            loc="left"
        )
        ax1.legend(framealpha=0.3, ncol=2, fontsize=7, markerscale=2, loc="upper left")
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"${v:,.0f}")
        )
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # ── Cluster size (DBSCAN) ─────────────────────────────────────────────
        counts = pd.Series(db_labels_arr).value_counts().sort_index()
        bar_labels = [f"Noise" if c == -1 else f"C{c}" for c in counts.index]
        bar_colors = [_DOWN if c == -1 else _CLUSTER_COLORS[c % len(_CLUSTER_COLORS)]
                      for c in counts.index]
        bars = ax2.bar(bar_labels, counts.values, color=bar_colors, alpha=0.85, width=0.6)
        for bar, val in zip(bars, counts.values):
            pct = val / len(db_labels_arr) * 100
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + len(db_labels_arr) * 0.005,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=8, color=_TEXT
            )
        ax2.set_title("DBSCAN Cluster Distribution", loc="left")
        ax2.set_ylabel("Number of Candles")

        summary_txt = (
            f"eps = {eps_auto}  |  min_samples = 5\n"
            f"Clusters: {n_clusters}  |  Noise: {n_noise} ({n_noise/len(db_labels_arr)*100:.1f}%)"
        )
        fig.text(0.5, 0.01, summary_txt, ha="center", fontsize=9,
                 color=_TEXT, style="italic")

        _title_block(
            fig,
            f"{symbol} — DBSCAN Clustering (Density-Based)",
            f"Timeframe: {tf}  |  Auto-tuned eps={eps_auto}  |  {n_clusters} clusters + noise"
        )
        _save(fig, out / "09d_dbscan.png")

        return db_labels

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 09e: PCA + t-SNE 2D Projections
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_projection(
        self,
        X: pd.DataFrame,
        km_labels: pd.Series,
        symbol: str, tf: str,
        out: Path, k: int,
    ) -> None:
        from sklearn.decomposition import PCA

        # PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X.values)
        var_exp = pca.explained_variance_ratio_ * 100

        fig, axes = plt.subplots(1, 2, figsize=(22, 9), constrained_layout=True)
        ax0, ax1 = axes

        # ── PCA scatter ───────────────────────────────────────────────────────
        for cl in range(k):
            mask  = km_labels.values == cl
            color = _CLUSTER_COLORS[cl % len(_CLUSTER_COLORS)]
            ax0.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                color=color, s=12, alpha=0.6, label=f"Cluster {cl}",
            )
        ax0.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var)")
        ax0.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var)")
        ax0.set_title(
            f"PCA 2D — K-Means Clusters (total var: {sum(var_exp):.1f}%)",
            loc="left"
        )
        ax0.legend(framealpha=0.3, markerscale=2)

        # ── t-SNE scatter ─────────────────────────────────────────────────────
        try:
            from sklearn.manifold import TSNE
            # Sample for speed (t-SNE is O(n^2))
            n_tsne  = min(2000, len(X))
            idx_s   = np.random.choice(len(X), n_tsne, replace=False)
            X_sub   = X.iloc[idx_s]
            lab_sub = km_labels.iloc[idx_s]

            tsne   = TSNE(n_components=2, perplexity=30, random_state=42,
                          n_iter=500, learning_rate="auto", init="pca")
            X_tsne = tsne.fit_transform(X_sub.values)

            for cl in range(k):
                mask  = lab_sub.values == cl
                color = _CLUSTER_COLORS[cl % len(_CLUSTER_COLORS)]
                ax1.scatter(
                    X_tsne[mask, 0], X_tsne[mask, 1],
                    color=color, s=12, alpha=0.6, label=f"Cluster {cl}",
                )
            ax1.set_xlabel("t-SNE dim 1")
            ax1.set_ylabel("t-SNE dim 2")
            ax1.set_title(
                f"t-SNE 2D — K-Means Clusters (n={n_tsne} sample)",
                loc="left"
            )
            ax1.legend(framealpha=0.3, markerscale=2)
        except Exception as e:
            ax1.text(0.5, 0.5, f"t-SNE failed:\n{e}",
                     transform=ax1.transAxes, ha="center", va="center",
                     fontsize=10, color=_DOWN)
            ax1.set_title("t-SNE 2D (failed)", loc="left")

        _title_block(
            fig,
            f"{symbol} — Cluster Projection: PCA & t-SNE",
            f"Timeframe: {tf}  |  K={k}  |  Features: {', '.join(X.columns.tolist()[:5])}..."
        )
        _save(fig, out / "09e_pca_tsne.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Text Report
    # ──────────────────────────────────────────────────────────────────────────

    def _save_report(
        self,
        results: dict,
        X: pd.DataFrame,
        km_labels: pd.Series,
        db_labels: pd.Series,
        feat_cols: list,
        out_dir: Path,
    ) -> None:
        """Lưu clustering text report."""
        symbol = results["symbol"]
        tf     = results["timeframe"]
        k      = results["best_k"]
        sil    = results.get("sil_scores", {})

        lines = [
            "=" * 70,
            "  CLUSTERING ANALYSIS REPORT",
            f"  Symbol: {symbol}  |  Timeframe: {tf}  |  N = {len(X):,}",
            "=" * 70,
            "",
            f"-- K-Means Configuration --------------------------------",
            f"  Best K        : {k}",
            f"  Features used : {', '.join(feat_cols)}",
            f"  Silhouette    : {sil.get(k, 'N/A')}",
            "",
            "-- Silhouette Scores by K --------------------------------",
        ]
        for ki, score in sorted(sil.items()):
            marker = "  <-- BEST" if ki == k else ""
            lines.append(f"  K={ki}  Silhouette = {score:.4f}{marker}")

        # K-Means cluster sizes
        if km_labels is not None and len(km_labels):
            lines += ["", "-- K-Means Cluster Sizes ---------------------------------"]
            counts = km_labels.value_counts().sort_index()
            for cl, cnt in counts.items():
                pct = cnt / len(km_labels) * 100
                lines.append(f"  Cluster {cl}: {cnt:>6,} candles ({pct:.1f}%)")

        # Feature profile per cluster
        if km_labels is not None and len(km_labels):
            lines += ["", "-- K-Means Feature Profile (normalized mean) -------------"]
            profile = X.copy()
            profile["cluster"] = km_labels.values
            means = profile.groupby("cluster")[feat_cols].mean()
            header = f"  {'Feature':<22}" + "".join(f"  Cl{c:<5}" for c in means.index)
            lines.append(header)
            for feat in feat_cols:
                row = f"  {feat:<22}"
                for c in means.index:
                    row += f"  {means.loc[c, feat]:+.3f}"
                lines.append(row)

        # DBSCAN summary
        if db_labels is not None and len(db_labels):
            n_cl    = len(set(db_labels.values)) - (1 if -1 in db_labels.values else 0)
            n_noise = (db_labels == -1).sum()
            lines += [
                "",
                "-- DBSCAN Summary ----------------------------------------",
                f"  Clusters detected : {n_cl}",
                f"  Noise / Outliers  : {n_noise} ({n_noise/len(db_labels)*100:.1f}%)",
            ]

        lines += ["", "=" * 70, ""]
        path = out_dir / "09_clustering_report.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"  ✔ Saved: {path.name}")
