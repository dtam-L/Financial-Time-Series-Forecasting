
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Optional, List

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

warnings.filterwarnings("ignore")

# ── Dark theme constants (nhất quán với visualized.py) ───────────────────────
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


def _apply_dark_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor":  _BG,
        "axes.facecolor":    _AXES_BG,
        "axes.edgecolor":    _EDGE,
        "axes.labelcolor":   _TEXT,
        "axes.grid":         True,
        "grid.color":        _GRID,
        "grid.alpha":        0.6,
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
        "figure.dpi":        130,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
        "savefig.facecolor": _BG,
        "font.family":       "DejaVu Sans",
        "font.size":         9,
        "axes.titlesize":    11,
        "axes.titlecolor":   _TEXT,
        "axes.titleweight":  "bold",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


_apply_dark_theme()


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=_BG, edgecolor="none")
    plt.close(fig)
    logger.info(f"  ✔ Saved: {path.name}")


def _title_block(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, fontsize=14, fontweight="bold", color=_TEXT, y=0.98)
    fig.text(0.5, 0.965, subtitle, ha="center", fontsize=9, color=_MUTED)


# ── Feature column selectors ─────────────────────────────────────────────────

_RETURN_FEATS = ["return_pct", "log_return"]
_MA_FEATS     = ["ma_7", "ma_25", "ma_99", "ema_12", "ema_26"]
_MACD_FEATS   = ["macd", "macd_signal", "macd_hist"]
_RSI_FEATS    = ["rsi_14"]
_BB_FEATS     = ["bb_mid", "bb_upper", "bb_lower", "bb_width", "bb_pct"]
_ATR_FEATS    = ["atr_14"]
_VOL_FEATS    = ["volume_ma_20", "volume_ratio"]
_RISK_FEATS   = ["rolling_vol_30", "price_zscore", "drawdown_pct"]
_CYCLICAL     = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
]

# Features phù hợp để tính VIF / MI (loại trừ price-level & cyclical raw)
_ANALYTICAL_FEATS = (
    _RETURN_FEATS
    + _MACD_FEATS
    + _RSI_FEATS
    + _BB_FEATS[3:]       # bb_width, bb_pct
    + _ATR_FEATS
    + _VOL_FEATS[1:]      # volume_ratio
    + _RISK_FEATS
)

_GROUP_LABELS = {
    **{f: "Returns"    for f in _RETURN_FEATS},
    **{f: "MA/EMA"     for f in _MA_FEATS},
    **{f: "MACD"       for f in _MACD_FEATS},
    **{f: "RSI"        for f in _RSI_FEATS},
    **{f: "Bollinger"  for f in _BB_FEATS},
    **{f: "ATR"        for f in _ATR_FEATS},
    **{f: "Volume"     for f in _VOL_FEATS},
    **{f: "Risk"       for f in _RISK_FEATS},
    **{f: "Cyclical"   for f in _CYCLICAL},
}

_GROUP_COLORS = {
    "Returns":   _BLUE,
    "MA/EMA":    _YELLOW,
    "MACD":      _ORANGE,
    "RSI":       _PURPLE,
    "Bollinger": _CYAN,
    "ATR":       _DOWN,
    "Volume":    _UP,
    "Risk":      "#ff6e6e",
    "Cyclical":  _MUTED,
}


def _available(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Return only the cols that exist in df."""
    return [c for c in cols if c in df.columns]


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class FeatureEDA:
    """
    Post-feature-engineering EDA: phân tích chuyên sâu các indicators và
    derived features sau khi đã chạy qua ``engineer_features()``.

    Parameters
    ----------
    output_root : Path
        Root directory cho output (reports/figures/).

    Usage
    -----
    >>> feda = FeatureEDA(output_root=Path("reports/figures"))
    >>> feda.run_all(df, "BTC/USDT", "1d")
    """

    def __init__(self, output_root: Optional[Path] = None) -> None:
        _apply_dark_theme()
        _root = Path(__file__).resolve().parent.parent
        self._out_root = output_root or (_root / "reports" / "figures")
        self._out_root.mkdir(parents=True, exist_ok=True)

    def _outdir(self, symbol: str, timeframe: str) -> Path:
        d = self._out_root / f"{symbol.replace('/', '_')}_{timeframe}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ──────────────────────────────────────────────────────────────────────────
    # Orchestrator
    # ──────────────────────────────────────────────────────────────────────────

    def run_all(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> dict:
        """
        Chạy toàn bộ Feature EDA pipeline và lưu tất cả charts.

        Parameters
        ----------
        df        : DataFrame sau ``engineer_features()`` — có DatetimeIndex.
        symbol    : e.g. 'BTC/USDT'
        timeframe : e.g. '1d'

        Returns
        -------
        dict  : summary statistics cho report text.
        """
        logger.info(f"[FeatureEDA] Starting post-FE EDA: {symbol} {timeframe}")
        out = self._outdir(symbol, timeframe)

        summary: dict = {}

        self._plot_correlation_heatmap(df, symbol, timeframe, out)
        self._plot_feature_distributions(df, symbol, timeframe, out)
        mi_scores = self._plot_mutual_information(df, symbol, timeframe, out)
        summary["mutual_information"] = mi_scores
        stat_results = self._plot_stationarity_overview(df, symbol, timeframe, out)
        summary["stationarity"] = stat_results
        self._plot_rolling_feature_stability(df, symbol, timeframe, out)
        vif_df = self._plot_vif(df, symbol, timeframe, out)
        summary["vif"] = vif_df

        self._save_text_report(summary, symbol, timeframe, out, len(df))
        logger.success(f"[FeatureEDA] Done -> {out}")
        return summary

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 10: Feature Correlation Heatmap
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_correlation_heatmap(
        self, df: pd.DataFrame, symbol: str, tf: str, out: Path
    ) -> None:
        """
        Vẽ correlation heatmap cho tất cả numeric features (trừ price-level MA).
        Sử dụng hierarchical clustering để nhóm features liên quan.
        """
        exclude_prefix = ("open", "high", "low", "close", "volume",
                          "ma_", "bb_mid", "bb_upper", "bb_lower",
                          "ema_", "volume_ma_", "year", "month",
                          "day_of", "hour", "regime", "vol_regime")
        num_cols = [
            c for c in df.select_dtypes(include=np.number).columns
            if not c.startswith(exclude_prefix)
        ]
        if len(num_cols) < 4:
            logger.warning("[FeatureEDA] Not enough features for correlation heatmap.")
            return

        corr_df = df[num_cols].dropna()
        corr    = corr_df.corr()

        # Hierarchical clustering sort
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
            dist = 1 - corr.abs()
            np.fill_diagonal(dist.values, 0)
            dist_sq = squareform(np.clip(dist.values, 0, None))
            lnk = linkage(dist_sq, method="ward")
            order = leaves_list(lnk)
            corr = corr.iloc[order, order]
        except Exception:
            pass  # fallback: original order

        n    = len(corr)
        size = max(14, n * 0.65)
        fig, ax = plt.subplots(figsize=(size, size * 0.85))

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(
            corr,
            ax=ax,
            mask=mask,
            cmap=cmap,
            vmin=-1, vmax=1, center=0,
            annot=n <= 22,
            fmt=".2f",
            annot_kws={"size": max(5, 10 - n // 5)},
            linewidths=0.4,
            linecolor=_EDGE,
            square=True,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        )

        # Color-code axis labels by feature group
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            feat = label.get_text()
            grp  = _GROUP_LABELS.get(feat, "Other")
            label.set_color(_GROUP_COLORS.get(grp, _TEXT))
            label.set_fontsize(8)

        ax.set_title(
            "Feature Correlation Matrix (Clustered)",
            loc="left", pad=12
        )
        ax.tick_params(axis="x", rotation=45)

        # Legend for groups
        from matplotlib.patches import Patch
        seen_groups = {_GROUP_LABELS.get(c, "Other") for c in corr.columns}
        patches = [
            Patch(facecolor=_GROUP_COLORS.get(g, _TEXT), label=g)
            for g in sorted(seen_groups)
        ]
        ax.legend(
            handles=patches, loc="upper right", framealpha=0.3,
            fontsize=7, title="Feature Group", title_fontsize=7,
        )

        _title_block(
            fig,
            f"{symbol} -- Feature Correlation Heatmap (Post-FE)",
            f"Timeframe: {tf}  |  N={len(corr_df):,} rows  |  {n} features"
        )
        _save(fig, out / "10_feature_correlation.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 11: Feature Distribution Grid
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_feature_distributions(
        self, df: pd.DataFrame, symbol: str, tf: str, out: Path
    ) -> None:
        """
        Grid plot: histogram + KDE cho mỗi engineered feature.
        Chú thích mean, std, skew, kurt trên mỗi subplot.
        """
        groups: dict = {
            "Returns":   _available(df, _RETURN_FEATS),
            "MACD":      _available(df, _MACD_FEATS),
            "RSI":       _available(df, _RSI_FEATS),
            "Bollinger": _available(df, ["bb_width", "bb_pct"]),
            "ATR":       _available(df, _ATR_FEATS),
            "Volume":    _available(df, ["volume_ratio"]),
            "Risk":      _available(df, _RISK_FEATS),
            "Cyclical":  _available(df, _CYCLICAL),
        }

        all_feats = [(grp, feat) for grp, feats in groups.items() for feat in feats]
        if not all_feats:
            return

        ncols = 4
        nrows = int(np.ceil(len(all_feats) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 5, nrows * 3.8),
            constrained_layout=True
        )
        axes_flat = np.array(axes).flatten()

        for idx, (grp, feat) in enumerate(all_feats):
            ax  = axes_flat[idx]
            col = df[feat].dropna()
            if len(col) < 10:
                ax.set_visible(False)
                continue

            color = _GROUP_COLORS.get(grp, _BLUE)

            ax.hist(
                col, bins=50, density=True,
                color=color, alpha=0.45, edgecolor="none"
            )

            # KDE overlay
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(col)
                x_min, x_max = col.quantile(0.001), col.quantile(0.999)
                x_kde = np.linspace(x_min, x_max, 200)
                ax.plot(x_kde, kde(x_kde), color=color, linewidth=1.8)
            except Exception:
                pass

            mu  = col.mean()
            sig = col.std()
            sk  = col.skew()
            ku  = col.kurtosis()
            ax.axvline(mu, color=_TEXT, linewidth=1.0, linestyle="--", alpha=0.7)

            stats_txt = f"mu={mu:.3g}  sigma={sig:.3g}\nSkew={sk:.2f}  Kurt={ku:.2f}"
            ax.text(
                0.97, 0.95, stats_txt,
                transform=ax.transAxes, va="top", ha="right",
                fontsize=7, color=_TEXT,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor=_BG, edgecolor=_EDGE, alpha=0.8
                )
            )

            ax.set_title(f"[{grp}]  {feat}", loc="left", fontsize=9)
            ax.set_xlabel("")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2g}"))

        for idx in range(len(all_feats), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        _title_block(
            fig,
            f"{symbol} -- Engineered Feature Distributions",
            f"Timeframe: {tf}  |  {len(all_feats)} features  |  N={len(df):,} rows"
        )
        _save(fig, out / "11_feature_distributions.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 12: Mutual Information
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_mutual_information(
        self, df: pd.DataFrame, symbol: str, tf: str, out: Path
    ) -> dict:
        """
        Tính Mutual Information giữa mỗi feature và:
          - log_return (current)
          - future_return (log_return shifted -1: "predict next candle")
        Vẽ horizontal bar chart, sorted by MI với future_return.
        """
        try:
            from sklearn.feature_selection import mutual_info_regression
        except ImportError:
            logger.warning("[FeatureEDA] scikit-learn not installed -- skipping MI chart.")
            return {}

        feats = _available(df, _ANALYTICAL_FEATS + _CYCLICAL)
        if not feats:
            return {}

        df_mi = df[feats + ["log_return"]].copy()
        df_mi["future_return"] = df_mi["log_return"].shift(-1)
        df_mi = df_mi.dropna()

        if len(df_mi) < 50:
            return {}

        X = df_mi[feats].values

        mi_current = mutual_info_regression(
            X, df_mi["log_return"].values, random_state=42
        )
        mi_future = mutual_info_regression(
            X, df_mi["future_return"].values, random_state=42
        )

        mi_df = pd.DataFrame({
            "feature":    feats,
            "mi_current": mi_current,
            "mi_future":  mi_future,
        }).sort_values("mi_future", ascending=True)

        n = len(mi_df)
        fig, (ax0, ax1) = plt.subplots(
            1, 2,
            figsize=(20, max(8, n * 0.38)),
            constrained_layout=True
        )

        colors_cur = [
            _GROUP_COLORS.get(_GROUP_LABELS.get(f, "Other"), _BLUE)
            for f in mi_df["feature"]
        ]
        ax0.barh(
            mi_df["feature"], mi_df["mi_current"],
            color=colors_cur, alpha=0.8, height=0.65
        )
        ax0.set_title("Mutual Information -- Current Log Return", loc="left")
        ax0.set_xlabel("MI Score")

        colors_fut = [
            _GROUP_COLORS.get(_GROUP_LABELS.get(f, "Other"), _ORANGE)
            for f in mi_df["feature"]
        ]
        ax1.barh(
            mi_df["feature"], mi_df["mi_future"],
            color=colors_fut, alpha=0.8, height=0.65
        )
        ax1.set_title("Mutual Information -- Future Log Return (t+1)", loc="left")
        ax1.set_xlabel("MI Score")

        for ax, col in [(ax0, "mi_current"), (ax1, "mi_future")]:
            for i, (_, row) in enumerate(mi_df.iterrows()):
                val = row[col]
                ax.text(
                    val + 0.001, i, f"{val:.4f}",
                    va="center", fontsize=7, color=_TEXT
                )
            ax.tick_params(axis="y", labelsize=8)

        _title_block(
            fig,
            f"{symbol} -- Feature Mutual Information (Predictive Power)",
            f"Timeframe: {tf}  |  Target: log_return & future_return  |  N={len(df_mi):,}"
        )
        _save(fig, out / "12_mutual_information.png")

        return mi_df.set_index("feature").to_dict(orient="index")

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 13: Stationarity Overview (ADF p-values)
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_stationarity_overview(
        self, df: pd.DataFrame, symbol: str, tf: str, out: Path
    ) -> dict:
        """
        Chạy ADF test cho từng numeric feature và vẽ heatmap p-values.
        p < 0.05 = stationary (green), ngược lại = non-stationary (red).
        """
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            logger.warning("[FeatureEDA] statsmodels not installed -- skipping stationarity chart.")
            return {}

        feats = _available(df, _ANALYTICAL_FEATS)
        results: dict = {}

        for feat in feats:
            series = df[feat].dropna()
            if len(series) < 30:
                continue
            try:
                adf_stat, p_val, _, _, _, _ = adfuller(series, autolag="AIC")
                results[feat] = {
                    "p_value":       round(p_val, 4),
                    "statistic":     round(adf_stat, 4),
                    "is_stationary": p_val < 0.05,
                }
            except Exception as e:
                logger.debug(f"ADF failed for {feat}: {e}")

        if not results:
            return {}

        stat_df = pd.DataFrame(results).T
        stat_df["group"] = stat_df.index.map(
            lambda f: _GROUP_LABELS.get(f, "Other")
        )
        stat_df = stat_df.sort_values(["group", "p_value"])
        ordered = stat_df.index.tolist()

        fig, (ax_bar, ax_hm) = plt.subplots(
            1, 2,
            figsize=(20, max(8, len(ordered) * 0.5)),
            gridspec_kw={"width_ratios": [2, 1]},
            constrained_layout=True
        )

        bar_colors = [
            _UP if stat_df.loc[f, "is_stationary"] else _DOWN
            for f in ordered
        ]
        ax_bar.barh(ordered, stat_df.loc[ordered, "p_value"].values,
                    color=bar_colors, alpha=0.8, height=0.65)
        ax_bar.axvline(0.05, color=_TEXT, linewidth=1.2, linestyle="--",
                       label="alpha = 0.05 threshold")
        ax_bar.set_xlabel("ADF p-value (lower = more stationary)")
        ax_bar.set_title("ADF Stationarity Test -- p-values", loc="left")
        ax_bar.legend(framealpha=0.3)

        for i, feat in enumerate(ordered):
            p = stat_df.loc[feat, "p_value"]
            is_s = stat_df.loc[feat, "is_stationary"]
            label = f"{p:.4f} OK" if is_s else f"{p:.4f} !!"
            ax_bar.text(
                float(p) + 0.002, i, label,
                va="center", fontsize=7,
                color=_UP if is_s else _DOWN
            )

        binary = stat_df.loc[ordered, "is_stationary"].astype(int).values.reshape(-1, 1)
        sns.heatmap(
            binary,
            ax=ax_hm,
            cmap=sns.color_palette([_DOWN, _UP], as_cmap=True),
            vmin=0, vmax=1,
            yticklabels=ordered,
            xticklabels=["Stationary?"],
            annot=[["YES" if v == 1 else "NO"] for v in binary.flatten()],
            fmt="",
            annot_kws={"size": 8},
            linewidths=0.5,
            linecolor=_EDGE,
            cbar=False,
        )
        ax_hm.tick_params(axis="y", labelsize=7)
        ax_hm.set_title("ADF Result (p < 0.05)", loc="left")

        n_stat    = sum(1 for r in results.values() if r["is_stationary"])
        n_nonstat = len(results) - n_stat
        fig.text(
            0.5, 0.01,
            f"Stationary: {n_stat}/{len(results)}  OK  |  Non-stationary: {n_nonstat}/{len(results)}  !!",
            ha="center", fontsize=9, color=_TEXT, style="italic"
        )

        _title_block(
            fig,
            f"{symbol} -- Feature Stationarity Overview (ADF Test)",
            f"Timeframe: {tf}  |  alpha = 0.05  |  {len(results)} features tested"
        )
        _save(fig, out / "13_stationarity_overview.png")

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 14: Rolling Feature Stability
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_rolling_feature_stability(
        self, df: pd.DataFrame, symbol: str, tf: str, out: Path
    ) -> None:
        """
        Vẽ rolling mean (window=90) của các key features để phát hiện
        concept drift / distribution shift theo thời gian.
        """
        key_features = _available(df, [
            "rsi_14", "bb_pct", "macd", "volume_ratio",
            "rolling_vol_30", "price_zscore", "atr_14", "bb_width",
        ])
        if not key_features:
            return

        window = min(90, max(20, len(df) // 10))
        ncols  = 2
        nrows  = int(np.ceil(len(key_features) / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 11, nrows * 3.5),
            constrained_layout=True
        )
        axes_flat = np.array(axes).flatten()

        for idx, feat in enumerate(key_features):
            ax    = axes_flat[idx]
            col   = df[feat].dropna()
            grp   = _GROUP_LABELS.get(feat, "Other")
            color = _GROUP_COLORS.get(grp, _BLUE)

            ax.plot(col.index, col, color=color, linewidth=0.5, alpha=0.25, zorder=1)

            roll_mean = col.rolling(window).mean()
            ax.plot(roll_mean.index, roll_mean,
                    color=color, linewidth=1.8, alpha=0.9, zorder=3,
                    label=f"Rolling mean ({window})")

            roll_std = col.rolling(window).std()
            ax.fill_between(
                roll_mean.index,
                roll_mean - roll_std, roll_mean + roll_std,
                color=color, alpha=0.10, zorder=2
            )

            overall_mean = col.mean()
            ax.axhline(overall_mean, color=_MUTED, linewidth=0.8,
                       linestyle="--", alpha=0.7,
                       label=f"Overall mean = {overall_mean:.3g}")

            ax.set_title(f"[{grp}]  {feat} -- Rolling Stability", loc="left", fontsize=9)
            ax.legend(framealpha=0.3, fontsize=7)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.tick_params(axis="x", rotation=30)

        for idx in range(len(key_features), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        _title_block(
            fig,
            f"{symbol} -- Feature Stability / Concept Drift Check",
            f"Timeframe: {tf}  |  Rolling window = {window} candles  |  N={len(df):,}"
        )
        _save(fig, out / "14_feature_stability.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 15: VIF — Variance Inflation Factor
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_vif(
        self, df: pd.DataFrame, symbol: str, tf: str, out: Path
    ) -> pd.DataFrame:
        """
        Tính VIF cho từng feature để phát hiện multicollinearity.
        VIF > 10 = cần xem xét loại bỏ, VIF > 5 = cảnh báo.
        """
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            logger.warning("[FeatureEDA] statsmodels not installed -- skipping VIF chart.")
            return pd.DataFrame()

        feats = _available(df, _ANALYTICAL_FEATS)
        if len(feats) < 3:
            return pd.DataFrame()

        X = df[feats].dropna()
        if len(X) < 50:
            return pd.DataFrame()

        X_norm = (X - X.mean()) / (X.std() + 1e-10)

        vif_data = []
        for i, feat in enumerate(feats):
            try:
                vif = variance_inflation_factor(X_norm.values, i)
                vif_data.append({"feature": feat, "VIF": round(float(vif), 2)})
            except Exception:
                vif_data.append({"feature": feat, "VIF": np.nan})

        vif_df = pd.DataFrame(vif_data).dropna()
        vif_df = vif_df.sort_values("VIF", ascending=True)

        n   = len(vif_df)
        fig, ax = plt.subplots(figsize=(14, max(8, n * 0.42)), constrained_layout=True)

        bar_colors = []
        for v in vif_df["VIF"]:
            if v > 10:
                bar_colors.append(_DOWN)
            elif v > 5:
                bar_colors.append(_ORANGE)
            else:
                bar_colors.append(_UP)

        ax.barh(
            vif_df["feature"], vif_df["VIF"],
            color=bar_colors, alpha=0.85, height=0.65
        )

        ax.axvline(5,  color=_ORANGE, linewidth=1.2, linestyle="--",
                   alpha=0.8, label="VIF = 5 (moderate concern)")
        ax.axvline(10, color=_DOWN,   linewidth=1.2, linestyle="--",
                   alpha=0.8, label="VIF = 10 (high multicollinearity)")
        ax.set_xlabel("Variance Inflation Factor (VIF)")
        ax.set_title("Feature Multicollinearity -- VIF Scores", loc="left")

        for i, (_, row) in enumerate(vif_df.iterrows()):
            v = row["VIF"]
            tag = "  HIGH" if v > 10 else ("  MOD" if v > 5 else "  OK")
            ax.text(
                v + 0.2, i,
                f"{v:.1f}{tag}",
                va="center", fontsize=7.5,
                color=_DOWN if v > 10 else (_ORANGE if v > 5 else _UP)
            )

        from matplotlib.patches import Patch
        legend_patches = [
            Patch(color=_UP,     label="VIF <= 5   -- OK"),
            Patch(color=_ORANGE, label="VIF 5-10   -- Moderate"),
            Patch(color=_DOWN,   label="VIF > 10   -- High Collinearity"),
        ]
        ax.legend(handles=legend_patches, framealpha=0.3, loc="lower right")

        n_high = (vif_df["VIF"] > 10).sum()
        n_mod  = ((vif_df["VIF"] > 5) & (vif_df["VIF"] <= 10)).sum()
        n_ok   = (vif_df["VIF"] <= 5).sum()

        fig.text(
            0.5, 0.01,
            f"OK (<=5): {n_ok}  |  Moderate (5-10): {n_mod}  |  High (>10): {n_high}",
            ha="center", fontsize=9, color=_TEXT, style="italic"
        )

        _title_block(
            fig,
            f"{symbol} -- Variance Inflation Factor (Multicollinearity)",
            f"Timeframe: {tf}  |  {n} features  |  Threshold: VIF > 10"
        )
        _save(fig, out / "15_vif_multicollinearity.png")

        return vif_df

    # ──────────────────────────────────────────────────────────────────────────
    # Text Report
    # ──────────────────────────────────────────────────────────────────────────

    def _save_text_report(
        self,
        summary: dict,
        symbol: str,
        tf: str,
        out_dir: Path,
        n_rows: int,
    ) -> None:
        """Lưu text summary report của Feature EDA."""
        lines = [
            "=" * 70,
            "  FEATURE EDA REPORT (Post-Feature-Engineering)",
            f"  Symbol: {symbol}  |  Timeframe: {tf}  |  N = {n_rows:,}",
            "=" * 70,
        ]

        mi = summary.get("mutual_information", {})
        if mi:
            lines += ["", "-- Top 10 Features by MI with Future Return -------------------------"]
            mi_sorted = sorted(
                mi.items(), key=lambda x: x[1].get("mi_future", 0), reverse=True
            )
            for rank, (feat, scores) in enumerate(mi_sorted[:10], 1):
                lines.append(
                    f"  {rank:>2}. {feat:<22}  "
                    f"MI(current)={scores['mi_current']:.4f}  "
                    f"MI(future)={scores['mi_future']:.4f}"
                )

        stat = summary.get("stationarity", {})
        if stat:
            n_stat    = sum(1 for r in stat.values() if r["is_stationary"])
            n_nonstat = len(stat) - n_stat
            lines += [
                "",
                "-- Stationarity Summary (ADF Test, alpha=0.05) ----------------------",
                f"  Stationary     : {n_stat}/{len(stat)} features",
                f"  Non-Stationary : {n_nonstat}/{len(stat)} features",
                "",
                "  Non-stationary features (may need differencing):",
            ]
            for feat, r in stat.items():
                if not r["is_stationary"]:
                    lines.append(f"    * {feat:<22}  p = {r['p_value']:.4f}")

        vif_df = summary.get("vif", pd.DataFrame())
        if isinstance(vif_df, pd.DataFrame) and not vif_df.empty:
            high_vif = vif_df[vif_df["VIF"] > 10]
            lines += [
                "",
                "-- Multicollinearity (VIF > 10) ---------------------------------------",
            ]
            if high_vif.empty:
                lines.append("  No features with VIF > 10 detected. OK")
            else:
                for _, row in high_vif.iterrows():
                    lines.append(f"  !! {row['feature']:<22}  VIF = {row['VIF']:.2f}")

        lines += ["", "=" * 70, ""]
        path = out_dir / "feature_eda_report.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"  ✔ Saved: {path.name}")
