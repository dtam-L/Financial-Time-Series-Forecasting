from __future__ import annotations
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — save-only mode

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy import stats
from loguru import logger

warnings.filterwarnings("ignore")

# ── Professional Dark Theme ───────────────────────────────────────────────────
_BG        = "#0d1117"
_AXES_BG   = "#161b22"
_EDGE      = "#30363d"
_TEXT      = "#c9d1d9"
_MUTED     = "#8b949e"
_GRID      = "#21262d"
_UP        = "#3fb950"   # green
_DOWN      = "#f85149"   # red
_BLUE      = "#58a6ff"
_ORANGE    = "#ff9500"
_PURPLE    = "#a371f7"
_YELLOW    = "#f0e68c"
_CYAN      = "#39d353"

_PALETTE = {
    "up":     _UP,
    "down":   _DOWN,
    "ma_7":   _BLUE,
    "ma_25":  _ORANGE,
    "ma_99":  _YELLOW,
    "bb":     _PURPLE,
    "rsi":    _PURPLE,
    "macd":   _BLUE,
    "signal": _ORANGE,
    "volume": "#388bfd",
    "neutral":_MUTED,
    "Bull":   _UP,
    "Bear":   _DOWN,
    "Sideways":_MUTED,
}

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_price(x: float) -> str:
    return f"${x:,.0f}" if x >= 100 else f"${x:,.4f}"

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=_BG, edgecolor="none")
    plt.close(fig)
    logger.info(f"  ✔ Saved: {path.name}")

def _title_block(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, fontsize=14, fontweight="bold", color=_TEXT, y=0.98)
    fig.text(0.5, 0.965, subtitle, ha="center", fontsize=9, color=_MUTED)

def _draw_candlesticks(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Draw OHLC candlesticks using matplotlib patches (no mplfinance needed)."""
    for i, (_, row) in enumerate(df.iterrows()):
        is_up  = float(row["close"]) >= float(row["open"])
        color  = _UP if is_up else _DOWN
        o, h, l, c = (float(row[k]) for k in ("open", "high", "low", "close"))
        body_bot = min(o, c)
        body_h   = max(abs(c - o), (h - l) * 0.005)

        ax.add_patch(Rectangle(
            (i - 0.35, body_bot), 0.7, body_h,
            facecolor=color, edgecolor=color,
            linewidth=0.3, zorder=3, alpha=0.9
        ))
        ax.plot([i, i], [l, h], color=color, linewidth=0.7, zorder=2)

    n = len(df)
    step = max(1, n // 10)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [df.index[t].strftime("%m/%d") for t in ticks],
        rotation=45, ha="right", fontsize=7
    )
    ax.set_xlim(-1, n)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_price(x)))


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class EDAVisualizer:
    """
    Generates and saves all EDA charts for a given (symbol, timeframe).

    Parameters
    ----------
    output_root : Path  Root directory for figure output (reports/figures/).
    dpi : int           Resolution for saved PNGs.
    candle_window : int How many latest candles to show in candlestick charts.
    """

    _CANDLE_WINDOW = {"1d": 180, "1h": 120, "4h": 150, "15m": 96}

    def __init__(
        self,
        output_root: Optional[Path] = None,
        dpi: int = 150,
    ) -> None:
        _apply_dark_theme()
        _root = Path(__file__).resolve().parent.parent
        self._out_root = output_root or (_root / "reports" / "figures")
        self._out_root.mkdir(parents=True, exist_ok=True)
        plt.rcParams["savefig.dpi"] = dpi

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
        df_other: Optional[pd.DataFrame] = None,
        other_symbol: Optional[str] = None,
    ) -> None:
        """
        Generate and save all 7 chart groups for (symbol, timeframe).

        Parameters
        ----------
        df          : DataFrame with features from OHLCVLoader.
        symbol      : e.g. 'BTC/USDT'
        timeframe   : e.g. '1d'
        df_other    : Optional second symbol DF for correlation chart.
        other_symbol: Label for the second symbol.
        """
        logger.info(f"Generating EDA charts: {symbol} {timeframe}")
        out = self._outdir(symbol, timeframe)

        self._plot_price_volume(df, symbol, timeframe, out)
        self._plot_technical(df, symbol, timeframe, out)
        self._plot_return_distribution(df, symbol, timeframe, out)
        self._plot_drawdown(df, symbol, timeframe, out)
        self._plot_seasonality(df, symbol, timeframe, out)
        self._plot_anomaly(df, symbol, timeframe, out)
        self._plot_regime(df, symbol, timeframe, out)

        if df_other is not None and other_symbol:
            self._plot_correlation(df, symbol, df_other, other_symbol, timeframe)

        logger.success(f"All charts saved → {out}")

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 01: Price & Volume
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_price_volume(
        self, df: pd.DataFrame, symbol: str, tf: str, out: Path
    ) -> None:
        cw = self._CANDLE_WINDOW.get(tf, 120)
        df_full  = df
        df_candle = df.iloc[-cw:] if len(df) > cw else df

        fig = plt.figure(figsize=(22, 14))
        gs  = fig.add_gridspec(
            3, 1, height_ratios=[2, 1.2, 0.8], hspace=0.12
        )

        # ── Row 0: Candlestick ────────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0])
        _draw_candlesticks(ax0, df_candle)

        x = np.arange(len(df_candle))
        for col, lbl, lw in [
            ("ma_7",  "MA 7",  1.2),
            ("ma_25", "MA 25", 1.5),
            ("ma_99", "MA 99", 1.8),
        ]:
            if col in df_candle.columns:
                ax0.plot(x, df_candle[col].values,
                         color=_PALETTE[col], label=lbl,
                         linewidth=lw, alpha=0.85, zorder=4)

        # Annotate current price
        last_close = float(df_candle["close"].iloc[-1])
        ax0.axhline(last_close, color=_BLUE, linewidth=0.8, linestyle="--", alpha=0.5)
        ax0.text(
            len(df_candle) - 1, last_close * 1.002,
            _fmt_price(last_close), color=_BLUE, fontsize=8, fontweight="bold"
        )
        ax0.set_title(
            f"{symbol}  •  Last {cw} candles ({tf})", loc="left", pad=6
        )
        ax0.legend(loc="upper left", framealpha=0.3)
        ax0.set_xticklabels([])

        # ── Row 1: Full history close line ────────────────────────────────────
        ax1 = fig.add_subplot(gs[1])
        ax1.plot(df_full.index, df_full["close"],
                 color=_BLUE, linewidth=0.8, alpha=0.9, label="Close")
        ax1.fill_between(
            df_full.index, df_full["close"].min(), df_full["close"],
            color=_BLUE, alpha=0.08
        )
        if "ma_99" in df_full.columns:
            ax1.plot(df_full.index, df_full["ma_99"],
                     color=_YELLOW, linewidth=1, alpha=0.7, label="MA 99")
        ax1.set_title("Full History — Close Price", loc="left", pad=4)
        ax1.legend(loc="upper left", framealpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: _fmt_price(v))
        )

        # ── Row 2: Volume ─────────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[2])
        colors = [
            _UP if float(r["close"]) >= float(r["open"]) else _DOWN
            for _, r in df_full.iterrows()
        ]
        ax2.bar(
            df_full.index, df_full["volume"],
            color=colors, width=pd.Timedelta("22h") if tf == "1d" else pd.Timedelta("45min"),
            alpha=0.7, label="Volume"
        )
        if "volume_ma_20" in df_full.columns:
            ax2.plot(df_full.index, df_full["volume_ma_20"],
                     color=_ORANGE, linewidth=1.2, label="Vol MA 20")
        ax2.set_title("Volume", loc="left", pad=4)
        ax2.legend(loc="upper left", framealpha=0.3)
        ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K")
        )
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # ── Stats box ─────────────────────────────────────────────────────────
        ytd_ret = (
            df_full["close"].iloc[-1] / df_full[df_full["year"] == df_full["year"].iloc[-1]]["close"].iloc[0] - 1
        ) * 100 if "year" in df_full.columns else 0

        stats_txt = (
            f"Current: {_fmt_price(last_close)}   "
            f"High: {_fmt_price(df_full['high'].max())}   "
            f"Low: {_fmt_price(df_full['low'].min())}   "
            f"YTD Return: {ytd_ret:+.1f}%"
        )
        fig.text(0.5, 0.01, stats_txt, ha="center", fontsize=9,
                 color=_TEXT, style="italic")

        _title_block(
            fig,
            f"{symbol} — Price & Volume Dashboard",
            f"Timeframe: {tf}  •  Generated {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )
        _save(fig, out / "01_price_volume.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 02: Technical Indicators (BB + RSI + MACD)
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_technical(
        self, df: pd.DataFrame, symbol: str, tf: str, out: Path
    ) -> None:
        cw  = self._CANDLE_WINDOW.get(tf, 120)
        d   = df.iloc[-cw:] if len(df) > cw else df
        idx = d.index

        fig, axes = plt.subplots(
            3, 1, figsize=(22, 16),
            gridspec_kw={"height_ratios": [2.5, 1.2, 1.2], "hspace": 0.08}
        )
        ax0, ax1, ax2 = axes

        # ── Bollinger Bands ───────────────────────────────────────────────────
        ax0.plot(idx, d["close"],   color=_TEXT,   linewidth=1.2, label="Close", zorder=3)
        ax0.plot(idx, d["bb_upper"], color=_PURPLE, linewidth=0.9, linestyle="--", label="BB Upper", alpha=0.8)
        ax0.plot(idx, d["bb_lower"], color=_PURPLE, linewidth=0.9, linestyle="--", label="BB Lower", alpha=0.8)
        ax0.plot(idx, d["bb_mid"],   color=_ORANGE, linewidth=0.8, linestyle=":", label="BB Mid", alpha=0.7)
        ax0.fill_between(idx, d["bb_lower"], d["bb_upper"],
                         color=_PURPLE, alpha=0.06, label="_nolegend_")

        # Highlight price outside bands
        outside_up   = d["close"] > d["bb_upper"]
        outside_down = d["close"] < d["bb_lower"]
        ax0.scatter(idx[outside_up],   d["close"][outside_up],
                    color=_DOWN, s=18, zorder=5, label="Above BB", marker="v")
        ax0.scatter(idx[outside_down], d["close"][outside_down],
                    color=_UP,   s=18, zorder=5, label="Below BB", marker="^")

        ax0.set_title("Bollinger Bands (20, 2σ)", loc="left")
        ax0.legend(loc="upper left", framealpha=0.3, ncol=3)
        ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _fmt_price(v)))
        ax0.set_xticklabels([])

        # ── RSI ───────────────────────────────────────────────────────────────
        ax1.plot(idx, d["rsi_14"], color=_PURPLE, linewidth=1.2, label="RSI 14")
        ax1.axhline(70, color=_DOWN, linewidth=0.8, linestyle="--", alpha=0.8, label="Overbought (70)")
        ax1.axhline(30, color=_UP,   linewidth=0.8, linestyle="--", alpha=0.8, label="Oversold (30)")
        ax1.axhline(50, color=_MUTED, linewidth=0.5, linestyle=":", alpha=0.5)
        ax1.fill_between(idx, 70, d["rsi_14"].clip(lower=70),
                         color=_DOWN, alpha=0.15)
        ax1.fill_between(idx, d["rsi_14"].clip(upper=30), 30,
                         color=_UP, alpha=0.15)
        ax1.set_ylim(0, 100)
        ax1.set_yticks([20, 30, 50, 70, 80])
        ax1.set_title("RSI 14", loc="left")
        ax1.legend(loc="upper left", framealpha=0.3, ncol=3)
        ax1.set_xticklabels([])

        # ── MACD ──────────────────────────────────────────────────────────────
        hist_colors = [_UP if v >= 0 else _DOWN for v in d["macd_hist"]]
        ax2.bar(idx, d["macd_hist"], color=hist_colors, alpha=0.7,
                width=pd.Timedelta("22h") if tf == "1d" else pd.Timedelta("45min"),
                label="Histogram")
        ax2.plot(idx, d["macd"],        color=_BLUE,   linewidth=1.2, label="MACD")
        ax2.plot(idx, d["macd_signal"], color=_ORANGE, linewidth=1.0, label="Signal")
        ax2.axhline(0, color=_MUTED, linewidth=0.5, linestyle="-")
        ax2.set_title("MACD (12/26/9)", loc="left")
        ax2.legend(loc="upper left", framealpha=0.3, ncol=3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m" if tf == "1d" else "%m/%d"))

        _title_block(
            fig,
            f"{symbol} — Technical Indicators",
            f"Timeframe: {tf}  •  Last {cw} candles"
        )
        _save(fig, out / "02_technical_indicators.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 03: Return Distribution
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_return_distribution(
        self, df: pd.DataFrame, symbol: str, tf: str, out: Path
    ) -> None:
        ret = df["log_return"].dropna()
        cum = (1 + df["return_pct"] / 100).cumprod()

        fig, axes = plt.subplots(2, 2, figsize=(20, 14), constrained_layout=True)
        (ax0, ax1), (ax2, ax3) = axes

        # ── Histogram + KDE + Normal overlay ─────────────────────────────────
        mu, sigma = ret.mean(), ret.std()
        x_norm = np.linspace(ret.min(), ret.max(), 200)
        y_norm = stats.norm.pdf(x_norm, mu, sigma)

        ax0.hist(ret, bins=80, density=True,
                 color=_BLUE, alpha=0.5, edgecolor="none", label="Returns")
        ax0.plot(x_norm, y_norm, color=_ORANGE, linewidth=1.5, label="Normal fit")

        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(ret)
        ax0.plot(x_norm, kde(x_norm), color=_PURPLE, linewidth=1.5, label="KDE")

        # VaR lines
        var5 = ret.quantile(0.05)
        ax0.axvline(var5, color=_DOWN, linewidth=1.2, linestyle="--",
                    label=f"VaR 5% = {var5:.4f}")

        sk, ku = ret.skew(), ret.kurtosis()
        stats_box = f"μ={mu:.4f}  σ={sigma:.4f}\nSkew={sk:.2f}  Kurt={ku:.2f}"
        ax0.text(0.97, 0.95, stats_box, transform=ax0.transAxes,
                 va="top", ha="right", fontsize=8, color=_TEXT,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=_AXES_BG, edgecolor=_EDGE))
        ax0.set_title("Log-Return Distribution", loc="left")
        ax0.legend(framealpha=0.3)
        ax0.set_xlabel("Log Return")

        # ── QQ Plot ───────────────────────────────────────────────────────────
        (osm, osr), (slope, intercept, r) = stats.probplot(ret, dist="norm")
        ax1.scatter(osm, osr, s=4, alpha=0.5, color=_BLUE, label="Returns")
        ax1.plot(osm, slope * np.array(osm) + intercept,
                 color=_ORANGE, linewidth=1.5, label="Normal line")
        ax1.set_title("Q-Q Plot vs Normal Distribution", loc="left")
        ax1.set_xlabel("Theoretical Quantiles")
        ax1.set_ylabel("Sample Quantiles")
        ax1.legend(framealpha=0.3)
        ax1.text(0.05, 0.95, f"R² = {r**2:.4f}", transform=ax1.transAxes,
                 va="top", fontsize=8, color=_MUTED)

        # ── Rolling Volatility ────────────────────────────────────────────────
        rv = df["rolling_vol_30"].dropna()
        ax2.plot(rv.index, rv * 100, color=_ORANGE, linewidth=1.0, alpha=0.9)
        ax2.fill_between(rv.index, 0, rv * 100, color=_ORANGE, alpha=0.12)
        rv_mean = rv.mean() * 100
        ax2.axhline(rv_mean, color=_MUTED, linewidth=0.8, linestyle="--",
                    label=f"Mean: {rv_mean:.1f}%")
        ax2.set_title("Rolling 30-Period Volatility (Annualised %)", loc="left")
        ax2.set_ylabel("Annualised Vol %")
        ax2.legend(framealpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # ── Cumulative Return ─────────────────────────────────────────────────
        cum.index = df.index[:len(cum)]
        ax3.plot(cum.index, (cum - 1) * 100, color=_BLUE, linewidth=1.0)
        ax3.fill_between(cum.index, 0, (cum - 1) * 100,
                         where=(cum >= 1), color=_UP,   alpha=0.12)
        ax3.fill_between(cum.index, 0, (cum - 1) * 100,
                         where=(cum <  1), color=_DOWN, alpha=0.12)
        ax3.axhline(0, color=_MUTED, linewidth=0.5)
        ax3.set_title("Cumulative Return (%)", loc="left")
        ax3.set_ylabel("Return %")
        ax3.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%")
        )
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        _title_block(
            fig,
            f"{symbol} — Return Distribution & Risk Metrics",
            f"Timeframe: {tf}  •  N={len(ret):,} observations"
        )
        _save(fig, out / "03_return_distribution.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 04: Drawdown
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_drawdown(
        self, df: pd.DataFrame, symbol: str, tf: str, out: Path
    ) -> None:
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(22, 12),
            gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.1}
        )

        # ── Close price + rolling max "underwater" ────────────────────────────
        rolling_max = df["close"].cummax()
        ax0.plot(df.index, df["close"], color=_BLUE, linewidth=1.0,
                 alpha=0.9, label="Close")
        ax0.plot(df.index, rolling_max, color=_MUTED, linewidth=0.8,
                 linestyle="--", alpha=0.6, label="Rolling Max (Peak)")
        ax0.fill_between(df.index, df["close"], rolling_max,
                         color=_DOWN, alpha=0.15, label="Underwater")

        # Find max drawdown period
        dd = df["drawdown_pct"]
        min_dd_idx = dd.idxmin()
        ax0.scatter([min_dd_idx], [df.loc[min_dd_idx, "close"]],
                    color=_DOWN, s=60, zorder=6, marker="v")
        ax0.annotate(
            f" Max DD: {dd.min():.1f}%",
            (min_dd_idx, df.loc[min_dd_idx, "close"]),
            fontsize=8, color=_DOWN, xytext=(10, -18), textcoords="offset points"
        )

        ax0.set_title("Price & Drawdown Analysis", loc="left")
        ax0.legend(framealpha=0.3, loc="upper left")
        ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _fmt_price(v)))
        ax0.set_xticklabels([])

        # ── Drawdown % timeline ───────────────────────────────────────────────
        ax1.fill_between(df.index, dd, 0, color=_DOWN, alpha=0.4)
        ax1.plot(df.index, dd, color=_DOWN, linewidth=0.8)
        ax1.set_ylim(dd.min() * 1.05, 5)
        ax1.set_title("Drawdown from Peak (%)", loc="left")
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.1f}%")
        )
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # Top 5 worst drawdowns annotation
        worst = dd.nsmallest(5)
        for ts, val in worst.items():
            ax1.annotate(
                f"{val:.1f}%",
                (ts, val), fontsize=7, color=_DOWN,
                xytext=(0, -12), textcoords="offset points", ha="center"
            )

        _title_block(
            fig, f"{symbol} — Drawdown Analysis",
            f"Max Drawdown: {dd.min():.2f}%  •  Timeframe: {tf}"
        )
        _save(fig, out / "04_drawdown.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 05: Seasonality
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_seasonality(
        self, df: pd.DataFrame, symbol: str, tf: str, out: Path
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)
        ax0, ax1 = axes

        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        df_s = df[["return_pct", "day_of_week", "hour", "month"]].dropna()

        # ── Returns by Day of Week ────────────────────────────────────────────
        dow_data = [
            df_s[df_s["day_of_week"] == d]["return_pct"].values
            for d in range(7)
        ]
        bp = ax0.boxplot(
            dow_data,
            patch_artist=True,
            medianprops=dict(color=_TEXT, linewidth=1.5),
            flierprops=dict(marker="o", markerfacecolor=_MUTED,
                            markersize=2, alpha=0.4),
            whiskerprops=dict(color=_MUTED, linewidth=0.8),
            capprops=dict(color=_MUTED, linewidth=0.8),
        )
        ax0.set_xticks(range(1, 8))
        ax0.set_xticklabels(day_names)
        means = [df_s[df_s["day_of_week"] == d]["return_pct"].mean() for d in range(7)]
        for patch, color in zip(
            bp["boxes"],
            [_UP if m >= 0 else _DOWN for m in means]
        ):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        ax0.axhline(0, color=_MUTED, linewidth=0.7, linestyle="--")
        ax0.set_title("Average Return by Day of Week", loc="left")
        ax0.set_ylabel("Return %")

        # Mean dots
        for d, m in enumerate(means):
            ax0.scatter(d + 1, m, color=_YELLOW, s=30, zorder=5)

        # ── 1h: Hour-of-Day Heatmap  |  1d: Monthly Heatmap ──────────────────
        if tf in ("1h", "4h", "15m"):
            pivot = (
                df_s.groupby(["day_of_week", "hour"])["return_pct"]
                .mean()
                .unstack(fill_value=0)
            )
            pivot.index = [day_names[d] for d in pivot.index]
            sns.heatmap(
                pivot, ax=ax1, cmap="RdYlGn", center=0,
                linewidths=0.3, linecolor=_EDGE,
                cbar_kws={"label": "Mean Return %"},
                annot=len(pivot.columns) <= 24,
                fmt=".2f", annot_kws={"size": 6}
            )
            ax1.set_title("Mean Return by Hour × Day (Heatmap)", loc="left")
            ax1.set_xlabel("Hour (UTC)")
        else:
            # Monthly heatmap for 1d timeframe
            df_s["year_col"] = df.loc[df_s.index, "year"] if "year" in df.columns else df_s.index.year
            pivot = (
                df_s.groupby(["year_col", "month"])["return_pct"]
                .mean()
                .unstack(fill_value=0)
            )
            month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]
            pivot.columns = [month_names[m - 1] for m in pivot.columns]
            sns.heatmap(
                pivot, ax=ax1, cmap="RdYlGn", center=0,
                linewidths=0.5, linecolor=_EDGE,
                cbar_kws={"label": "Mean Daily Return %"},
                annot=True, fmt=".2f", annot_kws={"size": 8}
            )
            ax1.set_title("Mean Return by Month × Year (Heatmap)", loc="left")

        _title_block(
            fig, f"{symbol} — Seasonality Patterns",
            f"Timeframe: {tf}  •  N={len(df_s):,} candles"
        )
        _save(fig, out / "05_seasonality.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 06: Anomaly Signals
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_anomaly(
        self, df: pd.DataFrame, symbol: str, tf: str, out: Path
    ) -> None:
        Z_THRESH  = 3.0
        VOL_THRESH = 3.0

        fig, axes = plt.subplots(
            3, 1, figsize=(22, 16),
            gridspec_kw={"height_ratios": [2, 1, 1], "hspace": 0.1}
        )
        ax0, ax1, ax2 = axes

        # ── Price + Z-score anomalies ─────────────────────────────────────────
        # Work on a clean sub-df aligned on price_zscore availability
        df_a = df[df["price_zscore"].notna()].copy()
        zs   = df_a["price_zscore"]
        anom_high = df_a[zs > Z_THRESH]
        anom_low  = df_a[zs < -Z_THRESH]

        ax0.plot(df.index, df["close"], color=_BLUE, linewidth=0.8, alpha=0.9)
        ax0.scatter(anom_high.index, anom_high["close"],
                    color=_DOWN, s=30, zorder=5, marker="v",
                    label=f"Z > +{Z_THRESH}σ ({len(anom_high)})")
        ax0.scatter(anom_low.index,  anom_low["close"],
                    color=_UP,   s=30, zorder=5, marker="^",
                    label=f"Z < -{Z_THRESH}σ ({len(anom_low)})")
        ax0.set_title("Price with Anomaly Events (Z-Score ±3σ)", loc="left")
        ax0.legend(framealpha=0.3, loc="upper left")
        ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _fmt_price(v)))
        ax0.set_xticklabels([])

        # ── Z-score timeline ──────────────────────────────────────────────────
        ax1.plot(zs.index, zs, color=_PURPLE, linewidth=0.8)
        ax1.fill_between(zs.index,  Z_THRESH,  zs.clip(lower=Z_THRESH),
                         color=_DOWN, alpha=0.3)
        ax1.fill_between(zs.index, zs.clip(upper=-Z_THRESH), -Z_THRESH,
                         color=_UP,   alpha=0.3)
        ax1.axhline( Z_THRESH,  color=_DOWN, linewidth=0.9, linestyle="--",
                     label=f"+{Z_THRESH}σ")
        ax1.axhline(-Z_THRESH,  color=_UP,   linewidth=0.9, linestyle="--",
                     label=f"-{Z_THRESH}σ")
        ax1.axhline(0, color=_MUTED, linewidth=0.5)
        ax1.set_title("Price Z-Score (60-period rolling)", loc="left")
        ax1.legend(framealpha=0.3, loc="upper left", ncol=2)
        ax1.set_xticklabels([])

        # ── Volume Spike ──────────────────────────────────────────────────────
        vr = df["volume_ratio"].dropna()
        colors = [_DOWN if v > VOL_THRESH else _BLUE for v in vr]
        ax2.bar(
            vr.index, vr,
            color=colors, alpha=0.7,
            width=pd.Timedelta("22h") if tf == "1d" else pd.Timedelta("45min")
        )
        ax2.axhline(VOL_THRESH, color=_DOWN, linewidth=1.0, linestyle="--",
                    label=f"Spike threshold ({VOL_THRESH}×)")
        ax2.axhline(1.0, color=_MUTED, linewidth=0.5)
        spikes = (vr > VOL_THRESH).sum()
        ax2.set_title(f"Volume Spike Detection (ratio vs 20-period MA) — {spikes} spikes",
                      loc="left")
        ax2.legend(framealpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        _title_block(
            fig, f"{symbol} — Anomaly Detection Signals",
            f"Timeframe: {tf}  •  Total events: Z-score={len(anom_high)+len(anom_low)}, Vol spikes={spikes}"
        )
        _save(fig, out / "06_anomaly_signals.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 07: Market Regime
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_regime(
        self, df: pd.DataFrame, symbol: str, tf: str, out: Path
    ) -> None:
        fig, axes = plt.subplots(
            3, 1, figsize=(22, 14),
            gridspec_kw={"height_ratios": [2, 1, 1], "hspace": 0.1}
        )
        ax0, ax1, ax2 = axes

        regime_order = ["Bull", "Bear", "Sideways"]

        # ── Regime-colored price line ─────────────────────────────────────────
        for reg in regime_order:
            mask = df["regime"] == reg
            segs = df[mask]
            ax0.scatter(
                segs.index, segs["close"],
                color=_PALETTE[reg], s=3, alpha=0.7,
                label=f"{reg} ({mask.sum():,} candles)"
            )
        ax0.set_title("Price Colored by Market Regime (MA 7/25/99)", loc="left")
        ax0.legend(framealpha=0.3, loc="upper left", markerscale=3)
        ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _fmt_price(v)))
        ax0.set_xticklabels([])

        # ── Regime distribution bar ───────────────────────────────────────────
        regime_counts = df["regime"].value_counts()
        ax1.barh(
            regime_counts.index,
            regime_counts.values,
            color=[_PALETTE[r] for r in regime_counts.index],
            alpha=0.8, height=0.5
        )
        for i, (reg, cnt) in enumerate(regime_counts.items()):
            pct = cnt / len(df) * 100
            ax1.text(cnt + 50, i, f"{pct:.1f}%", va="center",
                     color=_TEXT, fontsize=9)
        ax1.set_title("Time Spent per Regime (candle count)", loc="left")
        ax1.set_xlabel("Candles")
        ax1.grid(axis="x")

        # ── Volatility Regime ─────────────────────────────────────────────────
        vol_colors = {"Low": _UP, "Mid": _ORANGE, "High": _DOWN}
        for vr_name, vr_color in vol_colors.items():
            mask = df["vol_regime"] == vr_name
            ax2.scatter(
                df[mask].index, df[mask]["atr_14"],
                color=vr_color, s=3, alpha=0.6, label=vr_name
            )
        ax2.plot(df.index, df["atr_14"], color=_MUTED,
                 linewidth=0.5, alpha=0.4)
        ax2.set_title("Volatility Regime via ATR-14 (Low / Mid / High)", loc="left")
        ax2.legend(framealpha=0.3, markerscale=3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        _title_block(
            fig, f"{symbol} — Market Regime Analysis",
            f"Timeframe: {tf}"
        )
        _save(fig, out / "07_market_regime.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Chart 08: Cross-Symbol Correlation (BTC vs ETH)
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_correlation(
        self,
        df_a: pd.DataFrame, sym_a: str,
        df_b: pd.DataFrame, sym_b: str,
        tf: str,
    ) -> None:
        out = self._out_root / "cross_symbol"
        out.mkdir(parents=True, exist_ok=True)

        # Align on common index
        common = df_a.index.intersection(df_b.index)
        if len(common) < 30:
            logger.warning("Not enough common candles for correlation chart.")
            return
        ra = df_a.loc[common, "log_return"]
        rb = df_b.loc[common, "log_return"]

        fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
        ax0, ax1, ax2 = axes

        # ── Normalised Price ──────────────────────────────────────────────────
        pa = df_a.loc[common, "close"] / df_a.loc[common, "close"].iloc[0]
        pb = df_b.loc[common, "close"] / df_b.loc[common, "close"].iloc[0]
        ax0.plot(common, pa, color=_BLUE,   linewidth=1.0, label=sym_a)
        ax0.plot(common, pb, color=_ORANGE, linewidth=1.0, label=sym_b)
        ax0.set_title("Normalised Price (Base = 1.0)", loc="left")
        ax0.legend(framealpha=0.3)
        ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # ── Rolling 30-period Correlation ─────────────────────────────────────
        rolling_corr = ra.rolling(30).corr(rb)
        ax1.plot(common, rolling_corr, color=_PURPLE, linewidth=1.0)
        ax1.fill_between(common, 0.8, rolling_corr.clip(lower=0.8),
                         color=_UP, alpha=0.2, label="High correlation (>0.8)")
        ax1.fill_between(common, rolling_corr.clip(upper=0.4), 0.4,
                         color=_DOWN, alpha=0.2, label="Low correlation (<0.4)")
        ax1.axhline(rolling_corr.mean(), color=_YELLOW, linewidth=0.8,
                    linestyle="--",
                    label=f"Mean: {rolling_corr.mean():.2f}")
        ax1.set_ylim(-1, 1)
        ax1.set_title(f"Rolling 30-Period Correlation: {sym_a} vs {sym_b}",
                      loc="left")
        ax1.legend(framealpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # ── Scatter ───────────────────────────────────────────────────────────
        slope, intercept, r_val, p_val, _ = stats.linregress(ra, rb)
        x_line = np.linspace(ra.min(), ra.max(), 200)
        ax2.scatter(ra, rb, s=4, alpha=0.4, color=_BLUE)
        ax2.plot(x_line, slope * x_line + intercept,
                 color=_ORANGE, linewidth=1.5,
                 label=f"β={slope:.2f}  R²={r_val**2:.3f}")
        ax2.set_title(f"Return Scatter: {sym_a} vs {sym_b}", loc="left")
        ax2.set_xlabel(f"{sym_a} Log Return")
        ax2.set_ylabel(f"{sym_b} Log Return")
        ax2.legend(framealpha=0.3)

        _title_block(
            fig, f"Correlation: {sym_a} vs {sym_b}",
            f"Timeframe: {tf}  •  N={len(common):,} common candles"
        )
        _save(fig, out / f"08_correlation_{tf}.png")
