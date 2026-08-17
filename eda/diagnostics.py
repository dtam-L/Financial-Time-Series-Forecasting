"""
eda/diagnostics.py
==================
Statistical tests & diagnostics for financial time series.
Output: text report saved to reports/figures/diagnostics_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from tabulate import tabulate

# statsmodels optional imports
try:
    from statsmodels.tsa.stattools import adfuller, kpss, acf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy import stats as scipy_stats
    _STATSMODELS_OK = True
except ImportError:
    _STATSMODELS_OK = False
    logger.warning("statsmodels/scipy not found — some tests will be skipped.")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


class StatisticalDiagnostics:

    def run(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> dict:
        """
        Run all diagnostics tests and return a results dict.
        """
        ret = df["log_return"].dropna()
        price = df["close"].dropna()

        results = {
            "symbol":    symbol,
            "timeframe": timeframe,
            "n_obs":     len(ret),
            "period":    f"{df.index[0].date()} -> {df.index[-1].date()}",
        }

        if not _STATSMODELS_OK:
            logger.warning("Skipping statistical tests (statsmodels not installed).")
            return results

        logger.info(f"Running statistical diagnostics: {symbol} {timeframe}")

        results["descriptive"]  = self._descriptive(ret)
        results["adf_price"]    = self._adf(price, "Price level (ADF)")
        results["adf_return"]   = self._adf(ret,   "Log Returns (ADF)")
        results["kpss_price"]   = self._kpss(price, "Price level (KPSS)")
        results["kpss_return"]  = self._kpss(ret,   "Log Returns (KPSS)")
        results["ljung_box"]    = self._ljung_box(ret)
        results["jarque_bera"]  = self._jarque_bera(ret)
        results["hurst"]        = self._hurst(price)

        return results

    # ── Individual tests ─────────────────────────────────────────────────────

    def _descriptive(self, series: pd.Series) -> dict:
        return {
            "mean":     series.mean(),
            "std":      series.std(),
            "skewness": series.skew(),
            "kurtosis": series.kurtosis(),
            "min":      series.min(),
            "max":      series.max(),
            "var_5pct": series.quantile(0.05),
            "cvar_5pct":series[series <= series.quantile(0.05)].mean(),
        }

    def _adf(self, series: pd.Series, label: str) -> dict:
        """Augmented Dickey-Fuller test for unit root (non-stationarity)."""
        result = adfuller(series.dropna(), autolag="AIC")
        stat, pval = result[0], result[1]
        crit = result[4]
        return {
            "label":        label,
            "statistic":    round(stat, 4),
            "p_value":      round(pval, 4),
            "critical_1pct":round(crit["1%"], 4),
            "critical_5pct":round(crit["5%"], 4),
            "is_stationary": pval < 0.05,
            "interpretation": (
                "[OK] STATIONARY (p<0.05 -> reject unit root)"
                if pval < 0.05
                else "[!!] NON-STATIONARY (p>=0.05 -> cannot reject unit root)"
            ),
        }

    def _kpss(self, series: pd.Series, label: str) -> dict:
        """KPSS test — null: series IS stationary."""
        with np.errstate(all="ignore"):
            stat, pval, _, crit = kpss(series.dropna(), regression="c", nlags="auto")
        return {
            "label":        label,
            "statistic":    round(stat, 4),
            "p_value":      round(pval, 4),
            "critical_5pct":round(crit["5%"], 4),
            "is_stationary": pval > 0.05,
            "interpretation": (
                "[OK] STATIONARY (p>0.05 -> cannot reject stationarity)"
                if pval > 0.05
                else "[!!] NON-STATIONARY (p<=0.05 -> reject stationarity)"
            ),
        }

    def _ljung_box(self, series: pd.Series, lags: int = 10) -> dict:
        """Ljung-Box test for autocorrelation in returns."""
        lb = acorr_ljungbox(series.dropna(), lags=[lags], return_df=True)
        stat = float(lb["lb_stat"].iloc[-1])
        pval = float(lb["lb_pvalue"].iloc[-1])
        return {
            "lags":    lags,
            "statistic": round(stat, 4),
            "p_value":   round(pval, 4),
            "has_autocorr": pval < 0.05,
            "interpretation": (
                "[!!] AUTOCORRELATION DETECTED (p<0.05)"
                if pval < 0.05
                else "[OK] No significant autocorrelation (p>=0.05)"
            ),
        }

    def _jarque_bera(self, series: pd.Series) -> dict:
        """Jarque-Bera test for normality of returns."""
        stat, pval = scipy_stats.jarque_bera(series.dropna())
        return {
            "statistic": round(float(stat), 4),
            "p_value":   round(float(pval), 6),
            "is_normal": pval > 0.05,
            "interpretation": (
                "[OK] Normal distribution (p>0.05)"
                if pval > 0.05
                else "[!!] NON-NORMAL -- Fat tails / skewness present (p<0.05)"
            ),
        }

    def _hurst(self, series: pd.Series, max_lag: int = 100) -> dict:
        """
        Hurst Exponent via R/S analysis.
        H > 0.5 → trending / persistent
        H = 0.5 → random walk (GBM)
        H < 0.5 → mean-reverting
        """
        series = series.dropna().values
        lags = range(2, min(max_lag, len(series) // 2))
        tau = []
        for lag in lags:
            chunks = [series[i:i+lag] for i in range(0, len(series) - lag, lag)]
            if not chunks:
                continue
            rs_vals = []
            for chunk in chunks:
                mean_c = np.mean(chunk)
                dev    = np.cumsum(chunk - mean_c)
                rs = (dev.max() - dev.min()) / (np.std(chunk) + 1e-10)
                rs_vals.append(rs)
            tau.append(np.mean(rs_vals))

        if len(tau) < 2:
            return {"hurst": 0.5, "interpretation": "Insufficient data"}

        poly = np.polyfit(np.log(list(lags)[:len(tau)]), np.log(tau), 1)
        H = poly[0]
        if H > 0.55:
            interp = f"[UP] TRENDING / PERSISTENT (H={H:.3f} > 0.5)"
        elif H < 0.45:
            interp = f"[<>] MEAN-REVERTING (H={H:.3f} < 0.5)"
        else:
            interp = f"[~~] RANDOM WALK / EFFICIENT (H={H:.3f} ~ 0.5)"

        return {"hurst": round(H, 4), "interpretation": interp}

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_report(self, results: dict) -> None:
        """Pretty-print results to console (Windows cp1252-safe)."""
        text = self._format_report(results)
        # Encode to utf-8 bytes then decode with replacement for any
        # characters unsupported by the Windows terminal code page.
        safe = text.encode("utf-8", errors="replace").decode(
            "utf-8", errors="replace"
        )
        try:
            print(safe)
        except UnicodeEncodeError:
            print(safe.encode("ascii", errors="replace").decode("ascii"))

    def save_report(self, results: dict, output_dir: Path) -> None:
        """Save diagnostics report as a text file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "diagnostics_report.txt"
        path.write_text(self._format_report(results), encoding="utf-8")
        logger.info(f"  ✔ Saved: {path.name}")

    def _format_report(self, results: dict) -> str:
        lines = [
            "=" * 70,
            f"  STATISTICAL DIAGNOSTICS REPORT",
            f"  Symbol: {results['symbol']}  |  Timeframe: {results['timeframe']}",
            f"  Period: {results.get('period', 'N/A')}  |  N = {results.get('n_obs', 'N/A')}",
            "=" * 70,
        ]

        # Descriptive stats
        if "descriptive" in results:
            d = results["descriptive"]
            lines += [
                "\n── Descriptive Statistics (Log Returns) ────────────────────",
                tabulate([
                    ["Mean",     f"{d['mean']:.6f}"],
                    ["Std Dev",  f"{d['std']:.6f}"],
                    ["Skewness", f"{d['skewness']:.4f}"],
                    ["Kurtosis", f"{d['kurtosis']:.4f}  {'(fat tails)' if d['kurtosis']>3 else ''}"],
                    ["VaR 5%",   f"{d['var_5pct']:.6f}"],
                    ["CVaR 5%",  f"{d['cvar_5pct']:.6f}"],
                ], tablefmt="simple"),
            ]

        # Stationarity
        for key in ["adf_price", "adf_return", "kpss_price", "kpss_return"]:
            if key in results:
                r = results[key]
                lines += [
                    f"\n── {r['label']} ────────────────────────────────────",
                    f"  Statistic : {r['statistic']}   p-value: {r['p_value']}",
                    f"  Result    : {r['interpretation']}",
                ]

        # Autocorrelation
        if "ljung_box" in results:
            r = results["ljung_box"]
            lines += [
                f"\n── Ljung-Box Autocorrelation Test (lag={r['lags']}) ─────────",
                f"  Statistic : {r['statistic']}   p-value: {r['p_value']}",
                f"  Result    : {r['interpretation']}",
            ]

        # Normality
        if "jarque_bera" in results:
            r = results["jarque_bera"]
            lines += [
                "\n── Jarque-Bera Normality Test ───────────────────────────────",
                f"  Statistic : {r['statistic']}   p-value: {r['p_value']}",
                f"  Result    : {r['interpretation']}",
            ]

        # Hurst
        if "hurst" in results:
            r = results["hurst"]
            lines += [
                "\n── Hurst Exponent (R/S Analysis) ────────────────────────────",
                f"  Result    : {r['interpretation']}",
            ]

        lines += ["", "=" * 70, ""]
        return "\n".join(lines)
