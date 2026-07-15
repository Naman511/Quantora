"""
data_pipeline.py
================
Data Ingestion & Feature Engineering Pipeline for the Intelligent Stock
Screener & Portfolio Builder ("Quantora").

Responsibilities
-----------------
1. Download historical OHLCV price data and fundamental metrics via yfinance.
2. Engineer technical indicators (RSI, MACD, moving averages, volatility,
   average volume) with strict point-in-time discipline (no look-ahead bias).
3. Impute missing fundamental data using sector-median imputation.
4. Construct the forward-looking binary classification target
   (stock outperforms benchmark over the next ~3 months).

This module is intentionally defensive: any single ticker failing to download
(delisted, rate-limited, bad symbol, missing fundamentals, etc.) must not
crash the pipeline for the rest of the universe.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import ta
except ImportError:  # pragma: no cover
    ta = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | data_pipeline | %(message)s",
)
logger = logging.getLogger("data_pipeline")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

FORWARD_WINDOW_DAYS = 63          # ~3 trading months
LOOKBACK_PERIOD = "3y"
MIN_HISTORY_DAYS = 220            # need enough bars for a 200D MA
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 1.5

FUNDAMENTAL_FIELDS = {
    "marketCap": "market_cap",
    "trailingPE": "pe_ratio",
    "priceToBook": "pb_ratio",
    "returnOnEquity": "roe",
    "debtToEquity": "debt_to_equity",
    "dividendYield": "dividend_yield",
    "earningsQuarterlyGrowth": "eps_growth",
    "beta": "beta",
    "sector": "sector",
}


@dataclass
class TickerBundle:
    """Container holding everything the pipeline produced for one ticker."""

    ticker: str
    prices: pd.DataFrame                     # OHLCV + engineered technicals
    fundamentals: Dict[str, object]
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Aggregate output of a full pipeline run across the universe."""

    feature_table: pd.DataFrame              # one row per ticker, latest snapshot
    price_histories: Dict[str, pd.DataFrame] = field(default_factory=dict)
    benchmark: Optional[pd.DataFrame] = None
    failed_tickers: List[str] = field(default_factory=list)


class DataPipeline:
    """
    Orchestrates data download, technical feature engineering, fundamental
    imputation, and target construction for a universe of tickers.
    """

    def __init__(
        self,
        benchmark_ticker: str = "^GSPC",
        lookback_period: str = LOOKBACK_PERIOD,
        forward_window_days: int = FORWARD_WINDOW_DAYS,
    ):
        self.benchmark_ticker = benchmark_ticker
        self.lookback_period = lookback_period
        self.forward_window_days = forward_window_days
        self._sector_medians: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, tickers: List[str]) -> PipelineResult:
        """
        Execute the full pipeline for a list of tickers and return a
        PipelineResult with a unified feature table ready for the ML engine.
        """
        tickers = sorted(set(t.strip().upper() for t in tickers if t.strip()))
        logger.info("Starting pipeline for %d tickers", len(tickers))

        benchmark_df = self._download_benchmark()

        bundles: List[TickerBundle] = []
        for i, ticker in enumerate(tickers, start=1):
            logger.info("[%d/%d] Processing %s", i, len(tickers), ticker)
            bundle = self._process_single_ticker(ticker, benchmark_df)
            bundles.append(bundle)

        valid_bundles = [b for b in bundles if b.error is None]
        failed = [b.ticker for b in bundles if b.error is not None]
        if failed:
            logger.warning("Failed tickers (%d): %s", len(failed), failed)

        self._compute_sector_medians(valid_bundles)
        for bundle in valid_bundles:
            self._impute_fundamentals(bundle)

        feature_table = self._build_feature_table(valid_bundles)
        price_histories = {b.ticker: b.prices for b in valid_bundles}

        return PipelineResult(
            feature_table=feature_table,
            price_histories=price_histories,
            benchmark=benchmark_df,
            failed_tickers=failed,
        )

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _download_with_retry(self, ticker: str) -> Optional[pd.DataFrame]:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = yf.download(
                    ticker,
                    period=self.lookback_period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if df is None or df.empty:
                    raise ValueError("empty dataframe returned")

                # Drop a trailing INCOMPLETE session bar. If this runs while
                # a given ticker's exchange is still mid-session, yfinance
                # can return a final row for "today" with Close (and
                # sometimes other OHLC fields) as NaN or a stale partial
                # value, since the session hasn't closed yet. Left in place,
                # that row becomes `latest = prices.iloc[-1]` downstream and
                # every technical/price reading derived from it goes NaN —
                # this is exactly why US tickers can show NaN price while
                # Indian tickers (already closed for the day, IST-relative)
                # don't: it depends on each exchange's session state at
                # request time, not on the exchange itself.
                if "Close" in df.columns:
                    df = df[df["Close"].notna()]
                if df is None or df.empty:
                    raise ValueError("no complete price bars after dropping incomplete session")

                return df
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Download attempt %d/%d failed for %s: %s",
                    attempt, MAX_RETRIES, ticker, exc,
                )
                time.sleep(RETRY_SLEEP_SECONDS)
        return None

    def _download_benchmark(self) -> Optional[pd.DataFrame]:
        df = self._download_with_retry(self.benchmark_ticker)
        if df is None:
            logger.error(
                "Could not download benchmark %s; relative-performance "
                "target will fall back to absolute returns.",
                self.benchmark_ticker,
            )
            return None
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        return df

    def _fetch_fundamentals(self, ticker: str) -> Dict[str, object]:
        info: Dict[str, object] = {}
        try:
            yf_ticker = yf.Ticker(ticker)
            raw_info = yf_ticker.info or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fundamentals fetch failed for %s: %s", ticker, exc)
            raw_info = {}

        for raw_key, clean_key in FUNDAMENTAL_FIELDS.items():
            value = raw_info.get(raw_key, np.nan)
            info[clean_key] = value if value is not None else np.nan

        # Normalize dividend yield (yfinance sometimes returns it as a
        # fraction already, sometimes as a percent-like float).
        dy = info.get("dividend_yield")
        if isinstance(dy, (int, float)) and dy and dy > 1.0:
            info["dividend_yield"] = dy / 100.0

        if not isinstance(info.get("sector"), str) or not info.get("sector"):
            info["sector"] = "Unknown"

        return info

    # ------------------------------------------------------------------
    # Per-ticker processing
    # ------------------------------------------------------------------

    def _process_single_ticker(
        self, ticker: str, benchmark_df: Optional[pd.DataFrame]
    ) -> TickerBundle:
        prices = self._download_with_retry(ticker)
        if prices is None or len(prices) < MIN_HISTORY_DAYS:
            return TickerBundle(
                ticker=ticker,
                prices=pd.DataFrame(),
                fundamentals={},
                error="insufficient_price_history",
            )

        try:
            prices = self._engineer_technicals(prices)
            prices = self._attach_target(prices, benchmark_df)
        except Exception as exc:  # noqa: BLE001
            logger.error("Feature engineering failed for %s: %s", ticker, exc)
            return TickerBundle(
                ticker=ticker, prices=pd.DataFrame(), fundamentals={}, error=str(exc)
            )

        fundamentals = self._fetch_fundamentals(ticker)
        return TickerBundle(ticker=ticker, prices=prices, fundamentals=fundamentals)

    def _engineer_technicals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators strictly from data available up to (and
        including) each row's own date. Nothing here uses future information,
        so these columns are safe to use as model features as-is.
        """
        df = df.copy()
        close = df["Close"]

        # Log returns & rolling volatility (30-day std of log returns)
        df["log_return"] = np.log(close / close.shift(1))
        df["volatility_30d"] = df["log_return"].rolling(30).std() * np.sqrt(252)

        # Moving averages
        df["ma_20"] = close.rolling(20).mean()
        df["ma_50"] = close.rolling(50).mean()
        df["ma_200"] = close.rolling(200).mean()

        # Average volume
        if "Volume" in df.columns:
            df["avg_volume_30d"] = df["Volume"].rolling(30).mean()
        else:
            df["avg_volume_30d"] = np.nan

        if ta is not None:
            df["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            macd = ta.trend.MACD(close=close)
            df["macd_line"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_hist"] = macd.macd_diff()
        else:
            # Manual fallback if `ta` isn't installed.
            df["rsi_14"] = self._manual_rsi(close, window=14)
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            df["macd_line"] = ema12 - ema26
            df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
            df["macd_hist"] = df["macd_line"] - df["macd_signal"]

        # Rolling max drawdown (30-day) — useful both as a feature and as a
        # regression target for the downside-risk model.
        rolling_max = close.rolling(30, min_periods=5).max()
        drawdown = (close - rolling_max) / rolling_max
        df["max_drawdown_30d"] = drawdown.rolling(30, min_periods=5).min()

        # Semi-deviation (downside deviation) of daily returns, 30-day window
        neg_returns = df["log_return"].where(df["log_return"] < 0, 0.0)
        df["semi_deviation_30d"] = neg_returns.rolling(30).std()

        return df

    @staticmethod
    def _manual_rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _attach_target(
        self, df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Binary target: 1 if the stock's forward N-day return beats the
        benchmark's forward N-day return over the same window, else 0.

        Data-leakage prevention: the target for row t is computed using
        price data at t and t+N only, and is explicitly NaN for the final
        N rows (since their forward window doesn't exist yet). Those rows
        must be dropped before model training.
        """
        df = df.copy()
        n = self.forward_window_days
        close = df["Close"]

        fwd_return = close.shift(-n) / close - 1.0
        df["forward_return"] = fwd_return

        if benchmark_df is not None and not benchmark_df.empty:
            bench_aligned = benchmark_df["Close"].reindex(df.index).ffill()
            bench_fwd_return = bench_aligned.shift(-n) / bench_aligned - 1.0
            df["benchmark_forward_return"] = bench_fwd_return
            df["outperformance"] = df["forward_return"] - df["benchmark_forward_return"]
        else:
            df["benchmark_forward_return"] = np.nan
            df["outperformance"] = df["forward_return"]

        df["target"] = np.where(
            df["outperformance"].isna(), np.nan, (df["outperformance"] > 0).astype(float)
        )
        return df

    # ------------------------------------------------------------------
    # Fundamental imputation
    # ------------------------------------------------------------------

    def _compute_sector_medians(self, bundles: List[TickerBundle]) -> None:
        rows = []
        for b in bundles:
            row = dict(b.fundamentals)
            rows.append(row)
        if not rows:
            self._sector_medians = {}
            return

        df = pd.DataFrame(rows)
        numeric_cols = [c for c in df.columns if c != "sector"]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        medians = df.groupby("sector")[numeric_cols].median(numeric_only=True)
        global_median = df[numeric_cols].median(numeric_only=True)

        self._sector_medians = {
            sector: medians.loc[sector].to_dict() for sector in medians.index
        }
        self._sector_medians["__global__"] = global_median.to_dict()

    def _impute_fundamentals(self, bundle: TickerBundle) -> None:
        sector = bundle.fundamentals.get("sector", "Unknown")
        fallback = self._sector_medians.get("__global__", {})
        sector_medians = self._sector_medians.get(sector, fallback)

        for key, value in list(bundle.fundamentals.items()):
            if key == "sector":
                continue
            numeric_value = pd.to_numeric(value, errors="coerce")
            if pd.isna(numeric_value):
                imputed = sector_medians.get(key, fallback.get(key, np.nan))
                bundle.fundamentals[key] = imputed
                bundle.fundamentals[f"{key}_imputed"] = True
            else:
                bundle.fundamentals[key] = float(numeric_value)
                bundle.fundamentals[f"{key}_imputed"] = False

    # ------------------------------------------------------------------
    # Feature table assembly
    # ------------------------------------------------------------------

    def _build_feature_table(self, bundles: List[TickerBundle]) -> pd.DataFrame:
        """
        Build a single "latest snapshot" row per ticker combining the most
        recent technical readings with fundamentals. This table feeds PCA,
        clustering, KNN, anomaly detection, and (with the historical panel)
        the supervised models.
        """
        records = []
        for b in bundles:
            if b.prices.empty:
                continue
            latest = b.prices.iloc[-1]
            record = {
                "ticker": b.ticker,
                "last_price": latest.get("Close", np.nan),
                "rsi_14": latest.get("rsi_14", np.nan),
                "macd_line": latest.get("macd_line", np.nan),
                "macd_signal": latest.get("macd_signal", np.nan),
                "macd_hist": latest.get("macd_hist", np.nan),
                "ma_20": latest.get("ma_20", np.nan),
                "ma_50": latest.get("ma_50", np.nan),
                "ma_200": latest.get("ma_200", np.nan),
                "volatility_30d": latest.get("volatility_30d", np.nan),
                "avg_volume_30d": latest.get("avg_volume_30d", np.nan),
                "max_drawdown_30d": latest.get("max_drawdown_30d", np.nan),
                "semi_deviation_30d": latest.get("semi_deviation_30d", np.nan),
            }
            record.update(
                {k: v for k, v in b.fundamentals.items() if not k.endswith("_imputed")}
            )
            records.append(record)

        if not records:
            return pd.DataFrame()

        table = pd.DataFrame(records).set_index("ticker")
        return table

    def build_training_panel(
        self, price_histories: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Stack historical rows across all tickers into one long panel suitable
        for time-series-aware supervised training. Rows with a NaN target
        (the last `forward_window_days` rows of each ticker, where the
        forward window hasn't completed yet) are dropped to prevent leakage.
        """
        feature_cols = [
            "rsi_14", "macd_line", "macd_signal", "macd_hist",
            "ma_20", "ma_50", "ma_200", "volatility_30d", "avg_volume_30d",
            "max_drawdown_30d", "semi_deviation_30d",
        ]
        # NOTE: "max_drawdown_30d" is already inside feature_cols (it doubles
        # as both a model feature and the risk-regression target). Do not
        # add it again here — doing so previously created two columns with
        # the same name, which made `panel[target_col]` return a DataFrame
        # instead of a Series downstream and broke the risk model.
        cols_to_select = feature_cols + ["target"]

        frames = []
        for ticker, df in price_histories.items():
            if df.empty:
                continue
            sub = df[cols_to_select].copy()
            sub["ticker"] = ticker
            sub["date"] = df.index
            frames.append(sub)

        if not frames:
            return pd.DataFrame()

        panel = pd.concat(frames, axis=0)
        panel = panel.dropna(subset=["target"])
        panel = panel.sort_values("date").reset_index(drop=True)
        return panel