"""
portfolio.py
============
Financial Portfolio Optimization for the Intelligent Stock Screener &
Portfolio Builder.

Provides two allocation algorithms over a chosen set of tickers:

  * Maximum Sharpe Ratio (Modern Portfolio Theory, via SLSQP)
  * Risk Parity (equal risk contribution, via SLSQP)

Both take a dict of ticker -> price history DataFrames (as produced by
`data_pipeline.py`) and return weights that sum to 1.0, plus expected
annualized return / volatility / Sharpe ratio for the resulting portfolio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | portfolio | %(message)s",
)
logger = logging.getLogger("portfolio")

TRADING_DAYS_PER_YEAR = 252


@dataclass
class PortfolioResult:
    weights: pd.Series               # ticker -> weight, sums to 1.0
    expected_return: float           # annualized
    expected_volatility: float       # annualized
    sharpe_ratio: float
    method: str
    tickers_dropped: List[str]
    used_shrinkage: bool = True      # NEW — whether shrinkage estimators were used


class PortfolioOptimizer:
    """
    Builds optimal portfolios from historical daily price data using either
    a Maximum Sharpe Ratio (MPT) objective or a Risk Parity objective.
    """

    def __init__(self, risk_free_rate: float = 0.06):
        # Default risk-free assumption is India-centric (~10Y G-Sec yield);
        # override for other markets (e.g. ~0.045 for US Treasuries).
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Shared data prep
    # ------------------------------------------------------------------

    def _build_return_matrix(
        self, price_histories: Dict[str, pd.DataFrame], tickers: List[str]
    ) -> pd.DataFrame:
        series_dict = {}
        dropped = []
        for t in tickers:
            df = price_histories.get(t)
            if df is None or df.empty or "Close" not in df.columns:
                dropped.append(t)
                continue
            series_dict[t] = df["Close"]

        if not series_dict:
            raise ValueError("None of the requested tickers have usable price history.")

        price_df = pd.DataFrame(series_dict).dropna(how="all")
        price_df = price_df.ffill().dropna()
        returns = np.log(price_df / price_df.shift(1)).dropna()

        if dropped:
            logger.warning("Dropped tickers with insufficient data: %s", dropped)

        return returns

    def _estimate_moments(
        self,
        returns: pd.DataFrame,
        shrink_cov: bool = True,
        shrink_mean: bool = True,
        mean_shrinkage_intensity: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample mean/covariance from historical daily returns are notoriously
        noisy estimators — covariance especially, since it has O(n^2) free
        parameters estimated from a comparatively small number of daily
        observations. That estimation noise, not the optimizer, is usually
        what makes Max-Sharpe portfolios unstable (large weight swings from
        small changes in the lookback window) or overconcentrated in a few
        names with spuriously high historical mean return.

          * Ledoit-Wolf shrinks the sample covariance toward a structured
            target, which is provably better-conditioned and lower-error
            than the raw sample covariance, particularly as the number of
            assets approaches the number of observations.
          * Mean returns are shrunk toward the cross-sectional grand mean
            (a simple James-Stein-style shrinkage) since the sample mean
            is the single noisiest input to a Sharpe-ratio objective —
            optimizers chase noisy high means aggressively.
        """
        raw_mean = returns.mean().values

        if shrink_mean and len(raw_mean) > 1:
            grand_mean = float(np.mean(raw_mean))
            mean_returns = (
                (1 - mean_shrinkage_intensity) * raw_mean
                + mean_shrinkage_intensity * grand_mean
            )
        else:
            mean_returns = raw_mean

        if shrink_cov:
            cov_matrix = LedoitWolf().fit(returns.values).covariance_
        else:
            cov_matrix = returns.cov().values

        return mean_returns, cov_matrix

    def _portfolio_stats(
        self, weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> tuple[float, float, float]:
        port_return = float(np.dot(weights, mean_returns) * TRADING_DAYS_PER_YEAR)
        port_vol = float(
            np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 1e-9 else 0.0
        return port_return, port_vol, sharpe

    def _multi_start_minimize(
        self,
        objective,
        n: int,
        bounds: tuple,
        constraints: list,
        n_restarts: int = 8,
        maxiter: int = 500,
        ftol: float = 1e-9,
    ):
        """
        SLSQP is a local optimizer — a single fixed initial guess (equal
        weights) can land in different local minima depending on the
        objective's curvature, which is especially common for the
        non-convex risk-parity objective. Running several restarts from
        random feasible-ish starting points (Dirichlet-sampled, so they
        already sum to 1) and keeping the best result makes the outcome
        far less sensitive to the starting point.
        """
        rng = np.random.default_rng(42)
        starting_points = [np.repeat(1.0 / n, n)]
        for _ in range(max(n_restarts - 1, 0)):
            starting_points.append(rng.dirichlet(np.ones(n)))

        best_result = None
        for start in starting_points:
            result = minimize(
                objective, start, method="SLSQP", bounds=bounds, constraints=constraints,
                options={"maxiter": maxiter, "ftol": ftol},
            )
            if best_result is None:
                best_result = result
                continue
            candidate_better = result.success and (
                not best_result.success or result.fun < best_result.fun
            )
            if candidate_better:
                best_result = result

        if not best_result.success:
            logger.warning(
                "Optimization did not fully converge in any of %d restarts: %s",
                len(starting_points), best_result.message,
            )
        return best_result

    # ------------------------------------------------------------------
    # Max Sharpe Ratio
    # ------------------------------------------------------------------

    def optimize_max_sharpe(
        self,
        price_histories: Dict[str, pd.DataFrame],
        tickers: List[str],
        max_weight: float = 0.4,
        min_weight: float = 0.0,
        use_shrinkage: bool = True,
    ) -> PortfolioResult:
        returns = self._build_return_matrix(price_histories, tickers)
        used_tickers = list(returns.columns)
        dropped = [t for t in tickers if t not in used_tickers]
        n = len(used_tickers)

        # Feasibility check: with n assets each capped at max_weight, the
        # weights can sum to at most n * max_weight. If that's < 1.0 the
        # sum-to-1 constraint can never be satisfied and SLSQP will just
        # fail to converge with a confusing message — catch it up front.
        if n * max_weight < 1.0:
            raise ValueError(
                f"Infeasible constraints: {n} assets with max_weight={max_weight} "
                f"can sum to at most {n * max_weight:.2f}, need >= 1.0. "
                f"Raise max_weight or include more tickers."
            )

        mean_returns, cov_matrix = self._estimate_moments(
            returns, shrink_cov=use_shrinkage, shrink_mean=use_shrinkage
        )

        def neg_sharpe(w: np.ndarray) -> float:
            _, _, sharpe = self._portfolio_stats(w, mean_returns, cov_matrix)
            return -sharpe

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple((min_weight, max_weight) for _ in range(n))

        result = self._multi_start_minimize(neg_sharpe, n, bounds, constraints)

        weights = np.clip(result.x, 0, None)
        weights = weights / weights.sum()

        port_return, port_vol, sharpe = self._portfolio_stats(weights, mean_returns, cov_matrix)
        weight_series = pd.Series(weights, index=used_tickers, name="weight").sort_values(ascending=False)

        return PortfolioResult(
            weights=weight_series,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            method="Maximum Sharpe Ratio (MPT)",
            tickers_dropped=dropped,
            used_shrinkage=use_shrinkage,
        )

    # ------------------------------------------------------------------
    # Risk Parity
    # ------------------------------------------------------------------

    def optimize_risk_parity(
        self,
        price_histories: Dict[str, pd.DataFrame],
        tickers: List[str],
        use_shrinkage: bool = True,
    ) -> PortfolioResult:
        returns = self._build_return_matrix(price_histories, tickers)
        used_tickers = list(returns.columns)
        dropped = [t for t in tickers if t not in used_tickers]
        n = len(used_tickers)

        mean_returns, cov_matrix = self._estimate_moments(
            returns, shrink_cov=use_shrinkage, shrink_mean=use_shrinkage
        )

        def risk_contributions(w: np.ndarray) -> np.ndarray:
            port_var = w.T @ cov_matrix @ w
            marginal_contrib = cov_matrix @ w
            return w * marginal_contrib / (port_var + 1e-12)

        def risk_parity_objective(w: np.ndarray) -> float:
            contribs = risk_contributions(w)
            target = np.repeat(1.0 / n, n)
            return float(np.sum((contribs - target) ** 2))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple((0.001, 1.0) for _ in range(n))

        result = self._multi_start_minimize(
            risk_parity_objective, n, bounds, constraints, maxiter=1000, ftol=1e-12,
        )

        weights = np.clip(result.x, 0, None)
        weights = weights / weights.sum()

        port_return, port_vol, sharpe = self._portfolio_stats(weights, mean_returns, cov_matrix)
        weight_series = pd.Series(weights, index=used_tickers, name="weight").sort_values(ascending=False)

        return PortfolioResult(
            weights=weight_series,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            method="Risk Parity",
            tickers_dropped=dropped,
            used_shrinkage=use_shrinkage,
        )

    # ------------------------------------------------------------------
    # Convenience: auto-select top-N by model probability
    # ------------------------------------------------------------------

    @staticmethod
    def auto_select_top_n(buy_probabilities: pd.Series, top_n: int = 5) -> List[str]:
        return list(buy_probabilities.sort_values(ascending=False).head(top_n).index)