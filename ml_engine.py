"""
ml_engine.py
============
Machine Learning Core for the Intelligent Stock Screener & Portfolio Builder.

Implements seven ML tracks on top of the feature table / training panel
produced by `data_pipeline.py`:

  1. PCA               -> dimensionality reduction (15+ features -> 5 PCs)
  2. KMeans clustering -> auto-labeled stock archetypes + per-stock tags
  3. Classifier bake-off -> P(outperform benchmark over next ~3m), best of
                            XGBoost / LightGBM / RandomForest / LogisticRegression
  4. Random Forest regressor -> expected 30-day downside risk
  5. KNN peer finder   -> top-5 nearest alternative stocks
  6. Isolation Forest  -> anomaly / decoupling detection
  7. SHAP TreeExplainer -> explainability for the classifier (tree models only)

All fit methods are defensive against small universes, missing columns, and
NaNs, since real fundamental/technical data is frequently incomplete.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | ml_engine | %(message)s",
)
logger = logging.getLogger("ml_engine")


def _as_ndarray(x) -> np.ndarray:
    """
    Force any sklearn transform output (ndarray, DataFrame, or a pandas-output
    configured transformer's result — see sklearn.set_config(transform_output=...))
    into a plain numpy array.

    This exists because several places in this module rely on POSITIONAL
    integer-array slicing (e.g. `X_scaled[train_idx]` inside a
    TimeSeriesSplit loop). If `X_scaled` happens to be a pandas DataFrame
    instead of an ndarray, `df[train_idx]` is interpreted as a *column*
    lookup, not row selection — silently corrupting which rows go into which
    fold, or crashing with a KeyError / "feature names" mismatch at predict
    time. Normalizing to ndarray immediately after every fit_transform /
    transform call makes this module's behavior independent of any global
    sklearn output configuration.
    """
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_numpy()
    return np.asarray(x)


SNAPSHOT_FEATURES = [
    "rsi_14", "macd_line", "macd_signal", "macd_hist",
    "ma_20", "ma_50", "ma_200", "volatility_30d", "avg_volume_30d",
    "max_drawdown_30d", "semi_deviation_30d",
    "market_cap", "pe_ratio", "pb_ratio", "roe",
    "debt_to_equity", "dividend_yield", "eps_growth", "beta",
]

CLUSTER_ARCHETYPES = {
    "growth": "High Growth",
    "dividend": "Stable Dividend",
    "risky": "High Risk / High Beta",
    "value": "Undervalued Value Stock",
}


@dataclass
class PCAResult:
    components: pd.DataFrame
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    feature_names: List[str]


@dataclass
class ClusterResult:
    labels: pd.Series
    archetype_names: Dict[int, str]
    centroids: pd.DataFrame
    stock_tags: pd.DataFrame  # per-ticker percentile tags, universe-wide (not cluster-averaged)


@dataclass
class ClassifierResult:
    model: object
    probabilities: pd.Series
    auc_scores: List[float]
    mean_auc: float
    feature_names: List[str]
    scaler: StandardScaler
    model_name: str = "xgboost"
    candidate_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class RiskModelResult:
    model: object
    predictions: pd.Series
    mse: float
    r2: float
    baseline_mse: float
    baseline_r2: float
    feature_names: List[str]


@dataclass
class AnomalyResult:
    is_anomaly: pd.Series          # True/False per ticker
    anomaly_score: pd.Series       # lower = more anomalous


class MLEngine:
    """Bundles all seven ML tracks and keeps fitted artifacts for reuse."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.imputer: Optional[SimpleImputer] = None
        self.feature_names: List[str] = []
        self.pca_result: Optional[PCAResult] = None
        self.cluster_result: Optional[ClusterResult] = None
        self.classifier_result: Optional[ClassifierResult] = None
        self.risk_result: Optional[RiskModelResult] = None
        self.anomaly_result: Optional[AnomalyResult] = None
        self._knn_model: Optional[NearestNeighbors] = None
        self._knn_index: Optional[pd.Index] = None
        self._knn_scaled: Optional[np.ndarray] = None
        self._clf_imputer: Optional[SimpleImputer] = None
        self._clf_scaled_snapshot: Optional[np.ndarray] = None
        self.best_params_: Optional[dict] = None

    # ------------------------------------------------------------------
    # Shared preprocessing
    # ------------------------------------------------------------------

    def _prepare_snapshot_matrix(
        self, feature_table: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Select available snapshot features, impute missing values (median),
        and scale. Returns the cleaned (pre-scale) frame and the scaled
        numpy matrix, both aligned to feature_table.index.
        """
        cols = [c for c in SNAPSHOT_FEATURES if c in feature_table.columns]
        if not cols:
            raise ValueError("No usable snapshot features found in feature_table.")

        raw = feature_table[cols].apply(pd.to_numeric, errors="coerce")

        self.imputer = SimpleImputer(strategy="median")
        imputed = _as_ndarray(self.imputer.fit_transform(raw))
        imputed_df = pd.DataFrame(imputed, index=raw.index, columns=cols)

        self.scaler = StandardScaler()
        scaled = _as_ndarray(self.scaler.fit_transform(imputed_df))

        self.feature_names = cols
        return imputed_df, scaled

    # ------------------------------------------------------------------
    # 1. PCA
    # ------------------------------------------------------------------

    def run_pca(
        self, feature_table: pd.DataFrame, n_components: int = 5
    ) -> PCAResult:
        _, scaled = self._prepare_snapshot_matrix(feature_table)
        n_components = min(n_components, scaled.shape[1], max(scaled.shape[0] - 1, 1))
        n_components = max(n_components, 1)

        pca = PCA(n_components=n_components, random_state=self.random_state)
        components = pca.fit_transform(scaled)

        comp_df = pd.DataFrame(
            components,
            index=feature_table.index,
            columns=[f"PC{i+1}" for i in range(n_components)],
        )
        cumulative = np.cumsum(pca.explained_variance_ratio_)

        self.pca_result = PCAResult(
            components=comp_df,
            explained_variance_ratio=pca.explained_variance_ratio_,
            cumulative_variance=cumulative,
            feature_names=self.feature_names,
        )
        logger.info(
            "PCA fit: %d components explain %.1f%% of variance",
            n_components, cumulative[-1] * 100,
        )
        return self.pca_result

    # ------------------------------------------------------------------
    # 2. KMeans clustering with automatic archetype labeling
    #    + per-stock percentile tags (universe-wide, not cluster-averaged)
    # ------------------------------------------------------------------

    def run_clustering(
        self,
        feature_table: pd.DataFrame,
        n_clusters: int = 4,
        use_pca: bool = True,
    ) -> ClusterResult:
        if use_pca and self.pca_result is not None:
            X = self.pca_result.components.values
            index = self.pca_result.components.index
        else:
            imputed_df, X = self._prepare_snapshot_matrix(feature_table)
            index = imputed_df.index

        n_clusters = max(1, min(n_clusters, X.shape[0]))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        labels_series = pd.Series(labels, index=index, name="cluster")

        # Compute centroid characteristics in the ORIGINAL feature space
        # (not PCA space) so labeling logic is interpretable.
        imputed_df, _ = self._prepare_snapshot_matrix(feature_table)
        imputed_df = imputed_df.reindex(index)
        centroid_profile = imputed_df.groupby(labels_series.values).mean()

        archetype_names = self._label_clusters(centroid_profile)

        # Per-stock tags computed against the WHOLE universe, not the
        # cluster average. This is what surfaces a high-yield stock that
        # got clustered into "Growth" (because that's its dominant trait)
        # but should still show up when screening for dividend yield.
        stock_tags = self._compute_stock_level_tags(imputed_df)

        self.cluster_result = ClusterResult(
            labels=labels_series,
            archetype_names=archetype_names,
            centroids=centroid_profile,
            stock_tags=stock_tags,
        )
        return self.cluster_result

    def _label_clusters(self, centroid_profile: pd.DataFrame) -> Dict[int, str]:
        """
        Heuristically label each cluster centroid as one of four archetypes
        by ranking centroids on growth, dividend, risk, and value signals.
        This is a CLUSTER-level label only — see _compute_stock_level_tags
        for the per-ticker signal that doesn't get averaged away.
        """
        names: Dict[int, str] = {}
        cols = centroid_profile.columns

        def safe_col(name: str) -> pd.Series:
            if name in cols:
                return centroid_profile[name]
            return pd.Series(0.0, index=centroid_profile.index)

        growth_score = safe_col("eps_growth") - safe_col("pe_ratio").rank(pct=True)
        dividend_score = safe_col("dividend_yield") - safe_col("volatility_30d").rank(pct=True)
        risk_score = safe_col("volatility_30d") + safe_col("beta") + safe_col("debt_to_equity").rank(pct=True)
        value_score = -safe_col("pe_ratio").rank(pct=True) - safe_col("pb_ratio").rank(pct=True) + safe_col("roe").rank(pct=True)

        scores = pd.DataFrame(
            {
                "growth": growth_score,
                "dividend": dividend_score,
                "risky": risk_score,
                "value": value_score,
            }
        )

        assigned_labels: Dict[str, int] = {}
        remaining_clusters = list(scores.index)
        for archetype in ["risky", "growth", "dividend", "value"]:
            if not remaining_clusters:
                break
            candidate = scores.loc[remaining_clusters, archetype].idxmax()
            assigned_labels[archetype] = candidate
            remaining_clusters.remove(candidate)

        cluster_to_archetype = {v: k for k, v in assigned_labels.items()}
        for cluster_id in centroid_profile.index:
            archetype_key = cluster_to_archetype.get(cluster_id)
            names[cluster_id] = CLUSTER_ARCHETYPES.get(archetype_key, f"Cluster {cluster_id}")

        return names

    def _compute_stock_level_tags(
        self, imputed_df: pd.DataFrame, top_pct: float = 0.8
    ) -> pd.DataFrame:
        """
        Percentile-ranks every stock on the same growth/dividend/risk/value
        axes used for cluster labeling, but per-TICKER across the full
        universe — so a high-yield stock that got clustered into "Growth"
        still shows up as a dividend candidate when you screen for yield.

        Returns a DataFrame indexed like imputed_df with one percentile
        column per axis plus a human-readable `tags` column listing every
        archetype the stock scores in the top `top_pct` percentile on
        (independent of which KMeans cluster it landed in).
        """
        cols = imputed_df.columns

        def safe(col: str) -> pd.Series:
            return imputed_df[col] if col in cols else pd.Series(0.0, index=imputed_df.index)

        scores = pd.DataFrame({
            "growth": safe("eps_growth") - safe("pe_ratio").rank(pct=True),
            "dividend": safe("dividend_yield") - safe("volatility_30d").rank(pct=True),
            "risky": safe("volatility_30d") + safe("beta") + safe("debt_to_equity").rank(pct=True),
            "value": -safe("pe_ratio").rank(pct=True) - safe("pb_ratio").rank(pct=True) + safe("roe").rank(pct=True),
        })

        percentiles = scores.rank(pct=True)
        percentiles.columns = [f"{c}_percentile" for c in percentiles.columns]

        def tag_row(row) -> str:
            hits = [
                CLUSTER_ARCHETYPES[axis]
                for axis in ["growth", "dividend", "risky", "value"]
                if row[f"{axis}_percentile"] >= top_pct
            ]
            return ", ".join(hits) if hits else "—"

        percentiles["tags"] = percentiles.apply(tag_row, axis=1)
        return percentiles

    # ------------------------------------------------------------------
    # 3. Classifier bake-off (temporal split) — outperformance probability
    #    Tries XGBoost, LightGBM (if installed), RandomForest, and
    #    LogisticRegression, each tuned via RandomizedSearchCV over
    #    TimeSeriesSplit folds, and keeps whichever wins on mean CV AUC.
    # ------------------------------------------------------------------

    def _add_cross_sectional_rank_features(
        self, df: pd.DataFrame, feature_cols: List[str], date_col: Optional[str] = "date",
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Appends date-wise percentile-rank versions of each feature.

        Absolute values (PE=18) carry weak, non-stationary signal for
        predicting RELATIVE outperformance vs a benchmark across changing
        market regimes. Where a stock ranks *within its same-date universe*
        (cheaper than 80% of peers today) tends to be a much more stable
        predictor. This is usually the single biggest lever on AUC for
        cross-sectional equity models — bigger than hyperparameter tuning.
        """
        df = df.copy()
        rank_cols = []
        if date_col and date_col in df.columns:
            grouped = df.groupby(date_col)
            for col in feature_cols:
                rank_col = f"{col}_xrank"
                df[rank_col] = grouped[col].rank(pct=True)
                rank_cols.append(rank_col)
        else:
            for col in feature_cols:
                rank_col = f"{col}_xrank"
                df[rank_col] = df[col].rank(pct=True)
                rank_cols.append(rank_col)
        return df, rank_cols

    def train_classifier(
        self,
        training_panel: pd.DataFrame,
        latest_snapshot: pd.DataFrame,
        n_splits: int = 5,
        search_iter: int = 40,
    ) -> ClassifierResult:
        base_feature_cols = [c for c in SNAPSHOT_FEATURES if c in training_panel.columns]
        panel = training_panel.dropna(subset=["target"]).sort_values("date")

        # Cross-sectional rank features, computed date-by-date on the panel.
        panel, rank_cols = self._add_cross_sectional_rank_features(
            panel, base_feature_cols, date_col="date"
        )
        feature_cols = base_feature_cols + rank_cols

        X_raw = panel[feature_cols].apply(pd.to_numeric, errors="coerce")
        y = panel["target"].astype(int).values

        imputer = SimpleImputer(strategy="median")
        X_imputed = _as_ndarray(imputer.fit_transform(X_raw))
        scaler = StandardScaler()
        X_scaled = _as_ndarray(scaler.fit_transform(X_imputed))

        n_splits = max(2, min(n_splits, len(panel) // 50)) if len(panel) >= 100 else 2
        tscv = TimeSeriesSplit(n_splits=n_splits)

        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

        def fold_auc(model_factory) -> List[float]:
            scores = []
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    continue
                m = model_factory(y_train)
                m.fit(X_train, y_train)
                preds = m.predict_proba(X_test)[:, 1]
                try:
                    scores.append(roc_auc_score(y_test, preds))
                except ValueError:
                    pass
            return scores

        candidates: Dict[str, dict] = {}

        # --- XGBoost: tuned via RandomizedSearchCV --------------------------
        if xgb is not None:
            xgb_search = RandomizedSearchCV(
                estimator=xgb.XGBClassifier(
                    eval_metric="logloss", random_state=self.random_state,
                    scale_pos_weight=scale_pos_weight, n_jobs=-1,
                ),
                param_distributions={
                    "n_estimators": randint(100, 500),
                    "max_depth": randint(3, 8),
                    "learning_rate": uniform(0.01, 0.19),
                    "subsample": uniform(0.6, 0.4),
                    "colsample_bytree": uniform(0.6, 0.4),
                    "min_child_weight": randint(1, 10),
                    "gamma": uniform(0.0, 0.5),
                    "reg_alpha": uniform(0.0, 1.0),
                    "reg_lambda": uniform(0.5, 2.0),
                },
                n_iter=search_iter, scoring="roc_auc", cv=tscv,
                n_jobs=-1, random_state=self.random_state, refit=False, verbose=0,
            )
            xgb_search.fit(X_scaled, y)
            best_xgb_params = xgb_search.best_params_
            factory = lambda y_tr, p=best_xgb_params: xgb.XGBClassifier(
                **p, eval_metric="logloss", random_state=self.random_state,
                scale_pos_weight=((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)), n_jobs=-1,
            )
            candidates["xgboost"] = {
                "auc_scores": fold_auc(factory),
                "final_factory": lambda p=best_xgb_params: xgb.XGBClassifier(
                    **p, eval_metric="logloss", random_state=self.random_state,
                    scale_pos_weight=scale_pos_weight, n_jobs=-1,
                ),
            }
        else:
            logger.warning("xgboost not installed — skipping this candidate in the bake-off.")

        # --- LightGBM, if installed: tuned via RandomizedSearchCV -----------
        if lgb is not None:
            lgb_search = RandomizedSearchCV(
                estimator=lgb.LGBMClassifier(
                    random_state=self.random_state, class_weight="balanced", n_jobs=-1, verbosity=-1,
                ),
                param_distributions={
                    "n_estimators": randint(100, 500),
                    "max_depth": randint(3, 10),
                    "learning_rate": uniform(0.01, 0.19),
                    "subsample": uniform(0.6, 0.4),
                    "colsample_bytree": uniform(0.6, 0.4),
                    "num_leaves": randint(15, 63),
                    "reg_alpha": uniform(0.0, 1.0),
                    "reg_lambda": uniform(0.5, 2.0),
                },
                n_iter=search_iter, scoring="roc_auc", cv=tscv,
                n_jobs=-1, random_state=self.random_state, refit=False, verbose=0,
            )
            lgb_search.fit(X_scaled, y)
            best_lgb_params = lgb_search.best_params_
            factory = lambda y_tr, p=best_lgb_params: lgb.LGBMClassifier(
                **p, random_state=self.random_state, class_weight="balanced", n_jobs=-1, verbosity=-1,
            )
            candidates["lightgbm"] = {
                "auc_scores": fold_auc(factory),
                "final_factory": lambda p=best_lgb_params: lgb.LGBMClassifier(
                    **p, random_state=self.random_state, class_weight="balanced", n_jobs=-1, verbosity=-1,
                ),
            }

        # --- Random Forest: light random search ------------------------------
        rf_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(
                random_state=self.random_state, class_weight="balanced", n_jobs=-1,
            ),
            param_distributions={
                "n_estimators": randint(200, 600),
                "max_depth": randint(3, 15),
                "min_samples_leaf": randint(1, 10),
                "max_features": uniform(0.3, 0.7),
            },
            n_iter=min(search_iter, 25), scoring="roc_auc", cv=tscv,
            n_jobs=-1, random_state=self.random_state, refit=False, verbose=0,
        )
        rf_search.fit(X_scaled, y)
        best_rf_params = rf_search.best_params_
        factory = lambda y_tr, p=best_rf_params: RandomForestClassifier(
            **p, random_state=self.random_state, class_weight="balanced", n_jobs=-1,
        )
        candidates["random_forest"] = {
            "auc_scores": fold_auc(factory),
            "final_factory": lambda p=best_rf_params: RandomForestClassifier(
                **p, random_state=self.random_state, class_weight="balanced", n_jobs=-1,
            ),
        }

        # --- Logistic Regression: small C search, cheap sanity-check model --
        lr_search = RandomizedSearchCV(
            estimator=LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=self.random_state,
            ),
            param_distributions={"C": uniform(0.01, 10.0)},
            n_iter=15, scoring="roc_auc", cv=tscv,
            n_jobs=-1, random_state=self.random_state, refit=False, verbose=0,
        )
        lr_search.fit(X_scaled, y)
        best_lr_params = lr_search.best_params_
        factory = lambda y_tr, p=best_lr_params: LogisticRegression(
            **p, max_iter=2000, class_weight="balanced", random_state=self.random_state,
        )
        candidates["logistic_regression"] = {
            "auc_scores": fold_auc(factory),
            "final_factory": lambda p=best_lr_params: LogisticRegression(
                **p, max_iter=2000, class_weight="balanced", random_state=self.random_state,
            ),
        }

        if not candidates:
            raise RuntimeError("No classifier candidates available — install xgboost, lightgbm, or scikit-learn.")

        # --- Pick the winner on mean CV AUC ----------------------------------
        candidate_scores = {
            name: float(np.mean(c["auc_scores"])) if c["auc_scores"] else float("nan")
            for name, c in candidates.items()
        }
        valid_scores = {k: v for k, v in candidate_scores.items() if v == v}  # drop NaNs
        if not valid_scores:
            raise RuntimeError("None of the classifier candidates produced a usable AUC score.")
        best_name = max(valid_scores, key=valid_scores.get)
        best_auc_scores = candidates[best_name]["auc_scores"]
        final_model = candidates[best_name]["final_factory"]()
        final_model.fit(X_scaled, y)

        logger.info(
            "Model bake-off complete | scores=%s | winner=%s (mean AUC=%.3f)",
            {k: round(v, 3) for k, v in candidate_scores.items() if v == v},
            best_name, candidate_scores[best_name],
        )

        # Score the latest cross-sectional snapshot (rank features computed
        # against the CURRENT snapshot's own universe, single date).
        snap_df, _ = self._add_cross_sectional_rank_features(
            latest_snapshot, base_feature_cols, date_col=None
        )
        snap_X = snap_df.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce")
        snap_imputed = _as_ndarray(imputer.transform(snap_X))
        snap_scaled = _as_ndarray(scaler.transform(snap_imputed))
        probs = final_model.predict_proba(snap_scaled)[:, 1]
        prob_series = pd.Series(probs, index=latest_snapshot.index, name="buy_probability")

        mean_auc = float(np.mean(best_auc_scores)) if best_auc_scores else float("nan")

        self.classifier_result = ClassifierResult(
            model=final_model,
            probabilities=prob_series,
            auc_scores=best_auc_scores,
            mean_auc=mean_auc,
            feature_names=feature_cols,
            scaler=scaler,
            model_name=best_name,
            candidate_scores=candidate_scores,
        )
        self.best_params_ = candidates[best_name].get("final_factory")
        self._clf_imputer = imputer
        self._clf_scaled_snapshot = snap_scaled
        return self.classifier_result

    # ------------------------------------------------------------------
    # 4. Random Forest regressor — downside risk estimation
    # ------------------------------------------------------------------

    def train_risk_model(
        self, training_panel: pd.DataFrame, latest_snapshot: pd.DataFrame
    ) -> RiskModelResult:
        target_col = "max_drawdown_30d"
        feature_cols = [
            c for c in SNAPSHOT_FEATURES
            if c in training_panel.columns and c != target_col
        ]
        panel = training_panel.dropna(subset=[target_col]).sort_values("date")
        X_raw = panel[feature_cols].apply(pd.to_numeric, errors="coerce")

        target_data = panel[target_col]
        if isinstance(target_data, pd.DataFrame):
            # Defensive: a duplicated column name would make this a
            # DataFrame instead of a Series. Collapse to a single column
            # rather than letting pd.to_numeric raise on 2-D input.
            target_data = target_data.iloc[:, 0]
        y = pd.to_numeric(target_data, errors="coerce")

        valid_mask = ~y.isna()
        X_raw, y = X_raw[valid_mask], y[valid_mask]

        imputer = SimpleImputer(strategy="median")
        X_imputed = _as_ndarray(imputer.fit_transform(X_raw))

        split_point = int(len(X_imputed) * 0.8)
        X_train, X_test = X_imputed[:split_point], X_imputed[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        if len(X_test) > 0:
            preds_test = model.predict(X_test)
            mse = mean_squared_error(y_test, preds_test)
            r2 = r2_score(y_test, preds_test)
            baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
            baseline_mse = mean_squared_error(y_test, baseline_pred)
            baseline_r2 = r2_score(y_test, baseline_pred)
        else:
            mse = r2 = baseline_mse = baseline_r2 = float("nan")

        # Refit on all data, then score the latest snapshot per ticker.
        model.fit(X_imputed, y)
        snap_X = latest_snapshot.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce")
        snap_imputed = _as_ndarray(imputer.transform(snap_X))
        preds = model.predict(snap_imputed)
        pred_series = pd.Series(preds, index=latest_snapshot.index, name="expected_downside_risk")

        logger.info(
            "Risk model trained | MSE=%.5f (baseline %.5f) | R2=%.3f (baseline %.3f)",
            mse, baseline_mse, r2, baseline_r2,
        )

        self.risk_result = RiskModelResult(
            model=model,
            predictions=pred_series,
            mse=mse,
            r2=r2,
            baseline_mse=baseline_mse,
            baseline_r2=baseline_r2,
            feature_names=feature_cols,
        )
        return self.risk_result

    # ------------------------------------------------------------------
    # 5. KNN peer finder
    # ------------------------------------------------------------------

    def build_peer_finder(self, feature_table: pd.DataFrame) -> None:
        peer_features = [
            c for c in ["pe_ratio", "pb_ratio", "roe", "debt_to_equity",
                        "dividend_yield", "eps_growth", "beta", "volatility_30d"]
            if c in feature_table.columns
        ]
        if not peer_features:
            raise ValueError("No fundamental/volatility features available for peer finding.")

        raw = feature_table[peer_features].apply(pd.to_numeric, errors="coerce")
        imputer = SimpleImputer(strategy="median")
        imputed = _as_ndarray(imputer.fit_transform(raw))
        scaler = StandardScaler()
        scaled = _as_ndarray(scaler.fit_transform(imputed))

        n_neighbors = min(6, len(feature_table))
        model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        model.fit(scaled)

        self._knn_model = model
        self._knn_index = feature_table.index
        self._knn_scaled = scaled

    def find_peers(self, ticker: str, top_n: int = 5) -> List[Tuple[str, float]]:
        if self._knn_model is None or self._knn_index is None:
            raise RuntimeError("Call build_peer_finder() before find_peers().")
        if ticker not in self._knn_index:
            raise KeyError(f"{ticker} not found in feature table.")

        pos = self._knn_index.get_loc(ticker)
        query = self._knn_scaled[pos].reshape(1, -1)
        n_neighbors = min(top_n + 1, len(self._knn_index))
        distances, indices = self._knn_model.kneighbors(query, n_neighbors=n_neighbors)

        peers = []
        for dist, idx in zip(distances[0], indices[0]):
            candidate = self._knn_index[idx]
            if candidate == ticker:
                continue
            peers.append((candidate, float(dist)))
        return peers[:top_n]

    # ------------------------------------------------------------------
    # 6. Isolation Forest — anomaly detection
    # ------------------------------------------------------------------

    def run_anomaly_detection(
        self, feature_table: pd.DataFrame, contamination: float = 0.1
    ) -> AnomalyResult:
        _, scaled = self._prepare_snapshot_matrix(feature_table)
        contamination = min(max(contamination, 0.01), 0.4)

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=self.random_state,
        )
        preds = model.fit_predict(scaled)          # -1 = anomaly, 1 = normal
        scores = model.decision_function(scaled)    # lower = more anomalous

        is_anomaly = pd.Series(preds == -1, index=feature_table.index, name="is_anomaly")
        anomaly_score = pd.Series(scores, index=feature_table.index, name="anomaly_score")

        self.anomaly_result = AnomalyResult(is_anomaly=is_anomaly, anomaly_score=anomaly_score)
        logger.info(
            "Anomaly detection: %d/%d tickers flagged", int(is_anomaly.sum()), len(is_anomaly)
        )
        return self.anomaly_result

    # ------------------------------------------------------------------
    # 7. SHAP explainability for the winning classifier
    # ------------------------------------------------------------------

    def explain_ticker(self, ticker: str, feature_table: pd.DataFrame) -> pd.Series:
        """
        Returns a Series of SHAP contributions (feature -> value) for a
        single ticker's outperformance-probability prediction.

        Uses shap.TreeExplainer for tree-based winners (xgboost, lightgbm,
        random_forest) and falls back to shap.LinearExplainer for
        logistic_regression, since TreeExplainer only supports tree models.
        """
        if shap is None:
            raise ImportError("shap is required for the explainability track.")
        if self.classifier_result is None:
            raise RuntimeError("Train the classifier before requesting explanations.")
        if ticker not in feature_table.index:
            raise KeyError(f"{ticker} not found in feature table.")

        feature_cols = self.classifier_result.feature_names

        # feature_cols includes cross-sectional "_xrank" columns added during
        # training (see _add_cross_sectional_rank_features), but those are
        # never persisted onto feature_table itself — it only ever has the
        # base SNAPSHOT_FEATURES. Recompute the same ranks here, against the
        # current universe, mirroring exactly how the latest snapshot was
        # scored in train_classifier (date_col=None: rank within this single
        # cross-section since feature_table has one row per ticker, no date
        # column to group by).
        base_feature_cols = [c for c in SNAPSHOT_FEATURES if c in feature_table.columns]
        ranked_table, _ = self._add_cross_sectional_rank_features(
            feature_table, base_feature_cols, date_col=None
        )
        missing = [c for c in feature_cols if c not in ranked_table.columns]
        if missing:
            raise ValueError(
                f"Cannot compute SHAP explanation — engineered feature(s) missing "
                f"from the current snapshot: {missing}"
            )

        row = ranked_table.loc[[ticker], feature_cols].apply(pd.to_numeric, errors="coerce")
        row_imputed = _as_ndarray(self._clf_imputer.transform(row))
        row_scaled = _as_ndarray(self.classifier_result.scaler.transform(row_imputed))

        model = self.classifier_result.model
        if self.classifier_result.model_name == "logistic_regression":
            explainer = shap.LinearExplainer(model, self._clf_scaled_snapshot)
        else:
            explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(row_scaled)
        
        # 1. Handle cases where SHAP returns a list of arrays (one per class)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
            
        # 2. Handle cases where SHAP returns a 3D array (n_samples, n_features, n_classes)
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1] # Keep only the positive class

        # 3. Flatten the now-guaranteed 2D array (1, n_features) into 1D
        contributions = pd.Series(
            np.ravel(shap_values), index=feature_cols, name=ticker
        ).sort_values(key=lambda s: s.abs(), ascending=False)
        
        return contributions