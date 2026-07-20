"""Decision Tree Rule Extractor — extracts MQL5-compatible trading filters from strategy setups."""

from typing import Dict, Any, List, Tuple, Callable, Optional
import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeClassifier


class RuleExtractor:
    """Train a shallow Decision Tree to discover readable, compileable filtering rules."""

    def __init__(self, max_depth: int = 3, min_win_rate: float = 0.58, min_samples: int = 5) -> None:
        self.max_depth = max_depth
        self.min_win_rate = min_win_rate
        self.min_samples = min_samples

        # Mapping of python feature names to MQL5 variable names
        self.feature_mql_map = {
            "feature_bb_width_pct": "bbWidthPct",
            "feature_ema_slope": "emaSlope",
            "feature_atr_percentile": "atrPercentile",
            "feature_session_london": "sessionLondon",
            "feature_session_ny": "sessionNY",
            "feature_totalGap": "totalGap[1]",
        }

    def extract_rules(
        self,
        labeled_df: pl.DataFrame,
    ) -> Tuple[str, Callable[[Dict[str, Any]], bool], List[Dict[str, Any]]]:
        """Train a Decision Tree on key features and extract paths with high win rates.

        Returns:
            mql5_expression: The compiled boolean expression string for MetaTrader 5.
            python_filter_fn: A callable filter function for backtesting in Python.
            paths_details: A list of details for each winning path.
        """
        if labeled_df.is_empty() or len(labeled_df) < self.min_samples:
            return "true", lambda x: True, []

        # We select MQL5-compatible features present in the dataset
        candidate_features = [
            "feature_bb_width_pct",
            "feature_bb_width_percentile",
            "feature_ema_slope",
            "feature_atr_percentile",
            "feature_session_london",
            "feature_session_ny",
            "feature_totalGap",
        ]
        
        # Resolve actual columns present
        feature_cols = [c for c in candidate_features if c in labeled_df.columns]
        if not feature_cols:
            return "true", lambda x: True, []

        X = labeled_df.select(feature_cols).fill_null(0.0).to_numpy()
        y = labeled_df["tp_before_sl"].cast(pl.Int32).to_numpy()

        if len(np.unique(y)) < 2:
            # All setups won or all lost, no splits possible
            return "true", lambda x: True, []

        # Train a shallow decision tree
        dt = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
        dt.fit(X, y)

        tree = dt.tree_
        winning_paths = []

        def traverse(node_id: int, current_path: List[Tuple[str, str, float]]) -> None:
            # If leaf node
            if tree.feature[node_id] == -2:
                value = tree.value[node_id][0]  # shape (2,)
                samples = int(np.sum(value))
                if samples >= self.min_samples:
                    win_rate = float(value[1] / samples) if samples > 0 else 0.0
                    if win_rate >= self.min_win_rate:
                        winning_paths.append((current_path, win_rate, samples))
                return

            feat_idx = tree.feature[node_id]
            feat_name = feature_cols[feat_idx]
            threshold = float(tree.threshold[node_id])

            # Left child: feature <= threshold
            traverse(int(tree.children_left[node_id]), current_path + [(feat_name, "<=", threshold)])
            # Right child: feature > threshold
            traverse(int(tree.children_right[node_id]), current_path + [(feat_name, ">", threshold)])

        traverse(0, [])

        # Format details and expressions
        paths_details = []
        mql5_paths = []

        for path, wr, smp in winning_paths:
            paths_details.append({
                "path": path,
                "win_rate": wr,
                "samples": smp,
            })

            conditions = []
            for feat, op, thresh in path:
                mql_name = self.feature_mql_map.get(feat, feat.replace("feature_", ""))
                # Special formatting for binary indicators
                if "session" in feat:
                    val = 1 if thresh >= 0.5 else 0
                    if op == "<=":
                        conditions.append(f"{mql_name} == 0")
                    else:
                        conditions.append(f"{mql_name} == 1")
                else:
                    conditions.append(f"{mql_name} {op} {thresh:.4f}")
            
            mql5_paths.append("(" + " && ".join(conditions) + ")")

        # Compile final MQL5 string
        if mql5_paths:
            mql5_expression = " || ".join(mql5_paths)
        else:
            mql5_expression = "true"

        # Compile python filter function
        python_filter_fn = self._make_python_filter(winning_paths)

        return mql5_expression, python_filter_fn, paths_details

    def _make_python_filter(self, winning_paths: List[Tuple[List[Tuple[str, str, float]], float, int]]) -> Callable[[Dict[str, Any]], bool]:
        if not winning_paths:
            return lambda x: True

        def filter_fn(row: Dict[str, Any]) -> bool:
            for path, _, _ in winning_paths:
                match = True
                for feat, op, thresh in path:
                    # In setup features, keys are raw (no 'feature_' prefix)
                    # or they might be prefixed with 'feature_' depending on where it's called
                    val = row.get(feat)
                    if val is None:
                        val = row.get(feat.replace("feature_", ""))
                    if val is None:
                        val = 0.0

                    if op == "<=":
                        if not (float(val) <= thresh):
                            match = False
                            break
                    elif op == ">":
                        if not (float(val) > thresh):
                            match = False
                            break
                if match:
                    return True
            return False

        return filter_fn
