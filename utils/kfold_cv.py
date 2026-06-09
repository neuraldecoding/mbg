"""
K-Fold Cross Validation utilities for MBG model.

Designed for Q1 journal standards in EEG/BCI research:
- Subject-independent splits (GroupKFold) to avoid data leakage
- Leave-One-Subject-Out (LOSO) validation
- Stratified K-fold with proper class balance
- Statistical significance testing
- Comprehensive metrics reporting

CRITICAL: For EEG data, NEVER use random splits that mix data from the
same subject across train/test sets. This causes data leakage due to
temporal autocorrelation within subjects.
"""

import os
import json
import csv
import time
from typing import Optional, Dict, Any, List, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

try:
    from sklearn.model_selection import (
        StratifiedKFold,
        GroupKFold,
        LeaveOneGroupOut,
    )
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
        cohen_kappa_score,
        confusion_matrix,
        mean_squared_error,
        mean_absolute_error,
        r2_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not available. Install with: pip install scikit-learn")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARNING] scipy not available. Statistical tests will be unavailable.")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    task_type: str = 'classification',
) -> Dict[str, float]:
    """Compute comprehensive metrics for evaluation.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels (for classification) or values (for regression).
        y_prob: Predicted probabilities (optional, for AUC-ROC).
        task_type: 'classification' or 'regression'.

    Returns:
        Dictionary of metric name to value.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for compute_metrics")

    metrics = {}

    if task_type == 'classification':
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        # F1 score
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            metrics['f1'] = f1_score(y_true, y_pred, average='binary')
        else:
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')

        # AUC-ROC (requires probability scores)
        if y_prob is not None:
            try:
                if n_classes == 2:
                    if y_prob.ndim == 2:
                        metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
                    else:
                        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                else:
                    metrics['auc_roc'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='macro'
                    )
            except ValueError:
                # AUC not defined if only one class present in y_true
                metrics['auc_roc'] = float('nan')

    elif task_type == 'regression':
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)

    return metrics


def aggregate_fold_results(
    fold_results: List[Dict[str, float]],
    confidence_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Aggregate results across folds with mean, std, and confidence intervals.

    Args:
        fold_results: List of metric dictionaries, one per fold.
        confidence_level: Confidence level for CI (default 95%).

    Returns:
        Dictionary mapping metric name to {mean, std, ci_lower, ci_upper, n_folds}.
    """
    if not fold_results:
        return {}

    # Collect all metric names
    all_metrics = set()
    for result in fold_results:
        all_metrics.update(result.keys())

    aggregated = {}
    n_folds = len(fold_results)

    for metric_name in all_metrics:
        values = [r[metric_name] for r in fold_results if metric_name in r]
        values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]

        if not values:
            continue

        values_arr = np.array(values)
        mean_val = np.mean(values_arr)
        std_val = np.std(values_arr, ddof=1) if len(values_arr) > 1 else 0.0

        # Confidence interval
        if len(values_arr) > 1 and SCIPY_AVAILABLE:
            sem = std_val / np.sqrt(len(values_arr))
            t_crit = stats.t.ppf((1 + confidence_level) / 2, df=len(values_arr) - 1)
            ci_lower = mean_val - t_crit * sem
            ci_upper = mean_val + t_crit * sem
        else:
            ci_lower = mean_val
            ci_upper = mean_val

        aggregated[metric_name] = {
            'mean': float(mean_val),
            'std': float(std_val),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_folds': len(values_arr),
        }

    return aggregated


def statistical_test(
    results_a: List[float],
    results_b: List[float],
    test_type: str = 'both',
) -> Dict[str, Any]:
    """Perform statistical significance tests between two methods.

    Uses paired t-test (parametric) and Wilcoxon signed-rank test (non-parametric).

    Args:
        results_a: Per-fold results for method A.
        results_b: Per-fold results for method B.
        test_type: 'ttest', 'wilcoxon', or 'both'.

    Returns:
        Dictionary with test statistics and p-values.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for statistical_test")

    results_a = np.array(results_a)
    results_b = np.array(results_b)

    assert len(results_a) == len(results_b), \
        "Both result lists must have the same length (one per fold)"

    output = {
        'n_folds': len(results_a),
        'mean_a': float(np.mean(results_a)),
        'mean_b': float(np.mean(results_b)),
        'std_a': float(np.std(results_a, ddof=1)),
        'std_b': float(np.std(results_b, ddof=1)),
        'mean_difference': float(np.mean(results_a - results_b)),
    }

    if test_type in ('ttest', 'both'):
        t_stat, p_value = stats.ttest_rel(results_a, results_b)
        output['paired_ttest'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_005': bool(p_value < 0.05),
            'significant_001': bool(p_value < 0.01),
        }

    if test_type in ('wilcoxon', 'both'):
        try:
            w_stat, p_value = stats.wilcoxon(results_a, results_b)
            output['wilcoxon'] = {
                'w_statistic': float(w_stat),
                'p_value': float(p_value),
                'significant_005': bool(p_value < 0.05),
                'significant_001': bool(p_value < 0.01),
            }
        except ValueError as e:
            # Wilcoxon requires non-zero differences
            output['wilcoxon'] = {
                'error': str(e),
                'note': 'All differences are zero or sample too small',
            }

    return output


class KFoldCrossValidator:
    """K-Fold Cross Validation for EEG/BCI research.

    Supports:
    - Subject-independent splits (GroupKFold) - prevents data leakage
    - Leave-One-Subject-Out (LOSO) validation
    - Stratified K-fold for balanced class distribution
    - Per-fold checkpointing with resume support
    - Aggregated metrics with statistical reporting

    IMPORTANT: For EEG data, always use subject-independent splits.
    Random splitting mixes temporal segments from the same recording session,
    causing information leakage via autocorrelation.
    """

    def __init__(
        self,
        n_folds: int = 5,
        split_type: str = 'group',  # 'group', 'loso', 'stratified'
        random_seed: int = 42,
        results_dir: str = None,
    ):
        """Initialize KFoldCrossValidator.

        Args:
            n_folds: Number of folds (ignored for LOSO).
            split_type: Type of split strategy:
                - 'group': GroupKFold (subject-independent, recommended)
                - 'loso': Leave-One-Subject-Out
                - 'stratified': StratifiedKFold (use only for subject-level splits)
            random_seed: Random seed for reproducibility.
            results_dir: Directory to save results.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for KFoldCrossValidator")

        self.n_folds = n_folds
        self.split_type = split_type
        self.random_seed = random_seed
        self.results_dir = results_dir

        if results_dir:
            os.makedirs(results_dir, exist_ok=True)

        self.fold_results: List[Dict[str, float]] = []
        self.fold_predictions: List[Dict[str, np.ndarray]] = []

    def get_splits(
        self,
        labels: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test index splits.

        Args:
            labels: Array of labels for each sample.
            groups: Array of group/subject IDs for each sample.
                    Required for 'group' and 'loso' split types.

        Returns:
            List of (train_indices, test_indices) tuples.
        """
        n_samples = len(labels)
        X_dummy = np.zeros((n_samples, 1))  # Placeholder for sklearn API

        if self.split_type == 'group':
            if groups is None:
                raise ValueError("groups (subject IDs) required for GroupKFold")
            splitter = GroupKFold(n_splits=self.n_folds)
            splits = list(splitter.split(X_dummy, labels, groups=groups))

        elif self.split_type == 'loso':
            if groups is None:
                raise ValueError("groups (subject IDs) required for LOSO")
            splitter = LeaveOneGroupOut()
            splits = list(splitter.split(X_dummy, labels, groups=groups))
            self.n_folds = len(splits)  # Update n_folds to actual number

        elif self.split_type == 'stratified':
            splitter = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_seed,
            )
            splits = list(splitter.split(X_dummy, labels))

        else:
            raise ValueError(f"Unknown split_type: {self.split_type}")

        return splits

    def run(
        self,
        dataset: Any,
        labels: np.ndarray,
        groups: Optional[np.ndarray] = None,
        train_fn: Optional[Callable] = None,
        evaluate_fn: Optional[Callable] = None,
        task_type: str = 'classification',
    ) -> Dict[str, Any]:
        """Run full K-fold cross validation.

        Args:
            dataset: Dataset object (must support indexing).
            labels: Labels array.
            groups: Subject/group IDs for subject-independent splits.
            train_fn: Function(train_dataset, val_dataset, fold, config) -> model.
            evaluate_fn: Function(model, test_dataset) -> (y_pred, y_prob).
            task_type: 'classification' or 'regression'.

        Returns:
            Dictionary with per-fold and aggregated results.
        """
        splits = self.get_splits(labels, groups)

        print(f"\n{'='*60}")
        print(f"  K-Fold Cross Validation")
        print(f"  Split type: {self.split_type}")
        print(f"  Number of folds: {len(splits)}")
        if groups is not None:
            print(f"  Number of subjects: {len(np.unique(groups))}")
        print(f"{'='*60}\n")

        self.fold_results = []
        self.fold_predictions = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"\n--- Fold {fold_idx + 1}/{len(splits)} ---")
            print(f"  Train samples: {len(train_idx)}")
            print(f"  Test samples:  {len(test_idx)}")

            if groups is not None:
                train_subjects = np.unique(groups[train_idx])
                test_subjects = np.unique(groups[test_idx])
                print(f"  Train subjects: {len(train_subjects)}")
                print(f"  Test subjects:  {len(test_subjects)} ({test_subjects})")

                # Verify no overlap
                overlap = set(train_subjects) & set(test_subjects)
                assert len(overlap) == 0, \
                    f"DATA LEAKAGE: subjects {overlap} in both train and test!"

            # Create subsets
            if isinstance(dataset, (TensorDataset, torch.utils.data.Dataset)):
                train_dataset = Subset(dataset, train_idx)
                test_dataset = Subset(dataset, test_idx)
            else:
                # Assume numpy array
                train_dataset = dataset[train_idx]
                test_dataset = dataset[test_idx]

            # Train
            if train_fn is not None:
                model = train_fn(train_dataset, test_dataset, fold_idx, {})
            else:
                model = None

            # Evaluate
            if evaluate_fn is not None and model is not None:
                y_pred, y_prob = evaluate_fn(model, test_dataset)
            else:
                # If no functions provided, just store splits info
                y_pred = None
                y_prob = None

            # Compute metrics
            if y_pred is not None:
                y_true = labels[test_idx]
                fold_metrics = compute_metrics(y_true, y_pred, y_prob, task_type)
                self.fold_results.append(fold_metrics)
                self.fold_predictions.append({
                    'fold': fold_idx,
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'y_prob': y_prob,
                    'test_idx': test_idx,
                })

                print(f"  Results:")
                for name, value in fold_metrics.items():
                    print(f"    {name}: {value:.4f}")

        # Aggregate results
        aggregated = aggregate_fold_results(self.fold_results)

        print(f"\n{'='*60}")
        print(f"  Aggregated Results ({len(splits)}-fold)")
        print(f"{'='*60}")
        for name, stats_dict in aggregated.items():
            print(f"  {name}: {stats_dict['mean']:.4f} +/- {stats_dict['std']:.4f} "
                  f"(95% CI: [{stats_dict['ci_lower']:.4f}, {stats_dict['ci_upper']:.4f}])")

        # Save results
        results = {
            'split_type': self.split_type,
            'n_folds': len(splits),
            'random_seed': self.random_seed,
            'task_type': task_type,
            'fold_results': self.fold_results,
            'aggregated': aggregated,
        }

        if self.results_dir:
            self._save_results(results)

        return results

    def _save_results(self, results: Dict[str, Any]):
        """Save results to CSV and JSON."""
        # Save fold results as CSV
        csv_path = os.path.join(self.results_dir, 'fold_results.csv')
        if self.fold_results:
            fieldnames = ['fold'] + list(self.fold_results[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i, fold_result in enumerate(self.fold_results):
                    row = {'fold': i, **fold_result}
                    writer.writerow(row)
            print(f"\n[Results] Fold results saved: {csv_path}")

        # Save aggregated results as JSON
        json_path = os.path.join(self.results_dir, 'aggregated_results.json')
        # Convert numpy types for JSON serialization
        json_results = {
            'split_type': results['split_type'],
            'n_folds': results['n_folds'],
            'random_seed': results['random_seed'],
            'task_type': results['task_type'],
            'aggregated': results['aggregated'],
        }
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"[Results] Aggregated results saved: {json_path}")

    def generate_latex_table(
        self,
        method_name: str = 'MBG',
        metrics_to_show: Optional[List[str]] = None,
    ) -> str:
        """Generate a LaTeX table of results for paper.

        Args:
            method_name: Name of the method for the table.
            metrics_to_show: List of metrics to include. If None, shows all.

        Returns:
            LaTeX table string.
        """
        aggregated = aggregate_fold_results(self.fold_results)

        if metrics_to_show is None:
            metrics_to_show = list(aggregated.keys())

        # Build LaTeX
        n_cols = len(metrics_to_show) + 1  # +1 for method name
        col_spec = 'l' + 'c' * len(metrics_to_show)

        lines = []
        lines.append(r'\begin{table}[htbp]')
        lines.append(r'\centering')
        lines.append(r'\caption{Cross-validation results (' +
                     f'{self.n_folds}-fold, {self.split_type} split' + r')}')
        lines.append(r'\begin{tabular}{' + col_spec + '}')
        lines.append(r'\toprule')

        # Header
        header_names = [m.replace('_', ' ').title() for m in metrics_to_show]
        lines.append('Method & ' + ' & '.join(header_names) + r' \\')
        lines.append(r'\midrule')

        # Data row
        values = []
        for metric in metrics_to_show:
            if metric in aggregated:
                m = aggregated[metric]['mean']
                s = aggregated[metric]['std']
                values.append(f'{m:.4f} $\\pm$ {s:.4f}')
            else:
                values.append('--')

        lines.append(f'{method_name} & ' + ' & '.join(values) + r' \\')
        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\label{tab:cv_results}')
        lines.append(r'\end{table}')

        return '\n'.join(lines)


if __name__ == '__main__':
    print("Testing K-Fold Cross Validation utilities...")

    # Test compute_metrics
    print("\n1. Testing compute_metrics...")
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0])
    y_prob = np.random.rand(10, 2)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    metrics = compute_metrics(y_true, y_pred, y_prob, task_type='classification')
    print(f"  Metrics: {metrics}")
    assert 'accuracy' in metrics
    assert 'balanced_accuracy' in metrics
    assert 'cohen_kappa' in metrics

    # Test aggregate_fold_results
    print("\n2. Testing aggregate_fold_results...")
    fold_results = [
        {'accuracy': 0.85, 'f1': 0.83},
        {'accuracy': 0.87, 'f1': 0.85},
        {'accuracy': 0.82, 'f1': 0.80},
        {'accuracy': 0.89, 'f1': 0.87},
        {'accuracy': 0.86, 'f1': 0.84},
    ]
    agg = aggregate_fold_results(fold_results)
    print(f"  Accuracy: {agg['accuracy']['mean']:.4f} +/- {agg['accuracy']['std']:.4f}")
    assert abs(agg['accuracy']['mean'] - 0.858) < 0.01

    # Test statistical_test
    print("\n3. Testing statistical_test...")
    results_a = [0.85, 0.87, 0.82, 0.89, 0.86]
    results_b = [0.80, 0.82, 0.78, 0.84, 0.81]
    test_result = statistical_test(results_a, results_b)
    print(f"  Paired t-test p-value: {test_result['paired_ttest']['p_value']:.4f}")
    print(f"  Significant (p<0.05): {test_result['paired_ttest']['significant_005']}")

    # Test KFoldCrossValidator splits
    print("\n4. Testing KFoldCrossValidator splits...")
    n_samples = 100
    n_subjects = 10
    labels = np.random.randint(0, 2, n_samples)
    groups = np.repeat(np.arange(n_subjects), n_samples // n_subjects)

    cv = KFoldCrossValidator(n_folds=5, split_type='group', random_seed=42)
    splits = cv.get_splits(labels, groups)
    print(f"  Number of splits: {len(splits)}")

    # Verify no subject overlap
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_subjects = set(groups[train_idx])
        test_subjects = set(groups[test_idx])
        overlap = train_subjects & test_subjects
        assert len(overlap) == 0, f"Fold {fold_idx}: Subject overlap detected!"
    print("  No subject overlap - PASSED")

    # Test LOSO
    print("\n5. Testing LOSO splits...")
    cv_loso = KFoldCrossValidator(split_type='loso', random_seed=42)
    splits_loso = cv_loso.get_splits(labels, groups)
    print(f"  LOSO splits: {len(splits_loso)} (= number of subjects)")
    assert len(splits_loso) == n_subjects

    # Test LaTeX generation
    print("\n6. Testing LaTeX table generation...")
    cv.fold_results = fold_results
    latex = cv.generate_latex_table(method_name='MBG (Ours)')
    print(latex)

    print("\nAll kfold_cv tests passed!")
