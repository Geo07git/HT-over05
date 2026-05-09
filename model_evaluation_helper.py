import os
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def reliability_table(y, p, bins=10, method='quantile'):
    df = pd.DataFrame({'y': np.asarray(y), 'p': np.asarray(p)})
    df = df.dropna().copy()
    if method == 'quantile':
        df['bin'] = pd.qcut(df['p'], q=bins, duplicates='drop')
    else:
        edges = np.linspace(0, 1, bins + 1)
        df['bin'] = pd.cut(df['p'], bins=edges, include_lowest=True)
    g = df.groupby('bin', observed=False)
    out = pd.DataFrame({
        'bin': g.size().index.astype(str),
        'n': g.size().values,
        'p_mean': g['p'].mean().values,
        'y_mean': g['y'].mean().values,
    })
    out['gap_pp'] = (out['y_mean'] - out['p_mean']) * 100
    out['calibration_flag'] = np.where(out['gap_pp'] > 1.5, 'underconfident', np.where(out['gap_pp'] < -1.5, 'overconfident', 'well_calibrated'))
    return out


def expected_calibration_error(y, p, bins=10):
    rel = reliability_table(y, p, bins=bins, method='uniform')
    total = rel['n'].sum()
    ece = ((rel['n'] / total) * (rel['y_mean'] - rel['p_mean']).abs()).sum()
    return float(ece)


def summarize_probabilities(y, p):
    return {
        'auc': float(roc_auc_score(y, p)),
        'brier': float(brier_score_loss(y, p)),
        'logloss': float(log_loss(y, p)),
        'ece_10bins': float(expected_calibration_error(y, p, bins=10)),
        'mean_pred': float(np.mean(p)),
        'base_rate': float(np.mean(y)),
    }


def compare_with_bookmaker(df, over_col='odds_over', under_col='odds_under', p_col='p_model'):
    x = df[[over_col, under_col, p_col]].dropna().copy()
    x['p_over_raw'] = 1.0 / x[over_col]
    x['p_under_raw'] = 1.0 / x[under_col]
    z = x['p_over_raw'] + x['p_under_raw']
    x['p_book_fair'] = x['p_over_raw'] / z
    x['edge_pp'] = (x[p_col] - x['p_book_fair']) * 100
    return x


def save_evaluation_bundle(test_df, out_dir='evaluation_output'):
    os.makedirs(out_dir, exist_ok=True)
    metrics = summarize_probabilities(test_df['y'], test_df['p_model'])
    rel_q = reliability_table(test_df['y'], test_df['p_model'], bins=12, method='quantile')
    rel_u = reliability_table(test_df['y'], test_df['p_model'], bins=10, method='uniform')

    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, 'metrics_summary.csv'), index=False)
    rel_q.to_csv(os.path.join(out_dir, 'reliability_quantile.csv'), index=False)
    rel_u.to_csv(os.path.join(out_dir, 'reliability_uniform.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)

    if {'odds_over', 'odds_under'}.issubset(test_df.columns):
        book = compare_with_bookmaker(test_df)
        book.to_csv(os.path.join(out_dir, 'bookmaker_comparison.csv'), index=False)

    return metrics, rel_q, rel_u


if __name__ == '__main__':
    print('This file is a helper module. Import it from your training/evaluation workflow.')
