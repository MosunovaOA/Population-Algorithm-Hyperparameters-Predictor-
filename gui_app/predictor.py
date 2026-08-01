"""
Загрузка sklearn-модели и предсказание параметров
"""
import os
import numpy as np
from itertools import product
import pandas as pd

MAX_ITERATIONS = 300.0
GRID_VALUES = np.round(np.arange(0.1, 2.01, 0.1), 2)


class CascadePredictor:
    """Обёртка над каскадной sklearn-моделью из .pkl файлов."""

    def __init__(self):
        self.is_loaded = False

    def load(self, directory='saved_model_v10_simple'):
        import joblib
        for name in ['scaler_X', 'cls_alive', 'cls_good',
                      'reg_sr', 'reg_iter', 'meta_regressor']:
            path = os.path.join(directory, f'{name}.pkl')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Не найден: {path}")
            setattr(self, name, joblib.load(path))
        self.is_loaded = True

    def _predict_full(self, X_scaled):
        n = len(X_scaled)
        pa = self.cls_alive.predict_proba(X_scaled)[:, 1]
        pg = self.cls_good.predict_proba(X_scaled)[:, 1]
        sr_pred = np.clip(self.reg_sr.predict(X_scaled), 0, 1)
        it = np.clip(self.reg_iter.predict(X_scaled), 0, 1)

        meta_features = np.column_stack([
            pa, pg, pa * pg, sr_pred, it,
        ])

        predictions = self.meta_regressor.predict(meta_features)
        sr = np.clip(predictions[:, 0], 0, 1)
        iter_pred = np.clip(predictions[:, 1], 0, 1)

        return {
            'sr_median': sr,
            'iter_pred': iter_pred,
        }

    def _make_features(self, grid, dim, features_6):
        n = len(grid)
        return np.column_stack([
            np.full(n, features_6[0]),
            np.full(n, features_6[1]),
            np.full(n, features_6[2]),
            np.full(n, features_6[3]),
            np.full(n, features_6[4]),
            np.full(n, features_6[5]),
            np.full(n, int(dim == 3)),
            np.full(n, int(dim == 4)),
            np.full(n, int(dim == 5)),
            grid[:, 0], grid[:, 1], grid[:, 2],
        ])

    def find_optimal(self, features_6, dim, strategy='median', top_n=5):
        """Поиск оптимальных параметров на сетке 20×20×20."""
        w_vals = GRID_VALUES
        cc_vals = GRID_VALUES
        cs_vals = GRID_VALUES

        grid = np.array(list(product(w_vals, cc_vals, cs_vals)))

        X_raw = self._make_features(grid, dim, features_6)
        X_scaled = self.scaler_X.transform(X_raw)
        p = self._predict_full(X_scaled)

        df = pd.DataFrame({
            'w': grid[:, 0], 'c_cog': grid[:, 1], 'c_soc': grid[:, 2],
            'sr_median': p['sr_median'],
            'iter_pred': p['iter_pred'],
        })

        if strategy == 'robust':
            col = 'sr_median'
        elif strategy == 'stable':
            df['score'] = df['sr_median'] - 0.3 * (df['sr_median'] - df['sr_median'])
            col = 'score'
        else:
            col = 'sr_median'

        return df.sort_values([col, 'iter_pred'], ascending=[False, True]).head(top_n)


def load_predictor(model_dir='saved_model_v10_simple'):
    predictor = CascadePredictor()
    predictor.load(model_dir)
    return predictor


def predict_params(predictor, features_6, dim):
    """Предсказание оптимальных параметров."""
    result = predictor.find_optimal(features_6, dim, top_n=1).iloc[0]
    params = (result['w'], result['c_cog'], result['c_soc'])
    return params, result['sr_median'], result['iter_pred']