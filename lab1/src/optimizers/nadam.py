from typing import Tuple, List

import cupy as np

from src.base import Optimizer, Layer


class NAdam(Optimizer):
    def __init__(
        self, lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        """
        :param lr - learning rate
        :param beta1 -
        :param beta2 -
        :param eps - small value to avoid zero denominator
        """
        self._cache_v = {}
        self._cache_s = {}
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._t = 1

    def update(self, layers: List[Layer]) -> None:
        if len(self._cache_s) == 0 or len(self._cache_v) == 0:
            self._init_cache(layers)
            self._t = 1

        beta1 = self._beta1
        beta2 = self._beta2
        eps = self._eps
        
        for idx, layer in enumerate(layers):
            weights, gradients = layer.weights, layer.gradients
            if weights is None or gradients is None:
                continue
            
            (w, b), (dw, db) = weights, gradients
            dw_key, db_key = NAdam._get_cache_keys(idx)
            
            v = self._cache_v
            s = self._cache_s
            v_hat = {}
            s_hat = {}
            t = self._t

            v[dw_key] = beta1 * v[dw_key] + (1 - beta1) * dw
            v[db_key] = beta1 * v[db_key] + (1 - beta1) * db
            s[dw_key] = beta2 * s[dw_key] + (1 - beta2) * np.square(dw)
            s[db_key] = beta2 * s[db_key] + (1 - beta2) * np.square(db)
            
            v_hat[dw_key] = v[dw_key] / (1 - np.power(beta1, t)) + \
                (1 - beta1) * dw / (1 - np.power(beta1, t))
            v_hat[db_key] = v[db_key] / (1 - np.power(beta1, t)) + \
                (1 - beta1) * db / (1 - np.power(beta1, t))
            s_hat[dw_key] = s[dw_key] / (1 - np.power(beta2, t))
            s_hat[db_key] = s[db_key] / (1 - np.power(beta2, t))
            
            dw = v_hat[dw_key] / (np.sqrt(s_hat[dw_key]) + eps)
            db = v_hat[db_key] / (np.sqrt(s_hat[db_key]) + eps)

            layer.set_wights(
                w=w - self._lr * dw,
                b=b - self._lr * db
            )
            
            self._cache_v = v
            self._cache_s = s
            self._t += 1

    def _init_cache(self, layers: List[Layer]) -> None:
        for idx, layer in enumerate(layers):
            gradients = layer.gradients
            if gradients is None:
                continue

            dw, db = gradients
            dw_key, db_key = NAdam._get_cache_keys(idx)

            self._cache_v[dw_key] = np.zeros_like(dw)
            self._cache_v[db_key] = np.zeros_like(db)
            self._cache_s[dw_key] = np.zeros_like(dw)
            self._cache_s[db_key] = np.zeros_like(db)

    @staticmethod
    def _get_cache_keys(idx: int) -> Tuple[str, str]:
        """
        :param idx - index of layer
        """
        return f"dw{idx}", f"db{idx}"
