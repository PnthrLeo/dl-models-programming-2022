import cupy as np

from src.base import Layer


class SoftmaxLayer(Layer):
    def __init__(self):
        self._z = None

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        """
        :param a_prev - 2D tensor with shape (n, k)
        :output 2D tensor with shape (n, k)
        ------------------------------------------------------------------------
        n - number of examples in batch
        k - number of classes
        """
        e = np.exp(a_prev - a_prev.max(axis=1, keepdims=True))
        self._z = e / np.sum(e, axis=1, keepdims=True)
        return self._z

    def backward_pass(self, da_curr: np.array) -> np.array:
        # z = self._z
        # num_classes = z.shape[1]
        # mat = np.repeat(z[..., np.newaxis], repeats=num_classes, axis=2)
        # mat_t = np.transpose(mat, axes=(0, 2, 1))
        # i_mat = np.eye(num_classes)
        # da_curr = np.matmul(mat * (i_mat - mat_t), da_curr[..., np.newaxis])
        # da_curr = da_curr[:, :, 0]
        return da_curr
