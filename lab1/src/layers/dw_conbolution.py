from src.layers import ConvLayer2D
import cupy as np
from typing import Tuple
from src.errors import InvalidPaddingModeError


class DWConvLayer2d(ConvLayer2D):

    def __init__(
        self, w: np.array,
        b: np.array,
        padding: str = 'valid',
        padding_value: int = 0,
        stride: int = 1
    ):
        """
        :param w -  3D tensor with shape (h_f, w_f, c_f)
        :param b - 2D tensor with shape (c_f, )
        :param padding - flag describing type of activation padding valid/same
        :param stride - stride along width and height of input volume
        ------------------------------------------------------------------------
        h_f - height of filter volume
        w_f - width of filter volume
        c_f - number of filters
        """
        super().__init__(w, b, padding, padding_value, stride)
        
    @classmethod
    def initialize(
        cls: int,
        kernel_shape: Tuple[int, int, int],
        padding: str = 'valid',
        padding_value: int = 0,
        stride: int = 1
    ) -> ConvLayer2D:
        w = np.random.randn(*kernel_shape) * 0.1
        b = np.random.randn(kernel_shape[2]) * 0.1
        return cls(w=w, b=b, padding=padding, padding_value=padding_value, stride=stride)
    
    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        """
        :param a_prev - 4D tensor with shape (n, h_in, w_in, c)
        :output 4D tensor with shape (n, h_out, w_out, c)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        w_out - width of input volume
        h_out - width of input volume
        c - number of channels of the input volume
        """
        self._a_prev = np.array(a_prev, copy=True)
        output_shape = self.calculate_output_dims(input_dims=a_prev.shape)
        _, h_out, w_out, _ = output_shape
        h_f, w_f, _ = self._w.shape
        pad = self.calculate_pad_dims()
        a_prev_pad = self.pad(array=a_prev, pad=pad)
        output = np.zeros(output_shape)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_f
                w_start = j * self._stride
                w_end = w_start + w_f

                output[:, i, j, :] = np.sum(
                    a_prev_pad[:, h_start:h_end, w_start:w_end, :] *
                    self._w[np.newaxis, :, :, :],
                    axis=(1, 2)
                )

        return output + self._b

    def backward_pass(self, da_curr: np.array) -> np.array:
        """
        :param da_curr - 4D tensor with shape (n, h_out, w_out, c)
        :output 4D tensor with shape (n, h_in, w_in, c)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        w_out - width of input volume
        h_out - width of input volume
        c - number of channels of the input volume
        """
        _, h_out, w_out, _ = da_curr.shape
        n, h_in, w_in, _ = self._a_prev.shape
        h_f, w_f, _ = self._w.shape
        pad = self.calculate_pad_dims()
        a_prev_pad = self.pad(array=self._a_prev, pad=pad)
        output = np.zeros_like(a_prev_pad)

        self._db = da_curr.sum(axis=(0, 1, 2)) / n
        self._dw = np.zeros_like(self._w)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self._stride
                h_end = h_start + h_f
                w_start = j * self._stride
                w_end = w_start + w_f
                output[:, h_start:h_end, w_start:w_end, :] += self._w[np.newaxis, :, :, :] * \
                    da_curr[:, i:i+1, j:j+1, :]
                self._dw += np.sum(
                    a_prev_pad[:, h_start:h_end, w_start:w_end, :] *
                    da_curr[:, i:i+1, j:j+1, :],
                    axis=0
                )

        self._dw /= n
        return output[:, pad[0]:pad[0]+h_in, pad[1]:pad[1]+w_in, :]
    
    def calculate_output_dims(
        self, input_dims: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        :param input_dims - 4 element tuple (n, h_in, w_in, c)
        :output 4 element tuple (n, h_out, w_out, c)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        w_out - width of input volume
        h_out - width of input volume
        c - number of channels of the input volume
        """
        n, h_in, w_in, c = input_dims
        h_f, w_f, _ = self._w.shape
        if self._padding == 'same':
            return n, h_in, w_in, c
        elif self._padding == 'valid':
            h_out = (h_in - h_f + 2 * self._padding_value) // self._stride + 1
            w_out = (w_in - w_f + 2 * self._padding_value) // self._stride + 1
            return n, h_out, w_out, c
        else:
            raise InvalidPaddingModeError(
                f"Unsupported padding value: {self._padding}"
            )

    def calculate_pad_dims(self) -> Tuple[int, int]:
        """
        :output - 2 element tuple (h_pad, w_pad)
        ------------------------------------------------------------------------
        h_pad - single side padding on height of the volume
        w_pad - single side padding on width of the volume
        """
        if self._padding == 'same' and self._stride == 1:
            h_f, w_f, _, = self._w.shape
            return (h_f - 1) // 2, (w_f - 1) // 2
        elif self._padding == 'valid':
            return self._padding_value, self._padding_value
        else:
            raise InvalidPaddingModeError(
                f"Unsupported padding value: {self._padding}"
            )
