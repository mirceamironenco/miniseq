import pytest
import torch

from miniseq.utils import make_dtype


def test_make_dtype():
    assert make_dtype("float32") == torch.float32
    assert make_dtype("bfloat16") == torch.bfloat16
    assert make_dtype("float64") == torch.float64

    with pytest.raises(ValueError):
        make_dtype("invalid_dtype")
