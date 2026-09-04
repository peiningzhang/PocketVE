import sys
from pathlib import Path

import torch
from torch_scatter import scatter_softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.uni_transformer import scatter_softmax_jvp_compatible


def test_vectorized_scatter_softmax_jvp_matches_reference_with_empty_segments():
    torch.manual_seed(0)
    index = torch.tensor([0, 2, 0, 2, 2, 4])
    src = torch.randn(6, 3, dtype=torch.double)

    value, _ = torch.func.jvp(
        lambda x: scatter_softmax_jvp_compatible(x, index, dim_size=6),
        (src,),
        (torch.randn_like(src),),
    )
    expected = scatter_softmax(src, index, dim=0, dim_size=6)

    torch.testing.assert_close(value, expected)


def test_vectorized_scatter_softmax_jvp_matches_finite_difference():
    torch.manual_seed(1)
    index = torch.tensor([0, 0, 1, 3, 3, 3])
    src = torch.randn(6, 4, dtype=torch.double)
    tangent = torch.randn_like(src)

    _, actual_jvp = torch.func.jvp(
        lambda x: scatter_softmax_jvp_compatible(x, index, dim_size=5),
        (src,),
        (tangent,),
    )
    eps = 1e-6
    finite_difference = (
        scatter_softmax_jvp_compatible(src + eps * tangent, index, dim_size=5)
        - scatter_softmax_jvp_compatible(src - eps * tangent, index, dim_size=5)
    ) / (2 * eps)

    torch.testing.assert_close(actual_jvp, finite_difference, atol=1e-7, rtol=1e-5)


def test_vectorized_scatter_softmax_jvp_supports_1d_and_empty_inputs():
    index = torch.tensor([0, 1, 1])
    src = torch.tensor([1.0, -1.0, 2.0])
    value, tangent = torch.func.jvp(
        lambda x: scatter_softmax_jvp_compatible(x, index, dim_size=3),
        (src,),
        (torch.ones_like(src),),
    )
    torch.testing.assert_close(value, scatter_softmax(src, index, dim_size=3))
    torch.testing.assert_close(tangent, torch.zeros_like(tangent), atol=1e-6, rtol=0)

    empty = torch.empty(0, 2)
    empty_index = torch.empty(0, dtype=torch.long)
    empty_value, empty_jvp = torch.func.jvp(
        lambda x: scatter_softmax_jvp_compatible(x, empty_index, dim_size=0),
        (empty,),
        (empty,),
    )
    assert empty_value.shape == empty.shape
    assert empty_jvp.shape == empty.shape
