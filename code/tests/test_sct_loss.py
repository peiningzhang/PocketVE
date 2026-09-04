import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.sct_loss import (
    categorical_logit_sct_loss,
    categorical_tangent_normalize,
    coordinate_sct_residual,
    graph_coordinate_rms_normalize,
    sct_warmup_factor,
    sct_time_scale,
)


def test_graph_coordinate_rms_normalize_is_flat_graph_wise():
    tangent = torch.tensor(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    )
    batch = torch.tensor([0, 0, 1])
    normalized, rms = graph_coordinate_rms_normalize(tangent, batch, 0.1)

    torch.testing.assert_close(rms, torch.tensor([torch.sqrt(torch.tensor(25.0 / 6.0)), 1.0]))
    torch.testing.assert_close(
        normalized[0], tangent[0] / (torch.sqrt(torch.tensor(25.0 / 6.0)) + 0.1)
    )
    torch.testing.assert_close(normalized[2], tangent[2] / 1.1)


def test_graph_coordinate_rms_does_not_depend_on_other_graph_sizes():
    tangent = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    batch = torch.tensor([0, 1])
    _, rms = graph_coordinate_rms_normalize(tangent, batch, 0.1)
    torch.testing.assert_close(rms, torch.tensor([2.0 / 3.0**0.5, 3.0 / 3.0**0.5]))


def test_coordinate_residual_detaches_teacher_and_tangent():
    student = torch.nn.Parameter(torch.zeros(2, 3))
    teacher = torch.ones_like(student, requires_grad=True)
    tangent = torch.full_like(student, 2.0, requires_grad=True)
    residual = coordinate_sct_residual(student, teacher, tangent, 0.5, 0.2)
    residual.square().mean().backward()

    assert student.grad is not None
    assert teacher.grad is None
    assert tangent.grad is None


def test_coordinate_residual_uses_explicit_probe_step():
    student = torch.zeros(1, 3)
    teacher = torch.ones_like(student)
    tangent = torch.full_like(student, 2.0)
    residual = coordinate_sct_residual(student, teacher, tangent, 0.5, 0.25)
    torch.testing.assert_close(residual, torch.full_like(student, -0.75))


def test_sct_time_scale_linear_log_and_sigma_floor():
    sigma = torch.tensor([2.0, 0.5, 0.0001])
    torch.testing.assert_close(
        sct_time_scale(sigma, 0.001, "linear"), torch.ones_like(sigma)
    )
    torch.testing.assert_close(
        sct_time_scale(sigma, 0.001, "log"), torch.tensor([2.0, 0.5, 0.001])
    )


def test_sct_warmup_starts_at_one_over_warmup_and_reaches_one():
    assert sct_warmup_factor(0, 4) == 0.25
    assert sct_warmup_factor(3, 4) == 1.0
    assert sct_warmup_factor(100, 0) == 1.0


def test_joint_jvp_toy_matches_analytic_derivative():
    coords = torch.tensor([[1.0, 2.0, 3.0]])
    sigma = torch.tensor([0.5])
    direction = torch.tensor([[2.0, 3.0, 4.0]])

    def toy(path_coords, path_sigma):
        return path_coords.square() + path_sigma[:, None]

    value, tangent = torch.func.jvp(
        toy,
        (coords, sigma),
        (direction, torch.ones_like(sigma)),
    )
    torch.testing.assert_close(value, coords.square() + sigma[:, None])
    torch.testing.assert_close(tangent, 2.0 * coords * direction + 1.0)


def test_log_sigma_joint_jvp_scales_all_output_leaves():
    coords = torch.tensor([[1.0, 2.0, 3.0]])
    sigma = torch.tensor([0.5])
    epsilon = torch.tensor([[2.0, 3.0, 4.0]])

    def toy(path_coords, path_sigma):
        return (
            path_coords.square() + path_sigma[:, None],
            path_coords.sum(dim=-1, keepdim=True) * path_sigma[:, None],
            torch.cat([path_sigma[:, None], path_coords[:, :1]], dim=-1),
        )

    value, tangent = torch.func.jvp(
        toy,
        (coords, sigma),
        (sigma[:, None] * epsilon, sigma),
    )
    expected_tangent = (
        2.0 * coords * epsilon + 1.0,
        (epsilon.sum(dim=-1, keepdim=True) * sigma[:, None])
        + coords.sum(dim=-1, keepdim=True),
        torch.cat([torch.ones_like(sigma[:, None]), epsilon[:, :1]], dim=-1),
    )
    for actual, expected in zip(tangent, expected_tangent):
        torch.testing.assert_close(actual, sigma[:, None] * expected)
    torch.testing.assert_close(value[0], coords.square() + sigma[:, None])


def test_log_coordinate_normalization_scales_floor_with_sigma():
    tau = torch.tensor([[1.0, 2.0, 0.0], [0.5, -1.0, 2.0]])
    batch = torch.tensor([0, 0])
    sigma = torch.tensor([[3.0], [3.0]])
    linear, _ = graph_coordinate_rms_normalize(tau, batch, 0.2)
    log, _ = graph_coordinate_rms_normalize(
        sigma * tau,
        batch,
        0.2,
        norm_constant_scale=sigma,
        norm_mode="scale-c",
    )
    torch.testing.assert_close(log, linear)


def test_fixed_c_ignores_scale_and_no_norm_returns_raw_coordinate_tangent():
    tangent = torch.tensor([[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    batch = torch.tensor([0, 0])
    fixed, rms = graph_coordinate_rms_normalize(
        tangent,
        batch,
        0.5,
        norm_constant_scale=torch.tensor([[3.0], [3.0]]),
        norm_mode="fixed-c",
    )
    expected_rms = torch.sqrt(torch.tensor((4.0 + 16.0) / 6.0))
    torch.testing.assert_close(rms, expected_rms.expand(1))
    torch.testing.assert_close(fixed, tangent / (expected_rms + 0.5))

    raw, raw_rms = graph_coordinate_rms_normalize(
        tangent, batch, 0.5, norm_mode="no-norm"
    )
    torch.testing.assert_close(raw, tangent)
    torch.testing.assert_close(raw_rms, expected_rms.expand(1))


def test_graph_normalization_rejects_inconsistent_log_scale():
    tangent = torch.ones(2, 3)
    batch = torch.tensor([0, 0])
    with pytest.raises(ValueError):
        graph_coordinate_rms_normalize(
            tangent,
            batch,
            0.1,
            norm_constant_scale=torch.tensor([[1.0], [2.0]]),
            norm_mode="scale-c",
        )


def test_categorical_sct_has_student_gradient_and_simplex_target():
    student = torch.nn.Parameter(torch.zeros(5, 4))
    teacher = torch.randn(5, 4, requires_grad=True)
    tangent = torch.randn(5, 4, requires_grad=True)
    batch = torch.tensor([0, 0, 1, 1, 1])
    result = categorical_logit_sct_loss(
        student, teacher, tangent, batch, 0.1, 0.1, 1.0
    )
    result['loss'].backward()

    assert student.grad is not None
    assert teacher.grad is None
    assert tangent.grad is None
    assert torch.all(result['target_probs'] >= 0)
    torch.testing.assert_close(
        result['target_probs'].sum(dim=-1), torch.ones(5)
    )


def test_categorical_tangent_is_invariant_to_class_constant_shift():
    tangent = torch.randn(5, 4)
    normalized, _, _ = categorical_tangent_normalize(tangent, 0.1)
    shifted, _, _ = categorical_tangent_normalize(tangent + 7.0, 0.1)
    torch.testing.assert_close(normalized, shifted)


def test_log_categorical_normalization_scales_additive_floor():
    tau = torch.tensor([[1.0, -1.0, 0.5], [2.0, 0.0, -1.0]])
    sigma = torch.tensor([[2.0], [0.5]])
    linear, _, _ = categorical_tangent_normalize(tau, 0.2)
    log, _, _ = categorical_tangent_normalize(
        sigma * tau,
        0.2,
        norm_constant_scale=sigma,
        norm_mode="scale-c",
    )
    torch.testing.assert_close(log, linear)


def test_categorical_modes_keep_gauge_projection_contract():
    tangent = torch.tensor([[2.0, 0.0, 1.0]])
    centered = tangent - tangent.mean(dim=-1, keepdim=True)
    raw, _, _ = categorical_tangent_normalize(
        tangent, 0.2, norm_mode="no-norm"
    )
    torch.testing.assert_close(raw, centered)

    fixed, _, _ = categorical_tangent_normalize(
        tangent,
        0.2,
        norm_constant_scale=torch.tensor([[10.0]]),
        norm_mode="fixed-c",
    )
    expected = centered / (centered.square().mean(dim=-1, keepdim=True).sqrt() + 0.2)
    torch.testing.assert_close(fixed, expected)


def test_invalid_tangent_normalization_mode_is_rejected():
    with pytest.raises(ValueError):
        categorical_tangent_normalize(torch.ones(1, 3), 0.1, norm_mode="bad")
    with pytest.raises(ValueError):
        categorical_logit_sct_loss(
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            torch.zeros(1, dtype=torch.long),
            0.1,
            0.1,
            1.0,
            norm_mode="bad",
        )


def test_zero_categorical_tangent_matches_teacher_distribution():
    student = torch.randn(5, 3)
    teacher = torch.randn(5, 3)
    batch = torch.tensor([0, 0, 1, 1, 1])
    result = categorical_logit_sct_loss(
        student, teacher, torch.zeros_like(teacher), batch, 0.1, 0.1, 1.0
    )
    torch.testing.assert_close(result['target_probs'], torch.softmax(teacher, dim=-1))
    torch.testing.assert_close(
        result['virtual_teacher_kl'], torch.zeros(2), atol=1e-6, rtol=1e-6
    )


def test_categorical_reduction_is_graph_mean_not_token_mean():
    student = torch.tensor([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    teacher = torch.zeros_like(student)
    tangent = torch.zeros_like(student)
    batch = torch.tensor([0, 1, 1])
    result = categorical_logit_sct_loss(
        student, teacher, tangent, batch, 0.1, 0.1, 1.0
    )
    torch.testing.assert_close(result['loss'], result['loss_per_graph'].mean())
    assert result['loss_per_graph'].shape == (2,)


def test_categorical_bond_reduction_keeps_large_graph_from_dominating():
    student = torch.tensor(
        [[2.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )
    teacher = torch.zeros_like(student)
    tangent = torch.zeros_like(student)
    batch_bond = torch.tensor([0, 1, 1, 1, 1])
    result = categorical_logit_sct_loss(
        student, teacher, tangent, batch_bond, 0.1, 0.1, 1.0
    )
    torch.testing.assert_close(result['loss'], result['loss_per_graph'].mean())
    assert result['loss_per_graph'].shape == (2,)
