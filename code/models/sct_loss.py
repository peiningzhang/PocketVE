"""Pure loss helpers for single-time consistency training.

The helpers in this module deliberately do not know anything about the model
architecture.  PocketVE stores ligand atoms and bond tokens in a flat PyG
batch, so all reductions are performed per graph before taking the batch mean.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn.functional as F


_VALID_NORM_MODES = ("fixed-c", "scale-c", "no-norm")


def _check_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _check_float_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have a floating-point dtype, got {value.dtype}")
    _check_finite(name, value)


def _check_norm_constant(name: str, value) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        value = value.detach().item()
    if not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _validate_batch_index(batch_index: torch.Tensor, n_items: int, name: str = "batch_index") -> None:
    if not isinstance(batch_index, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if batch_index.ndim != 1 or batch_index.numel() != n_items:
        raise ValueError(
            f"{name} must have shape [{n_items}], got {tuple(batch_index.shape)}"
        )
    if batch_index.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"{name} must have an integer dtype, got {batch_index.dtype}")
    if batch_index.numel() and (batch_index < 0).any():
        raise ValueError(f"{name} must contain non-negative graph indices")


def _unique_batch(batch_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sorted graph ids and an inverse index for flat batch entries."""

    if batch_index.numel() == 0:
        return batch_index.new_empty((0,)), batch_index.new_empty((0,))
    return torch.unique(batch_index, sorted=True, return_inverse=True)


def _graph_mean(values: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    """Mean a flat value tensor inside each graph."""

    if values.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got {tuple(values.shape)}")
    _validate_batch_index(batch_index, values.numel())
    if values.numel() == 0:
        return values.new_empty((0,))

    _, inverse = _unique_batch(batch_index)
    n_graphs = int(inverse.max().item()) + 1
    sums = values.new_zeros(n_graphs)
    counts = values.new_zeros(n_graphs)
    sums.index_add_(0, inverse, values)
    counts.index_add_(0, inverse, torch.ones_like(values))
    return sums / counts.clamp_min(1)


def sct_warmup_factor(step, warmup_steps):
    """Return the scalar tangent warmup factor.

    ``step`` is zero-based: step 0 contributes ``1 / warmup_steps``.  A
    non-positive warmup disables warmup and returns 1.0.
    """

    if isinstance(step, torch.Tensor):
        if step.numel() != 1:
            raise ValueError("step must be a scalar")
        step = step.detach().item()
    if isinstance(warmup_steps, torch.Tensor):
        if warmup_steps.numel() != 1:
            raise ValueError("warmup_steps must be a scalar")
        warmup_steps = warmup_steps.detach().item()

    if not isinstance(step, Real) or not math.isfinite(float(step)):
        raise ValueError("step must be finite")
    if not isinstance(warmup_steps, Real) or not math.isfinite(float(warmup_steps)):
        raise ValueError("warmup_steps must be finite")
    step = float(step)
    warmup_steps = float(warmup_steps)
    if step < 0:
        raise ValueError("step must be >= 0")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if warmup_steps == 0:
        return 1.0
    return min(1.0, (step + 1.0) / warmup_steps)


def sct_time_scale(sigma, sigma_min, time_parameterization):
    """Return ``d sigma / d q`` for the shared sCT time coordinate.

    PocketVE's linear consistency coordinate is ``q = sigma``.  Its log
    coordinate is ``q = log(sigma)`` (not the model's already transformed
    noise embedding), so the chain-rule scale is ``max(sigma, sigma_min)``.
    The returned tensor keeps the input shape/device/dtype and is therefore
    directly broadcastable with flat ligand coordinates.
    """

    _check_float_tensor("sigma", sigma)
    sigma_min = _check_norm_constant("sigma_min", sigma_min)
    if time_parameterization not in ("linear", "log"):
        raise ValueError(
            "time_parameterization must be 'linear' or 'log', got "
            f"{time_parameterization!r}"
        )

    if time_parameterization == "linear":
        scale = torch.ones_like(sigma)
    else:
        scale = sigma.clamp_min(sigma_min)
    _check_finite("sCT time scale", scale)
    return scale


def _check_norm_mode(norm_mode):
    if norm_mode not in _VALID_NORM_MODES:
        raise ValueError(
            "norm_mode must be 'fixed-c', 'scale-c' or 'no-norm', got "
            f"{norm_mode!r}"
        )
    return norm_mode


def _normalize_scale(
    scale,
    *,
    shape,
    device,
    dtype,
    name="norm_constant_scale",
):
    """Convert a scalar/per-token scale to a flat token-shaped tensor."""

    if scale is None:
        return None
    if isinstance(scale, torch.Tensor):
        scale_tensor = scale.to(device=device, dtype=dtype)
    else:
        try:
            scale_tensor = torch.as_tensor(scale, device=device, dtype=dtype)
        except (TypeError, ValueError):
            raise TypeError(f"{name} must be a scalar or tensor") from None
    _check_finite(name, scale_tensor)
    if scale_tensor.ndim == 0:
        scale_tensor = scale_tensor.expand(shape)
    elif tuple(scale_tensor.shape) == tuple(shape):
        pass
    elif scale_tensor.ndim == 1 and tuple(shape) == (scale_tensor.shape[0], 1):
        scale_tensor = scale_tensor.view(-1, 1)
    elif scale_tensor.ndim == 2 and scale_tensor.shape[-1] == 1 and len(shape) == 1:
        if scale_tensor.shape[0] != shape[0]:
            raise ValueError(
                f"{name} must have shape {tuple(shape)}, got {tuple(scale_tensor.shape)}"
            )
        scale_tensor = scale_tensor.reshape(shape)
    else:
        raise ValueError(
            f"{name} must be a scalar or have token shape {tuple(shape)}, "
            f"got {tuple(scale_tensor.shape)}"
        )
    if torch.any(scale_tensor <= 0):
        raise ValueError(f"{name} must be > 0")
    return scale_tensor


def _graph_scale_from_token_scale(
    token_scale: torch.Tensor,
    inverse: torch.Tensor,
    n_graphs: int,
    *,
    name="norm_constant_scale",
):
    """Collapse a per-token scale to graph scale after checking consistency."""

    graph_scale = token_scale.new_empty(n_graphs)
    for graph_idx in range(n_graphs):
        values = token_scale[inverse == graph_idx]
        if values.numel() == 0:
            raise ValueError(f"{name} has a graph with no tokens")
        graph_value = values[0]
        if not torch.allclose(values, graph_value.expand_as(values), rtol=1e-5, atol=1e-6):
            raise ValueError(
                f"{name} must be constant within each graph; graph {graph_idx} has "
                "multiple values"
            )
        graph_scale[graph_idx] = graph_value
    return graph_scale


def graph_coordinate_rms_normalize(
    tangent: torch.Tensor,
    batch_ligand: torch.Tensor,
    norm_constant,
    norm_constant_scale=None,
    norm_mode="fixed-c",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a flat coordinate tangent independently for each graph.

    The RMS is over all three coordinates of all atoms in a graph, i.e.
    ``sqrt(sum(tangent**2) / (3 * N_graph))``.  ``fixed-c`` uses the
    time-independent floor ``c``; ``scale-c`` uses ``alpha * c`` per graph;
    and ``no-norm`` returns the raw tangent.  The returned second value is the
    pre-normalization graph RMS and is useful for diagnostics.
    """

    _check_float_tensor("tangent", tangent)
    if tangent.ndim != 2 or tangent.shape[-1] != 3:
        raise ValueError(
            "tangent must have shape [num_ligand_atoms, 3], "
            f"got {tuple(tangent.shape)}"
        )
    _validate_batch_index(batch_ligand, tangent.shape[0], "batch_ligand")
    norm_constant = _check_norm_constant("norm_constant", norm_constant)
    norm_mode = _check_norm_mode(norm_mode)
    token_scale = None
    if norm_mode == "scale-c":
        token_scale = _normalize_scale(
            norm_constant_scale,
            shape=(tangent.shape[0], 1),
            device=tangent.device,
            dtype=tangent.dtype,
        )

    if tangent.shape[0] == 0:
        return tangent.clone(), tangent.new_empty((0,))

    _, inverse = _unique_batch(batch_ligand)
    n_graphs = int(inverse.max().item()) + 1
    sum_sq = tangent.new_zeros(n_graphs)
    counts = tangent.new_zeros(n_graphs)
    sum_sq.index_add_(0, inverse, tangent.square().sum(dim=-1))
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=tangent.dtype))
    rms = torch.sqrt(sum_sq / (counts * tangent.shape[-1]).clamp_min(1.0))
    if norm_mode == "no-norm":
        return tangent, rms
    if token_scale is None:
        floor = tangent.new_tensor(norm_constant)
    else:
        graph_scale = _graph_scale_from_token_scale(
            token_scale.reshape(-1), inverse, n_graphs
        )
        floor = norm_constant * graph_scale[inverse]
    normalized = tangent / (rms[inverse].unsqueeze(-1) + floor.unsqueeze(-1))
    _check_finite("normalized tangent", normalized)
    return normalized, rms


def categorical_tangent_normalize(
    tangent_logits: torch.Tensor,
    norm_constant,
    norm_constant_scale=None,
    norm_mode="fixed-c",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center and RMS-normalize a categorical tangent independently per token.

    The class-mean removal makes the result invariant to the categorical
    logits' class-constant gauge direction.  ``fixed-c`` uses the constant
    floor ``c``, ``scale-c`` uses ``alpha * c`` per token, and ``no-norm``
    keeps only the gauge projection.  The second and third return values are
    the raw and normalized per-token RMS values.
    """

    _check_float_tensor("tangent_logits", tangent_logits)
    if tangent_logits.ndim < 2 or tangent_logits.shape[-1] < 2:
        raise ValueError(
            "tangent_logits must have shape [..., num_classes] with at least "
            f"two classes, got {tuple(tangent_logits.shape)}"
        )
    norm_constant = _check_norm_constant("norm_constant", norm_constant)
    norm_mode = _check_norm_mode(norm_mode)
    token_scale = None
    if norm_mode == "scale-c":
        token_scale = _normalize_scale(
            norm_constant_scale,
            shape=tangent_logits.shape[:-1],
            device=tangent_logits.device,
            dtype=tangent_logits.dtype,
        )

    centered = tangent_logits - tangent_logits.mean(dim=-1, keepdim=True)
    raw_rms = centered.square().mean(dim=-1).sqrt()
    if norm_mode == "no-norm":
        # Class-constant logit shifts do not affect softmax probabilities;
        # retain the gauge projection while skipping RMS/floor scaling.
        normalized = centered
        normalized_rms = normalized.square().mean(dim=-1).sqrt()
        _check_finite("normalized categorical tangent", normalized)
        return normalized, raw_rms, normalized_rms
    if token_scale is None:
        floor = tangent_logits.new_tensor(norm_constant)
    else:
        floor = norm_constant * token_scale
    normalized = centered / (raw_rms + floor).unsqueeze(-1)
    normalized_rms = normalized.square().mean(dim=-1).sqrt()
    _check_finite("normalized categorical tangent", normalized)
    return normalized, raw_rms, normalized_rms


def coordinate_sct_residual(
    student: torch.Tensor,
    teacher: torch.Tensor,
    normalized_tangent: torch.Tensor,
    warmup,
    probe_step=1.0,
) -> torch.Tensor:
    """Construct the detached-teacher surrogate coordinate residual."""

    _check_float_tensor("student", student)
    _check_float_tensor("teacher", teacher)
    _check_float_tensor("normalized_tangent", normalized_tangent)
    if student.shape != teacher.shape or student.shape != normalized_tangent.shape:
        raise ValueError(
            "student, teacher and normalized_tangent must have identical shapes: "
            f"student={tuple(student.shape)}, teacher={tuple(teacher.shape)}, "
            f"tangent={tuple(normalized_tangent.shape)}"
        )
    if isinstance(warmup, torch.Tensor):
        if warmup.numel() != 1:
            raise ValueError("warmup must be a scalar")
        warmup_tensor = warmup.detach().to(device=student.device, dtype=student.dtype)
    else:
        if not isinstance(warmup, Real) or not math.isfinite(float(warmup)):
            raise ValueError("warmup must be finite")
        warmup_tensor = student.new_tensor(float(warmup))
    _check_finite("warmup", warmup_tensor)
    if warmup_tensor.item() < 0:
        raise ValueError("warmup must be >= 0")
    if isinstance(probe_step, torch.Tensor):
        if probe_step.numel() != 1:
            raise ValueError("probe_step must be a scalar")
        probe_step = probe_step.detach().item()
    if not isinstance(probe_step, Real) or not math.isfinite(float(probe_step)):
        raise ValueError("probe_step must be finite")
    probe_step = float(probe_step)
    if probe_step <= 0:
        raise ValueError("probe_step must be > 0")

    residual = student - teacher.detach() + (
        probe_step * normalized_tangent * warmup_tensor
    ).detach()
    _check_finite("coordinate sCT residual", residual)
    return residual


def categorical_logit_sct_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    tangent_logits: torch.Tensor,
    batch_index: torch.Tensor,
    probe_step,
    norm_constant,
    warmup,
    norm_constant_scale=None,
    norm_mode="fixed-c",
) -> dict[str, torch.Tensor]:
    """Compute graph-mean categorical logit-sCT KL and diagnostics.

    The virtual teacher is
    ``softmax(detached_teacher - probe_step * detached_normalized_tangent)``.
    Only ``student_logits`` remains attached to the returned loss.  The
    reduction is token-mean within each graph followed by graph-mean, so a
    large ``N_g**2`` bond matrix cannot dominate a batch merely by having more
    tokens.
    """

    _check_float_tensor("student_logits", student_logits)
    _check_float_tensor("teacher_logits", teacher_logits)
    _check_float_tensor("tangent_logits", tangent_logits)
    if student_logits.ndim != 2 or student_logits.shape[-1] < 2:
        raise ValueError(
            "categorical logits must have shape [num_tokens, num_classes] with "
            f"at least two classes, got {tuple(student_logits.shape)}"
        )
    if student_logits.shape != teacher_logits.shape or student_logits.shape != tangent_logits.shape:
        raise ValueError(
            "student_logits, teacher_logits and tangent_logits must have identical shapes: "
            f"student={tuple(student_logits.shape)}, teacher={tuple(teacher_logits.shape)}, "
            f"tangent={tuple(tangent_logits.shape)}"
        )
    _validate_batch_index(batch_index, student_logits.shape[0])
    norm_constant = _check_norm_constant("norm_constant", norm_constant)
    norm_mode = _check_norm_mode(norm_mode)
    if isinstance(probe_step, torch.Tensor):
        if probe_step.numel() != 1:
            raise ValueError("probe_step must be a scalar")
        probe_step = probe_step.detach().item()
    if not isinstance(probe_step, Real) or not math.isfinite(float(probe_step)):
        raise ValueError("probe_step must be finite")
    probe_step = float(probe_step)
    if probe_step <= 0:
        raise ValueError("probe_step must be > 0")
    if isinstance(warmup, torch.Tensor):
        if warmup.numel() != 1:
            raise ValueError("warmup must be a scalar")
        warmup_tensor = warmup.detach().to(
            device=student_logits.device, dtype=student_logits.dtype
        )
    else:
        if not isinstance(warmup, Real) or not math.isfinite(float(warmup)):
            raise ValueError("warmup must be finite")
        warmup_tensor = student_logits.new_tensor(float(warmup))
    _check_finite("warmup", warmup_tensor)
    if warmup_tensor.item() < 0:
        raise ValueError("warmup must be >= 0")

    if student_logits.shape[0] == 0:
        zero = student_logits.sum() * 0.0
        empty = student_logits.new_empty((0,))
        return {
            "loss": zero,
            "loss_per_graph": empty,
            "loss_per_sample": empty,
            "raw_tangent_rms": empty,
            "normalized_tangent_rms": empty,
            "virtual_teacher_kl": empty,
            "teacher_entropy": empty,
            "target_entropy": empty,
            "argmax_flip": empty,
            "target_probs": student_logits.new_empty(student_logits.shape),
        }

    normalized_tangent, raw_rms, normalized_rms = categorical_tangent_normalize(
        tangent_logits.detach(),
        norm_constant,
        norm_constant_scale=norm_constant_scale,
        norm_mode=norm_mode,
    )
    normalized_tangent = normalized_tangent * warmup_tensor
    teacher_logits = teacher_logits.detach()
    target_log_probs = F.log_softmax(
        teacher_logits - probe_step * normalized_tangent.detach(), dim=-1
    )
    target_probs = target_log_probs.exp().detach()
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    token_kl = (
        target_probs * (target_log_probs.detach() - student_log_probs)
    ).sum(dim=-1) / probe_step
    virtual_teacher_kl = (
        target_probs
        * (target_log_probs.detach() - F.log_softmax(teacher_logits, dim=-1))
    ).sum(dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    teacher_entropy = -(
        teacher_probs * F.log_softmax(teacher_logits, dim=-1)
    ).sum(dim=-1)
    target_entropy = -(target_probs * target_log_probs.detach()).sum(dim=-1)
    argmax_flip = (
        target_probs.argmax(dim=-1) != teacher_logits.argmax(dim=-1)
    ).to(dtype=token_kl.dtype)

    loss_per_graph = _graph_mean(token_kl, batch_index)
    return {
        "loss": loss_per_graph.mean(),
        "loss_per_graph": loss_per_graph,
        "loss_per_sample": loss_per_graph,
        "raw_tangent_rms": _graph_mean(raw_rms, batch_index),
        "normalized_tangent_rms": _graph_mean(normalized_rms * warmup_tensor, batch_index),
        "virtual_teacher_kl": _graph_mean(virtual_teacher_kl, batch_index),
        "teacher_entropy": _graph_mean(teacher_entropy, batch_index),
        "target_entropy": _graph_mean(target_entropy, batch_index),
        "argmax_flip": _graph_mean(argmax_flip, batch_index),
        "target_probs": target_probs,
    }
