"""Unit tests for LossFunction classes."""

import warnings

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.autograd import grad

from graphnet.training.loss_functions import (
    LogCoshLoss,
    VonMisesFisher3DLoss,
    VonMisesFisherLoss,
)
from graphnet.utilities.maths import eps_like


# Utility method(s)
def _compute_elementwise_gradient(outputs: Tensor, inputs: Tensor) -> Tensor:
    """Compute gradient of each element in `outptus` wrt. `inputs`.

    It is assumed that each element in `inputs` only affects the corresponding
    element in `outputs`. This should be the result of any vectorised
    calculation (as used in tests).
    """
    # Check(s)
    assert inputs.dim() == 1
    assert outputs.dim() == 1
    assert inputs.size(dim=0) == outputs.size(dim=0)

    # Compute list of elementwise gradients
    nb_elements = inputs.size(dim=0)
    elementwise_gradients = torch.stack(
        [
            grad(outputs=outputs[ix], inputs=inputs, retain_graph=True,)[
                0
            ][ix]
            for ix in range(nb_elements)
        ]
    )
    return elementwise_gradients


# Unit test(s)
def test_log_cosh(dtype: torch.dtype = torch.float32) -> None:
    """Test agreement of the two ways to calculate this loss."""
    # Prepare test data
    x = torch.tensor([-100, -10, -1, 0, 1, 10, 100], dtype=dtype).unsqueeze(
        1
    )  # Shape [N, 1]
    y = 0.0 * x.clone()  # Shape [N,1]
    # Calculate losses using loss function, and manually
    log_cosh_loss = LogCoshLoss()
    losses = log_cosh_loss(x, y, return_elements=True)
    losses_reference = torch.log(torch.cosh(x - y))

    # (1) Loss functions should not return  `inf` losses, even for large
    #     differences between prediction and target. This is not necessarily
    #     true for the directly calculated loss (reference) where cosh(x) may go
    #     to `inf` for x >~ 100.
    assert torch.all(torch.isfinite(losses))

    # (2) For the inputs where the reference loss _is_ valid, the two
    #     calculations should agree exactly.
    reference_is_valid = torch.isfinite(losses_reference)
    assert torch.allclose(
        losses_reference[reference_is_valid], losses[reference_is_valid]
    )


def test_von_mises_fisher_exact_m3(dtype: torch.dtype = torch.float64) -> None:
    """Test implementaion of exact von-Mises Fisher loss with m=3.

    See https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
    for exact, simplified reference.
    """
    # Define test parameters
    m = 3
    k = torch.tensor(
        data=[0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
        requires_grad=True,
        dtype=dtype,
    )

    # Compute values
    res_reference = (
        torch.log(k) - k - torch.log(2 * np.pi * (1 - torch.exp(-2 * k)))
    )
    res_exact = VonMisesFisherLoss.log_cmk_exact(m, k)

    # Compute gradients
    grads_reference = _compute_elementwise_gradient(res_reference, k)
    grads_exact = _compute_elementwise_gradient(res_exact, k)

    # Test that values agree
    assert torch.allclose(res_exact, res_reference)

    # Test that gradients agree
    assert torch.allclose(grads_reference, grads_exact)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_von_mises_fisher_exact_m3_at_zero(dtype: torch.dtype) -> None:
    """Test finite values and gradients around the removable singularity."""
    k = torch.tensor(
        data=[0.0, 1e-10, 5e-7, 1e-3, 1.0],
        requires_grad=True,
        dtype=dtype,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = VonMisesFisherLoss.log_cmk_exact(3, k)
        gradients = grad(result.sum(), k)[0]

    expected = (
        -np.log(4 * np.pi)
        - k.detach()[:3] ** 2 / 6
        + k.detach()[:3] ** 4 / 180
    )
    expected_gradients = -k.detach()[:3] / 3

    assert not caught_warnings
    assert torch.all(torch.isfinite(result))
    assert torch.all(torch.isfinite(gradients))
    assert torch.allclose(result[:3], expected)
    assert torch.allclose(gradients[:3], expected_gradients)


def test_von_mises_fisher_exact_m3_continuous_at_small_kappa() -> None:
    """Test continuity where the small-kappa series switches to exact form."""
    k = torch.tensor(
        data=[1e-6 * (1 - 1e-3), 1e-6 * (1 + 1e-3)],
        dtype=torch.float64,
    )
    result = VonMisesFisherLoss.log_cmk_exact(3, k)

    assert torch.all(torch.isfinite(result))
    assert torch.allclose(result[0], result[1], rtol=0.0, atol=1e-12)


def test_von_mises_fisher_3d_loss_at_zero_kappa() -> None:
    """Test the complete 3D loss at zero predicted concentration."""
    prediction = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)

    loss = VonMisesFisher3DLoss()(prediction, target, return_elements=True)
    gradient = grad(loss.sum(), prediction)[0]

    assert torch.all(torch.isfinite(loss))
    assert torch.all(torch.isfinite(gradient))
    assert torch.allclose(
        loss, torch.tensor([np.log(4 * np.pi)], dtype=torch.float64)
    )


@pytest.mark.parametrize("m", [2, 3])
def test_von_mises_fisher_approximation(
    m: int, dtype: torch.dtype = torch.float64
) -> None:
    """Test approximate calculation of $log C_{m}(k)$.

    See [1812.04616], Section 8.2 for approximation.
    """
    # Check(s)
    assert isinstance(m, int)
    assert m > 1

    # Define test parameters
    k = torch.tensor(
        data=[0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
        requires_grad=True,
        dtype=dtype,
    )

    # Compute values
    res_approx = VonMisesFisherLoss.log_cmk_approx(m, k)
    res_exact = VonMisesFisherLoss.log_cmk_exact(m, k)

    C = (
        res_exact[0] - res_approx[0]
    )  # Normalisation constant from integrating gradient
    res_approx += C - eps_like(C)

    # Compute gradients
    grads_approx = _compute_elementwise_gradient(res_approx, k)
    grads_exact = _compute_elementwise_gradient(res_exact, k)

    # Test inequality in [1812.04616] Sec. 8.2
    assert torch.all(res_exact >= res_approx), (m, res_exact, res_approx)

    # Test value approximation
    assert torch.allclose(res_approx, res_exact, rtol=1e0, atol=1e-01)

    # Test gradient approximation
    assert torch.allclose(grads_approx, grads_exact, rtol=1e0)


@pytest.mark.parametrize("m", [2, 3])
def test_von_mises_fisher_approximation_large_kappa(
    m: int, dtype: torch.dtype = torch.float64
) -> None:
    """Test approximate calculation of $log C_{m}(k)$ for large kappa values.

    See [1812.04616], Section 8.2 for approximation.
    """
    # Check(s)
    assert isinstance(m, int)
    assert m > 1

    # Define test parameters
    k = torch.tensor(
        data=[100.0, 200.0, 300.0, 500.0, 1000.0],
        requires_grad=True,
        dtype=dtype,
    )

    # Compute values
    res_approx = VonMisesFisherLoss.log_cmk_approx(m, k)
    res_exact = VonMisesFisherLoss.log_cmk_exact(m, k)

    C = res_exact[0] - res_approx[0]  # Normalisation constant
    res_approx += C

    # Compute gradients
    grads_approx = _compute_elementwise_gradient(res_approx, k)
    grads_exact = _compute_elementwise_gradient(res_exact, k)

    exact_is_valid = torch.isfinite(res_exact)

    # Test value approximation
    assert torch.allclose(
        res_approx[exact_is_valid], res_exact[exact_is_valid], rtol=1e-2
    )

    # Test gradient approximation
    assert torch.allclose(
        grads_approx[exact_is_valid], grads_exact[exact_is_valid], rtol=1e-2
    )
