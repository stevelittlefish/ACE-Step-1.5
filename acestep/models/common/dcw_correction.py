"""Differential Correction in Wavelet domain (DCW) for flow-matching sampling.

Implements the sampler-side correction from:

    Meng Yu, Lei Sun, Jianhao Zeng, Xiangxiang Chu, Kun Zhan.
    "Elucidating the SNR-t Bias of Diffusion Probabilistic Models",
    CVPR 2026.  arXiv:2604.16044.  https://github.com/AMAP-ML/DCW

The paper decomposes the current latent ``x_next`` and the predicted clean
sample ``denoised = x - v * t`` with a single-level DWT, then pushes
``x_next``'s frequency band(s) away from the denoised estimate:

    xL, xH = DWT(x_next)
    yL, yH = DWT(denoised)
    xL     = xL + s * (xL - yL)        # "low" mode
    x_next = IDWT(xL, xH)

ACE-Step's DiT latents are 1-D temporal tensors of shape ``[B, T, C]`` at
25 Hz, so we apply a 1-D DWT along the ``T`` axis.  The module imports
``pytorch_wavelets`` lazily — if the user enables DCW without installing
it, we log one clear warning and fall back to a no-op instead of
crashing the pipeline.

This file holds the :class:`DCWCorrector` wrapper that the sampler loop
calls each step.  The wavelet primitives live in
:mod:`acestep.models.common.dcw_primitives` and the lazy ``pytorch_wavelets``
loader lives in :mod:`acestep.models.common.dcw_loader` (split out per
the project's 200-LOC module cap).

Usage inside a sampler step::

    corrector = DCWCorrector(mode="low", scaler=0.1, wavelet="haar")
    # ... regular sampler computes x_next and denoised ...
    x_next = corrector.apply(x_next, denoised, t_curr)
"""

from __future__ import annotations

import os

import torch
from loguru import logger

from .dcw_primitives import dcw_double, dcw_high, dcw_low, dcw_pix

__all__ = [
    "VALID_DCW_MODES",
    "DCWCorrector",
    "dcw_low",
    "dcw_high",
    "dcw_double",
    "dcw_pix",
]

VALID_DCW_MODES = ("low", "high", "double", "pix")


class DCWCorrector:
    """Stateful wrapper that applies DCW per sampler step.

    Encapsulates the mode / scaler / wavelet choice so the sampler loop
    only needs to call ``corrector.apply(x_next, denoised, t_curr)``.

    The per-step coefficient follows the paper's Eq. 20 / 21 (EDM
    reference in ``AMAP-ML/DCW/generate.py``):

    * ``low``: ``λ = t * scaler`` — strongest at high noise / early steps,
      decays to 0 as ``t → 0``.  Matches the intuition that low-frequency
      content is painted first.
    * ``high``: ``λ = (1 - t) * scaler`` — **complementary** schedule,
      strongest at low noise / late steps when the network is actually
      painting high-frequency detail.
    * ``double``: low band uses ``t * scaler``, high band uses
      ``(1 - t) * high_scaler`` independently.
    * ``pix``: raw ``scaler`` (no ``t`` modulation), matching the reference
      FLUX scheduler's pixel-space baseline.
    """

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "double",
        scaler: float = 0.05,
        high_scaler: float = 0.02,
        wavelet: str = "haar",
    ) -> None:
        if mode not in VALID_DCW_MODES:
            raise ValueError(
                f"Invalid dcw_mode='{mode}'. Expected one of {VALID_DCW_MODES}."
            )
        self.enabled = bool(enabled)

        # LOCAL CHANGE: ACESTEP_DCW_ENABLED=false forces DCW off everywhere.
        #
        # It is applied here, in the corrector itself, rather than at any of the
        # callers, because `dcw_enabled: bool = True` is re-defaulted at every
        # layer of the chain — inference.py's params dataclass,
        # handler/generate_music.py:212, service_generate.py:54, and again as
        # generate_kwargs.get("dcw_enabled", True) in
        # service_generate_execute.py:241. Patching any one of them is silently
        # overridden by the next, which is exactly what happened on the first
        # attempt: inference.py was patched, the container rebuilt, and
        # "[DCW] Active" still appeared in the log. This is the one place that
        # cannot be bypassed.
        #
        # Why turn it off at all: DCW rewrites the latent once per sampler step
        # and is not exposed in the REST API. xl-sft on this deployment gets
        # worse as inference_steps rises — best at 4 or 5, chaotic by 10 —
        # which is backwards for a flow model and is what a per-step
        # perturbation accumulating looks like rather than a trajectory
        # converging. Classifier-free guidance was ruled out by a run at
        # guidance_scale=1.0, where do_cfg is false and the sampler never
        # enters the CFG branch, and the divergence persisted. DCW was the only
        # other per-step term. xl-turbo escapes it by clamping to 8 steps.
        if os.getenv("ACESTEP_DCW_ENABLED", "true").strip().lower() not in {
            "1", "true", "yes", "y", "on",
        }:
            if self.enabled:
                logger.info("[DCW] Disabled by ACESTEP_DCW_ENABLED")
            self.enabled = False
        self.mode = mode
        self.scaler = float(scaler)
        self.high_scaler = float(high_scaler)
        self.wavelet = wavelet
        if self.is_active:
            # One-line receipt so users can confirm the UI values actually
            # reached the sampler.  If a wavelet change in the UI seems
            # inert, grep this log line to see what was applied.
            logger.info(
                "[DCW] Active — mode={}, scaler={:.4f}, high_scaler={:.4f}, wavelet={!r}",
                self.mode, self.scaler, self.high_scaler, self.wavelet,
            )

    @property
    def is_active(self) -> bool:
        """``True`` if the corrector will actually modify the latent."""
        if not self.enabled:
            return False
        if self.mode == "double":
            return self.scaler != 0.0 or self.high_scaler != 0.0
        return self.scaler != 0.0

    def apply(
        self, x_next: torch.Tensor, denoised: torch.Tensor, t_curr: float
    ) -> torch.Tensor:
        """Apply the configured DCW correction.

        Args:
            x_next: Latent produced by the sampler step, shape ``[B, T, C]``.
            denoised: Predicted clean sample ``x - v * t``, shape ``[B, T, C]``.
            t_curr: Current timestep in ``[0, 1]`` (flow-matching convention).
                Drives the per-band schedule (see class docstring).

        Returns:
            Corrected latent with the same shape and dtype as ``x_next``.
        """
        if not self.is_active:
            return x_next
        t = float(t_curr)
        low_s = t * self.scaler
        high_s = (1.0 - t) * self.scaler
        double_high_s = (1.0 - t) * self.high_scaler
        if self.mode == "low":
            return dcw_low(x_next, denoised, low_s, self.wavelet)
        if self.mode == "high":
            return dcw_high(x_next, denoised, high_s, self.wavelet)
        if self.mode == "double":
            return dcw_double(x_next, denoised, low_s, double_high_s, self.wavelet)
        if self.mode == "pix":
            return dcw_pix(x_next, denoised, self.scaler)
        raise RuntimeError(f"unreachable dcw_mode={self.mode}")
