"""High-Dimensional Leverage Scoring for Outlier-Aware Mixed-Precision Quantization.

Grounding: Chapter 21 (Identification of High Leverage Points in High Dimensional
Sparse and Non-Sparse Data, Zahariah & Habshah Midi).

Mathematical Formulation:
In high-dimensional transformer activations X in R^(N x D) where D >> N, inverting
the Hat matrix H = X(X^T X)^(-1) X^T is intractable and numerically singular.

Chapter 21 calculates Fast Diagnostic Leverage Distances without matrix inversion:
1. Robust Global Center & Dispersion:
   med_global = Median(X)
   MAD_global = Median(|X - med_global|) * 1.4826
2. Subspace Projection on K << D principal directions (via randomized SVD):
   X_centered = X - Median(X, dim=0)
   X_normed = X_centered / (MAD_global + eps)
   U, S, V = torch.pca_lowrank(X_normed, q=K, center=False)
3. Single-Pass Diagnostic Leverage Score:
   Lev_j = ( max_i |X_{ij}| / (MAD_global + eps) ) * sum_{k=1}^K ( V_{jk}^2 * S_k^2 )

Isolates top outlier channels (0.1% - 0.5% of hidden dimension) in sub-second time,
routing them to protected bfloat16 storage while quantizing 99.5% normal channels to INT4.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class HighDimensionalLeverageScorer:
    """Computes single-pass high-dimensional leverage scores for outlier channel protection."""

    def __init__(self, k_components: int = 4, eps: float = 1e-6) -> None:
        self.k_components = k_components
        self.eps = eps

    @torch.no_grad()
    def compute_leverage_scores(self, X: torch.Tensor) -> torch.Tensor:
        """Computes diagnostic leverage scores for all D channels in X.

        Args:
            X: Activation tensor of shape (N, D) or (B, N, D)

        Returns:
            Leverage score tensor of shape (D,)
        """
        # Flatten batch dimensions if necessary -> (N, D)
        if X.dim() == 3:
            X_flat = X.reshape(-1, X.shape[-1]).to(torch.float32)
        else:
            X_flat = X.to(torch.float32)

        N, D = X_flat.shape
        q = min(self.k_components, min(N, D) - 1)
        if q < 1:
            q = 1

        # Step 1: Global robust dispersion
        med_global = torch.median(X_flat)
        abs_dev = torch.abs(X_flat - med_global)
        mad_global = torch.median(abs_dev) * 1.4826
        mad_clamped = torch.clamp(mad_global, min=self.eps)

        # Center per channel, normalize by global scale
        med_channel = torch.median(X_flat, dim=0).values
        X_centered = X_flat - med_channel.unsqueeze(0)
        X_normed = X_centered / mad_clamped

        # Step 2: Randomized SVD projection onto principal subspace
        _, S, V = torch.pca_lowrank(X_normed, q=q, center=False)

        # Step 3: Diagnostic Leverage computation
        # Subspace energy weighted by singular values: sum_k (V_{jk}^2 * S_k^2)
        subspace_energy = torch.sum((V**2) * (S.unsqueeze(0) ** 2), dim=-1)  # Shape: (D,)

        # Marginal extreme deviation factor
        max_abs = torch.max(torch.abs(X_centered), dim=0).values
        marginal_factor = max_abs / mad_clamped

        # Composite Chapter 21 Diagnostic Leverage Score
        leverage_scores = marginal_factor * subspace_energy
        return leverage_scores

    @torch.no_grad()
    def identify_outlier_channels(
        self,
        X: torch.Tensor,
        top_k: int = 16,
        ratio_threshold: float | None = None,
    ) -> torch.Tensor:
        """Returns sorted indices of top outlier channels to protect in bfloat16.

        Args:
            X: Activation tensor (N, D)
            top_k: Maximum number of channels to protect
            ratio_threshold: Optional multiplier above median leverage

        Returns:
            Tensor of integer channel indices of shape (K_protected,)
        """
        scores = self.compute_leverage_scores(X)
        D = scores.shape[0]
        k = min(top_k, D)

        if ratio_threshold is not None:
            med_score = torch.median(scores).item()
            cutoff = med_score * ratio_threshold
            outlier_mask = scores >= cutoff
            indices = torch.nonzero(outlier_mask, as_tuple=False).squeeze(-1)
            if indices.numel() > k:
                # Top k by score
                _, top_sub = torch.topk(scores[indices], k=k, largest=True)
                indices = indices[top_sub]
            return torch.sort(indices).values

        # Default top_k selection
        _, top_indices = torch.topk(scores, k=k, largest=True)
        return torch.sort(top_indices).values


class MixedPrecisionW4A16Packer:
    """Splits weight tensors into packed INT4 normal channels and BF16 protected channels."""

    def __init__(self, group_size: int = 128) -> None:
        self.group_size = group_size

    @torch.no_grad()
    def split_weights(
        self,
        W: torch.Tensor,
        outlier_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Splits weight matrix W into normal INT4 channels and protected BF16 channels.

        Args:
            W: Weight matrix of shape (D_in, D_out)
            outlier_indices: 1D tensor of channel indices along D_in to protect

        Returns:
            Tuple of:
            - W_normal: Remaining channels of shape (D_in - k, D_out)
            - W_protected: Full precision slice of shape (k, D_out) in bfloat16
            - normal_indices: 1D tensor of remaining channel indices
        """
        D_in, D_out = W.shape
        device = W.device
        all_indices = torch.arange(D_in, device=device)

        mask = torch.ones(D_in, dtype=torch.bool, device=device)
        mask[outlier_indices] = False
        normal_indices = all_indices[mask]

        W_protected = W[outlier_indices, :].to(torch.bfloat16)
        W_normal = W[normal_indices, :]

        return W_normal, W_protected, normal_indices

    @torch.no_grad()
    def quantize_normal_channels(
        self,
        W_normal: torch.Tensor,
        calibration_mode: Literal[ssi, naive_max] = "ssi",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantizes normal channel weights to symmetric 4-bit with group scales.

        Args:
            W_normal: Remaining normal channel weight tensor
            calibration_mode: "ssi" for Stress-Strength Interference optimal scaling,
                             "naive_max" for standard max magnitude scaling.

        Returns:
            - W_q: Quantized integer tensor in [-8, 7]
            - scales: Group scale tensor of shape (D_in_normal // group_size, D_out)
        """
        D_in, D_out = W_normal.shape
        # Pad to multiple of group_size if necessary
        pad_in = (self.group_size - (D_in % self.group_size)) % self.group_size
        if pad_in > 0:
            W_padded = F.pad(W_normal, (0, 0, 0, pad_in))
        else:
            W_padded = W_normal

        D_in_padded = W_padded.shape[0]
        n_groups = D_in_padded // self.group_size
        grouped = W_padded.view(n_groups, self.group_size, D_out)

        max_abs = torch.max(torch.abs(grouped), dim=1, keepdim=True).values.clamp(min=1e-5)

        if calibration_mode == "ssi":
            # Chapter 8.5 Stress-Strength Interference closed-form optimal boundary
            mean = torch.mean(grouped, dim=1, keepdim=True)
            var = torch.var(grouped, dim=1, unbiased=True, keepdim=True).clamp(min=1e-6)
            std = torch.sqrt(var)
            centered = grouped - mean
            m4 = torch.mean(centered**4, dim=1, keepdim=True)
            kurt = (m4 / (var**2)).clamp(min=1.0, max=25.0)

            k_ssi = 2.2 + 0.45 * torch.sqrt(kurt)
            boundary = torch.minimum(k_ssi * std, max_abs)
            scales = boundary / 7.0
        else:
            scales = max_abs / 7.0

        # Quantize and clamp
        q_grouped = torch.clamp(torch.round(grouped / scales), -8.0, 7.0)
        W_q = q_grouped.view(D_in_padded, D_out)[:D_in, :]

        return W_q.to(torch.int8), scales.squeeze(1).to(torch.bfloat16)

    @torch.no_grad()
    def reconstruct_mixed_precision(
        self,
        W_q: torch.Tensor,
        scales: torch.Tensor,
        W_protected: torch.Tensor,
        outlier_indices: torch.Tensor,
        normal_indices: torch.Tensor,
        original_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Reconstructs full dense float tensor from mixed-precision representation for SNR testing."""
        D_in, D_out = original_shape
        device = W_q.device
        W_rec = torch.zeros(D_in, D_out, dtype=torch.bfloat16, device=device)

        # Dequantize normal channels
        D_norm = W_q.shape[0]
        n_groups = scales.shape[0]
        pad_in = (n_groups * self.group_size) - D_norm
        if pad_in > 0:
            W_q_padded = F.pad(W_q.to(torch.float32), (0, 0, 0, pad_in))
        else:
            W_q_padded = W_q.to(torch.float32)

        grouped_q = W_q_padded.view(n_groups, self.group_size, D_out)
        dequant_grouped = grouped_q * scales.unsqueeze(1).to(torch.float32)
        dequant_normal = dequant_grouped.view(-1, D_out)[:D_norm, :].to(torch.bfloat16)

        W_rec[normal_indices, :] = dequant_normal
        W_rec[outlier_indices, :] = W_protected

        return W_rec
