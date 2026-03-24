"""Loss functions for QAFC-PhysFormer.

Includes:
- QualityRankingLoss: Self-supervised ranking loss for quality scores
- QAFCLoss: Combined loss with uncertainty weighting (Kendall et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson


class QualityRankingLoss(nn.Module):
    """
    Self-supervised quality ranking loss.

    For two compressed versions of the same video (different CRF),
    the higher quality version (lower CRF) should have higher quality score.

    Uses margin ranking loss:
        loss = max(0, score_low - score_high + margin)

    This encourages: score_high > score_low + margin
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, score_high: torch.Tensor, score_low: torch.Tensor) -> torch.Tensor:
        """
        Args:
            score_high: Quality score for higher quality video [B, 1]
            score_low: Quality score for lower quality video [B, 1]
        Returns:
            Ranking loss (scalar)
        """
        # score_high should be > score_low
        # F.margin_ranking_loss(score_high, score_low, target=1) computes:
        # max(0, -1 * (score_high - score_low) + margin) = max(0, score_low - score_high + margin)
        target = torch.ones(score_high.shape[0], 1, device=score_high.device)
        loss = F.margin_ranking_loss(score_high, score_low, target, margin=self.margin)
        return loss


class QAFCLoss(nn.Module):
    """
    Combined loss for QAFC training.

    Uses uncertainty weighting (Kendall et al., 2018) to balance tasks:
        L = exp(-w1) * L1 + w1 + exp(-w2) * L2 + w2

    where w = log(sigma^2) is learned uncertainty.

    Components:
    1. rPPG loss: Negative Pearson correlation (primary task)
    2. Quality ranking loss: Self-supervised quality learning
    """

    def __init__(self, ranking_margin: float = 0.1):
        super().__init__()
        self.rppg_loss = Neg_Pearson()
        self.ranking_loss = QualityRankingLoss(margin=ranking_margin)

        # Learnable uncertainty weights (log variance)
        # Initialized to 0 (uncertainty = 1)
        self.log_var_rppg = nn.Parameter(torch.tensor(0.0))
        self.log_var_ranking = nn.Parameter(torch.tensor(0.0))

    def _weighted(self, loss: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Apply uncertainty weighting to a loss term."""
        return torch.exp(-log_var) * loss + log_var

    def forward(
        self,
        pred_rppg: torch.Tensor,
        gt_rppg: torch.Tensor,
        quality_score_high: torch.Tensor,
        quality_score_low: torch.Tensor
    ) -> dict:
        """
        Args:
            pred_rppg: Predicted rPPG signal [B, T]
            gt_rppg: Ground truth rPPG signal [B, T]
            quality_score_high: Quality score for high quality video [B, 1]
            quality_score_low: Quality score for low quality video [B, 1]
        Returns:
            Dictionary with 'total', 'rppg', 'ranking' losses
        """
        l_rppg = self.rppg_loss(pred_rppg, gt_rppg)
        l_rank = self.ranking_loss(quality_score_high, quality_score_low)

        total = (
            self._weighted(l_rppg, self.log_var_rppg) +
            self._weighted(l_rank, self.log_var_ranking)
        )

        return {
            'total': total,
            'rppg': l_rppg.detach(),
            'ranking': l_rank.detach(),
        }
