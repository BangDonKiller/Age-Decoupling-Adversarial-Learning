import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
	def __init__(self, m=0.25, gamma=256.0):
		super().__init__()
		self.m = m
		self.gamma = gamma

	def forward(self, feat1_or_scores, feat2_or_same_label, same_label=None):
		"""
		Pairwise Circle Loss for speaker verification.

		Args:
			Legacy mode:
				feat1_or_scores: Tensor [B, D], assumed to be L2-normalized.
				feat2_or_same_label: Tensor [B, D], assumed to be L2-normalized.
				same_label: Tensor [B], 1 for same speaker, 0 for different speakers.

			Score mode:
				feat1_or_scores: Tensor [B], cosine-like similarity scores.
				feat2_or_same_label: Tensor [B], 1 for same speaker, 0 for different speakers.
				same_label: None
		"""
		# Backward-compatible dual interface:
		# 1) forward(feat1, feat2, same_label)
		# 2) forward(scores, same_label)
		if same_label is None:
			cos_sim = feat1_or_scores
			same_label = feat2_or_same_label
		else:
			feat1 = feat1_or_scores
			feat2 = feat2_or_same_label
			# Compute cosine similarity directly because embeddings are assumed normalized.
			cos_sim = torch.sum(feat1 * feat2, dim=1)

		# Use float32 for the loss math to improve numerical stability under mixed precision.
		cos_sim = cos_sim.float()
		same_label = same_label.float()

		# Split the batch into positive and negative pairs.
		pos_mask = same_label > 0.5
		neg_mask = ~pos_mask

		# Circle Loss margin targets.
		delta_p = 1.0 - self.m
		delta_n = self.m

		loss_terms = []

		# Positive pairs: want similarity larger than 1 - m.
		if torch.any(pos_mask):
			s_p = cos_sim[pos_mask]
			alpha_p = F.relu(1.0 + self.m - s_p)
			logit_p = -self.gamma * alpha_p * (s_p - delta_p)
			# log(sum(exp(logit_p))) in a stable way.
			pos_logsumexp = torch.logsumexp(logit_p, dim=0)
			loss_terms.append(pos_logsumexp)

		# Negative pairs: want similarity smaller than m.
		if torch.any(neg_mask):
			s_n = cos_sim[neg_mask]
			alpha_n = F.relu(s_n + self.m)
			logit_n = self.gamma * alpha_n * (s_n - delta_n)
			# log(sum(exp(logit_n))) in a stable way.
			neg_logsumexp = torch.logsumexp(logit_n, dim=0)
			loss_terms.append(neg_logsumexp)

		# Handle edge cases where a batch contains only positive pairs or only negative pairs.
		if len(loss_terms) == 2:
			# Original Circle Loss formulation: log(1 + sum(exp(logit_n)) * sum(exp(logit_p))).
			loss = F.softplus(loss_terms[0] + loss_terms[1])
		elif len(loss_terms) == 1:
			# If only one pair type exists, fall back to the corresponding one-sided objective.
			loss = F.softplus(loss_terms[0])
		else:
			# No valid pairs in the batch.
			loss = cos_sim.new_tensor(0.0)

		return loss
