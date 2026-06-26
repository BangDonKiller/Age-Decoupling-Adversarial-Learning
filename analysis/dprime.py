import argparse
from pathlib import Path

import pandas as pd


def compute_dprime(target_scores: pd.Series, non_target_scores: pd.Series) -> float:
	"""Compute d-prime under equal-variance assumption."""
	mu_target = target_scores.mean()
	mu_non_target = non_target_scores.mean()
	var_target = target_scores.var(ddof=1)
	var_non_target = non_target_scores.var(ddof=1)

	pooled_std = (0.5 * (var_target + var_non_target)) ** 0.5
	if pooled_std == 0:
		raise ValueError("Pooled standard deviation is zero. d-prime is undefined.")

	return (mu_target - mu_non_target) / pooled_std


def main() -> None:
	parser = argparse.ArgumentParser(description="Compute d-prime for expert score columns.")
	parser.add_argument(
		"--csv",
		type=Path,
		default=Path("logs/det_curve/Vox-CA5_three_expert_scores.csv"),
		help="Path to score CSV file.",
	)
	parser.add_argument(
		"--label-col",
		type=str,
		default="label",
		help="Column name for labels (1 for target, 0 for non-target).",
	)
	parser.add_argument(
		"--score-cols",
		nargs="+",
		default=["Small_score", "Medium_score", "Large_score"],
		help="Score column names to compute d-prime.",
	)
	args = parser.parse_args()

	if not args.csv.exists():
		raise FileNotFoundError(f"CSV file not found: {args.csv}")

	df = pd.read_csv(args.csv)

	if args.label_col not in df.columns:
		raise KeyError(f"Missing label column: {args.label_col}")

	missing_score_cols = [c for c in args.score_cols if c not in df.columns]
	if missing_score_cols:
		raise KeyError(f"Missing score columns: {missing_score_cols}")

	labels = df[args.label_col]
	if not labels.isin([0, 1]).all():
		raise ValueError("Label column must contain only 0 and 1.")

	target_mask = labels == 1
	non_target_mask = labels == 0

	if target_mask.sum() == 0 or non_target_mask.sum() == 0:
		raise ValueError("Both target (1) and non-target (0) samples are required.")

	print(f"CSV: {args.csv}")
	print("d-prime results:")
	for col in args.score_cols:
		target_scores = df.loc[target_mask, col]
		non_target_scores = df.loc[non_target_mask, col]
		dprime = compute_dprime(target_scores, non_target_scores)
		print(f"  {col}: {dprime:.6f}")


if __name__ == "__main__":
	main()
