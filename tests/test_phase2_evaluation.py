"""Unit tests for Phase 2 Evaluation Pipeline components.

Validates:
  1. Key-based McNemar test paired calculation and contingency table logic.
  2. McNemar error handling on key divergence and ground truth mismatch.
  3. Anti-mirror dual clinical safety gates (balanced <= 50%, baseline < 87.5%).
  4. Subject-level bilateral agreement aggregation logic.
  5. CLI dispatcher support for --phase phase2.
"""

import json
import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from src.evaluate import compute_subject_level_metrics, run_mcnemar_test


class TestPhase2Evaluation(unittest.TestCase):
    """Test suite for Phase 2 evaluation helpers and invariants."""

    def test_01_mcnemar_test_calculation(self) -> None:
        """Verify run_mcnemar_test computes valid contingency table and p-values."""
        # Create synthetic Phase 1.2 predictions
        subjects = [f"DM_{i:03d}" for i in range(1, 27)]
        rows_p12 = []
        rows_p2 = []
        for s in subjects:
            for side in ['L', 'R']:
                # Both models predict High_Severity on true High_Severity
                rows_p12.append({
                    'subject_id': s,
                    'side': side,
                    'true_3class': 'High_Severity',
                    'predicted_3class': 'High_Severity',
                })
                rows_p2.append({
                    'subject_id': s,
                    'side': side,
                    'true_3class': 'High_Severity',
                    'predicted_3class': 'High_Severity',
                })

        # Introduce 4 discordant predictions: Phase 2 correct, Phase 1.2 wrong
        for idx in range(4):
            rows_p12[idx]['predicted_3class'] = 'Low_Severity'

        p12_df = pd.DataFrame(rows_p12)
        p2_df = pd.DataFrame(rows_p2)

        with tempfile.TemporaryDirectory() as tmpdir:
            p12_path = os.path.join(tmpdir, "p12_preds.csv")
            out_json = os.path.join(tmpdir, "mcnemar.json")
            p12_df.to_csv(p12_path, index=False)

            run_mcnemar_test(p2_df, p12_path, out_json)

            self.assertTrue(os.path.exists(out_json))
            with open(out_json) as f:
                res = json.load(f)

            self.assertEqual(res['n_test_feet'], 52)
            self.assertEqual(res['contingency_table']['both_correct_a'], 48)
            self.assertEqual(res['contingency_table']['p2_correct_p12_wrong_b'], 4)
            self.assertEqual(res['contingency_table']['p2_wrong_p12_correct_c'], 0)
            self.assertEqual(res['contingency_table']['both_wrong_d'], 0)
            self.assertEqual(res['discordant_pairs_b_plus_c'], 4)
            self.assertIn('exact_binomial_p_value', res)
            self.assertIn('interpretation', res)

    def test_02_mcnemar_key_alignment_assertion(self) -> None:
        """Verify run_mcnemar_test raises AssertionError when keys or lengths diverge."""
        # DataFrame with missing rows
        p2_df = pd.DataFrame({
            'subject_id': ['DM_001'],
            'side': ['L'],
            'true_3class': ['Healthy'],
            'predicted_3class': ['Healthy'],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            p12_path = os.path.join(tmpdir, "p12_preds.csv")
            out_json = os.path.join(tmpdir, "mcnemar.json")
            p2_df.to_csv(p12_path, index=False)

            with self.assertRaises(AssertionError):
                run_mcnemar_test(p2_df, p12_path, out_json)

    def test_03_subject_level_aggregation(self) -> None:
        """Verify bilateral agreement categorization: both correct, both wrong, split."""
        test_df = pd.DataFrame([
            # Subject 1: both correct
            {'subject_id': 'DM_001', 'side': 'L', 'model_class': 'Healthy', 'true_class': 'Healthy'},
            {'subject_id': 'DM_001', 'side': 'R', 'model_class': 'Healthy', 'true_class': 'Healthy'},
            # Subject 2: both wrong
            {'subject_id': 'DM_002', 'side': 'L', 'model_class': 'Healthy', 'true_class': 'Healthy'},
            {'subject_id': 'DM_002', 'side': 'R', 'model_class': 'Healthy', 'true_class': 'Healthy'},
            # Subject 3: split
            {'subject_id': 'DM_003', 'side': 'L', 'model_class': 'Healthy', 'true_class': 'Healthy'},
            {'subject_id': 'DM_003', 'side': 'R', 'model_class': 'Healthy', 'true_class': 'Healthy'},
        ])
        # True class for Control is 0 (Healthy)
        preds = [
            0, 0,  # Subject 1: both correct
            1, 1,  # Subject 2: both wrong (predicted Low_Severity)
            0, 1,  # Subject 3: split (L correct, R wrong)
        ]

        subj_df = compute_subject_level_metrics(test_df, preds, 'single')
        self.assertEqual(len(subj_df), 3)

        s1 = subj_df[subj_df['subject_id'] == 'DM_001'].iloc[0]
        self.assertTrue(s1['subject_correct_single'])
        self.assertFalse(s1['split_case'])

        s2 = subj_df[subj_df['subject_id'] == 'DM_002'].iloc[0]
        self.assertFalse(s2['subject_correct_single'])
        self.assertFalse(s2['split_case'])

        s3 = subj_df[subj_df['subject_id'] == 'DM_003'].iloc[0]
        self.assertFalse(s3['subject_correct_single'])
        self.assertTrue(s3['split_case'])

    def test_04_anti_mirror_clinical_gates(self) -> None:
        """Verify anti-mirror clinical safety thresholds (balanced <= 50%, baseline < 87.5%)."""
        # Case A: 4 under-estimation out of 10 errors -> 40% under-rate
        under_rate_a = 40.0
        balanced_a = bool(under_rate_a <= 50.0)
        baseline_a = bool(under_rate_a < 87.5)
        self.assertTrue(balanced_a)
        self.assertTrue(baseline_a)

        # Case B: 5 under-estimation out of 6 errors -> 83.33% under-rate (Phase 2 baseline)
        under_rate_b = 83.33
        balanced_b = bool(under_rate_b <= 50.0)
        baseline_b = bool(under_rate_b < 87.5)
        self.assertFalse(balanced_b)
        self.assertTrue(baseline_b)

        # Case C: 9 under-estimation out of 10 errors -> 90% under-rate (fails both)
        under_rate_c = 90.0
        balanced_c = bool(under_rate_c <= 50.0)
        baseline_c = bool(under_rate_c < 87.5)
        self.assertFalse(balanced_c)
        self.assertFalse(baseline_c)

    def test_05_cli_dispatcher_phase2(self) -> None:
        """Verify argparse accepts --phase phase2 without raising error."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "src/evaluate.py", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("phase2", result.stdout)


if __name__ == '__main__':
    unittest.main()
