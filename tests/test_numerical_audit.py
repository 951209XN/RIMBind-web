import unittest

from backend.numerical_audit import compare_probabilities


class NumericalAudit(unittest.TestCase):
    def compare(self, candidate, reference, historical, **kwargs):
        return compare_probabilities(candidate, reference, historical,
            candidate_runtime=kwargs.get('candidate_runtime', 'gpu-A/torch-X/assets-Y'),
            reference_runtime=kwargs.get('reference_runtime', 'gpu-A/torch-X/assets-Y'))

    def test_same_runtime_pass_does_not_erase_historical_failure(self):
        result = self.compare([.2, .7], [.2, .7], [.21, .69])
        self.assertTrue(result['same_runtime_equivalence_passed'])
        self.assertFalse(result['historical_equivalence_passed'])
        self.assertFalse(result['release_authorized'])
        self.assertFalse(result['policy_change_approved'])

    def test_different_runtime_not_mislabeled_same(self):
        with self.assertRaises(ValueError):
            self.compare([.2], [.2], [.2], reference_runtime='gpu-B/torch-X/assets-Y')

    def test_threshold_not_relaxed(self):
        result = self.compare([.2], [.20002], [.2])
        self.assertFalse(result['same_runtime_equivalence_passed'])
        self.assertEqual(result['same_runtime_tolerance'], 1e-5)
        self.assertTrue(result['historical_equivalence_passed'])

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            self.compare([float('nan')], [.2], [.2])

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.compare([.2], [.2, .3], [.2])

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            self.compare([2], [.2], [.2])


if __name__ == '__main__':
    unittest.main()
