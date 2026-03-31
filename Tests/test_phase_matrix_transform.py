import unittest

import numpy as np

from phase_matrix_calculator import PhaseMatrixCalculator


class PhaseMatrixTransformTests(unittest.TestCase):
    def test_zero_rotation_keeps_diagonal_sparse_matrix_unchanged(self) -> None:
        transformed = PhaseMatrixCalculator.transform_diagonal_phase_matrix_to_principal(
            P11=3.0,
            P22=1.5,
            sigma1=0.0,
            sigma2=0.0,
        )

        expected = np.diag([3.0, 1.5, 0.0, 0.0])
        np.testing.assert_allclose(transformed, expected)

    def test_sparse_helper_matches_general_transform(self) -> None:
        phase_matrix_12 = np.diag([2.0, 0.5, 0.0, 0.0])
        sigma1 = 0.2
        sigma2 = -0.4

        general = PhaseMatrixCalculator.transform_phase_matrix_to_principal(
            phase_matrix_12=phase_matrix_12,
            sigma1=sigma1,
            sigma2=sigma2,
        )
        sparse = PhaseMatrixCalculator.transform_diagonal_phase_matrix_to_principal(
            P11=2.0,
            P22=0.5,
            sigma1=sigma1,
            sigma2=sigma2,
        )

        np.testing.assert_allclose(general, sparse)

    def test_nonzero_rotation_couples_qu_subblock_for_unequal_diagonal_terms(self) -> None:
        transformed = PhaseMatrixCalculator.transform_diagonal_phase_matrix_to_principal(
            P11=4.0,
            P22=1.0,
            sigma1=0.3,
            sigma2=-0.2,
        )

        self.assertNotAlmostEqual(float(transformed[1, 2]), 0.0)
        self.assertNotAlmostEqual(float(transformed[2, 1]), 0.0)
        self.assertAlmostEqual(float(transformed[3, 3]), 0.0)

    def test_equal_diagonal_case_preserves_trace_under_same_left_right_rotation(self) -> None:
        transformed = PhaseMatrixCalculator.transform_diagonal_phase_matrix_to_principal(
            P11=2.5,
            P22=2.5,
            sigma1=0.8,
            sigma2=0.8,
        )

        qu_block = transformed[1:3, 1:3]
        np.testing.assert_allclose(qu_block, qu_block.T, atol=1e-12)
        self.assertAlmostEqual(float(np.trace(qu_block)), 2.5)

    def test_batch_inputs_broadcast_to_common_shape(self) -> None:
        transformed = PhaseMatrixCalculator.transform_diagonal_phase_matrix_to_principal(
            P11=np.array([1.0, 2.0]),
            P22=np.array([0.5, 1.5]),
            sigma1=np.array([0.0, 0.2]),
            sigma2=0.1,
        )

        self.assertEqual(transformed.shape, (2, 4, 4))

    def test_inverse_transform_recovers_original_matrix(self) -> None:
        original = np.diag([5.0, 2.0, 0.0, 0.0])
        sigma1 = 0.15
        sigma2 = -0.35

        principal = PhaseMatrixCalculator.transform_phase_matrix_to_principal(
            phase_matrix_12=original,
            sigma1=sigma1,
            sigma2=sigma2,
        )
        recovered = (
            PhaseMatrixCalculator.stokes_rotation_matrix(-sigma2)
            @ principal
            @ PhaseMatrixCalculator.stokes_rotation_matrix(sigma1)
        )

        np.testing.assert_allclose(recovered, original, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
