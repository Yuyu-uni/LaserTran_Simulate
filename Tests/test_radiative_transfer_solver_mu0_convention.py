import contextlib
import io
import unittest

import numpy as np

from radiative_transfer_solver import RadiativeTransferSolver


class RadiativeTransferSolverMu0ConventionTests(unittest.TestCase):
    def test_brdf_positive_for_downward_incident_direction_convention(self) -> None:
        n_angles = 90
        phase_angles = np.linspace(
            np.pi / (2.0 * n_angles),
            np.pi - np.pi / (2.0 * n_angles),
            n_angles,
        )
        p11 = np.full(n_angles, 0.05, dtype=float)
        p22 = np.full(n_angles, 0.05, dtype=float)

        solver = RadiativeTransferSolver(
            extinction_coefficient=1.0,
            phase_angles=phase_angles,
            p11=p11,
            p22=p22,
            n_streams=8,
            fourier_order=2,
            n_phi_quadrature=64,
        )

        # Run quietly in tests to avoid terminal encoding side effects.
        with contextlib.redirect_stdout(io.StringIO()):
            solution = solver.run_simulation(
                solar_zenith_deg=30.0,
                solar_azimuth_deg=0.0,
            )

        self.assertAlmostEqual(solution.mu0, float(np.cos(np.deg2rad(30.0))), places=12)
        brdf = solver.compute_brdf(mu_s=0.5, phi_s=0.0, solution=solution)
        self.assertGreater(brdf, 0.0)


if __name__ == "__main__":
    unittest.main()
