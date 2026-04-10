import math
import unittest

from scipy.special import erfinv

from basic_utils import convert_ssa_re_to_medium_params


def _ssa_from_medium_params(mean_wave_number: float, b: float, fv: float) -> float:
    erf_term = erfinv(1.0 - 2.0 * fv)
    coeff = (
        2.0
        * math.exp(-(erf_term**2))
        / (0.918 * math.pi * math.sqrt(3.0) * fv)
        * math.sqrt((b + 2.0) / (b + 1.0))
    )
    return coeff * mean_wave_number


class BasicUtilsParameterConversionTests(unittest.TestCase):
    def test_convert_from_ssa_and_re_recovers_mean_wave_number(self) -> None:
        target_mean_wave_number = 5349.7
        b = 1.345
        fv = 0.194
        ssa = _ssa_from_medium_params(target_mean_wave_number, b=b, fv=fv)
        r_e = 3.0 / (0.918 * ssa)

        result = convert_ssa_re_to_medium_params(ssa=ssa, r_e=r_e, b=b, fv=fv)

        self.assertAlmostEqual(result["mean_waveNumber"], target_mean_wave_number, places=10)
        self.assertAlmostEqual(result["b"], b, places=12)
        self.assertAlmostEqual(result["fv"], fv, places=12)

    def test_convert_from_re_only(self) -> None:
        b = 1.345
        fv = 0.194
        r_e = 0.00015

        result = convert_ssa_re_to_medium_params(r_e=r_e, b=b, fv=fv)

        expected_ssa = 3.0 / (0.918 * r_e)
        self.assertAlmostEqual(result["SSA"], expected_ssa, places=10)
        self.assertAlmostEqual(result["R_e"], r_e, places=12)

    def test_reject_inconsistent_ssa_and_re(self) -> None:
        with self.assertRaises(ValueError):
            convert_ssa_re_to_medium_params(ssa=100.0, r_e=0.1, b=1.345, fv=0.194)


if __name__ == "__main__":
    unittest.main()
