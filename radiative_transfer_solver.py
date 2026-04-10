from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class RTESolution:
    """保存单个入射条件下的傅里叶模态解。"""

    mu0: float
    phi0: float
    incident_stokes: np.ndarray
    mode_0: np.ndarray
    mode_cos: list[np.ndarray]
    mode_sin: list[np.ndarray]


class RadiativeTransferSolver:
    """
    矢量辐射传输方程求解器（基于离散坐标 + 傅里叶展开 + 特征值法）。

    说明：
    1. 当前项目已提供的相矩阵数据为 P11(Theta) 与 P22(Theta)。
    2. 本实现将 1-2 参考系相矩阵近似为仅对角项非零：diag(P11, P22, 0, 0)。
    3. 在此基础上，按 .agent/MD_EqSolveGuide.md 的流程实现傅里叶分解和 ODE 求解。
    """

    def __init__(
        self,
        extinction_coefficient: float,
        phase_angles: np.ndarray,
        p11: np.ndarray,
        p22: np.ndarray,
        n_streams: int = 16,
        fourier_order: int = 8,
        n_phi_quadrature: int = 96,
    ) -> None:
        self.kappa_e = float(extinction_coefficient)
        self.phase_angles = np.asarray(phase_angles, dtype=float)
        self.p11 = np.asarray(p11, dtype=float)
        self.p22 = np.asarray(p22, dtype=float)
        self.n_streams = int(n_streams)
        self.fourier_order = int(fourier_order)
        self.n_phi_quadrature = int(n_phi_quadrature)

        self._validate_inputs()
        self._init_quadrature()

        self.phase_m0: Optional[np.ndarray] = None
        self.phase_m_cos: Optional[np.ndarray] = None
        self.phase_m_sin: Optional[np.ndarray] = None

        self.last_solution: Optional[RTESolution] = None
        self.last_products: Optional[dict[str, np.ndarray | float]] = None

    # ========================
    # 初始化与基础工具
    # ========================
    def _validate_inputs(self) -> None:
        if self.kappa_e <= 0.0:
            raise ValueError("extinction_coefficient 必须为正数。")
        if self.phase_angles.ndim != 1:
            raise ValueError("phase_angles 必须是一维数组。")
        if self.p11.shape != self.phase_angles.shape or self.p22.shape != self.phase_angles.shape:
            raise ValueError("p11/p22 与 phase_angles 维度不一致。")
        if np.any(np.diff(self.phase_angles) <= 0):
            raise ValueError("phase_angles 必须严格递增。")
        if self.n_streams <= 0 or self.n_streams % 2 != 0:
            raise ValueError("n_streams 必须为正偶数（确保上下半球离散点对称）。")
        if self.fourier_order < 0:
            raise ValueError("fourier_order 必须 >= 0。")
        if self.n_phi_quadrature < 8:
            raise ValueError("n_phi_quadrature 建议至少为 8。")

    @staticmethod
    def _leggauss_interval(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
        """将 [-1, 1] 上的 Gauss-Legendre 节点映射到 [a, b]。"""
        x, w = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (b - a) * x + 0.5 * (b + a)
        weights = 0.5 * (b - a) * w
        return nodes, weights

    def _init_quadrature(self) -> None:
        # 极角离散：mu in [-1, 1]
        self.mu_nodes, self.mu_weights = np.polynomial.legendre.leggauss(self.n_streams)
        self.mu_nodes = self.mu_nodes.astype(float)
        self.mu_weights = self.mu_weights.astype(float)

        # 方位角离散：phi in [0, 2pi]
        self.phi_nodes, self.phi_weights = self._leggauss_interval(
            0.0, 2.0 * np.pi, self.n_phi_quadrature
        )

        self._inv_mu_diag = np.repeat(1.0 / self.mu_nodes, 4)

        self.down_dir_indices = np.where(self.mu_nodes < 0.0)[0]
        self.up_dir_indices = np.where(self.mu_nodes > 0.0)[0]
        self.down_component_indices = np.concatenate(
            [np.arange(4 * i, 4 * i + 4) for i in self.down_dir_indices]
        )
        self.up_component_indices = np.concatenate(
            [np.arange(4 * i, 4 * i + 4) for i in self.up_dir_indices]
        )

        if self.fourier_order > 0:
            m = np.arange(1, self.fourier_order + 1, dtype=float)[:, None]
            self.cos_m_phi = np.cos(m * self.phi_nodes[None, :])
            self.sin_m_phi = np.sin(m * self.phi_nodes[None, :])
        else:
            self.cos_m_phi = np.empty((0, self.n_phi_quadrature), dtype=float)
            self.sin_m_phi = np.empty((0, self.n_phi_quadrature), dtype=float)

    def _phase_interp(self, scattering_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """插值获取 P11/P22(Theta)。"""
        p11_values = np.interp(
            scattering_angles,
            self.phase_angles,
            self.p11,
            left=float(self.p11[0]),
            right=float(self.p11[-1]),
        )
        p22_values = np.interp(
            scattering_angles,
            self.phase_angles,
            self.p22,
            left=float(self.p22[0]),
            right=float(self.p22[-1]),
        )
        return p11_values, p22_values

    # ========================
    # 相矩阵傅里叶系数
    # ========================
    def _compute_pair_fourier(
        self,
        mu_out: float,
        mu_in: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算单个 (mu_out, mu_in) 对应的相矩阵傅里叶系数。

        返回：
        - m=0: (4,4)
        - cos 系数: (M,4,4)
        - sin 系数: (M,4,4)
        """
        sin_out = np.sqrt(max(0.0, 1.0 - mu_out * mu_out))
        sin_in = np.sqrt(max(0.0, 1.0 - mu_in * mu_in))

        cos_gamma = mu_out * mu_in + sin_out * sin_in * np.cos(self.phi_nodes)
        cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
        gamma = np.arccos(cos_gamma)

        p11_vals, p22_vals = self._phase_interp(gamma)
        wp11 = self.phi_weights * p11_vals
        wp22 = self.phi_weights * p22_vals

        p0 = np.zeros((4, 4), dtype=float)
        p0[0, 0] = np.sum(wp11) / (2.0 * np.pi)
        p0[1, 1] = np.sum(wp22) / (2.0 * np.pi)

        p_cos = np.zeros((self.fourier_order, 4, 4), dtype=float)
        p_sin = np.zeros((self.fourier_order, 4, 4), dtype=float)

        if self.fourier_order > 0:
            # 按论文公式：P_mc = (1/pi) \\sum w_i P cos(mphi_i)
            #             P_ms = (1/pi) \\sum w_i P sin(mphi_i)
            cos_terms = self.cos_m_phi
            sin_terms = self.sin_m_phi

            p_cos[:, 0, 0] = (cos_terms @ wp11) / np.pi
            p_cos[:, 1, 1] = (cos_terms @ wp22) / np.pi
            p_sin[:, 0, 0] = (sin_terms @ wp11) / np.pi
            p_sin[:, 1, 1] = (sin_terms @ wp22) / np.pi

        return p0, p_cos, p_sin

    def _build_phase_fourier_tables(self) -> None:
        """构建离散 mu_j, mu_k 上的 P0/Pmc/Pms 系数表。"""
        if self.phase_m0 is not None:
            return

        jn = self.n_streams
        self.phase_m0 = np.zeros((jn, jn, 4, 4), dtype=float)
        self.phase_m_cos = np.zeros((self.fourier_order, jn, jn, 4, 4), dtype=float)
        self.phase_m_sin = np.zeros((self.fourier_order, jn, jn, 4, 4), dtype=float)

        for j, mu_out in enumerate(self.mu_nodes):
            for k, mu_in in enumerate(self.mu_nodes):
                p0, p_cos, p_sin = self._compute_pair_fourier(mu_out=mu_out, mu_in=mu_in)
                self.phase_m0[j, k] = p0
                if self.fourier_order > 0:
                    self.phase_m_cos[:, j, k] = p_cos
                    self.phase_m_sin[:, j, k] = p_sin

    def _build_incident_phase_fourier(
        self,
        mu0: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构建源项中的 P(mu_j; mu0) 傅里叶系数。"""
        jn = self.n_streams
        p0_inc = np.zeros((jn, 4, 4), dtype=float)
        pcos_inc = np.zeros((self.fourier_order, jn, 4, 4), dtype=float)
        psin_inc = np.zeros((self.fourier_order, jn, 4, 4), dtype=float)

        for j, mu_out in enumerate(self.mu_nodes):
            p0, p_cos, p_sin = self._compute_pair_fourier(mu_out=mu_out, mu_in=mu0)
            p0_inc[j] = p0
            if self.fourier_order > 0:
                pcos_inc[:, j] = p_cos
                psin_inc[:, j] = p_sin

        return p0_inc, pcos_inc, psin_inc

    # ========================
    # ODE 系统装配与求解
    # ========================
    def _build_system_matrix(self, p_mode: np.ndarray, mode_order: int) -> np.ndarray:
        """
        根据给定傅里叶模态的相矩阵构建 B 矩阵。

        dI/dz = B I + F exp(-kappa_e z / mu0)
        """
        dim = 4 * self.n_streams
        mat = -self.kappa_e * np.eye(dim, dtype=float)

        # 傅里叶模态的方位角积分归一化因子：
        # m=0:  2π
        # m>=1: π
        azimuth_factor = 2.0 * np.pi if mode_order == 0 else np.pi

        # 组装散射耦合项：sum_k w_k P(j,k) I(k)
        for j in range(self.n_streams):
            rs = slice(4 * j, 4 * j + 4)
            for k in range(self.n_streams):
                cs = slice(4 * k, 4 * k + 4)
                mat[rs, cs] += azimuth_factor * self.mu_weights[k] * p_mode[j, k]

        # 左乘 M^{-1}
        b_mat = self._inv_mu_diag[:, None] * mat
        return b_mat

    def _build_source_vector(
        self,
        p_incident_mode: np.ndarray,
        incident_stokes: np.ndarray,
    ) -> np.ndarray:
        """构建源项向量 F = M^{-1}Q。"""
        dim = 4 * self.n_streams
        q_vec = np.zeros(dim, dtype=float)
        for j in range(self.n_streams):
            rs = slice(4 * j, 4 * j + 4)
            q_vec[rs] = p_incident_mode[j] @ incident_stokes

        f_vec = self._inv_mu_diag * q_vec
        return f_vec

    @staticmethod
    def _select_stable_modes(eigvals: np.ndarray, n_required: int) -> np.ndarray:
        """
        选取稳定模态。

        说明：
        - 在当前实现的 z 方向和边界条件约定下，满足深层衰减的分支对应 Re(lambda) > 0。
        - 若数量不足，则退化为选择实部最大的 n_required 个，保证方程可解。
        """
        real_part = np.real(eigvals)
        stable = np.where(real_part > 1e-10)[0]

        if stable.size >= n_required:
            order = np.argsort(real_part[stable])[::-1]
            return stable[order[:n_required]]

        # 数值退化场景下兜底
        return np.argsort(real_part)[-n_required:]

    def _solve_single_mode_surface(
        self,
        p_mode: np.ndarray,
        p_incident_mode: np.ndarray,
        mode_order: int,
        mu0: float,
        incident_stokes: np.ndarray,
    ) -> np.ndarray:
        """
        解单个傅里叶模态在 z=0 的漫射 Stokes 向量。

        返回形状：(n_streams, 4)
        """
        b_mat = self._build_system_matrix(p_mode=p_mode, mode_order=mode_order)
        f_vec = self._build_source_vector(p_incident_mode, incident_stokes)

        dim = b_mat.shape[0]
        eye = np.eye(dim, dtype=float)
        shift = self.kappa_e / mu0

        # 特解系数 C = -(B + shift*E)^(-1) F
        lhs = b_mat + shift * eye
        try:
            c_particular = -np.linalg.solve(lhs, f_vec)
        except np.linalg.LinAlgError:
            # 奇异时使用最小二乘兜底
            c_particular = -np.linalg.lstsq(lhs, f_vec, rcond=None)[0]

        eigvals, eigvecs = np.linalg.eig(b_mat)
        n_down = self.down_component_indices.size
        select_idx = self._select_stable_modes(eigvals=eigvals, n_required=n_down)

        v_selected = eigvecs[:, select_idx]
        v_down = v_selected[self.down_component_indices, :]
        rhs = -c_particular[self.down_component_indices]

        # 边界条件：I^(D)(mu<0, z=0) = 0
        coeff = np.linalg.lstsq(v_down, rhs, rcond=None)[0]
        surface_flat = v_selected @ coeff + c_particular
        surface_flat = np.real_if_close(surface_flat, tol=1e5).astype(float)

        return surface_flat.reshape(self.n_streams, 4)

    # ========================
    # 对外求解接口
    # ========================
    def run_simulation(
        self,
        solar_zenith_deg: float,
        solar_azimuth_deg: float = 0.0,
        incident_stokes: Optional[np.ndarray] = None,
    ) -> RTESolution:
        """
        运行辐射传输求解（默认只计算 z=0 的漫射上行辐射）。

        :param solar_zenith_deg: 入射天顶角，单位度，范围 [0, 89.9]
        :param solar_azimuth_deg: 入射方位角，单位度
        :param incident_stokes: 入射 Stokes 向量，默认 [0.5, 0.5, 0, 0]^T
        """
        theta0_rad = np.deg2rad(solar_zenith_deg)
        mu0_abs = float(np.cos(theta0_rad))
        if mu0_abs <= 0.0:
            raise ValueError("solar_zenith_deg 必须小于 90 度。")
        # 与推导文档保持一致：mu<0 对应向下方向。
        # 因此内部求解使用 mu0_internal = cos(pi-theta0) = -cos(theta0)。
        mu0_internal = float(np.cos(np.pi - theta0_rad))

        phi0 = float(np.deg2rad(solar_azimuth_deg))

        if incident_stokes is None:
            incident_stokes = np.array([0.5, 0.5, 0.0, 0.0], dtype=float)
        else:
            incident_stokes = np.asarray(incident_stokes, dtype=float)
            if incident_stokes.shape != (4,):
                raise ValueError("incident_stokes 维度必须是 (4,)。")

        self._build_phase_fourier_tables()
        p0_inc, pcos_inc, psin_inc = self._build_incident_phase_fourier(mu0=mu0_internal)

        print("=" * 58)
        print("🚀 启动矢量辐射传输方程求解")
        print(f"   κ_e = {self.kappa_e:.4f} m^-1")
        print(f"   离散流数 n_streams = {self.n_streams}")
        print(f"   傅里叶阶数 M = {self.fourier_order}")
        print(
            f"   入射角 θ0 = {solar_zenith_deg:.2f}°, "
            f"μ0_internal = {mu0_internal:.6f}, |μ0| = {mu0_abs:.6f}"
        )
        print("=" * 58)

        mode_0 = self._solve_single_mode_surface(
            p_mode=self.phase_m0,
            p_incident_mode=p0_inc,
            mode_order=0,
            mu0=mu0_internal,
            incident_stokes=incident_stokes,
        )

        mode_cos: list[np.ndarray] = []
        mode_sin: list[np.ndarray] = []

        for m_idx in range(self.fourier_order):
            m_order = m_idx + 1
            print(f"🔄 正在求解傅里叶 m={m_order} 的 cos/sin 模态...")

            m_cos = self._solve_single_mode_surface(
                p_mode=self.phase_m_cos[m_idx],
                p_incident_mode=pcos_inc[m_idx],
                mode_order=m_order,
                mu0=mu0_internal,
                incident_stokes=incident_stokes,
            )
            m_sin = self._solve_single_mode_surface(
                p_mode=self.phase_m_sin[m_idx],
                p_incident_mode=psin_inc[m_idx],
                mode_order=m_order,
                mu0=mu0_internal,
                incident_stokes=incident_stokes,
            )

            mode_cos.append(m_cos)
            mode_sin.append(m_sin)

        self.last_solution = RTESolution(
            mu0=mu0_abs,
            phi0=phi0,
            incident_stokes=incident_stokes,
            mode_0=mode_0,
            mode_cos=mode_cos,
            mode_sin=mode_sin,
        )

        return self.last_solution

    # ========================
    # 表面辐射重建
    # ========================
    def _interp_mode_by_mu(self, mode_values: np.ndarray, mu_target: float) -> np.ndarray:
        """在离散 mu 点上对某个模态进行线性插值。"""
        mu_clamped = float(np.clip(mu_target, -1.0, 1.0))
        out = np.zeros(4, dtype=float)
        for comp in range(4):
            out[comp] = np.interp(mu_clamped, self.mu_nodes, mode_values[:, comp])
        return out

    def evaluate_surface_stokes(
        self,
        mu_s: float,
        phi_s: float,
        solution: Optional[RTESolution] = None,
    ) -> np.ndarray:
        """根据傅里叶模态重建指定方向的 z=0 漫射 Stokes 向量。"""
        if solution is None:
            if self.last_solution is None:
                raise RuntimeError("请先调用 run_simulation()。")
            solution = self.last_solution

        delta_phi = float(phi_s - solution.phi0)
        stokes = self._interp_mode_by_mu(solution.mode_0, mu_s)

        for m_idx in range(self.fourier_order):
            m = m_idx + 1
            cos_term = np.cos(m * delta_phi)
            sin_term = np.sin(m * delta_phi)
            stokes += self._interp_mode_by_mu(solution.mode_cos[m_idx], mu_s) * cos_term
            stokes += self._interp_mode_by_mu(solution.mode_sin[m_idx], mu_s) * sin_term

        return stokes

    # ========================
    # 物理量导出接口
    # ========================
    def compute_brdf(
        self,
        mu_s: float,
        phi_s: float,
        solution: Optional[RTESolution] = None,
    ) -> float:
        """计算 BRDF：rho = (I1 + I2) / mu0。"""
        if solution is None:
            if self.last_solution is None:
                raise RuntimeError("请先调用 run_simulation()。")
            solution = self.last_solution

        stokes = self.evaluate_surface_stokes(mu_s=mu_s, phi_s=phi_s, solution=solution)
        return float((stokes[0] + stokes[1]) / solution.mu0)

    def compute_polarized_brdf(
        self,
        mu_s: float,
        phi_s: float,
        solution: Optional[RTESolution] = None,
    ) -> float:
        """计算偏振 BRDF。"""
        if solution is None:
            if self.last_solution is None:
                raise RuntimeError("请先调用 run_simulation()。")
            solution = self.last_solution

        stokes = self.evaluate_surface_stokes(mu_s=mu_s, phi_s=phi_s, solution=solution)
        numerator = np.sqrt(
            (stokes[0] - stokes[1]) ** 2 + stokes[2] ** 2 + stokes[3] ** 2
        )
        return float(numerator / solution.mu0)

    def compute_albedo(self, solution: Optional[RTESolution] = None) -> float:
        """
        计算半球反照率。

        利用完整方位角积分后仅 m=0 项保留的性质：
        alpha = 2pi * integral_{mu>0} rho_bar(mu) * mu dmu
        """
        if solution is None:
            if self.last_solution is None:
                raise RuntimeError("请先调用 run_simulation()。")
            solution = self.last_solution

        albedo = 0.0
        for idx in self.up_dir_indices:
            mu = self.mu_nodes[idx]
            w = self.mu_weights[idx]
            stokes_avg_phi = solution.mode_0[idx]
            brdf_avg_phi = (stokes_avg_phi[0] + stokes_avg_phi[1]) / solution.mu0
            albedo += w * brdf_avg_phi * mu

        albedo *= 2.0 * np.pi
        return float(max(albedo, 0.0))

    @staticmethod
    def compute_arf_from_brdf(brdf: np.ndarray, albedo: float) -> np.ndarray:
        """
        计算 ARF（论文定义）：
        ARF = BRF / albedo = pi * BRDF / albedo
        """
        safe_albedo = max(float(albedo), 1e-12)
        return np.pi * brdf / safe_albedo

    @staticmethod
    def integrate_albedo_from_brdf_grid(
        theta_samples_deg: np.ndarray,
        phi_samples_deg: np.ndarray,
        brdf: np.ndarray,
    ) -> float:
        """
        基于离散 BRDF 网格做半球积分计算 albedo。

        alpha = integral_0^{2pi} integral_0^{pi/2} BRDF(theta,phi) * cos(theta) * sin(theta) dtheta dphi
        """
        theta = np.deg2rad(np.asarray(theta_samples_deg, dtype=float))
        phi = np.deg2rad(np.asarray(phi_samples_deg, dtype=float))
        brdf_grid = np.asarray(brdf, dtype=float)

        mu = np.cos(theta)
        sin_theta = np.sin(theta)
        weight = mu[:, None] * sin_theta[:, None]
        integrand = np.maximum(brdf_grid, 0.0) * weight

        # 先沿 phi 积分，再沿 theta 积分
        phi_integrated = np.trapezoid(integrand, phi, axis=1)
        alpha = np.trapezoid(phi_integrated, theta, axis=0)
        return float(max(alpha, 0.0))

    def export_reflectance_products(
        self,
        theta_samples_deg: Optional[np.ndarray] = None,
        phi_samples_deg: Optional[np.ndarray] = None,
        output_npz_path: Optional[str] = None,
        solution: Optional[RTESolution] = None,
    ) -> dict[str, np.ndarray | float]:
        """
        导出 BRDF / 偏振BRDF / albedo / ARF 网格结果。
        """
        if solution is None:
            if self.last_solution is None:
                raise RuntimeError("请先调用 run_simulation()。")
            solution = self.last_solution

        if theta_samples_deg is None:
            theta_samples_deg = np.linspace(0.0, 85.0, 36)
        if phi_samples_deg is None:
            phi_samples_deg = np.linspace(0.0, 360.0, 73)

        theta_samples_deg = np.asarray(theta_samples_deg, dtype=float)
        phi_samples_deg = np.asarray(phi_samples_deg, dtype=float)

        brdf = np.zeros((theta_samples_deg.size, phi_samples_deg.size), dtype=float)
        pol_brdf = np.zeros_like(brdf)

        for i, theta_deg in enumerate(theta_samples_deg):
            mu_s = float(np.cos(np.deg2rad(theta_deg)))
            if mu_s <= 0.0:
                continue
            for j, phi_deg in enumerate(phi_samples_deg):
                phi_rad = float(np.deg2rad(phi_deg))
                stokes = self.evaluate_surface_stokes(mu_s=mu_s, phi_s=phi_rad, solution=solution)
                brdf_val = (stokes[0] + stokes[1]) / solution.mu0
                pol_val = np.sqrt((stokes[0] - stokes[1]) ** 2 + stokes[2] ** 2 + stokes[3] ** 2) / solution.mu0
                brdf[i, j] = max(float(brdf_val), 0.0)
                pol_brdf[i, j] = max(float(pol_val), 0.0)

        # 优先使用与导出 BRDF 一致的离散半球积分口径，避免 mode_0 近似带来的符号误差。
        albedo_from_grid = self.integrate_albedo_from_brdf_grid(
            theta_samples_deg=theta_samples_deg,
            phi_samples_deg=phi_samples_deg,
            brdf=brdf,
        )
        albedo_from_mode0 = self.compute_albedo(solution=solution)
        albedo = albedo_from_grid if albedo_from_grid > 0.0 else albedo_from_mode0
        arf = self.compute_arf_from_brdf(brdf=brdf, albedo=albedo)

        products: dict[str, np.ndarray | float] = {
            "theta_samples_deg": theta_samples_deg,
            "phi_samples_deg": phi_samples_deg,
            "brdf": brdf,
            "polarized_brdf": pol_brdf,
            "albedo": albedo,
            "albedo_from_grid": albedo_from_grid,
            "albedo_from_mode0": albedo_from_mode0,
            "arf": arf,
            "mu0": solution.mu0,
            "theta0_deg": float(np.rad2deg(np.arccos(solution.mu0))),
            "phi0_deg": float(np.rad2deg(solution.phi0)),
        }

        if output_npz_path:
            np.savez(output_npz_path, **products)
            print(f"💾 反射物理量结果已保存: {output_npz_path}")

        self.last_products = products
        return products

    # ========================
    # 可视化接口
    # ========================
    def plot_reflectance_products(
        self,
        products: Optional[dict[str, np.ndarray | float]] = None,
        output_prefix: str = "Results/rte_reflectance",
        show: bool = True,
    ) -> None:
        """
        可视化 BRDF、ARF 和偏振 BRDF。

        图风格参考论文展示习惯：
        - 使用极坐标展示方位角/观测角分布
        - 使用二维曲线展示主平面上的方向特征
        """
        if products is None:
            if self.last_products is None:
                raise RuntimeError("请先调用 export_reflectance_products()。")
            products = self.last_products

        try:
            import matplotlib.pyplot as plt
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
            plt.rcParams["axes.unicode_minus"] = False  # 避免负号显示成方块

        except Exception as exc:
            raise RuntimeError("当前环境无法导入 matplotlib，无法进行可视化。") from exc

        # 读取数据并确保类型正确
        theta_deg = np.asarray(products["theta_samples_deg"], dtype=float)
        phi_deg = np.asarray(products["phi_samples_deg"], dtype=float)
        brdf = np.asarray(products["brdf"], dtype=float)
        arf = np.asarray(products["arf"], dtype=float)
        pol_brdf = np.asarray(products["polarized_brdf"], dtype=float)
        albedo = float(products["albedo"])
        theta0_deg = float(products["theta0_deg"])

        phi_rad = np.deg2rad(phi_deg)
        phi_mesh, theta_mesh = np.meshgrid(phi_rad, theta_deg)

        # 创建多子图
        fig = plt.figure(figsize=(15, 11))
        fig.suptitle(
            f"RTE 求解反射率分布 (太阳天顶角方向$\\theta_0$={theta0_deg:.1f}°, albedo={albedo:.4f})",
            fontsize=13,
        )

        ax1 = fig.add_subplot(2, 2, 1, projection="polar")
        c1 = ax1.contourf(phi_mesh, theta_mesh, brdf, levels=40, cmap="viridis")
        ax1.set_title("BRDF")
        ax1.set_theta_zero_location("N")
        ax1.set_theta_direction(-1)
        ax1.set_rlim(theta_deg.min(), theta_deg.max())
        plt.colorbar(c1, ax=ax1, pad=0.12, shrink=0.9)

        ax2 = fig.add_subplot(2, 2, 2, projection="polar")
        c2 = ax2.contourf(phi_mesh, theta_mesh, arf, levels=40, cmap="plasma")
        ax2.set_title("ARF = $\\frac{\\pi \\cdot BRDF}{Albedo}$")
        ax2.set_theta_zero_location("N")
        ax2.set_theta_direction(-1)
        ax2.set_rlim(theta_deg.min(), theta_deg.max())
        plt.colorbar(c2, ax=ax2, pad=0.12, shrink=0.9)

        ax3 = fig.add_subplot(2, 2, 3, projection="polar")
        c3 = ax3.contourf(phi_mesh, theta_mesh, pol_brdf, levels=40, cmap="magma")
        ax3.set_title("Polarized BRDF")
        ax3.set_theta_zero_location("N")
        ax3.set_theta_direction(-1)
        ax3.set_rlim(theta_deg.min(), theta_deg.max())
        plt.colorbar(c3, ax=ax3, pad=0.12, shrink=0.9)

        ax4 = fig.add_subplot(2, 2, 4)
        principal_idx = np.argmin(np.abs(phi_deg - 0.0))
        opposite_idx = np.argmin(np.abs(phi_deg - 180.0))
        ax4.plot(theta_deg, brdf[:, principal_idx], label="BRDF (方位角$\\phi=0°$)", linewidth=1.8)
        ax4.plot(theta_deg, brdf[:, opposite_idx], label="BRDF (方位角$\\phi=180°$)", linewidth=1.8)
        ax4.plot(theta_deg, arf[:, principal_idx], "--", label="ARF (方位角$\\phi=0°$)", linewidth=1.4)
        ax4.plot(theta_deg, arf[:, opposite_idx], "--", label="ARF (方位角$\\phi=180°$)", linewidth=1.4)
        ax4.set_xlabel("Viewing Zenith Angle (deg)")
        ax4.set_ylabel("Value")
        ax4.grid(True, linestyle=":", alpha=0.6)
        ax4.legend()
        ax4.set_title("Principal Plane Profiles")

        plt.tight_layout()

        png_path = f"{output_prefix}.png"
        fig.savefig(png_path, dpi=180)
        print(f"📊 可视化已保存: {png_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def compute_albedo_vs_sza(
        self,
        solar_zenith_deg_samples: np.ndarray,
        solar_azimuth_deg: float = 0.0,
        incident_stokes: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算 albedo-入射天顶角关系曲线。"""
        sza = np.asarray(solar_zenith_deg_samples, dtype=float)
        albedo_values = np.zeros_like(sza)

        for i, theta0 in enumerate(sza):
            sol = self.run_simulation(
                solar_zenith_deg=float(theta0),
                solar_azimuth_deg=solar_azimuth_deg,
                incident_stokes=incident_stokes,
            )
            albedo_values[i] = self.compute_albedo(solution=sol)

        return sza, albedo_values

    def plot_albedo_vs_sza(
        self,
        solar_zenith_deg_samples: np.ndarray,
        output_path: str = "Results/rte_albedo_vs_sza.png",
        show: bool = True,
    ) -> None:
        """绘制 albedo 随入射天顶角变化曲线。"""
        try:
            import matplotlib.pyplot as plt
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
            plt.rcParams["axes.unicode_minus"] = False  # 避免负号显示成方块
        except Exception as exc:
            raise RuntimeError("当前环境无法导入 matplotlib，无法进行可视化。") from exc

        sza, albedo = self.compute_albedo_vs_sza(solar_zenith_deg_samples=solar_zenith_deg_samples)

        plt.figure(figsize=(7.6, 5.2))
        plt.plot(sza, albedo, "o-", linewidth=1.8)
        plt.xlabel("Solar Zenith Angle (deg)")
        plt.ylabel("Albedo")
        plt.title("Albedo vs Solar Zenith Angle")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path, dpi=180)
        print(f"📈 Albedo 曲线已保存: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    # ========================
    # 调试与摘要
    # ========================
    def summarize(self, products: Optional[dict[str, Any]] = None) -> dict[str, float]:
        """返回便于日志记录的关键物理量摘要。"""
        if products is None:
            if self.last_products is None:
                raise RuntimeError("请先调用 export_reflectance_products()。")
            products = self.last_products

        brdf = np.asarray(products["brdf"], dtype=float)
        arf = np.asarray(products["arf"], dtype=float)
        pol = np.asarray(products["polarized_brdf"], dtype=float)

        return {
            "albedo": float(products["albedo"]),
            "brdf_max": float(np.max(brdf)),
            "brdf_min": float(np.min(brdf)),
            "arf_max": float(np.max(arf)),
            "arf_min": float(np.min(arf)),
            "polarized_brdf_max": float(np.max(pol)),
        }
