"""
A rocket powered landing with successive convexification

author: Sven Niederberger
        Atsushi Sakai

Reference:
- Python implementation of 'Successive Convexification for 6-DoF Mars Rocket Powered Landing with Free-Final-Time' paper
by Michael Szmuk and Behcet Acıkmese.

- EmbersArc/SuccessiveConvexificationFreeFinalTime: Implementation of "Successive Convexification for 6-DoF Mars Rocket Powered Landing with Free-Final-Time" https://github.com/EmbersArc/SuccessiveConvexificationFreeFinalTime

"""
import warnings
from time import time, perf_counter_ns
import numpy as np
from scipy.integrate import odeint, solve_ivp
import cvxpy
import matplotlib.pyplot as plt
from functools import lru_cache, wraps
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List, Tuple
from enum import Enum

# Trajectory points
K = 50

# Max solver iterations
iterations = 30

# Weight constants
W_SIGMA = 1  # flight time
W_DELTA = 1e-3  # difference in state/input
W_DELTA_SIGMA = 1e-1  # difference in flight time
W_NU = 1e5  # virtual control

print(cvxpy.installed_solvers())
solver = 'ECOS'
verbose_solver = False

show_animation = True


class ConvergenceMode(Enum):
    STRICT = 1
    RELAXED = 2
    ADAPTIVE = 3


@dataclass
class OptimizationMetrics:
    iteration: int
    delta_norm: float
    sigma_norm: float
    nu_norm: float
    solve_time: float
    converged: bool = False


def timing_decorator(threshold_ms: float = 10.0):
    """Сложный декоратор с замыканием для замера времени выполнения (строки 57-69)"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = perf_counter_ns()
            result = func(*args, **kwargs)
            elapsed_ms = (perf_counter_ns() - start) / 1e6
            
            if elapsed_ms > threshold_ms:
                print(f"🕒 {func.__name__} took {elapsed_ms:.2f} ms (exceeded {threshold_ms} ms)")
            
            # Нетипичное: побочный эффект в декораторе - модификация аргументов
            if kwargs and 'verbose' in kwargs and kwargs['verbose']:
                kwargs['verbose'] = False  # Необычное изменение флага
                
            return result
        return wrapper
    return decorator


class Rocket_Model_6DoF:
    """
    A 6 degree of freedom rocket landing problem.
    """

    def __init__(self, rng):
        """
        A large r_scale for a small scale problem will
        lead to numerical problems as parameters become excessively small
        and (it seems) precision is lost in the dynamics.
        """
        self.n_x = 14
        self.n_u = 3

        # Mass
        self.m_wet = 3.0  # 30000 kg
        self.m_dry = 2.2  # 22000 kg

        # Flight time guess
        self.t_f_guess = 10.0  # 10 s

        # State constraints
        self.r_I_final = np.array((0., 0., 0.))
        self.v_I_final = np.array((-1e-1, 0., 0.))
        self.q_B_I_final = self.euler_to_quat((0, 0, 0))
        self.w_B_final = np.deg2rad(np.array((0., 0., 0.)))

        self.w_B_max = np.deg2rad(60)

        # Angles
        max_gimbal = 20
        max_angle = 90
        glidelslope_angle = 20

        self.tan_delta_max = np.tan(np.deg2rad(max_gimbal))
        self.cos_theta_max = np.cos(np.deg2rad(max_angle))
        self.tan_gamma_gs = np.tan(np.deg2rad(glidelslope_angle))

        # Thrust limits
        self.T_max = 5.0
        self.T_min = 0.3

        # Angular moment of inertia
        self.J_B = 1e-2 * np.diag([1., 1., 1.])

        # Gravity
        self.g_I = np.array((-1, 0., 0.))

        # Fuel consumption
        self.alpha_m = 0.01

        # Vector from thrust point to CoM
        self.r_T_B = np.array([-1e-2, 0., 0.])

        # Кэш для матриц (нетипичное для этого кода) - строка 122
        self._matrix_cache: Dict[str, np.ndarray] = {}
        self._cache_enabled = True

        self.set_random_initial_state(rng)

        self.x_init = np.concatenate(
            ((self.m_wet,), self.r_I_init, self.v_I_init, self.q_B_I_init, self.w_B_init))
        self.x_final = np.concatenate(
            ((self.m_dry,), self.r_I_final, self.v_I_final, self.q_B_I_final, self.w_B_final))

        self.r_scale = np.linalg.norm(self.r_I_init)
        self.m_scale = self.m_wet

    def set_random_initial_state(self, rng):
        if rng is None:
            rng = np.random.default_rng()

        # Сложная вложенность с условиями (строки 133-152)
        def compute_initial_component(base_val: float, 
                                    spread: float,
                                    constraint_func: Optional[Callable] = None) -> float:
            """Рекурсивная функция с замыканием для вычисления начальных условий"""
            val = base_val + rng.uniform(-spread, spread)
            
            if constraint_func:
                max_attempts = 5
                for attempt in range(max_attempts):
                    if constraint_func(val):
                        return val
                    val = base_val + rng.uniform(-spread, spread)
                
                # Необычный паттерн: fallback с лямбда-выражением
                fallback = (lambda x: x * 0.5 if x > 0 else x * 2.0)(base_val)
                return fallback if not constraint_func or constraint_func(fallback) else base_val
            return val

        self.r_I_init = np.array((0., 0., 0.))
        self.r_I_init[0] = compute_initial_component(3.5, 0.5, lambda x: 3 <= x <= 4)
        
        # Сложная генерация с использованием map и filter (строка 150)
        init_vals = list(map(
            lambda idx: compute_initial_component(0, 2, lambda v: -2 <= v <= 2),
            range(2)
        ))
        self.r_I_init[1:3] = np.array(init_vals)

        self.v_I_init = np.array((0., 0., 0.))
        self.v_I_init[0] = compute_initial_component(-0.75, 0.25, lambda x: -1 <= x <= -0.5)
        
        # Нетипичное: использование генератора с условием
        v_components = (self.r_I_init[i] * rng.uniform(-0.5, -0.2) 
                       for i in range(1, 3))
        self.v_I_init[1:3] = np.fromiter(v_components, dtype=float)

        self.q_B_I_init = self.euler_to_quat((0,
                                              rng.uniform(-30, 30),
                                              rng.uniform(-30, 30)))
        self.w_B_init = np.deg2rad((0,
                                    rng.uniform(-20, 20),
                                    rng.uniform(-20, 20)))

    @timing_decorator(threshold_ms=1.0)
    def f_func(self, x, u):
        """Декорированная версия с замером времени"""
        cache_key = f"f_{hash(x.tobytes())}_{hash(u.tobytes())}"
        if self._cache_enabled and cache_key in self._matrix_cache:
            return self._matrix_cache[cache_key]

        m, _, _, _, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz = x[0], x[1], x[
            2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]
        ux, uy, uz = u[0], u[1], u[2]

        result = np.array([
            [-0.01 * np.sqrt(ux**2 + uy**2 + uz**2)],
            [vx],
            [vy],
            [vz],
            [(-1.0 * m - ux * (2 * q2**2 + 2 * q3**2 - 1) - 2 * uy
              * (q0 * q3 - q1 * q2) + 2 * uz * (q0 * q2 + q1 * q3)) / m],
            [(2 * ux * (q0 * q3 + q1 * q2) - uy * (2 * q1**2
                                                   + 2 * q3**2 - 1) - 2 * uz * (q0 * q1 - q2 * q3)) / m],
            [(-2 * ux * (q0 * q2 - q1 * q3) + 2 * uy
              * (q0 * q1 + q2 * q3) - uz * (2 * q1**2 + 2 * q2**2 - 1)) / m],
            [-0.5 * q1 * wx - 0.5 * q2 * wy - 0.5 * q3 * wz],
            [0.5 * q0 * wx + 0.5 * q2 * wz - 0.5 * q3 * wy],
            [0.5 * q0 * wy - 0.5 * q1 * wz + 0.5 * q3 * wx],
            [0.5 * q0 * wz + 0.5 * q1 * wy - 0.5 * q2 * wx],
            [0],
            [1.0 * uz],
            [-1.0 * uy]
        ])
        
        if self._cache_enabled:
            self._matrix_cache[cache_key] = result
            # Ограничение размера кэша (необычно для этого кода)
            if len(self._matrix_cache) > 1000:
                self._matrix_cache.clear()
        
        return result

    def A_func(self, x, u):
        cache_key = f"A_{hash(x.tobytes())}_{hash(u.tobytes())}"
        if self._cache_enabled and cache_key in self._matrix_cache:
            return self._matrix_cache[cache_key]

        m, _, _, _, _, _, _, q0, q1, q2, q3, wx, wy, wz = x[0], x[1], x[
            2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]
        ux, uy, uz = u[0], u[1], u[2]

        result = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [(ux * (2 * q2**2 + 2 * q3**2 - 1) + 2 * uy * (q0 * q3 - q1 * q2) - 2 * uz * (q0 * q2 + q1 * q3)) / m**2, 0, 0, 0, 0, 0, 0, 2 * (q2 * uz
                                                                                                                                             - q3 * uy) / m, 2 * (q2 * uy + q3 * uz) / m, 2 * (q0 * uz + q1 * uy - 2 * q2 * ux) / m, 2 * (-q0 * uy + q1 * uz - 2 * q3 * ux) / m, 0, 0, 0],
            [(-2 * ux * (q0 * q3 + q1 * q2) + uy * (2 * q1**2 + 2 * q3**2 - 1) + 2 * uz * (q0 * q1 - q2 * q3)) / m**2, 0, 0, 0, 0, 0, 0, 2 * (-q1 * uz
                                                                                                                                              + q3 * ux) / m, 2 * (-q0 * uz - 2 * q1 * uy + q2 * ux) / m, 2 * (q1 * ux + q3 * uz) / m, 2 * (q0 * ux + q2 * uz - 2 * q3 * uy) / m, 0, 0, 0],
            [(2 * ux * (q0 * q2 - q1 * q3) - 2 * uy * (q0 * q1 + q2 * q3) + uz * (2 * q1**2 + 2 * q2**2 - 1)) / m**2, 0, 0, 0, 0, 0, 0, 2 * (q1 * uy
                                                                                                                                             - q2 * ux) / m, 2 * (q0 * uy - 2 * q1 * uz + q3 * ux) / m, 2 * (-q0 * ux - 2 * q2 * uz + q3 * uy) / m, 2 * (q1 * ux + q2 * uy) / m, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -0.5 * wx, -0.5 * wy,
             - 0.5 * wz, -0.5 * q1, -0.5 * q2, -0.5 * q3],
            [0, 0, 0, 0, 0, 0, 0, 0.5 * wx, 0, 0.5 * wz,
             - 0.5 * wy, 0.5 * q0, -0.5 * q3, 0.5 * q2],
            [0, 0, 0, 0, 0, 0, 0, 0.5 * wy, -0.5 * wz, 0,
             0.5 * wx, 0.5 * q3, 0.5 * q0, -0.5 * q1],
            [0, 0, 0, 0, 0, 0, 0, 0.5 * wz, 0.5 * wy,
             - 0.5 * wx, 0, -0.5 * q2, 0.5 * q1, 0.5 * q0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        
        if self._cache_enabled:
            self._matrix_cache[cache_key] = result
        
        return result

    def B_func(self, x, u):
        cache_key = f"B_{hash(x.tobytes())}_{hash(u.tobytes())}"
        if self._cache_enabled and cache_key in self._matrix_cache:
            return self._matrix_cache[cache_key]

        m, _, _, _, _, _, _, q0, q1, q2, q3, _, _, _ = x[0], x[1], x[
            2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]
        ux, uy, uz = u[0], u[1], u[2]

        result = np.array([
            [-0.01 * ux / np.sqrt(ux**2 + uy**2 + uz**2),
             -0.01 * uy / np.sqrt(ux ** 2 + uy**2 + uz**2),
             -0.01 * uz / np.sqrt(ux**2 + uy**2 + uz**2)],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [(-2 * q2**2 - 2 * q3**2 + 1) / m, 2
             * (-q0 * q3 + q1 * q2) / m, 2 * (q0 * q2 + q1 * q3) / m],
            [2 * (q0 * q3 + q1 * q2) / m, (-2 * q1**2 - 2
                                           * q3**2 + 1) / m, 2 * (-q0 * q1 + q2 * q3) / m],
            [2 * (-q0 * q2 + q1 * q3) / m, 2 * (q0 * q1 + q2 * q3)
             / m, (-2 * q1**2 - 2 * q2**2 + 1) / m],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1.0],
            [0, -1.0, 0]
        ])
        
        if self._cache_enabled:
            self._matrix_cache[cache_key] = result
        
        return result

    def euler_to_quat(self, a):
        a = np.deg2rad(a)

        cy = np.cos(a[1] * 0.5)
        sy = np.sin(a[1] * 0.5)
        cr = np.cos(a[0] * 0.5)
        sr = np.sin(a[0] * 0.5)
        cp = np.cos(a[2] * 0.5)
        sp = np.sin(a[2] * 0.5)

        q = np.zeros(4)

        q[0] = cy * cr * cp + sy * sr * sp
        q[1] = cy * sr * cp - sy * cr * sp
        q[3] = cy * cr * sp + sy * sr * cp
        q[2] = sy * cr * cp - cy * sr * sp

        return q

    def skew(self, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def dir_cosine(self, q):
        return np.array([
            [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2]
                                                   + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])],
            [2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2
             * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])],
            [2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3]
                                                   - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]
        ])

    def omega(self, w):
        return np.array([
            [0, -w[0], -w[1], -w[2]],
            [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]],
            [w[2], w[1], -w[0], 0],
        ])

    def initialize_trajectory(self, X, U):
        """
        Initialize the trajectory with linear approximation.
        """
        K = X.shape[1]

        # Сложная рекурсивная функция для интерполяции (нетипично для этого кода) - строка 335
        def recursive_interpolate(k_idx, depth=0, max_depth=3):
            """Рекурсивная интерполяция с ограничением глубины"""
            if depth > max_depth or k_idx == 0 or k_idx == K-1:
                alpha1 = (K - k_idx) / K
                alpha2 = k_idx / K
                
                m_k = (alpha1 * self.x_init[0] + alpha2 * self.x_final[0],)
                r_I_k = alpha1 * self.x_init[1:4] + alpha2 * self.x_final[1:4]
                v_I_k = alpha1 * self.x_init[4:7] + alpha2 * self.x_final[4:7]
                q_B_I_k = np.array([1, 0, 0, 0])
                w_B_k = alpha1 * self.x_init[11:14] + alpha2 * self.x_final[11:14]
                
                return np.concatenate((m_k, r_I_k, v_I_k, q_B_I_k, w_B_k)), m_k * -self.g_I
            else:
                # Рекурсивное вычисление
                left_state, left_control = recursive_interpolate(k_idx-1, depth+1, max_depth)
                right_state, right_control = recursive_interpolate(k_idx+1, depth+1, max_depth)
                
                # Необычное усреднение с весами
                weight_left = 0.7 - 0.1 * depth
                weight_right = 0.3 + 0.1 * depth
                
                state_avg = weight_left * left_state + weight_right * right_state
                control_avg = weight_left * left_control + weight_right * right_control
                
                return state_avg, control_avg

        for k in range(K):
            if k % 5 == 0:  # Каждый 5-й шаг используем рекурсивную интерполяцию
                X[:, k], U[:, k] = recursive_interpolate(k)
            else:
                alpha1 = (K - k) / K
                alpha2 = k / K

                m_k = (alpha1 * self.x_init[0] + alpha2 * self.x_final[0],)
                r_I_k = alpha1 * self.x_init[1:4] + alpha2 * self.x_final[1:4]
                v_I_k = alpha1 * self.x_init[4:7] + alpha2 * self.x_final[4:7]
                q_B_I_k = np.array([1, 0, 0, 0])
                w_B_k = alpha1 * self.x_init[11:14] + alpha2 * self.x_final[11:14]

                X[:, k] = np.concatenate((m_k, r_I_k, v_I_k, q_B_I_k, w_B_k))
                U[:, k] = m_k * -self.g_I

        return X, U

    def get_constraints(self, X_v, U_v, X_last_p, U_last_p):
        """
        Get model specific constraints.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :return: A list of cvx constraints
        """
        # Boundary conditions:
        constraints = [
            X_v[0, 0] == self.x_init[0],
            X_v[1:4, 0] == self.x_init[1:4],
            X_v[4:7, 0] == self.x_init[4:7],
            # X_v[7:11, 0] == self.x_init[7:11],  # initial orientation is free
            X_v[11:14, 0] == self.x_init[11:14],

            # X_[0, -1] final mass is free
            X_v[1:, -1] == self.x_final[1:],
            U_v[1:3, -1] == 0,
        ]

        constraints += [
            # State constraints:
            X_v[0, :] >= self.m_dry,  # minimum mass
            cvxpy.norm(X_v[2: 4, :], axis=0) <= X_v[1, :] / \
            self.tan_gamma_gs,  # glideslope
            cvxpy.norm(X_v[9:11, :], axis=0) <= np.sqrt(
                (1 - self.cos_theta_max) / 2),  # maximum angle
            # maximum angular velocity
            cvxpy.norm(X_v[11: 14, :], axis=0) <= self.w_B_max,

            # Control constraints:
            cvxpy.norm(U_v[1:3, :], axis=0) <= self.tan_delta_max * \
            U_v[0, :],  # gimbal angle constraint
            cvxpy.norm(U_v, axis=0) <= self.T_max,  # upper thrust constraint
        ]

        # linearized lower thrust constraint
        # Сложная генерация списка с условием (строка 408)
        rhs = [
            (U_last_p[:, k] / cvxpy.norm(U_last_p[:, k]) @ U_v[:, k]
             if cvxpy.norm(U_last_p[:, k]).value > 1e-6
             else self.T_min * 0.9)  # Fallback значение
            for k in range(X_v.shape[1])
        ]
        constraints += [
            self.T_min <= cvxpy.vstack(rhs)
        ]

        return constraints


class Integrator:
    def __init__(self, m, K):
        self.K = K
        self.m = m
        self.n_x = m.n_x
        self.n_u = m.n_u

        self.A_bar = np.zeros([m.n_x * m.n_x, K - 1])
        self.B_bar = np.zeros([m.n_x * m.n_u, K - 1])
        self.C_bar = np.zeros([m.n_x * m.n_u, K - 1])
        self.S_bar = np.zeros([m.n_x, K - 1])
        self.z_bar = np.zeros([m.n_x, K - 1])

        # vector indices for flat matrices
        x_end = m.n_x
        A_bar_end = m.n_x * (1 + m.n_x)
        B_bar_end = m.n_x * (1 + m.n_x + m.n_u)
        C_bar_end = m.n_x * (1 + m.n_x + m.n_u + m.n_u)
        S_bar_end = m.n_x * (1 + m.n_x + m.n_u + m.n_u + 1)
        z_bar_end = m.n_x * (1 + m.n_x + m.n_u + m.n_u + 2)
        self.x_ind = slice(0, x_end)
        self.A_bar_ind = slice(x_end, A_bar_end)
        self.B_bar_ind = slice(A_bar_end, B_bar_end)
        self.C_bar_ind = slice(B_bar_end, C_bar_end)
        self.S_bar_ind = slice(C_bar_end, S_bar_end)
        self.z_bar_ind = slice(S_bar_end, z_bar_end)

        self.f, self.A, self.B = m.f_func, m.A_func, m.B_func

        # integration initial condition
        self.V0 = np.zeros((m.n_x * (1 + m.n_x + m.n_u + m.n_u + 2),))
        self.V0[self.A_bar_ind] = np.eye(m.n_x).reshape(-1)

        self.dt = 1. / (K - 1)
        
        # Нетипичное: кэш для результатов интегрирования
        self._integration_cache: Dict[Tuple[int, float], Any] = {}

    def calculate_discretization(self, X, U, sigma):
        """
        Calculate discretization for given states, inputs and total time.

        :param X: Matrix of states for all time points
        :param U: Matrix of inputs for all time points
        :param sigma: Total time
        :return: The discretization matrices
        """
        # Сложная проверка кэша с использованием хэшей (строка 470)
        cache_key = (hash(X.tobytes()) % 1000000, 
                    hash(U.tobytes()) % 1000000, 
                    int(sigma * 1000))
        
        if cache_key in self._integration_cache:
            print("📦 Using cached discretization")
            return self._integration_cache[cache_key]

        for k in range(self.K - 1):
            self.V0[self.x_ind] = X[:, k]
            
            # Необычный подход: использование solve_ivp как альтернативы odeint
            try:
                # Нетипичное: использование нескольких методов интегрирования
                if k % 10 == 0:
                    # Каждый 10-й шаг используем метод Рунге-Кутты 4-5 порядка
                    sol = solve_ivp(
                        lambda t, V: self._ode_dVdt(V, t, U[:, k], U[:, k + 1], sigma),
                        (0, self.dt),
                        self.V0,
                        method='RK45',
                        rtol=1e-6,
                        atol=1e-9
                    )
                    V = sol.y[:, -1]
                else:
                    # Обычный метод
                    V = np.array(odeint(self._ode_dVdt, self.V0, (0, self.dt),
                                        args=(U[:, k], U[:, k + 1], sigma))[1, :])
            except Exception as e:
                # Fallback на простую эйлерову схему (нетипичное резервное решение)
                print(f"⚠️ Integration warning at k={k}, using Euler fallback: {e}")
                V = self.V0 + self.dt * self._ode_dVdt(
                    self.V0, 0, U[:, k], U[:, k + 1], sigma)

            # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
            # flatten matrices in column-major (Fortran) order for CVXPY
            Phi = V[self.A_bar_ind].reshape((self.n_x, self.n_x))
            self.A_bar[:, k] = Phi.flatten(order='F')
            self.B_bar[:, k] = np.matmul(Phi, V[self.B_bar_ind].reshape(
                (self.n_x, self.n_u))).flatten(order='F')
            self.C_bar[:, k] = np.matmul(Phi, V[self.C_bar_ind].reshape(
                (self.n_x, self.n_u))).flatten(order='F')
            self.S_bar[:, k] = np.matmul(Phi, V[self.S_bar_ind])
            self.z_bar[:, k] = np.matmul(Phi, V[self.z_bar_ind])

        result = (self.A_bar, self.B_bar, self.C_bar, self.S_bar, self.z_bar)
        
        # Кэширование с ограничением размера
        if len(self._integration_cache) > 50:
            # Удаляем самый старый ключ (простая стратегия)
            oldest_key = next(iter(self._integration_cache))
            del self._integration_cache[oldest_key]
        
        self._integration_cache[cache_key] = result
        return result

    def _ode_dVdt(self, V, t, u_t0, u_t1, sigma):
        """
        ODE function to compute dVdt.

        :param V: Evaluation state V = [x, Phi_A, B_bar, C_bar, S_bar, z_bar]
        :param t: Evaluation time
        :param u_t0: Input at start of interval
        :param u_t1: Input at end of interval
        :param sigma: Total time
        :return: Derivative at current time and state dVdt
        """
        alpha = (self.dt - t) / self.dt
        beta = t / self.dt
        x = V[self.x_ind]
        u = u_t0 + beta * (u_t1 - u_t0)

        # using \Phi_A(\tau_{k+1},\xi) = \Phi_A(\tau_{k+1},\tau_k)\Phi_A(\xi,\tau_k)^{-1}
        # and pre-multiplying with \Phi_A(\tau_{k+1},\tau_k) after integration
        Phi_A_xi = np.linalg.inv(
            V[self.A_bar_ind].reshape((self.n_x, self.n_x)))

        A_subs = sigma * self.A(x, u)
        B_subs = sigma * self.B(x, u)
        f_subs = self.f(x, u)

        dVdt = np.zeros_like(V)
        dVdt[self.x_ind] = sigma * f_subs.transpose()
        dVdt[self.A_bar_ind] = np.matmul(
            A_subs, V[self.A_bar_ind].reshape((self.n_x, self.n_x))).reshape(-1)
        dVdt[self.B_bar_ind] = np.matmul(Phi_A_xi, B_subs).reshape(-1) * alpha
        dVdt[self.C_bar_ind] = np.matmul(Phi_A_xi, B_subs).reshape(-1) * beta
        dVdt[self.S_bar_ind] = np.matmul(Phi_A_xi, f_subs).transpose()
        z_t = -np.matmul(A_subs, x) - np.matmul(B_subs, u)
        dVdt[self.z_bar_ind] = np.dot(Phi_A_xi, z_t.T).flatten()

        return dVdt


class SCProblem:
    """
    Defines a standard Successive Convexification problem and
            adds the model specific constraints and objectives.

    :param m: The model object
    :param K: Number of discretization points
    """

    def __init__(self, m, K):
        # Variables:
        self.var = dict()
        self.var['X'] = cvxpy.Variable((m.n_x, K))
        self.var['U'] = cvxpy.Variable((m.n_u, K))
        self.var['sigma'] = cvxpy.Variable(nonneg=True)
        self.var['nu'] = cvxpy.Variable((m.n_x, K - 1))
        self.var['delta_norm'] = cvxpy.Variable(nonneg=True)
        self.var['sigma_norm'] = cvxpy.Variable(nonneg=True)
        
        # Нетипичное: дополнительные переменные для усложнения
        self.var['slack'] = cvxpy.Variable((K,), nonneg=True)

        # Parameters:
        self.par = dict()
        self.par['A_bar'] = cvxpy.Parameter((m.n_x * m.n_x, K - 1))
        self.par['B_bar'] = cvxpy.Parameter((m.n_x * m.n_u, K - 1))
        self.par['C_bar'] = cvxpy.Parameter((m.n_x * m.n_u, K - 1))
        self.par['S_bar'] = cvxpy.Parameter((m.n_x, K - 1))
        self.par['z_bar'] = cvxpy.Parameter((m.n_x, K - 1))

        self.par['X_last'] = cvxpy.Parameter((m.n_x, K))
        self.par['U_last'] = cvxpy.Parameter((m.n_u, K))
        self.par['sigma_last'] = cvxpy.Parameter(nonneg=True)

        self.par['weight_sigma'] = cvxpy.Parameter(nonneg=True)
        self.par['weight_delta'] = cvxpy.Parameter(nonneg=True)
        self.par['weight_delta_sigma'] = cvxpy.Parameter(nonneg=True)
        self.par['weight_nu'] = cvxpy.Parameter(nonneg=True)
        
        # Дополнительный параметр для усложнения
        self.par['relaxation_factor'] = cvxpy.Parameter(nonneg=True, value=1.0)

        # Constraints:
        constraints = []

        # Model:
        constraints += m.get_constraints(
            self.var['X'], self.var['U'], self.par['X_last'], self.par['U_last'])

        # Dynamics:
        # x_t+1 = A_*x_t+B_*U_t+C_*U_T+1*S_*sigma+zbar+nu
        # Сложная вложенность с дополнительными условиями (строка 596)
        dynamics_constraints = []
        for k in range(K - 1):
            lhs = self.var['X'][:, k + 1]
            rhs = (cvxpy.reshape(self.par['A_bar'][:, k], (m.n_x, m.n_x), order='F') @ self.var['X'][:, k] +
                   cvxpy.reshape(self.par['B_bar'][:, k], (m.n_x, m.n_u), order='F') @ self.var['U'][:, k] +
                   cvxpy.reshape(self.par['C_bar'][:, k], (m.n_x, m.n_u), order='F') @ self.var['U'][:, k + 1] +
                   self.par['S_bar'][:, k] * self.var['sigma'] +
                   self.par['z_bar'][:, k] +
                   self.var['nu'][:, k] +
                   self.var['slack'][k] * 0.01)  # Небольшое слабое слагаемое
            
            # Необычное: добавление условия с логическим оператором
            if k % 7 == 0:  # Каждое 7-е ограничение делаем более строгим
                dynamics_constraints.append(lhs == rhs)
            else:
                dynamics_constraints.append(cvxpy.norm(lhs - rhs, 2) <= self.var['slack'][k])
        
        constraints += dynamics_constraints

        # Trust regions:
        dx = cvxpy.sum(cvxpy.square(
            self.var['X'] - self.par['X_last']), axis=0)
        du = cvxpy.sum(cvxpy.square(
            self.var['U'] - self.par['U_last']), axis=0)
        ds = self.var['sigma'] - self.par['sigma_last']
        
        # Сложное условие с использованием нескольких норм
        trust_region_expr = (cvxpy.norm(dx + du, 1) * self.par['relaxation_factor'] + 
                           cvxpy.norm(self.var['slack'], 1) * 0.01)
        constraints += [trust_region_expr <= self.var['delta_norm']]
        
        constraints += [cvxpy.norm(ds, 'inf') <= self.var['sigma_norm']]

        # Flight time positive:
        constraints += [self.var['sigma'] >= 0.1]

        # Objective:
        sc_objective = cvxpy.Minimize(
            self.par['weight_sigma'] * self.var['sigma'] +
            self.par['weight_nu'] * cvxpy.norm(self.var['nu'], 'inf') +
            self.par['weight_delta'] * self.var['delta_norm'] +
            self.par['weight_delta_sigma'] * self.var['sigma_norm'] +
            cvxpy.norm(self.var['slack'], 1) * 0.001  # Штраф за слабые переменные
        )

        objective = sc_objective

        self.prob = cvxpy.Problem(objective, constraints)

    def set_parameters(self, **kwargs):
        """
        All parameters have to be filled before calling solve().
        Takes the following arguments as keywords:

        A_bar
        B_bar
        C_bar
        S_bar
        z_bar
        X_last
        U_last
        sigma_last
        E
        weight_sigma
        weight_nu
        radius_trust_region
        """

        # Нетипичное: использование getattr и setattr динамически
        for key, value in kwargs.items():
            if hasattr(self.par, key) if hasattr(self.par, '__getitem__') else key in self.par:
                self.par[key].value = value
            else:
                # Необычное: создание параметра на лету (осторожно!)
                if not hasattr(self, '_dynamic_params'):
                    self._dynamic_params = {}
                if key not in self._dynamic_params:
                    self._dynamic_params[key] = cvxpy.Parameter(value=value)
                print(f'⚠️ Dynamic parameter \'{key}\' created.')

    def get_variable(self, name):
        if name in self.var:
            return self.var[name].value
        else:
            # Рекурсивный поиск вложенных структур (усложнение)
            def recursive_search(obj, path):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        result = recursive_search(v, f"{path}.{k}")
                        if result is not None:
                            return result
                elif hasattr(obj, 'value'):
                    return obj.value
                return None
            
            result = recursive_search(self.var, 'var')
            if result is None:
                print(f'Variable \'{name}\' does not exist.')
            return result

    def solve(self, **kwargs):
        error = False
        try:
            with warnings.catch_warnings():  # For User warning from solver
                warnings.simplefilter('ignore')
                # Нетипичное: перехват и обработка разных исключений
                try:
                    self.prob.solve(verbose=verbose_solver,
                                    solver=solver,
                                    max_iters=200,
                                    feastol=1e-6,
                                    reltol=1e-5)
                except cvxpy.error.SolverError as e:
                    print(f"🔧 Solver error: {e}, trying with relaxed parameters")
                    # Пытаемся решить с другими параметрами
                    self.par['relaxation_factor'].value = 1.5
                    self.prob.solve(verbose=False,
                                   solver=solver,
                                   max_iters=300,
                                   feastol=1e-5)
        except cvxpy.SolverError:
            error = True

        stats = self.prob.solver_stats

        info = {
            'setup_time': stats.setup_time,
            'solver_time': stats.solve_time,
            'iterations': stats.num_iters,
            'solver_error': error,
            'status': self.prob.status
        }

        return info


def axis3d_equal(X, Y, Z, ax):

    max_range = np.array([X.max() - X.min(), Y.max()
                          - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                                    - 1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                                    - 1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2,
                                    - 1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')


def plot_animation(X, U):  # pragma: no cover

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # for stopping simulation with the esc key.
    fig.canvas.mpl_connect('key_release_event',
                           lambda event: [exit(0) if event.key == 'escape' else None])

    for k in range(K):
        plt.cla()
        ax.plot(X[2, :], X[3, :], X[1, :])  # trajectory
        ax.scatter3D([0.0], [0.0], [0.0], c="r",
                     marker="x")  # target landing point
        axis3d_equal(X[2, :], X[3, :], X[1, :], ax)

        rx, ry, rz = X[1:4, k]
        # vx, vy, vz = X[4:7, k]
        qw, qx, qy, qz = X[7:11, k]

        CBI = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy + qw * qz),
             2 * (qx * qz - qw * qy)],
            [2 * (qx * qy - qw * qz), 1 - 2
             * (qx ** 2 + qz ** 2), 2 * (qy * qz + qw * qx)],
            [2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx),
             1 - 2 * (qx ** 2 + qy ** 2)]
        ])

        Fx, Fy, Fz = np.dot(np.transpose(CBI), U[:, k])
        dx, dy, dz = np.dot(np.transpose(CBI), np.array([1., 0., 0.]))

        # attitude vector
        ax.quiver(ry, rz, rx, dy, dz, dx, length=0.5, linewidth=3.0,
                  arrow_length_ratio=0.0, color='black')

        # thrust vector
        ax.quiver(ry, rz, rx, -Fy, -Fz, -Fx, length=0.1,
                  arrow_length_ratio=0.0, color='red')

        ax.set_title("Rocket powered landing")
        plt.pause(0.5)


def main(rng=None):
    print("start!!")
    
    # Нетипичное: создание замыкания для адаптивного обновления весов
    def create_weight_updater(base_weight: float, mode: ConvergenceMode = ConvergenceMode.ADAPTIVE):
        """Фабрика функций для обновления весов с памятью"""
        history = []
        
        def update_weight(current_value: float, iteration: int, metrics: Dict[str, float]) -> float:
            history.append((iteration, current_value, metrics))
            
            if mode == ConvergenceMode.STRICT:
                return current_value * 1.5
            elif mode == ConvergenceMode.RELAXED:
                return current_value * (1.2 if iteration < 10 else 1.1)
            else:  # ADAPTIVE
                if len(history) < 3:
                    return current_value * 1.3
                
                # Сложная логика на основе истории
                recent_improvement = all(
                    history[-i][2]['delta_norm'] < history[-i-1][2]['delta_norm'] 
                    for i in range(1, min(3, len(history)))
                )
                
                if recent_improvement and metrics['delta_norm'] < 0.1:
                    return current_value * 1.7  # Ускоряем сходимость
                elif metrics['delta_norm'] > 0.5:
                    return current_value * 1.1  # Замедляем
                else:
                    return current_value * 1.5
        
        return update_weight

    m = Rocket_Model_6DoF(rng)

    # state and input list
    X = np.empty(shape=[m.n_x, K])
    U = np.empty(shape=[m.n_u, K])

    # INITIALIZATION
    sigma = m.t_f_guess
    X, U = m.initialize_trajectory(X, U)

    integrator = Integrator(m, K)
    problem = SCProblem(m, K)

    converged = False
    w_delta = W_DELTA
    
    # Использование замыкания для обновления весов
    update_w_delta = create_weight_updater(W_DELTA, ConvergenceMode.ADAPTIVE)
    
    metrics_history: List[OptimizationMetrics] = []

    for it in range(iterations):
        t0_it = time()
        print('-' * 18 + f' Iteration {str(it + 1).zfill(2)} ' + '-' * 18)

        A_bar, B_bar, C_bar, S_bar, z_bar = integrator.calculate_discretization(
            X, U, sigma)

        problem.set_parameters(A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, S_bar=S_bar, z_bar=z_bar,
                               X_last=X, U_last=U, sigma_last=sigma,
                               weight_sigma=W_SIGMA, weight_nu=W_NU,
                               weight_delta=w_delta, weight_delta_sigma=W_DELTA_SIGMA,
                               relaxation_factor=1.0 + 0.05 * it)  # Постепенное увеличение релаксации
        
        info = problem.solve()

        X = problem.get_variable('X')
        U = problem.get_variable('U')
        sigma = problem.get_variable('sigma')

        delta_norm = problem.get_variable('delta_norm')
        sigma_norm = problem.get_variable('sigma_norm')
        nu_norm = np.linalg.norm(problem.get_variable('nu'), np.inf)

        print('delta_norm', delta_norm)
        print('sigma_norm', sigma_norm)
        print('nu_norm', nu_norm)
        
        # Сохраняем метрики
        metrics = OptimizationMetrics(
            iteration=it,
            delta_norm=delta_norm,
            sigma_norm=sigma_norm,
            nu_norm=nu_norm,
            solve_time=info['solver_time']
        )
        metrics_history.append(metrics)

        # Сложное условие сходимости с несколькими критериями (строка 843)
        convergence_condition = (
            delta_norm < 1e-3 and 
            sigma_norm < 1e-3 and 
            nu_norm < 1e-7 and
            (it > 5 or (delta_norm < 5e-3 and all(m.delta_norm < 1e-2 for m in metrics_history[-3:])))
        ) or (
            it > 15 and 
            np.std([m.delta_norm for m in metrics_history[-5:]]) < 1e-4 and
            all(m.delta_norm < 2e-3 for m in metrics_history[-3:])
        )
        
        if convergence_condition:
            converged = True
            metrics_history[-1].converged = True

        # Адаптивное обновление веса
        w_delta = update_w_delta(w_delta, it, {
            'delta_norm': delta_norm,
            'sigma_norm': sigma_norm,
            'nu_norm': nu_norm
        })

        print('Time for iteration', time() - t0_it, 's')

        if converged:
            print(f'🎯 Converged after {it + 1} iterations.')
            
            # Необычное: вывод дополнительной статистики
            if len(metrics_history) > 1:
                avg_time = np.mean([m.solve_time for m in metrics_history])
                print(f'📊 Average solve time: {avg_time:.3f}s')
                print(f'📈 Final delta_norm improvement: '
                      f'{metrics_history[0].delta_norm / metrics_history[-1].delta_norm:.1f}x')
            
            break

    if not converged:
        print('⚠️ Did not converge within iteration limit')
        # Нетипичное: попытка последнего решения с релаксацией
        print('🔄 Trying final relaxed solve...')
        problem.par['relaxation_factor'].value = 2.0
        problem.solve()

    if show_animation:  # pragma: no cover
        plot_animation(X, U)

    print("done!!")


if __name__ == '__main__':
    # Нетипичное: запуск с разными сидами для тестирования
    seeds = [42, 123, 456]
    for i, seed in enumerate(seeds):
        if i > 0:
            print(f"\n{'='*50}")
            print(f"Running with seed {seed}")
            print('='*50)
        rng = np.random.default_rng(seed)
        main(rng)
