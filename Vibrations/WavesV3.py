from abc import ABC, abstractmethod
from manimlib import *
from itertools import product
import numpy as np
import copy
import cmath
import pickle
import hashlib
from scipy import integrate
from scipy.special import jn_zeros, jv, lpmn, spherical_jn
import scipy.optimize as opt
from scipy.interpolate import Rbf
from typing import Callable, List, Tuple, Optional, Union, Type
from QMVlib import * 

class WaveConfig:
    """Configuration settings for wave properties"""
    def __init__(
        self,
        length: float = 5.0,
        resolution: int = 10,
        dampening: float = 0.0,
        amplitude: float = 1.0,
        wave_speed: float = 1.0
    ):
        self.length = length
        self.resolution = resolution
        self.dampening = dampening
        self.amplitude = amplitude
        self.wave_speed = wave_speed


class WaveFunction:
    """Encapsulates wave function calculations"""
    @staticmethod
    def calculate_standing_wave(x: float, length: float, n: int) -> float:
        return np.sin(n * PI * x / length)
    
    @staticmethod
    def calculate_time_dependence(t: float, omega: complex) -> complex:
        return cmath.exp(1j * t * omega)


class WaveMobject(TipableVMobject, ABC):
    """Abstract base class for all 1D wave objects"""
    def __init__(self, config: WaveConfig, t_initial=0, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.start = LEFT * (config.length / 2)
        self.end = RIGHT * (config.length / 2)
        self.set_points_by_wave_function(t_initial)

    @abstractmethod
    def wave_function(self, x: float, t: float) -> float:
        pass

    def set_points_by_wave_function(self, t: float) -> None:
        self.clear_points()
        x_values = np.linspace(0, self.config.length, 
                             int(self.config.resolution * self.config.length))
        y_values = np.array([self.wave_function(x, t) for x in x_values])
        points = np.column_stack((x_values, y_values, np.zeros_like(x_values)))
        self.set_points_smoothly(points)
        self.put_start_and_end_on(self.start, self.end)

    def attach_value_tracker(self, tracker):
        self.add_updater(lambda m: m.set_points_by_wave_function(tracker.get_value())) #type: ignore 
        return self


class SummedWave(WaveMobject):
    def __init__(self, *wave_mobjects, config: WaveConfig, **kwargs):
        self.wave_funcs = [wave_mob.wave_function for wave_mob in wave_mobjects]
        super().__init__(config, **kwargs)

    def wave_function(self, x: float, t: float) -> float:
        return sum(wave_func(x, t) for wave_func in self.wave_funcs)


class StandingWave(WaveMobject):
    """A standing wave fixed at both ends"""
    def __init__(self, mode: int, config: WaveConfig, **kwargs):
        self.mode = mode
        self.k = self.mode * PI / config.length
        super().__init__(config, **kwargs)

    def wave_function(self, x: float, t: float) -> float:
        omega = self.config.wave_speed * self.k
        if self.config.dampening < 0.001:
            standing_wave = WaveFunction.calculate_standing_wave(x, self.config.length, self.mode)
            return self.config.amplitude * standing_wave * np.cos(omega * t)
        
        omega_damped = self._calculate_damped_omega(self.k, self.config.wave_speed)
        standing_wave = WaveFunction.calculate_standing_wave(x, self.config.length, self.mode)
        time_dependence = WaveFunction.calculate_time_dependence(t, omega_damped)
        return self.config.amplitude * (standing_wave * time_dependence).real

    def _calculate_damped_omega(self, k: float, c: float) -> complex:
        return cmath.sqrt((c * k)**2 - self.config.dampening**2) + self.config.dampening * 1j


class FourierAnalyzer:
    """Handles Fourier series calculations"""
    @staticmethod
    def calculate_coefficients(y_of_x_or_pts: Union[Callable, List[Tuple[float, float]]], 
                             N: int, length: float) -> List[float]:
        if callable(y_of_x_or_pts):
            return FourierAnalyzer._coefficients_from_function(y_of_x_or_pts, N, length)
        return FourierAnalyzer._coefficients_from_points(y_of_x_or_pts, N, length)

    @staticmethod
    def _coefficients_from_function(y_of_x: Callable, N: int, length: float) -> List[float]:
        return [
            (2 / length) * integrate.quad(
                lambda x: y_of_x(x) * np.sin((n * PI * x) / length), 
                0, length)[0]
            for n in range(1, N + 1)
        ]

    @staticmethod
    def _coefficients_from_points(points: List[Tuple[float, float]], 
                                N: int, length: float) -> List[float]:
        coefficients = []
        for n in range(1, N + 1):
            An = sum(FourierAnalyzer._segment_contribution(points[i], points[i + 1], n, length)
                    for i in range(len(points) - 1))
            coefficients.append(An)
        return coefficients

    @staticmethod
    def _segment_contribution(p1: Tuple[float, float], p2: Tuple[float, float], 
                            n: int, length: float) -> float:
        x1, y1 = p1
        x2, y2 = p2
        def indef_integral(x):
            pi_n = PI * n
            dx = x2 - x1
            dy = y2 - y1
            numerator = 2 * (length * dy * np.sin(pi_n * x / length) - 
                           pi_n * (y2 * (x - x1) - y1 * (x - x2)) * 
                           np.cos(pi_n * x / length))
            denominator = PI ** 2 * n ** 2 * dx
            return numerator / denominator
        return indef_integral(x2) - indef_integral(x1)


class FourierWave(WaveMobject):
    """A wave represented by its Fourier series"""
    def __init__(self, N: int, y_of_x_or_pts: Union[Callable, List[Tuple[float, float]]], 
                 config: WaveConfig, **kwargs):
        self.N = N
        self.coefficients = FourierAnalyzer.calculate_coefficients(
            y_of_x_or_pts, N, config.length)
        self.omegas = self._calculate_omegas(config)
        super().__init__(config, **kwargs)

    def _calculate_omegas(self, config: WaveConfig) -> List[complex]:
        if config.dampening < 0.001:
            return [PI * n * config.wave_speed / config.length for n in range(1, self.N + 1)]
        return [
            cmath.sqrt((config.wave_speed * n * PI / config.length)**2 - 
                      config.dampening**2) + config.dampening * 1j
            for n in range(1, self.N + 1)
        ]

    def wave_function(self, x: float, t: float) -> float:
        return sum(
            self.coefficients[n] * 
            WaveFunction.calculate_standing_wave(x, self.config.length, n + 1) * 
            (np.cos(self.omegas[n] * t) if self.config.dampening < 0.001 
             else WaveFunction.calculate_time_dependence(t, self.omegas[n]).real)
            for n in range(self.N)
        )


class TransverseWave(WaveMobject):
    """A wave that represents a static shape or function"""
    def __init__(self, y_of_x_or_pts: Union[Callable, List[Tuple[float, float]]], 
                 config: WaveConfig, **kwargs):
        self.y_of_x = self._create_wave_function(y_of_x_or_pts)
        super().__init__(config, **kwargs)

    def _create_wave_function(self, y_of_x_or_pts: Union[Callable, List[Tuple[float, float]]]) -> Callable:
        if callable(y_of_x_or_pts):
            return y_of_x_or_pts
        
        if not isinstance(y_of_x_or_pts, list) or not all(isinstance(item, tuple) for item in y_of_x_or_pts):
            raise ValueError("Input must be either a callable or a list of point tuples")
        
        def interpolate_points(x: float) -> float:
            points = y_of_x_or_pts
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                if x1 <= x <= x2:
                    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
            raise ValueError(f"x={x} is out of bounds of the provided points")
        
        return interpolate_points

    def wave_function(self, x: float, t: float) -> float:
        return self.y_of_x(x)


class WaveVisualizationConfig:
    """Configuration for wave visualization properties"""
    def __init__(self, num_waves: int, colors: Optional[List[str]] = None):
        self.colors = colors or [BLUE_B, BLUE_E]
        self.color_gradient = self._create_color_gradient(num_waves)
        
    def _create_color_gradient(self, num_waves: int) -> List[str]:
        return color_gradient(self.colors, num_waves)
    
    def get_wave_style(self, wave_index: int, total_waves: int, for_vmobject = False) -> dict:
        if for_vmobject:
            return {
                'stroke_color': self.color_gradient[wave_index],
                'stroke_width': DEFAULT_STROKE_WIDTH * (0.2 + (total_waves - wave_index) * (0.6 / total_waves)),
                'opacity': 0.2 + (total_waves - wave_index) * (0.4 / total_waves)
            }
        else:
            return {
                'color': self.color_gradient[wave_index],
                'opacity': 0.2 + (total_waves - wave_index) * (0.4 / total_waves)
            }


class FourierGroup(VGroup):
    """A group of waves showing Fourier decomposition"""
    def __init__(self, 
                 num_harmonics: int, 
                 wave_function: Union[Callable, List[Tuple[float, float]]], 
                 vis_config: Optional[WaveVisualizationConfig] = None,
                 wave_config: Optional[WaveConfig] = None,
                 **kwargs):
        super().__init__()
        
        self.wave_config = wave_config or WaveConfig(**kwargs)
        self.vis_config = vis_config or WaveVisualizationConfig(num_harmonics)
        
        # Create component waves
        self.transverse_wave = TransverseWave(wave_function, config=self.wave_config)
        self.fourier_wave = FourierWave(num_harmonics, wave_function, 
                                        config=self.wave_config)
        
        # Create individual harmonic waves
        self.harmonic_waves = self._create_harmonic_waves(num_harmonics)
        
        # Add all components to the group
        self.add(self.transverse_wave, self.fourier_wave, self.harmonic_waves)

    def _create_harmonic_waves(self, num_harmonics: int) -> VGroup:
        harmonic_waves = []
        for n in range(1, num_harmonics + 1):
            nth_wave_config = copy.deepcopy(self.wave_config)
            nth_wave_config.amplitude = self.fourier_wave.coefficients[n - 1]
            nth_wave_config.wave_speed = self.fourier_wave.omegas[n - 1] / (n * PI / nth_wave_config.length)
            harmonic_waves.append(
                StandingWave(
                    mode=n,
                    config=nth_wave_config,
                    **self.vis_config.get_wave_style(n - 1, num_harmonics)
                )
            )
        return VGroup(*harmonic_waves)

    def attach_value_tracker(self, tracker) -> FourierWave:
        """Attach a time tracker to animate all waves"""
        self.transverse_wave.attach_value_tracker(tracker)
        self.fourier_wave.attach_value_tracker(tracker)
        for wave in self.harmonic_waves:
            wave.attach_value_tracker(tracker)
        return self


class WaveAnimationScene(InteractiveScene):
    """Scene for demonstrating wave animations"""
    def construct(self):
        # Setup t_tracker
        t_tracker = ValueTracker(0)

        # Create and show standing wave
        wave_config = WaveConfig(dampening=0.2)
        wave = StandingWave(mode=2, config=wave_config, color=BLUE_B)
        self.play(ShowCreation(wave))
        wave.attach_value_tracker(t_tracker)
        self.play(
            t_tracker.animate.set_value(10),
            run_time=10,
            rate_func=linear
        )
        self.remove(wave)

        # Create fourier wave
        L = 10
        points_A = [(0, 0)]
        points_A += [(2, 3)]
        points_A += [(L, 0)]
        points_A = sorted(points_A)
        points_B = [(0, 0)]
        points_B += [
            (1, 2),
            (2, 3),
            (3, 0),
            (3.01, 2),
            (6, 2),
            (6.01, 0),
            (9, 1),
            (9.95, 3),
        ]
        points_B += [(L, 0)]
        points_B = sorted(points_B)
        example_func = lambda x: 1/3 * x * np.sin(x)
        chosen_example = example_func
        N = 25
        wave_config = WaveConfig(length=L, dampening=0.1, wave_speed=3)
        fourier_group = FourierGroup(
            num_harmonics=N,
            wave_function=chosen_example,
            wave_config=wave_config
        )
        
        # Show original wave
        self.play(ShowCreation(fourier_group.transverse_wave), run_time=1)
        self.wait()
        self.play(FadeOut(fourier_group.transverse_wave), run_time=0.5)
        
        # Show harmonic waves
        self.play(
            LaggedStart(*[
                ShowCreation(wave) 
                for wave in fourier_group.harmonic_waves
            ]),
            run_time=5,
            rate_func=linear
        )
        
        # Create intermediate waves for transformation
        intermediate_waves = [
            FourierWave(
                n, 
                chosen_example,
                config=wave_config
            ) 
            for n in range(1, N + 1)
        ]
        
        # Transform to final wave
        self.play(
            LaggedStart(
                TransformFromCopy(orig_wave, int_wave)
                for orig_wave, int_wave in zip(fourier_group.harmonic_waves, intermediate_waves)
            ),
            run_time=3,
            rate_func=slow_into
        )
        self.play(
            FadeOut(VGroup(intermediate_waves)),
            FadeIn(fourier_group.fourier_wave)
        )
        
        # Animate final wave
        t_tracker = ValueTracker(0)
        fourier_group.attach_value_tracker(t_tracker)
        self.play(
            t_tracker.animate.set_value(20),
            run_time=20,
            rate_func=linear
        )


class MembraneConfig:
    """Configuration settings for membrane properties"""
    def __init__(
        self,
        u_range: Tuple[float, float] = [0, 1],
        v_range: Tuple[float, float] = [0, 1],
        resolution: Tuple[int, int] = (10, 10),
        dampening: float = 0.0,
        amplitude: float = 1.0,
        phase: float = 0.0,
        wave_speed: float = 1.0
    ):
        self.u_range = u_range
        self.v_range = v_range
        self.resolution = resolution
        self.dampening = dampening
        self.amplitude = amplitude
        self.wave_speed = wave_speed
        self.phase = phase


class MembraneMesh(SurfaceMesh):
    def attach_value_tracker(self, tracker) -> None:
        def mesh_updater(mesh):
            mesh.uv_surface.set_uv_to_wave_function(tracker.get_value())
            new_mesh = SurfaceMesh(mesh.uv_surface, resolution=mesh.uv_surface.config.resolution).match_style(mesh)
            mesh.become(new_mesh)
        self.add_updater(mesh_updater)
        return self


class MembraneWaveMobject(ParametricSurface, ABC):
    """Abstract base class for all 2D wave objects"""
    def __init__(self, config: MembraneConfig, **kwargs):
        self.config = config
        self.mesh = None
        super().__init__(
            lambda u, v: self.wave_function(u, v, 0),
            u_range=self.config.u_range,
            v_range=self.config.v_range,
            resolution = self.config.resolution,
            **kwargs
        )

    def always_update_color_for_height(self, low_color, high_color):
        color_map = get_colormap_from_colors([low_color, high_color])
        self.add_updater(
            lambda m: m.set_color_by_rgba_func(lambda point: color_map([0.5 + point[2] / self.config.amplitude])[0])
        )

    @abstractmethod
    def spatial_component(self, u, v) -> float:
        """Calculate the time-independent spatial components of the wave function."""
        pass

    @abstractmethod
    def temporal_component(self, t) -> float:
        """Calculate the time-dependent component of the wave function."""
        pass

    def wave_function(self, u, v, t):
        x_val, y_val, z_spatial = self.spatial_component(u, v)
        time_factor = self.temporal_component(t)
        return x_val, y_val, z_spatial * time_factor

    @Mobject.affects_data
    def set_uv_to_wave_function(self, t: float) -> None:
        self.uv_func = lambda u, v: self.wave_function(u, v, t)
        super().init_points()

    def attach_value_tracker(self, tracker) -> None:
        self.add_updater(lambda m: m.set_uv_to_wave_function(tracker.get_value()))
        if self.mesh is not None:
            self.mesh.attach_value_tracker(tracker)
        return self

    def get_mesh(self, **kwargs) -> MembraneMesh:
        if self.mesh is None:
            self.mesh = MembraneMesh(
                self,
                resolution=self.config.resolution,
                **kwargs
            )
        return self.mesh


class StandingRectangularMembrane(MembraneWaveMobject):
    def __init__(self, m: int, n: int, width: float = 8, height: float = 10, config: Optional[MembraneConfig] = None, **kwargs):
        self.m = m
        self.n = n
        self.width = width
        self.height = height
        self.config = config or MembraneConfig()
        self.config.u_range = (-width / 2, width / 2)
        self.config.v_range = (-height / 2, height / 2)

        # Precompute constants
        self.c = self.config.wave_speed
        self.omegamn = PI * self.c * np.sqrt((n / width) ** 2 + (m / height) ** 2)
        if self.config.dampening > 0.001:
            betamn = self.config.dampening
            self.omegamn = np.sqrt(self.omegamn ** 2 - betamn ** 2)
            self.betamn = betamn
        else:
            self.betamn = 0
        self.kx = PI * n / width
        self.ky = PI * m / height
        self.phimn = self.config.phase
        self.Amn = self.config.amplitude
        
        super().__init__(self.config, **kwargs)

    def spatial_component(self, u, v):
        """Calculate the time-independent spatial components of the wave function."""
        y_comp = np.sin(self.ky * v) if self.m % 2 == 0 else np.cos(self.ky * v)
        x_comp = np.sin(self.kx * u) if self.n % 2 == 0 else np.cos(self.kx * u)
        return u, v, self.Amn * x_comp * y_comp


    def temporal_component(self, t):
        """Calculate the time-dependent component of the wave function."""
        time_factor = np.cos(self.omegamn * t)
        if self.betamn > 0.001:
            time_factor *= np.exp(-self.betamn * t)
        return time_factor


class StandingCircularMembrane(MembraneWaveMobject):
    def __init__(self, m: int, n: int, radius: float = 5, config: Optional[MembraneConfig] = None, **kwargs):
        self.m = m
        self.n = n
        self.r = radius
        self.config = config or MembraneConfig()
        self.config.u_range = (0, radius)
        self.config.v_range = (0, TAU)
        if n == 0 and m > 0:
            self.jmn = 0
        elif n > 0 and m >= 0: 
            self.jmn = jn_zeros(m, n)[-1]
        else:
            raise ValueError(f'Values of m ({m}) and n ({n}) are invalid')
        self.kmn = self.jmn / self.r
        self.c = self.config.wave_speed
        self.omegamn = self.kmn * self.c
        if self.config.dampening > 0.001:
            betamn = self.config.dampening
            self.omegamn = np.sqrt(self.omegamn ** 2 - betamn ** 2)
            self.betamn = betamn
        else:
            self.betamn = 0
        
        # Precalculate time-independent components
        self.Amn = self.config.amplitude
        self.phi_mn = self.config.phase
        
        super().__init__(self.config, **kwargs)

    def spatial_component(self, u, v):
        """Calculate the time-independent spatial components of the wave function."""
        x_val = u * np.cos(v)
        y_val = u * np.sin(v)
        bessel = jv(self.m, self.kmn * u)
        z_spatial = self.Amn * bessel * np.cos(self.m * v + self.phi_mn)
        return x_val, y_val, z_spatial

    def temporal_component(self, t):
        """Calculate the time-dependent component of the wave function."""
        time_factor = np.cos(self.omegamn * t)
        if self.betamn > 0.001:
            time_factor *= np.exp(-self.betamn * t)
        return time_factor


class FourierMembrane(MembraneWaveMobject):
    def __init__(
        self,
        surface_z: Callable,
        membrane_type: str,
        M: int = 5,
        N: int = 5,
        config: Optional[MembraneConfig] = None,
        sub_wave_config: Optional[MembraneConfig] = None,
        **kwargs #For membrane_type consructor
    ):
        self.config = config or MembraneConfig()
        if membrane_type == 'radial':
            if 'radius' in kwargs:
                self.radius = kwargs.pop('radius', None)
            else:
                self.radius = 5
            self.membrane_class = StandingCircularMembrane
            self.config.u_range = (0, self.radius)
            self.config.v_range = (0, TAU)
            self.mode_range = product(range(M + 1), range(1, N + 1))
            self.num_harmonics = (M + 1) * N
            sha = hashlib.sha224()
            sha.update(membrane_type.encode('ascii'))
            sha.update(str(int(100 * self.radius)).encode('ascii'))
            uuid = sha.hexdigest()
            fourier_calc = Fourier2DCalculator(surface_z, uuid)
        elif membrane_type == 'rectangular':
            if 'width' in kwargs:
                self.width = kwargs.pop('width', None)
            if 'height' in kwargs:
                self.height = kwargs.pop('height', None)
            self.membrane_class = StandingRectangularMembrane
            self.config.u_range = (-self.width/2, self.width/2)
            self.config.v_range = (-self.height/2, self.height/2)
            self.mode_range = product(range(1, M + 1), range(1, N + 1))
            self.num_harmonics = M * N
            sha = hashlib.sha224()
            sha.update(membrane_type.encode('ascii'))
            sha.update(str(int(100 * self.width)).encode('ascii'))
            sha.update(str(int(100 * self.height)).encode('ascii'))
            uuid = sha.hexdigest()
            fourier_calc = Fourier2DCalculator(surface_z, uuid)

        else:
            raise TypeError(f'membrane_type must be radial or rectangular. Not {membrane_type}.')

        self.sub_wave_config = sub_wave_config or self.config
        self.standing_waves = []

        # Calculate Fourier coefficients once during initialization
        for m, n in self.mode_range:
            if membrane_type == 'radial':
                sub_config = fourier_calc.radial_fourier_coefficients(m, n, self.radius, self.sub_wave_config)
                self.standing_waves.append(
                    self.membrane_class(
                        m, n, radius=self.radius, config=sub_config, **kwargs
                    )
                )
            elif membrane_type == 'rectangular':
                sub_config = fourier_calc.rectangular_fourier_coefficients(m, n, self.width, self.height, self.sub_wave_config)
                self.standing_waves.append(
                    self.membrane_class(
                        m, n, width=self.width, height=self.height, config=sub_config, **kwargs
                    )
                )
        fourier_calc.update_cache()
        self.standing_waves = SGroup(*self.standing_waves)
        super().__init__(self.config, **kwargs)

    def calculate_spatial_components(self, u, v):
        """Calculate and cache time-independent spatial components"""
        return [wave.spatial_component(u, v) for wave in self.standing_waves]

    def spatial_component(self, u, v):
        pass

    def temporal_component(self, t):
        pass

    def wave_function(self, u, v, t):
        x_vals, y_vals, z_vals = zip(*self.calculate_spatial_components(u, v))
        # Sum the time-dependent components
        temporal_z = 0
        for idx, wave in enumerate(self.standing_waves):
            z_temporal = wave.temporal_component(t)
            temporal_z += z_vals[idx] * z_temporal
        return x_vals[0], y_vals[0], temporal_z

    def attach_value_tracker(self, tracker) -> None:
        for wave in self.standing_waves:
            wave.attach_value_tracker(tracker)
        super().attach_value_tracker(tracker)
        return self


class Fourier2DCalculator:
    def __init__(self, surface_z, uuid):
        self.PATH_RELATIVE = 'animations/2025/Vibrations/fourier_coeff_cache.pkl'
        try:
            with open(self.PATH_RELATIVE, 'rb') as f:
                self.fourier_coeff_cache = pickle.load(f)
        except FileNotFoundError:
            self.fourier_coeff_cache = {uuid: {}}
        if uuid not in self.fourier_coeff_cache:
            self.fourier_coeff_cache[uuid] = {}
        self.surface_z = surface_z
        self.uuid = uuid
        self.should_update_cache = False

    @staticmethod
    def convolution_radial_mode(f, m, k_mn, R):
        if m == 0:
            # When there is radial symmetry, no need for phase
            S_mn, _ = integrate.dblquad(
                lambda r, theta: (jv(m, k_mn * r))**2 * r,
                0, TAU,
                lambda _: 0, lambda _: R
            )
            A_mn, _ = integrate.dblquad(
                lambda r, theta: f(r, theta) * jv(m, k_mn * r) * r,
                0, TAU,
                lambda _: 0, lambda _: R
            )
            return A_mn / S_mn, 0
        
        # Compute normalization constants
        S_mn, _ = integrate.dblquad(
            lambda r, theta: (jv(m, k_mn * r) * np.cos(m * theta))**2 * r,
            0, TAU,
            lambda _: 0, lambda _: R
        )
        T_mn, _ = integrate.dblquad(
            lambda r, theta: (jv(m, k_mn * r) * np.sin(m * theta))**2 * r,
            0, TAU,
            lambda _: 0, lambda _: R
        )
        
        # Compute the coefficients A_mn, B_mn
        A_mn, _ = integrate.dblquad(
            lambda r, theta: f(r, theta) * jv(m, k_mn * r) * np.cos(m * theta) * r,
            0, TAU,
            lambda _: 0, lambda _: R
        )
        B_mn, _ = integrate.dblquad(
            lambda r, theta: f(r, theta) * jv(m, k_mn * r) * np.sin(m * theta) * r,
            0, TAU,
            lambda _: 0, lambda _: R
        )
        # Normalize
        A_mn = A_mn / S_mn
        B_mn = B_mn / T_mn

        #Amplitude and phase
        C_mn = np.sqrt(A_mn ** 2 + B_mn ** 2)
        phi_mn = np.arctan2(-B_mn, A_mn)
        return C_mn, phi_mn

    @staticmethod
    def convolution_rectangular_mode(f, m, n, width, height, k_x, k_y):
        # Compute normalization constants
        S_mn = (width * height) / 4
        def phimn(u, v):
            y_comp = np.sin(k_y * v) if m % 2 == 0 else np.cos(k_y * v)
            x_comp = np.sin(k_x * u) if n % 2 == 0 else np.cos(k_x * u)
            return x_comp * y_comp
        
        # Compute the coefficients A_mn, B_mn

        A_mn, acc = integrate.dblquad(
            lambda x, y: f(x, y) * phimn(x, y),
            -height/2, height/2,
            lambda _: -width/2, lambda _: width/2
        )
        return A_mn / S_mn


    def radial_fourier_coefficients(self, m: int, n: int, R, config: MembraneConfig):
        harmonic_config_copy = copy.deepcopy(config)
        if (m, n) not in self.fourier_coeff_cache[self.uuid]:
            print(f'Calculating new FC: {(m, n, R)}')
            jmn = jn_zeros(m, n)[-1]
            kmn = jmn / R
            self.fourier_coeff_cache[self.uuid][(m, n)] = Fourier2DCalculator.convolution_radial_mode(self.surface_z, m, kmn, R)
            self.should_update_cache = True
        Amn, phimn = self.fourier_coeff_cache[self.uuid][(m, n)]
        harmonic_config_copy.amplitude = Amn
        harmonic_config_copy.phase = phimn
        return harmonic_config_copy

    def rectangular_fourier_coefficients(self, m: int, n: int, width, height, config: MembraneConfig):
        harmonic_config_copy = copy.deepcopy(config)
        if (m, n) not in self.fourier_coeff_cache[self.uuid]:
            print(f'Calculating new FC: {(m, n, width, height)}')
            kx = PI * n / width
            ky = PI * m / height
            self.fourier_coeff_cache[self.uuid][(m, n)] = Fourier2DCalculator.convolution_rectangular_mode(self.surface_z, m, n, width, height, kx, ky)
            self.should_update_cache = True
        Amn = self.fourier_coeff_cache[self.uuid][(m, n)]
        harmonic_config_copy.amplitude = Amn
        return harmonic_config_copy

    def update_cache(self):
        if self.should_update_cache:
            with open(self.PATH_RELATIVE, 'wb+') as f:
                pickle.dump(self.fourier_coeff_cache, f)
            self.should_update_cache = False

class PictureScene(InteractiveScene):
    def construct(self):
        # Settup Scene
        t_tracker = ValueTracker(0)
        light = self.camera.light_source
        light.move_to(OUT * 2.5)
        light_indicator = Sphere(radius=0.1).move_to(light.get_center())
        self.frame.reorient(43, 60, 1, IN, 10)
        self.frame.add_updater(lambda m, dt: m.increment_theta(dt * 3 * DEGREES)) #type: ignore
        config = MembraneConfig(
            resolution = (40, 40),
            dampening = 0.0,
            amplitude = 3,
            wave_speed = 5,
        )

        membrane_type = StandingCircularMembrane
        membrane_str = 'radial'
        membrane_kwargs = {'radius': 4}
        M, N = 8, 8

        # membrane_type = StandingRectangularMembrane
        # membrane_str = 'rectangular'
        # membrane_kwargs = {'width': 8, 'height':10}
        # M, N = 15, 15


        # Create some standing waves
        standing_2D_waves = [
            membrane_type(0, 4, shading=(0.2, 0.2, 0.1), config=config, **membrane_kwargs)
        ]
        wave_mesh = standing_2D_waves[0].get_mesh(
            stroke_color=WHITE,
            stroke_opacity=0.7,
            stroke_width=0.7
        )
        for standing_2D_wave in standing_2D_waves:
            standing_2D_wave.attach_value_tracker(t_tracker)
            standing_2D_wave.always_update_color_for_height(QMV_PURPLE_B, QMV_BLUE_A)
            self.add(standing_2D_wave, wave_mesh)
            self.play(t_tracker.animate.increment_value(1.2), run_time=1.2, rate_func=linear)

        # Freeze Frame  
        t_tracker.set_value(1.23)
        self.frame.set_euler_angles(0.84837751, 0.83608644, 0.01745329)


class MembraneAnimationScene(InteractiveScene):
    def construct(self):
        # Settup Scene
        t_tracker = ValueTracker(0)
        light = self.camera.light_source
        light.move_to(OUT * 2.5)
        light_indicator = Sphere(radius=0.1).move_to(light.get_center())
        self.frame.reorient(43, 60, 1, IN, 10)
        self.frame.add_updater(lambda m, dt: m.increment_theta(dt * 3 * DEGREES)) #type: ignore
        config = MembraneConfig(
            resolution = (50, 80),
            dampening = 0.0,
            amplitude = 3,
            wave_speed = 5,
        )

        membrane_type = StandingCircularMembrane
        membrane_str = 'radial'
        membrane_kwargs = {'radius': 5}
        M, N = 8, 8

        # membrane_type = StandingRectangularMembrane
        # membrane_str = 'rectangular'
        # membrane_kwargs = {'width': 8, 'height':10}
        # M, N = 15, 15


        # Create some standing waves
        standing_2D_waves = [
            membrane_type(0, 1, shading=(0.4, 0.2, 0.1), config=config, **membrane_kwargs),
            membrane_type(0, 2, shading=(0.4, 0.2, 0.1), config=config, **membrane_kwargs),
            membrane_type(1, 2, shading=(0.4, 0.2, 0.1), config=config, **membrane_kwargs),
        ]
        for standing_2D_wave in standing_2D_waves:
            standing_2D_wave.attach_value_tracker(t_tracker)
            standing_2D_wave.always_update_color_for_height(QMV_BLUE_A, QMV_PINK_A)
            self.add(standing_2D_wave)
            self.play(t_tracker.animate.increment_value(3), run_time=3, rate_func=linear)
            self.remove(standing_2D_wave)

        
        # Import face shape
        if membrane_str == 'radial':
            obama_points = ObamaFace(max_dist_from_center=4).get_3D_points_on_face().get_all_points()
            '''obama_dots = [Sphere(radius=0.1).move_to(point) for point in obama_points]'''
            R = membrane_kwargs['radius']
            pts_to_surface_z = ObamaFace().get_3D_param_radial(R)
            static_shape = ParametricSurface(
                lambda u, v: (u * np.cos(v), u * np.sin(v), pts_to_surface_z(u, v)),
                u_range = (0, R),
                v_range = (0, TAU),
                resolution = config.resolution,
                color=GREY_C
            )
            static_shape_mesh = SurfaceMesh(static_shape, resolution = config.resolution, stroke_color=YELLOW)
            static_shape_mesh.save_state()
        if membrane_str == 'rectangular':
            width, height = membrane_kwargs['width'], membrane_kwargs['height']
            pts_to_surface_z = ObamaFace().get_3D_param_rectangular(width, height)
            static_shape = ParametricSurface(
                lambda u, v: (u, v, pts_to_surface_z(u, v)),
                u_range = (-width/ 2, width/ 2),
                v_range = (-height / 2, height / 2),
                resolution = config.resolution,
                color = GREY_C
            )
            static_shape_mesh = SurfaceMesh(static_shape, resolution = config.resolution, stroke_color=YELLOW)
            static_shape_mesh.save_state()
        else:
            raise ValueError()

        # Show face on Membrane
        self.frame.set_euler_angles(-1.91454528,  1.0585782,  0.01745329)
        self.play(ShowCreation(static_shape_mesh), run_time=1)
        self.wait(6)
        self.play(self.frame.animate.set_euler_angles(-1.11549034,  0.57298149,  0.01745329))
        self.play(FadeIn(static_shape))
        self.wait(2)
        self.play(Uncreate(static_shape_mesh))
        static_shape_mesh.restore()
        self.wait(2)
        self.play(FadeOut(static_shape))


        # Create fourier mobject...
        low_res_config = copy.deepcopy(config)
        low_res_config.resolution = (20, 30)
        fourier_wave = FourierMembrane(
            pts_to_surface_z, membrane_str, M, N,
            **membrane_kwargs,
            config=config,
            sub_wave_config = low_res_config,
        )
        fourier_wave.save_state()


        # Extract mesh components and style
        fourier_wave_mesh = fourier_wave.get_mesh(stroke_color=WHITE, stroke_opacity=0.7)
        standing_waves = fourier_wave.standing_waves
        standing_waves = sorted(standing_waves, key=lambda wave: wave.config.amplitude, reverse=True)
        stroke_colors = color_gradient([BLUE_B, BLUE_E], fourier_wave.num_harmonics)
        stroke_opacity_range = [0.8, 0.1]
        stroke_width_range = [2, 0.5]
        max_fourier_coeff = max([wave.config.amplitude for wave in standing_waves])
        stroke_opacities = np.geomspace(*stroke_opacity_range, fourier_wave.num_harmonics) #type: ignore
        stroke_widths = np.geomspace(*stroke_width_range, fourier_wave.num_harmonics) #type: ignore
        standing_waves_mesh = [
            wave.get_mesh(
                stroke_color=stroke_colors[idx],
                stroke_opacity=stroke_opacities[idx],
                stroke_width=stroke_widths[idx]
            ) for idx, wave in enumerate(standing_waves)
        ]
        for wave in standing_waves_mesh:
            wave.save_state()
        self.play(
            LaggedStart(*[
                ShowCreation(wave) 
                for wave in standing_waves_mesh
            ]),
            run_time=5,
            rate_func=linear
        )
        self.play(
            LaggedStart(*[
                Uncreate(wave) 
                for wave in standing_waves_mesh
            ]),
            run_time=2
        )
        for wave in standing_waves_mesh:
            wave.restore()


        # Show again static_shape and leave there
        static_shape.save_state()
        static_shape_mesh.save_state()
        self.play(ShowCreation(static_shape_mesh), FadeIn(static_shape), run_time=2)
        self.wait()
        self.play(FadeOut(static_shape))
        self.play(self.frame.animate.set_euler_angles(-1.91454528,  1.0585782 ,  0.01745329))
        self.wait()
        self.play(static_shape_mesh.animate.set_stroke(opacity=0.6))



        '''
        # Create intermediate_waves_to Illustrate sum
        individual_trackers = [ValueTracker(0) for _ in range(fourier_wave.num_harmonics)]
        if membrane_str == 'radial':
            circular_range = [(m, n, m * N + (n - 1)) for n in range(1, N + 1) for m in range(min(n, M + 1))]
            sum_harmonic_range = circular_range
        elif membrane_str == 'rectangular':
            rectangular_range = [(m, n, (m - 1) * N + (n - 1)) for n in range(1, N + 1) for m in range(1, min(n, M + 1))]
            sum_harmonic_range = rectangular_range
        intermediate_fourier_waves = [
            FourierMembrane(
                pts_to_surface_z, membrane_str, M, N,
                **membrane_kwargs,
                config=config,
                sub_wave_config = low_res_config,
            )
            for m, n, _ in sum_harmonic_range
        ]
        intermediate_standing_waves = intermediate_fourier_waves[-1].standing_waves
        intermediate_standing_waves_mesh = [
            intermediate_standing_waves[idx].get_mesh(
                stroke_color=BLUE_B,
                stroke_opacity=1,
                stroke_width=2
            ) 
            for _, _, idx in sum_harmonic_range
        ]


        # Show summing of standing waves.
        idx = 0
        wave = intermediate_standing_waves_mesh[idx]
        int_wave = intermediate_fourier_waves[idx].get_mesh(stroke_opacity=0.7)
        tracker = individual_trackers[idx]
        self.play(DrawBorderThenFill(wave), run_time=2)
        wave.attach_value_tracker(tracker)
        self.play(tracker.animate.set_value(3), rate_func=linear, run_time=3)
        wave.clear_updaters()
        self.play(ReplacementTransform(wave, int_wave))
        int_wave.attach_value_tracker(tracker)
        self.play(int_wave.animate.set_stroke(opacity=0.3), tracker.animate.set_value(0))
        int_wave.clear_updaters()
        prev_int_wave = int_wave
        self.play(self.frame.animate.set_euler_angles(phi=75 * DEGREES))
        for idx in range(1, len(intermediate_fourier_waves)):
            wave = intermediate_standing_waves_mesh[idx]
            int_wave = intermediate_fourier_waves[idx].get_mesh(stroke_opacity=0.7)
            tracker = individual_trackers[idx]
            self.play(DrawBorderThenFill(wave), run_time=2)
            wave.attach_value_tracker(tracker)
            self.play(tracker.animate.set_value(2), rate_func=linear, run_time=3)
            wave.clear_updaters()
            self.play(prev_int_wave.animate.set_stroke(opacity=0.7))
            self.play(ReplacementTransform(wave, int_wave), ReplacementTransform(prev_int_wave, int_wave), run_time=2)
            tracker.set_value(0)
            int_wave.attach_value_tracker(tracker)
            self.play(tracker.animate.set_value(2), rate_func=linear, run_time=3)
            self.play(int_wave.animate.set_stroke(opacity=0.3), tracker.animate.set_value(0))
            int_wave.clear_updaters()
            prev_int_wave = int_wave
        '''

        tracker = ValueTracker(0)
        prev_int_wave = fourier_wave_mesh
        self.play(self.frame.animate.set_euler_angles(phi=0.99273458))
        self.play(
            LaggedStart(*[
                ShowCreation(wave) 
                for wave in standing_waves_mesh
            ]),
            run_time=3,
            rate_func=linear
        )
        self.play(
            LaggedStart(*[
                ReplacementTransform(wave, prev_int_wave) 
                for wave in standing_waves_mesh
            ]),
            run_time=2
        )
        for wave in standing_waves_mesh:
            wave.restore()


        # Into showing last mesh animation
        self.frame.clear_updaters()
        self.frame.add_updater(lambda m, dt: m.increment_theta(dt * 5 * DEGREES)) #type: ignore
        self.play(self.frame.animate.set_euler_angles(phi=30 * DEGREES))
        self.play(prev_int_wave.animate.set_stroke(opacity=0.8))
        self.play(FadeOut(static_shape_mesh))
        self.wait()
        prev_int_wave.attach_value_tracker(tracker)
        self.play(tracker.animate.increment_value(10), rate_func=linear, run_time=10)
        for wave in standing_waves_mesh:
            wave.clear_updaters()
            wave.restore()
        prev_int_wave.suspend_updating()
        self.play(LaggedStart(ShowCreation(wave) for wave in standing_waves_mesh), run_time=2)
        prev_int_wave.resume_updating()
        for wave in standing_waves_mesh:
            wave.attach_value_tracker(tracker)

        self.play(tracker.animate.increment_value(3), rate_func=linear, run_time=3)
        self.play(self.frame.animate.set_euler_angles(phi=80 * DEGREES),
            tracker.animate.increment_value(1), rate_func=linear, run_time=1)
        self.play(tracker.animate.increment_value(10), rate_func=linear, run_time=10)
        prev_int_wave.clear_updaters()
        for wave in standing_waves_mesh:
            wave.clear_updaters()
        self.play(
            FadeOut(prev_int_wave),
            *[FadeOut(wave) for wave in standing_waves_mesh]
        )


        # Show final wave surface (sped up)
        t_tracker = ValueTracker(0)
        config.dampening = 0.1
        fourier_wave = FourierMembrane(
                pts_to_surface_z, membrane_str, M, N,
                **membrane_kwargs,
                config=config,
                sub_wave_config=low_res_config,
            )
        self.frame.set_euler_angles(phi=45 * DEGREES)
        self.play(FadeIn(fourier_wave))
        fourier_wave.attach_value_tracker(t_tracker)
        fourier_wave.always_update_color_for_height(RED, YELLOW)
        self.wait(2)
        self.play(
            t_tracker.animate.set_value(50),
            run_time=30,
            rate_func=linear
        )
        self.wait()



class PressureWaveConfig:
    """Configuration settings for pressure wave properties"""
    def __init__(
        self,
        u_range: Tuple[float, float] = (0, 1),
        v_range: Tuple[float, float] = (0, 1),
        w_range: Tuple[float, float] = (0, 1),
        resolution: Tuple[int, int, int] = (5, 5, 5),
        dampening: float = 0.0,
        phase: float = 0.0,
        wave_speed: float = 2.0,
        radius_range: Tuple[float, float] = (0.01, 0.1),
        dot_opacity_range: Tuple[float, float] = (0.2, 0.95),
        color_range = [QMV_PURPLE_A, QMV_PURPLE_A, GREY, QMV_BLUE_A, QMV_BLUE_A],
        base_color = GREY,
    ):
        self.u_range = u_range
        self.v_range = v_range
        self.w_range = w_range
        self.resolution = resolution
        self.dampening = dampening
        self.wave_speed = wave_speed
        self.phase = phase
        self.radius_range = radius_range
        self.dot_opacity_range = dot_opacity_range
        self.color_range = color_range
        self.base_color = base_color

class PressureWaveMobject(DotCloud, ABC):
    def __init__(self, wave_config: Optional[PressureWaveConfig] = None, **kwargs):
        self.wave_config = wave_config or PressureWaveConfig()
        self.point_centers = self.calculate_point_centers(self.wave_config.u_range, self.wave_config.v_range, self.wave_config.w_range)
        super().__init__(
            self.point_centers,
            radius=self.wave_config.radius_range[0],
            opacity=self.wave_config.dot_opacity_range[0],
            anti_alias_width=self.wave_config.radius_range[0] / 1,
            **kwargs
        )
        self.center().make_3d()
        self.add_updater(lambda m: m.update_dot_properties())

    @Mobject.affects_data
    def update_dot_properties(self):
        points = self.data['point']
        wave_values = np.array([self.wave_function(x, y, z) for x, y, z in points])
        wave_values_alpha = (wave_values + 1) * 0.5 #Scaled to [0, 1]
        wave_values_abs = np.abs(wave_values)
        radii = wave_values_abs * (self.wave_config.radius_range[1] - self.wave_config.radius_range[0])
        radii += self.wave_config.radius_range[0]
        colors = get_colormap_from_colors(self.wave_config.color_range)(wave_values_alpha)
        colors[:, 3] = wave_values_abs * (self.wave_config.dot_opacity_range[1] - self.wave_config.dot_opacity_range[0])
        colors[:, 3] += self.wave_config.dot_opacity_range[0] 

        self.set_radii(radii)
        self.data['rgba'][:] = colors


    def always_sort_to_camera(self, camera: Camera):
        def updater(p_wave: PressureWaveMobject):
            p_wave.sort_points(lambda vec: np.linalg.norm(vec - camera.get_location()))
        self.add_updater(updater)
        return self
    
    @abstractmethod
    def calculate_point_centers(self, u_range, v_range, w_range):
        pass

    @abstractmethod
    def wave_function(self, x, y, z) -> float:
        '''Should return between -1 and 1'''
        pass

class RectangularPressureWaveMedium(PressureWaveMobject):
    def __init__(self, X=6, Y=4, Z=8, wave_config: Optional[PressureWaveConfig] = None, **kwargs):
        self.wave_config = wave_config or PressureWaveConfig()
        self.X = X
        self.Y = Y
        self.Z = Z
        self.wave_config.u_range = (0, X)
        self.wave_config.v_range = (0, Y)
        self.wave_config.w_range = (0, Z)
        self.functional_center = np.array([X / 2, Y / 2, Z / 2])
        self.wave_functions_to_render = {}
        super().__init__(wave_config=self.wave_config, **kwargs)

    def _standing_wave_function(self, m, n, l) -> Callable:
        kx = l * PI / self.X
        ky = m * PI / self.Y
        kz = n * PI / self.Z
        omegalmn = self.wave_config.wave_speed * np.sqrt(kx**2 + ky**2 + kz**2)
        def standing_wave_func(x, y, z, t):
            return np.sin(kx * x) * np.sin(ky * y) * np.sin(kz * z) * np.cos(omegalmn * t)
        return standing_wave_func

    def add_standing_wave(self, m, n, l, amplitude=1, t_tracker: Optional[ValueTracker] = None) -> ValueTracker:
        t_tracker = t_tracker or ValueTracker(0)
        def standing_wave(x, y, z):
            working_coord = np.array([x, y, z]) + self.functional_center
            return amplitude * self._standing_wave_function(m, n, l)(*working_coord, t_tracker.get_value())
        self.wave_functions_to_render[(m, n, l)] = standing_wave
        return t_tracker

    def remove_standing_wave(self, m, n, l):
        return self.wave_functions_to_render.pop((m, n, l))

    def wave_function(self, x, y, z) -> float:
        '''Should return between -1 and 1'''
        return sum([wave(x, y, z) for wave in self.wave_functions_to_render.values()])

    def calculate_point_centers(self, u_range, v_range, w_range):
        u = np.linspace(*u_range, int(self.wave_config.resolution[0] * (u_range[1] - u_range[0])))
        v = np.linspace(*v_range, int(self.wave_config.resolution[1] * (v_range[1] - v_range[0])))
        w = np.linspace(*w_range, int(self.wave_config.resolution[2] * (w_range[1] - w_range[0])))
        return np.array(list(product(u, v, w)))

class SphericalPressureWaveMedium(PressureWaveMobject):
    def __init__(self, R=3, wave_config: Optional[PressureWaveConfig] = None, **kwargs):
        self.wave_config = wave_config or PressureWaveConfig()
        self.wave_config.u_range = (0, 2 * R)
        self.wave_config.v_range = (0, 2 * R)
        self.wave_config.w_range = (0, 2 * R)
        self.functional_center = np.array([0, 0, 0])
        self.wave_functions_to_render = {}
        self.sphere_radius = R
        super().__init__(wave_config=self.wave_config, **kwargs)

    '''returns nth extreme and max value'''
    def _get_spherical_bessel_info(self, l, n):
        func = lambda r: spherical_jn(l, r)
        extrema = []
        sign = -1 if l != 0 else 1 # Start by assuming the first extrema is a minimum
        x0 = 0  # Initial guess, assuming 0 is an extremum
        for _ in range(n):
            # Find the next extrema using the sign and previous point
            res = opt.minimize_scalar(lambda x: sign * func(x), bounds=(x0, x0 + 10), method='bounded')
            x0 = res.x
            extrema.append((x0, res.fun))
            sign *= -1  # Alternate between maxima and minima

        return extrema[-1][0], abs(extrema[0][1])

    '''Allow l >= 0, different m (other than 0) values are degenerate must have m <= l, n >= 1.'''
    def _standing_wave_function(self, l, m, n) -> Callable:
        print(l, m, n)
        bessel_info = self._get_spherical_bessel_info(l, n)
        print(bessel_info)
        kln =  bessel_info[0]/ self.sphere_radius
        gamma = self.wave_config.phase
        omegalmn = self.wave_config.wave_speed * kln
        if l == 0:
            A = 1
        else:
            A = 1 / bessel_info[1]
        def standing_wave_func(x, y, z, t, kln=kln, A=A, gamma=gamma, omegalmn=omegalmn):
            # print(x, y, z)
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(y, x)
            phi = np.arctan2(np.sqrt(x * x + y * y), z)
            # print(r, theta, phi)
            # print(kln * r)
            r_sol = spherical_jn(l, kln * r)
            theta_sol = lpmn(m, l, np.cos(theta))[0][0][-1]
            phi_sol = np.cos(m * phi + gamma)
            time_sol = np.cos(omegalmn * t)
            # print(r_sol, theta_sol, phi_sol, time_sol)
            if A * r_sol * theta_sol * phi_sol * time_sol > 1:
                print(A * r_sol * theta_sol * phi_sol * time_sol)
                print(A, r_sol, theta_sol, phi_sol, time_sol)
            return A * r_sol * theta_sol * phi_sol * time_sol
        return standing_wave_func


    def add_standing_wave(self, m, n, l, amplitude=1, t_tracker: Optional[ValueTracker] = None) -> ValueTracker:
        t_tracker = t_tracker or ValueTracker(0)
        func_away_center = self._standing_wave_function(m, n, l)
        def standing_wave(x, y, z, func_away_center=func_away_center):
            working_coord = np.array([x, y, z]) + self.functional_center
            return amplitude * func_away_center(*working_coord, t_tracker.get_value())
        self.wave_functions_to_render[(m, n, l)] = standing_wave
        return t_tracker

    def remove_standing_wave(self, m, n, l):
        return self.wave_functions_to_render.pop((m, n, l))

    def wave_function(self, x, y, z) -> float:
        '''Should return between -1 and 1'''
        return sum([wave(x, y, z) for wave in self.wave_functions_to_render.values()])

    def calculate_point_centers(self, u_range, v_range, w_range):
        u = np.linspace(*u_range, int(self.wave_config.resolution[0] * (u_range[1] - u_range[0])))
        v = np.linspace(*v_range, int(self.wave_config.resolution[1] * (v_range[1] - v_range[0])))
        w = np.linspace(*w_range, int(self.wave_config.resolution[2] * (w_range[1] - w_range[0])))
        points = np.array(list(product(u, v, w)))
        points -= self.sphere_radius
        filtered_points = points[np.linalg.norm(points, axis=1) <= self.sphere_radius]
        return filtered_points


class PressureWaveAnimationScene(InteractiveScene):
    def construct(self):
        p_wave = SphericalPressureWaveMedium(R=3.5)
        m_label = VGroup(
            Tex(R"m = ", t2c = {"m": QMV_PURPLE_B}),
            Integer(0)
        ).arrange(RIGHT)
        n_label = VGroup(
            Tex(R"n = ", t2c = {"n": QMV_PURPLE_C}),
            Integer(0)
        ).arrange(RIGHT)
        l_label = VGroup(
            Tex(R"l = ", t2c = {"l": QMV_PURPLE_A}),
            Integer(0)
        ).arrange(RIGHT)
        mode_labels = VGroup(l_label, m_label, n_label).arrange(DOWN).fix_in_frame().to_corner(UL)
        self.add(p_wave, mode_labels)
        l_val_tracker = ValueTracker(0)
        m_val_tracker = ValueTracker(0)
        n_val_tracker = ValueTracker(0)
        l_label.add_updater(lambda m: m[1].set_value(int(l_val_tracker.get_value())))
        m_label.add_updater(lambda m: m[1].set_value(int(m_val_tracker.get_value())))
        n_label.add_updater(lambda m: m[1].set_value(int(n_val_tracker.get_value())))
        p_wave.always_sort_to_camera(self.camera)
        prev_mode = None
        # Mediocre rectagular mode list
        sorted_modes = sorted(list(product(range(1, 5), range(1, 5), range(1, 5))), key=lambda tup: np.linalg.norm(np.array(tup)))
        # Better spherical mode list
        sorted_modes = [(l, m, n) for l in range(0, 5) for m in range(0, l + 1) for n in range(1, 5)]
        self.frame.set_euler_angles(phi=60  * DEGREES)
        self.frame.add_ambient_rotation(10 * DEGREES)
        some_modes = [(1, 0, 3), (1, 1, 2), (2, 0, 2), (2, 2, 1), (2, 2, 2), (3, 0, 2)]
        for mode in some_modes:
            l_val_tracker.set_value(mode[0])
            m_val_tracker.set_value(mode[1])
            n_val_tracker.set_value(mode[2])
            if prev_mode is not None:
                p_wave.remove_standing_wave(*prev_mode)
            t_tracker = p_wave.add_standing_wave(*mode)
            self.play(t_tracker.animate.increment_value(3), rate_func=linear, run_time=3)
            prev_mode = mode


class PressureSphereShowcase(InteractiveScene):
    def construct(self):
        p_wave = SphericalPressureWaveMedium(R=3.5)
        m_label = VGroup(
            Tex(R"m = ", t2c = {"m": QMV_PURPLE_B}),
            Integer(0)
        ).arrange(RIGHT)
        n_label = VGroup(
            Tex(R"n = ", t2c = {"n": QMV_PURPLE_C}),
            Integer(0)
        ).arrange(RIGHT)
        l_label = VGroup(
            Tex(R"l = ", t2c = {"l": QMV_PURPLE_A}),
            Integer(0)
        ).arrange(RIGHT)
        mode_labels = VGroup(l_label, m_label, n_label).arrange(DOWN).fix_in_frame().to_corner(UL)
        self.add(p_wave, mode_labels)
        l_val_tracker = ValueTracker(0)
        m_val_tracker = ValueTracker(0)
        n_val_tracker = ValueTracker(0)
        l_label.add_updater(lambda m: m[1].set_value(int(l_val_tracker.get_value())))
        m_label.add_updater(lambda m: m[1].set_value(int(m_val_tracker.get_value())))
        n_label.add_updater(lambda m: m[1].set_value(int(n_val_tracker.get_value())))
        p_wave.always_sort_to_camera(self.camera)
        prev_mode = None
        # Mediocre rectagular mode list
        sorted_modes = sorted(list(product(range(1, 5), range(1, 5), range(1, 5))), key=lambda tup: np.linalg.norm(np.array(tup)))
        # Better spherical mode list
        sorted_modes = [(l, m, n) for l in range(0, 5) for m in range(0, l + 1) for n in range(1, 5)]
        self.frame.set_euler_angles(phi=60  * DEGREES)
        self.frame.add_ambient_rotation(10 * DEGREES)
        some_modes = [(1, 0, 3), (1, 1, 2), (2, 0, 2), (2, 2, 1), (2, 2, 2), (3, 0, 2)]
        mode = (1, 0, 3)
        l_val_tracker.set_value(mode[0])
        m_val_tracker.set_value(mode[1])
        n_val_tracker.set_value(mode[2])
        t_tracker = p_wave.add_standing_wave(*mode)
        self.play(t_tracker.animate.increment_value(3), rate_func=linear, run_time=3)
        for mode in some_modes[1:]:
            new_p_wave = SphericalPressureWaveMedium(R=3.5)
            t_tracker = new_p_wave.add_standing_wave(*mode)
            new_p_wave.update()
            p_wave.suspend_updating()
            new_p_wave.suspend_updating()
            self.play(
                ReplacementTransform(p_wave, new_p_wave),
                l_val_tracker.animate.set_value(mode[0]),
                m_val_tracker.animate.set_value(mode[1]),
                n_val_tracker.animate.set_value(mode[2]),
                run_time=2
            )
            new_p_wave.resume_updating()
            self.play(t_tracker.animate.increment_value(3), rate_func=linear, run_time=3)
            self.remove(p_wave)
            self.add(new_p_wave)
            p_wave = new_p_wave



    