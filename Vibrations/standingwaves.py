from abc import ABC, abstractmethod
from manimlib import *
import scipy.integrate as integrate
import cmath

DEFAULT_WAVE_LENGTH = 5
# Abstract Wave Mobject
class WaveMobject(TipableVMobject, ABC):
    """
    Abstract base class for all 1D wave objects (e.g., string waves). test

    Parameters:
    - length (float): The length of the wave's domain.
    - resolution (int): The number of points to sample per unit length. Default is 10.
    - dampening (float): A value between 0 and 1 representing the level of wave dampening. Default is 0.0.
    """
    def __init__(self, length=DEFAULT_WAVE_LENGTH, resolution=10, dampening=0.0, **kwargs):
        super().__init__(**kwargs)
        self.start = LEFT * (length / 2)
        self.end = RIGHT * (length / 2)
        self.length = length
        self.resolution = resolution
        self.dampening = dampening
        self.set_points_by_wave_function(0)

    @abstractmethod
    def wave_function(self, x, t):
        """
        Define the wave function for the specific wave.

        Parameters:
        - x (float): The position along the wave's domain.
        - t (float): The time.

        Returns:
        - float: The wave's displacement at position `x` and time `t`.
        """
        pass

    def set_points_by_wave_function(self, t):
        """
        Generate an array of points based on the wave function.

        Parameters:
        - t (float): The time at which to sample the wave.
        """
        self.clear_points()
        num_points = self.resolution * self.length
        x_values = np.linspace(0, self.length, num_points)
        y_values = np.array([self.wave_function(x, t) for x in x_values])
        points = np.column_stack((x_values, y_values, np.zeros_like(x_values)))
        self.set_points_smoothly(points)
        self.put_start_and_end_on(self.start, self.end)

    def attach_value_tracker(self, value_tracker):
        """
        Attach a ValueTracker to dynamically update the wave over time.

        Parameters:
        - value_tracker (ValueTracker): A tracker for the time variable `t`.
        """
        def update_wave(mobject):
            t = value_tracker.get_value()
            mobject.set_points_by_wave_function(t)

        self.add_updater(update_wave)


# 1D Wave Types (String Waves)
class WaveFixedAtBothEnds(WaveMobject):
    """
    A standing wave on a string fixed at both ends.

    Parameters:
    - n (int): The harmonic mode of the wave.
    - amplitude (float): The maximum displacement of the wave. Default is 1.
    - omega (float): The angular frequency of the wave. Default is TAU (2π).
    """
    def __init__(self, n, amplitude=1.0, omega=TAU, **kwargs):
        self.n = n
        self.amplitude = amplitude
        self.omega = omega
        super().__init__(**kwargs)


    def wave_function(self, x, t):
        """
        Define the standing wave function for a string fixed at both ends.

        Parameters:
        - x (float): The position along the string.
        - t (float): The time.

        Returns:
        - float: The wave's displacement at position `x` and time `t`.
        """
        if self.dampening < 0.001:
            standing_wave = np.sin(self.n * PI * x / self.length)
            time_dependence = np.cos(self.omega * t)
            return self.amplitude * standing_wave * time_dependence
        else:
            k = self.n * PI / self.length
            c = self.omega / k
            beta = self.dampening
            omega_damped = cmath.sqrt((c * k)**2 - beta**2) + beta * 1j
            standing_wave =  np.sin(k * x) 
            time_dependence = cmath.exp(1j * t * omega_damped)
            return self.amplitude * ((standing_wave * time_dependence).real)



class FourierWave(WaveMobject):
    def __init__(self, N, y_of_x_or_pts, c, **kwargs):
        self.N = N
        if 'length' in kwargs.keys():
            self.length = kwargs['length']
        else:
            self.length = DEFAULT_WAVE_LENGTH

        if callable(y_of_x_or_pts):
            self.y_of_x = y_of_x_or_pts
            self.A = [
                (2 / self.length) * integrate.quad(lambda x: self.y_of_x(x) * np.sin((n * PI * x) / self.length), 0, self.length)[0]
                for n in range(1, self.N + 1)
            ]
        elif isinstance(y_of_x_or_pts, list) and all(isinstance(item, tuple) for item in y_of_x_or_pts):
            points = y_of_x_or_pts
            self.A = []
            for n in range(1, self.N + 1):
                An = 0
                for i in range(len(points) - 1):
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    def custom_indef_integral(x):
                        pi_n = PI * n
                        x_diff = x2 - x1
                        y_diff = y2 - y1
                        L = self.length
                        numerator = 2 * (L * y_diff * np.sin(pi_n * x / L) - pi_n * (y2 * (x - x1) - y1 * (x - x2)) * np.cos(pi_n * x / L))
                        denominator = PI ** 2 * n ** 2 * x_diff
                        return numerator / denominator
                    An += custom_indef_integral(x2) - custom_indef_integral(x1)
                self.A.append(An)

        if 'dampening' in kwargs.keys():
            self.dampening = kwargs['dampening']
        else:
            self.dampening = 0.0
        if self.dampening < 0.001:
            self.omega = [
                PI * n * c / self.length
                    for n in range(1, self.N + 1)
            ]
        else:
            beta = self.dampening
            self.omega = [
                cmath.sqrt((c * n * PI / self.length)**2 - beta**2) + beta * 1j
                for n in range(1, self.N + 1)
            ]

        super().__init__(**kwargs)

    def wave_function(self, x, t):
        y_val = 0
        for n in range(1, self.N + 1):
            if self.dampening < 0.001:
                standing_wave = np.sin(n * PI * x / self.length)
                time_dependence = np.cos(self.omega[n - 1] * t)
                yn = self.A[n - 1] * standing_wave * time_dependence
            else:
                standing_wave = np.sin(n * PI * x / self.length)
                time_dependence = cmath.exp(1j * t * self.omega[n - 1])
                yn = self.A[n - 1] * standing_wave * time_dependence.real
            y_val += yn
        return y_val


class TransverseWave(WaveMobject):
    def __init__(self, y_of_x_or_pts, **kwargs):
        if callable(y_of_x_or_pts):
            self.y_of_x = y_of_x_or_pts
        elif isinstance(y_of_x_or_pts, list) and all(isinstance(item, tuple) for item in y_of_x_or_pts):
            points = y_of_x_or_pts
            def y_of_x(x):
                for i in range(len(points) - 1):
                    if points[i][0] <= x <= points[i + 1][0]:
                        x1, y1 = points[i]
                        x2, y2 = points[i + 1]
                        break
                else:
                    raise ValueError("x is out of bounds.")        
                return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
            self.y_of_x = y_of_x
        super().__init__(**kwargs)

    def wave_function(self, x, t):
        return self.y_of_x(x)


class FourierGroup(VGroup):
    def __init__(self, N, y_of_x_or_pts, c=1, colors_for_gradient=None, **kwargs):
        if colors_for_gradient is None:
            colors_for_gradient = [BLUE_B, BLUE_E]
        colors = color_gradient(colors_for_gradient, N)
        self.c = c
        self.y_of_x_or_pts = y_of_x_or_pts
        self.still_wave = TransverseWave(y_of_x_or_pts, **kwargs)
        self.fourier_wave = FourierWave(N, y_of_x_or_pts, c=c, **kwargs)
        self.sin_waves = VGroup(
            WaveFixedAtBothEnds(
                n,
                amplitude=self.fourier_wave.A[n - 1],
                omega = self.fourier_wave.omega[n - 1],
                color=colors[n - 1],
                stroke_width= DEFAULT_STROKE_WIDTH * (0.2 + (N - n) * (0.6 / N)),
                opacity= 0.2 + (N - n) * (0.8 / N),
                **kwargs
                ) for n in range(1, N + 1)
            )
        super().__init__(self.transverse_wave, self.fourier_wave, self.sin_waves)

    def attach_value_tracker(self, value_tracker):
        self.transverse_wave.attach_value_tracker(value_tracker)
        self.fourier_wave.attach_value_tracker(value_tracker)
        for wave in self.sin_waves:
            wave.attach_value_tracker(value_tracker)


# Abstract Membrane Mobject
class MembraneMobject(WaveMobject, ABC):
    """
    Abstract class for all 2D wave objects (e.g., membrane waves).

    Parameters:
    - length: The length of the membrane's domain along the x-axis.
    - width: The length of the membrane's domain along the y-axis.
    - resolution: The resolution for both axes. The number of points per unit length. Default is 10.
    - dampening: A float between 0 and 1 that represents the level of dampening. Default is 0.0.
    """
    def __init__(self, length=5, width=5, resolution=10, dampening=0.0, **kwargs):
        super().__init__(length=length, resolution=resolution, dampening=dampening, **kwargs)
        self.width = width

    @abstractmethod
    def wave_function(self, x, y, t):
        """
        Abstract method for defining the wave function. The derived class must implement this.

        Parameters:
        - x: The position (float) along the x-axis of the membrane.
        - y: The position (float) along the y-axis of the membrane.
        - t: The time (float).

        Returns:
        - float: The displacement at position (x, y) and time t.
        """
        pass
    
    def generate_points(self, t):
        """
        Generate an array of 3D points based on the wave function.

        This function samples the wave at `resolution * length` evenly spaced positions along the wave's domain.
        Both `x` and `y` are sampled along the membrane's domain.

        Parameters:
        - t: The time (float) at which to sample the wave.

        Returns:
        - np.ndarray: An array of 3D points representing the membrane's displacement at each sampled position.
        """
        num_points_x = self.resolution * self.length
        num_points_y = self.resolution * self.width
        # Use np.meshgrid to create the 2D grid of points for x and y
        x_values, y_values = np.meshgrid(np.linspace(0, self.length, num_points_x), 
                                         np.linspace(0, self.width, num_points_y))
        # Vectorize the wave_function for efficient computation across the grid
        displacement = np.vectorize(self.wave_function)(x_values, y_values, t)
        # Stack x, y, and displacement to create 3D points
        points = np.dstack((x_values, y_values, displacement))
        # Return the reshaped points as a 2D array where each row is [x, y, z]
        return points.reshape(-1, 3)


# 2D Membrane Types
class RectangularMembraneWave(MembraneMobject):
    """
    A wave on a rectangular membrane.

    Parameters:
    - length: The length of the membrane along the x-axis. Default is 5.
    - width: The length of the membrane along the y-axis. Default is 5.
    - resolution: The resolution for both axes. Default is 10.
    - dampening: The dampening factor. Default is 0.0.
    """
    def __init__(self, length=5, width=5, resolution=10, dampening=0.0, **kwargs):
        super().__init__(length=length, width=width, resolution=resolution, dampening=dampening, **kwargs)


class CircularMembraneWave(MembraneMobject):
    """
    A wave on a circular membrane.

    Parameters:
    - radius: The radius of the circular membrane. Default is 5.
    - resolution: The resolution for the wave. Default is 10.
    - dampening: The dampening factor. Default is 0.0.
    """
    def __init__(self, radius=5, resolution=10, dampening=0.0, **kwargs):
        super().__init__(length=radius, width=radius, resolution=resolution, dampening=dampening, **kwargs)
        self.radius = radius


class AnnularMembraneWave(MembraneMobject):
    """
    A wave on an annular (donut-shaped) membrane.

    Parameters:
    - inner_radius: The inner radius of the annular membrane. Default is 3.
    - outer_radius: The outer radius of the annular membrane. Default is 5.
    - resolution: The resolution for the wave. Default is 10.
    - dampening: The dampening factor. Default is 0.0.
    """
    def __init__(self, inner_radius=3, outer_radius=5, resolution=10, dampening=0.0, **kwargs):
        super().__init__(length=outer_radius, width=outer_radius, resolution=resolution, dampening=dampening, **kwargs)
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius


class UnboundedMembraneWave(MembraneMobject):
    """
    A wave on an unbounded (infinite) membrane.

    Parameters:
    - length: The length of the membrane's domain along the x-axis. Default is 5.
    - width: The length of the membrane's domain along the y-axis. Default is 5.
    - resolution: The resolution for both axes. Default is 10.
    - dampening: The dampening factor. Default is 0.0.
    """
    def __init__(self, length=5, width=5, resolution=10, dampening=0.0, **kwargs):
        super().__init__(length=length, width=width, resolution=resolution, dampening=dampening, **kwargs)


class WaveAnimationScene(InteractiveScene):
    def construct(self):
        # Settup value tracker and overlay
        t_tracker = ValueTracker(0)

        # Create basic sine wave
        fixed_end_wave = WaveFixedAtBothEnds(2, dampening=0.2, color=BLUE_B)
        self.play(ShowCreation(fixed_end_wave))


        # Animate basic sine wave
        fixed_end_wave.attach_value_tracker(t_tracker)
        self.play(t_tracker.animate.set_value(10), run_time=10, rate_func=linear)
        self.remove(fixed_end_wave)


        # Examples of functions to interpolate
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
        func_ex = lambda x: 1/3 * x * np.sin(x)
        

        # Create fourier setup
        N = 100
        fourier_gp = FourierGroup(N, func_ex, c=3, length=L, dampening=0.0)
        self.play(ShowCreation(fourier_gp[0]), run_time=1)
        self.wait()
        self.play(FadeOut(fourier_gp[0]), run_time=0.5)
        self.play(LaggedStart(ShowCreation(fourier_gp[2][i]) for i in range(N)), run_time=5, rate_func=linear)
        intermediate_fourier_waves = [FourierWave(n, fourier_gp.y_of_x_or_pts, fourier_gp.c, length=L, dampening=0.1) for n in range(1, N + 1)]
        self.play(LaggedStart(TransformFromCopy(fourier_gp[2][i], intermediate_fourier_waves[i]) for i in range(len(intermediate_fourier_waves))), run_time=5, rate_func=slow_into)
        self.play(FadeOut(VGroup(intermediate_fourier_waves)), FadeIn(fourier_gp[1]))


        # Animate fourier wave gp
        t_tracker = ValueTracker(0)
        fourier_gp.attach_value_tracker(t_tracker)
        self.play(t_tracker.animate.set_value(20), run_time=20, rate_func=linear)



