from manimlib import *
import numpy as np
from scipy.special import jv


Jmn = [
    [0, 2.40, 5.52, 8.65, 11.79, 14.93],
    [0, 3.83, 7.02, 10.17, 13.32, 16.47],
    [0, 5.14, 8.42, 11.62, 14.80, 17.96],
    [0, 6.38, 9.76, 13.02, 16.22, 19.41],
    [0, 7.59, 11.06, 14.37, 17.62, 20.83],
    [0, 8.77, 12.34, 15.70, 18.98, 22.22]
]
Amn = 1
gmn = 0
phimn = 0
c = 1

def drum_equation(r, theta, t, m, n, a):
    jmn = Jmn[m][n]
    kmn = jmn/a
    omegamn = kmn * c
    return Amn * jv(m, r * kmn) * np.cos(m * theta + gmn) * np.cos(omegamn * t + phimn)

class DrumRings(VGroup3D):
    def __init__(self, m, n, a, t, rings=75, theta_res=50, colors = None, **kwargs):
        super().__init__(**kwargs)
        self.m = m
        self.n = n
        self.a = a
        self.rings = rings
        self.theta_res = theta_res
        self.radial_delta = a / rings
        self.theta_delta = TAU / theta_res
        if colors is None:
            self.colors = color_gradient([BLUE_C, BLUE_A], rings)
        else:
            self.colors = colors

        for i in range(1, rings + 1):
            r = self.radial_delta * i
            ring_pts = self.points_on_ring(r, t)
            ring_mob = VMobject(color=self.colors[i - 1])
            ring_mob.set_points_smoothly(ring_pts)
            self.add(ring_mob)

    def points_on_ring(self, r, t):
        pts = []
        for j in range(self.theta_res + 1):
            theta = self.theta_delta * (j + 1)
            point = [r * np.cos(theta), r * np.sin(theta), drum_equation(r, theta, t, self.m, self.n, self.a)]
            pts.append(point)
        return pts

    def set_t_tracker(self, t_tracker):
        def drum_updater(m):
            for i in range(1, m.rings + 1):
                r = self.radial_delta * i
                m[i - 1].set_points_smoothly(m.points_on_ring(r, t_tracker.get_value()))
            return m
        self.add_updater(drum_updater)

class DrumSurface(ParametricSurface):
    def __init__(self, m, n, a, t, rings=75, theta_res=50, color=None, **kwargs):
        self.m = m
        self.n = n
        self.a = a
        self.t = t
        self.rings = rings
        self.theta_res = theta_res
        if color is None:
            self.color=BLUE_D
        else:
            self.color=color

        super().__init__(
            lambda r, theta: [
                r * np.cos(theta),
                r * np.sin(theta),
                drum_equation(r, theta, t, m, n, a)
            ],
            u_range=[0, a],
            v_range=[0, TAU],
            color=self.color,
            resolution=(rings, theta_res + 1),
            shading=(0.7, 0.1, 0.7),
            **kwargs
        )

    def get_mesh(self):
        return SurfaceMesh(self, resolution=(self.rings, self.theta_res + 1), stroke_color=BLUE_B, stroke_width=2)

class CircularMembraneSmall(InteractiveScene):
    def construct(self):

        # Extract system args: manimgl -wm --filename StandingWaveFixedRim.m$m.n$n.i$i vibrations.py CircularMembraneSmall $m $n $i
        import sys
        args = sys.argv
        i = int(args[-1])
        n = int(args[-2])
        m = int(args[-3])

        # Add the value tracker
        t_val = ValueTracker(0)
        def write_equations():
            equations = VGroup(
                VGroup(Tex("t = "), DecimalNumber(t_val.get_value(), num_decimal_places=2)).arrange(RIGHT),
                VGroup(Tex("m = "), Integer(m)).arrange(RIGHT),
                VGroup(Tex("n = "), Integer(n)).arrange(RIGHT)
            ).arrange(DOWN, aligned_edge=LEFT)
            equations.fix_in_frame()
            equations.to_corner(UL)
            equations.set_backstroke()
            return equations
        equations = write_equations()
        self.play(Write(equations))
        equations.add_updater(lambda m: m.become(write_equations()))

        # Make the drum
        a = 5
        self.frame.reorient(43, 45, 1, IN, 10)
        self.frame.add_updater(lambda m, dt: m.increment_theta(dt * 3 * DEGREES))
        if i == 0:
            drum = DrumRings(m, n, a, t_val.get_value())
        elif i == 1:
            drum_surface = DrumSurface(m, n, a, t_val.get_value())
            drum_surface_mesh = drum_surface.get_mesh()
            drum = drum_surface_mesh
        elif i == 2:
            drum_surface = DrumSurface(m, n, a, t_val.get_value())
            drum_surface.always_sort_to_camera(self.camera)
            drum = drum_surface
        self.play(ShowCreation(drum), run_time=2)
        if i == 0:
            drum.set_t_tracker(t_val)
        elif i == 1:
            drum.add_updater(lambda drum: drum.become(DrumSurface(m, n, a, t_val.get_value()).get_mesh()))
        elif i == 2:
            drum.add_updater(lambda drum: drum.become(DrumSurface(m, n, a, t_val.get_value())))

        # Show drum humming
        self.play(t_val.animate.increment_value(2.5), run_time=10, rate_func=linear)
        self.wait()




class CircularMembrane(InteractiveScene):
    def construct(self):
        
        # Add the value trackers
        t_val = ValueTracker(0)
        m_val = ValueTracker(0)
        n_val = ValueTracker(1)
        def write_equations():
            equations = VGroup(
                VGroup(Tex("t = "), DecimalNumber(t_val.get_value(), num_decimal_places=2)).arrange(RIGHT),
                VGroup(Tex("m = "), Integer(m_val.get_value())).arrange(RIGHT),
                VGroup(Tex("n = "), Integer(n_val.get_value())).arrange(RIGHT)
            ).arrange(DOWN, aligned_edge=LEFT)
            equations.fix_in_frame()
            equations.to_corner(UL)
            equations.set_backstroke()
            return equations
        equations = write_equations()
        self.play(Write(equations))
        equations.add_updater(lambda m: m.become(write_equations()))

        # Settup drum from circles
        a = 5
        drum_equation(2, 2, 1, 0, 1, 5)
        drum = DrumRings(0, 1, a, t_val.get_value())
        self.frame.reorient(43, 45, 1, IN, 10)
        self.frame.add_updater(lambda m, dt: m.increment_theta(dt * 3 * DEGREES))
        self.play(ShowCreation(drum))
        drum.set_t_tracker(t_val)

        # Cycle through various m and n, animate movement
        def animate_and_cycle(drum):
            for m in range(1, 4):
                for n in range(1, 4):
                    if isinstance(drum, DrumSurface):
                        new_drum = DrumSurface(m, n, a, t_val.get_value())
                        drum.clear_updaters()
                        self.play(
                            ReplacementTransform(drum, new_drum),
                            m_val.animate.set_value(m),
                            n_val.animate.set_value(n)
                        )
                    else:
                        if isinstance(drum, DrumRings):
                            new_drum = DrumRings(m, n, a, t_val.get_value())
                            new_drum.set_t_tracker(t_val)
                        elif isinstance(drum, SurfaceMesh):
                            new_drum = DrumSurface(m, n, a, t_val.get_value()).get_mesh()
                            drum.clear_updaters()
                        self.play(
                            TransformMatchingParts(drum, new_drum, matched_pairs=zip(drum, new_drum)),
                            m_val.animate.set_value(m),
                            n_val.animate.set_value(n),
                        )
                    drum = new_drum

                    if isinstance(drum, DrumSurface):
                        drum.add_updater(lambda drum: drum.become(DrumSurface(m, n, a, t_val.get_value())))
                    elif isinstance(drum, SurfaceMesh):
                        drum.add_updater(lambda drum: drum.become(DrumSurface(m, n, a, t_val.get_value()).get_mesh()))
                    self.play(t_val.animate.increment_value(1), run_time=3, rate_func=linear)
            return drum

        # Animate ring drum
        drum = animate_and_cycle(drum)
        # Switch to Mesh model
        drum_surface = DrumSurface(int(m_val.get_value()), int(n_val.get_value()), a, t_val.get_value())
        drum_surface_mesh = drum_surface.get_mesh()
        self.play(TransformMatchingShapes(drum, drum_surface_mesh, matched_pairs=zip(drum, drum_surface_mesh)), run_time=3)
        drum = drum_surface_mesh

        # Animate mesh drum
        drum = animate_and_cycle(drum)

        # Switch to Surface model
        drum_surface = DrumSurface(int(m_val.get_value()), int(n_val.get_value()), a, t_val.get_value())
        drum_surface.always_sort_to_camera(self.camera)
        drum.clear_updaters()
        self.play(TransformMatchingParts(drum, drum_surface, run_time=3), self.camera.light_source.animate.move_to(OUT * 3))
        drum = drum_surface

        # Animate surface drum
        animate_and_cycle(drum)


