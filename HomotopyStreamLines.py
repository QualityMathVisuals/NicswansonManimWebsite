from manim import *
from collections.abc import Iterable, Sequence
from math import ceil, floor
from typing import Callable
import random
DEFAULT_SCALAR_FIELD_COLORS: list = [BLUE, PURPLE, PINK]


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return(x, y)

class HomotopyStreamLines(VectorField):
    def __init__(
        self,
        func: Callable[[np.complex128, float], np.complex128],
        color: ParsableManimColor | None = None,
        color_scheme: Callable[[np.ndarray], float] | None = None,
        min_color_scheme_value: float = 0,
        max_color_scheme_value: float = 2,
        colors: Sequence[ParsableManimColor] = DEFAULT_SCALAR_FIELD_COLORS,
        # Determining stream line starting positions:
        x_range: Sequence[float] = None,
        y_range: Sequence[float] = None,
        noise_factor: float | None = None,
        n_repeats=1,
        padding=3,
        num_anchors_per_line=100,
        # Determining stream line appearance:
        stroke_width=1,
        opacity=1,
        **kwargs,
    ):
        self.x_range = x_range or [
            floor(-config["frame_width"] / 2),
            ceil(config["frame_width"] / 2),
        ]
        self.y_range = y_range or [
            floor(-config["frame_height"] / 2),
            ceil(config["frame_height"] / 2),
        ]
        self.ranges = [self.x_range, self.y_range]

        for i in range(len(self.ranges)):
            if len(self.ranges[i]) == 2:
                self.ranges[i] += [0.5]
            self.ranges[i][1] += self.ranges[i][2]

        self.x_range, self.y_range = self.ranges

        vector_valued_function = complex_func_to_R3_func(lambda z: func(z, 1))

        super().__init__(
            vector_valued_function,
            color,
            color_scheme,
            min_color_scheme_value,
            max_color_scheme_value,
            colors,
            **kwargs,
        )

        self.noise_factor = (
            noise_factor if noise_factor is not None else self.y_range[2] / 2
        )
        self.n_repeats = n_repeats
        self.padding = padding
        self.stroke_width = stroke_width

        half_noise = self.noise_factor / 2
        start_points = np.array(
            [
                (x - half_noise) * RIGHT
                + (y - half_noise) * UP
                + self.noise_factor * np.random.random(3)
                for n in range(self.n_repeats)
                for x in np.arange(*self.x_range)
                for y in np.arange(*self.y_range)
            ],
        )

        def outside_box(p):
            return (
                p[0] < self.x_range[0] - self.padding
                or p[0] > self.x_range[1] + self.padding - self.x_range[2]
                or p[1] < self.y_range[0] - self.padding
                or p[1] > self.y_range[1] + self.padding - self.y_range[2]
            )

        if not self.single_color:
            self.background_img = self.get_colored_background_image()

        self.virtual_time = 1
        
        for point in start_points:
            z = complex(*point[:-1])
            intermediate_points = [point]
            for i in range(num_anchors_per_line):
                t = i / (num_anchors_per_line - 1)
                f_z = func(z, t)
                new_point = RIGHT * f_z.real + UP * f_z.imag + OUT * point[2]
                if outside_box(new_point):
                    break
                intermediate_points.append(new_point)

            line = VMobject()
            line.duration = 1
            line.set_points_smoothly(intermediate_points)
            if self.single_color:
                line.set_stroke(
                    color=self.color, width=self.stroke_width, opacity=opacity
                )
            else:
                if config.renderer == RendererType.OPENGL:
                    # scaled for compatibility with cairo
                    line.set_stroke(width=self.stroke_width / 4.0)
                    norms = np.array(
                        [np.linalg.norm(self.func(point)) for point in line.points],
                    )
                    line.set_rgba_array_direct(
                        self.values_to_rgbas(norms, opacity),
                        name="stroke_rgba",
                    )
                else:
                    line.set_stroke(
                        [self.pos_to_color(p) for p in line.get_anchors()],
                    )
                    line.set_stroke(width=self.stroke_width, opacity=opacity)
            self.add(line)
        self.stream_lines = [*self.submobjects]

    def create(
        self,
        lag_ratio: float | None = None,
        run_time: Callable[[float], float] | None = None,
        **kwargs,
    ) -> AnimationGroup:
        if run_time is None:
            run_time = self.virtual_time
        if lag_ratio is None:
            lag_ratio = run_time / 2 / len(self.submobjects)

        animations = [
            Create(line, run_time=run_time, **kwargs) for line in self.stream_lines
        ]
        random.shuffle(animations)
        return AnimationGroup(*animations, lag_ratio=lag_ratio)

    def start_animation(
        self,
        warm_up: bool = True,
        flow_speed: float = 1,
        time_width: float = 0.3,
        rate_func: Callable[[float], float] = linear,
        line_animation_class: type[ShowPassingFlash] = ShowPassingFlash,
        **kwargs,
    ) -> None:
        for line in self.stream_lines:
            run_time = line.duration / flow_speed
            line.anim = line_animation_class(
                line,
                run_time=run_time,
                rate_func=rate_func,
                time_width=time_width,
                **kwargs,
            )
            line.anim.begin()
            line.time = random.random() * self.virtual_time
            if warm_up:
                line.time *= -1
            self.add(line.anim.mobject)

        def updater(mob, dt):
            for line in mob.stream_lines:
                line.time += dt * flow_speed
                if line.time >= self.virtual_time:
                    line.time -= self.virtual_time
                line.anim.interpolate(np.clip(line.time / line.anim.run_time, 0, 1))

        self.add_updater(updater)
        self.flow_animation = updater
        self.flow_speed = flow_speed
        self.time_width = time_width


    def end_animation(self) -> AnimationGroup:
        if self.flow_animation is None:
            raise ValueError("You have to start the animation before fading it out.")

        def hide_and_wait(mob, alpha):
            if alpha == 0:
                mob.set_stroke(opacity=0)
            elif alpha == 1:
                mob.set_stroke(opacity=1)

        def finish_updater_cycle(line, alpha):
            line.time += dt * self.flow_speed
            line.anim.interpolate(min(line.time / line.anim.run_time, 1))
            if alpha == 1:
                self.remove(line.anim.mobject)
                line.anim.finish()

        max_run_time = self.virtual_time / self.flow_speed
        creation_rate_func = utils.rate_functions.ease_out_sine
        creation_staring_speed = creation_rate_func(0.001) * 1000
        creation_run_time = (
            max_run_time / (1 + self.time_width) * creation_staring_speed
        )
        # creation_run_time is calculated so that the creation animation starts at the same speed
        # as the regular line flash animation but eases out.

        dt = 1 / config["frame_rate"]
        animations = []
        self.remove_updater(self.flow_animation)
        self.flow_animation = None

        for line in self.stream_lines:
            create = Create(
                line,
                run_time=creation_run_time,
                rate_func=creation_rate_func,
            )
            if line.time <= 0:
                animations.append(
                    Succession(
                        UpdateFromAlphaFunc(
                            line,
                            hide_and_wait,
                            run_time=-line.time / self.flow_speed,
                        ),
                        create,
                    ),
                )
                self.remove(line.anim.mobject)
                line.anim.finish()
            else:
                remaining_time = max_run_time - line.time / self.flow_speed
                animations.append(
                    Succession(
                        UpdateFromAlphaFunc(
                            line,
                            finish_updater_cycle,
                            run_time=remaining_time,
                        ),
                        create,
                    ),
                )
        return AnimationGroup(*animations)


class zSquared(Scene):
    def construct(self):
        # slc = HomotopyStreamLines(
        #     lambda z, t: z**(1 + t),
        #     x_range=[-7, 7, 0.25],
        #     y_range=[-4, 4, 0.25],
        #     max_color_scheme_value=50,
        # )
        # slc = HomotopyStreamLines(
        #     lambda z, t: np.exp((1 - t) * np.log(z) + t * np.log(z**2)),
        #     x_range=[-7, 7, 0.25],
        #     y_range=[-4, 4, 0.25],
        #     max_color_scheme_value=50,
        # )
        # slc = HomotopyStreamLines(
        #     lambda z, t: t*z**2 + (1 - t) * z,
        #     x_range=[-7, 7, 0.25],
        #     y_range=[-4, 4, 0.25],
        #     max_color_scheme_value=50,
        # )
        slc = HomotopyStreamLines(
            lambda z, t: np.abs(z)**(1 + t) * np.exp(1j * (1 + t) * (np.angle(z) % (2 * np.pi))),
            x_range=[-2, 2, 0.25],
            y_range=[-2, 0, 0.25],
            max_color_scheme_value=50,
        )

        slc.start_animation(
            warm_up=True,
            flow_speed=0.5,
            time_width=0.3
        )
        self.add(slc)
        self.wait(1)


manim_configuration = {
    "quality": "low_quality",
    "preview": False,
    "output_file": "PreviewVideo",
    "disable_caching": True,
    "max_files_cached": 1000,
    "write_to_movie": True,
    "show_file_in_browser": False,
}
if __name__ == "__main__":
    with tempconfig(manim_configuration):
        np.random.seed(0)
        scene = zSquared()
        scene.render()
