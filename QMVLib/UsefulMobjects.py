from manim import *

class SmileyFace(VMobject):
    """
    A class that represents 2D smileyface.
    The face consists of a circle, an arc, and two dots.
    """

    def __init__(self, radius=1, color=YELLOW_A, **kwargs):
        VMobject.__init__(self)

        circle = Circle(radius=radius, color=color)
        arc = Arc(radius=radius * 0.8, start_angle=PI * 5 / 4, angle=PI / 2, color=color)
        eyes = [Dot(point=[radius * 0.3, radius * 0.6, 0], color=color),
                Dot(point=[radius * -0.3, radius * 0.6, 0], color=color)]
        self.add(circle, arc, *eyes)

class SpeechBubble(VMobject):
    """
    A class that represents a speech bubble.
    The bubble consists of a rounded rectangle with a tail.
    """

    def __init__(self, width=4, height=2, tail_length=1, **kwargs):
        VMobject.__init__(self)

        # Create the rounded rectangle
        rect = RoundedRectangle(width=width, height=height, corner_radius=0.5)

        # Create the tail
        rounding_corner_length = 0.5
        tail = Polygon(
            [width / 2 - rounding_corner_length, -height / 2, 0],
            [width / 2 + tail_length - rounding_corner_length, -height / 2 - tail_length, 0],
            [width / 2 - rounding_corner_length - tail_length, -height / 2, 0]
        )

        # Add the rectangle and tail to the speech bubble
        self.add(rect, tail)

class CoffeeMug(VMobject):
    """
    A class that represents a 3D coffee mug object.
    The mug is composed of a cylinder base and a torus handle.
    """

    def __init__(self, resolution=10, base_color='#B1945C', light_gray_color=LIGHT_GRAY, fill_opacity=0.9,
                 stroke_width=7, stroke_color='#B1945C', stroke_opacity=0.8, **kwargs):
        """
        Parameters:
            resolution (int): The number of edges on the torus and cylinder.
            base_color (str): The color of the base of the mug.
            light_gray_color (str): The color of the handle of the mug.
            fill_opacity (float): The opacity of the fill color.
            stroke_width (float): The width of the stroke.
            stroke_color (str): The color of the stroke.
            stroke_opacity (float): The opacity of the stroke.
        """
        VMobject.__init__(self)
        self.resolution = resolution
        self.base_color = base_color
        self.light_gray_color = light_gray_color
        self.fill_opacity = fill_opacity
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.stroke_opacity = stroke_opacity

        base = Cylinder(radius=1, height=2, resolution=self.resolution,
                        checkerboard_colors=[self.base_color, self.light_gray_color],
                        fill_opacity=self.fill_opacity,
                        fill_color=self.light_gray_color,
                        stroke_width=self.stroke_width,
                        stroke_color=self.stroke_color,
                        stroke_opacity=self.stroke_opacity)
        handle = Torus(major_radius=0.6, minor_radius=0.2, resolution=self.resolution,
                       checkerboard_colors=[self.base_color, self.light_gray_color],
                       fill_opacity=self.fill_opacity,
                       fill_color=self.light_gray_color,
                       stroke_width=self.stroke_width / 2,
                       stroke_color=self.stroke_color,
                       stroke_opacity=self.stroke_opacity)
        handle.rotate(TAU / 4, axis=RIGHT)
        handle.rotate(TAU / 4, axis=IN)
        handle.shift(0.85 * UP)
        self.add(base, handle)


class ArrowDoubleEnded3D(Arrow3D):
    """An arrow made out of a cylindrical line and two conical tips.

    Parameters
    ----------
    start
        The start position of the arrow.
    end
        The end position of the arrow.
    thickness
        The thickness of the arrow.
    height
        The height of the conical tip.
    base_radius
        The base radius of the conical tip.
    color
        The color of the arrow.

    Examples
    --------
    .. manim:: ExampleArrow3D
        :save_last_frame:

        class ExampleArrow3D(ThreeDScene):
            def construct(self):
                axes = ThreeDAxes()
                arrow = ArrowDoubleEnded3D(
                    start=np.array([0, 0, 0]),
                    end=np.array([2, 2, 2]),
                    resolution=8
                )
                self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                self.add(axes, arrow)
    """

    def __init__(
            self,
            start: np.ndarray = LEFT,
            end: np.ndarray = RIGHT,
            thickness: float = 0.02,
            height: float = 0.3,
            base_radius: float = 0.08,
            show_start_arrow: bool = True,
            color=WHITE,
            **kwargs,
    ):
        super().__init__(
            start=start, end=end, thickness=thickness, height=height, base_radius=base_radius, color=color, **kwargs
        )

        self.cone_2 = Cone(
            direction=(-1) * self.direction, base_radius=base_radius, height=height, **kwargs
        )
        self.cone_2.shift(start)
        self.add(self.cone_2)

        if not show_start_arrow:
            self.cone_2.set_opacity(0)

        self.cone_2.set_color(color)

    def get_start_arrow(
            self,
    ):
        """Returns the cone at the start of the arrow.
        """
        return self.cone_2

    def show_start_arrow(self):
        self.cone_2.set_opacity(1)

    @classmethod
    def get_end_arrow(
            cls,
    ):
        """Returns the cone at the start of the arrow.
        """
        return cls.cone


def WriteAndFadeOut(mobject, delay=1, run_time=2):
    write_fade_time = (run_time - delay) / 2
    write = Write(mobject, run_time=write_fade_time)
    wait = Wait(delay)
    fadeout = FadeOut(mobject, run_time=write_fade_time)
    return LaggedStart(write, wait, fadeout, lag_ratio=1)

class BillBoard(VMobject):
    def __init__(self, width=4, height=2, **kwargs):
        VMobject.__init__(self)
        self.board = RoundedRectangle(
            width=width, height=height, **kwargs
        )
        self.legs = VGroup(
            Line(ORIGIN, DOWN * height / 4, **kwargs).shift(LEFT * width / 4 + DOWN * height / 2),
            Line(ORIGIN, DOWN * height / 4, **kwargs).shift(RIGHT * width / 4 + DOWN * height / 2)
        )
        self.add(self.board, self.legs)

class Heart(VMobject):
    def __init__(self, x=0, y=0, color=RED_A):
        VMobject.__init__(self)
        el1 = CubicBezier([x, y, 0], [x, y - .3, 0], [x - .5, y - .3, 0], [x - 0.5, y, 0])
        el2 = CubicBezier([x - .5, y, 0], [x - .5, y + .3, 0], [x, y + 0.35, 0], [x, y + 0.6, 0])
        el3 = CubicBezier([x, y + .6, 0], [x, y + .35, 0], [x + .5, y + .3, 0], [x + .5, y, 0])
        el4 = CubicBezier([x + .5, y, 0], [x + .5, y - .3, 0], [x, y - .3, 0], [x, y, 0])

        self.add(VGroup(el1, el2, el3, el4).rotate(180 * DEGREES).set_color(RED_A))
