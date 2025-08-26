from manim import *

class StickFigure(VMobject):
    """
    A class that represents 2D Stickfigure
    The face consists of a circle, an arc, and two dots.
    """

    def __init__(self, height=2.0, color=WHITE, **kwargs):
        super().__init__(**kwargs)

        self.height = height
        self.head_radius = height * 0.3
        self.head = Circle(radius=self.head_radius, color=color)
        self.body = Line(ORIGIN, DOWN * height, color=color).next_to(self.head, DOWN, buff=0)
        self.left_arm = Line(ORIGIN, DOWN * height * 0.5 + LEFT * height * 0.3, color=color).next_to(self.body, LEFT, buff=0)
        self.right_arm = Line(ORIGIN, DOWN * height * 0.5 + RIGHT * height * 0.3, color=color).next_to(self.body, RIGHT, buff=0)
        self.left_leg = Line(ORIGIN, DOWN * height * 0.5 + LEFT * height * 0.2, color=color).next_to(self.body, DOWN, buff=0).shift(LEFT * height * 0.1)
        self.right_leg = Line(ORIGIN, DOWN * height * 0.5 + RIGHT * height * 0.2, color=color).next_to(self.body, DOWN, buff=0).shift(RIGHT * height * 0.1)

        self.add(self.head, self.body, self.right_arm, self.left_arm, self.right_leg, self.left_leg).center()

class FacedStickFigure(StickFigure):
    """
    A class that represents 2D Stickfigure
    The face consists of a circle, an arc, and two dots.
    """

    def __init__(self, facing=ORIGIN, **kwargs):
        super().__init__(**kwargs)
        eye_length = (self.head_radius / 3)
        self.eye_length = eye_length
        self.facing = facing
        self.set_smile_face()

    def set_smile_face(self):
        eye_length = self.eye_length
        face = VGroup(
            Line(ORIGIN, UP * eye_length).shift((UP + LEFT) * (eye_length / 3)),
            Line(ORIGIN, UP * eye_length).shift((UP + RIGHT) * (eye_length / 3)),
            Arc(radius=self.head_radius / 3, angle=PI / 3).rotate(-PI * 2 / 3).shift(
                DOWN * (self.head_radius / 2.5) + LEFT * (eye_length / 1.5))
        ).move_to(self.head.get_center() + self.facing * (self.head_radius / 6))
        face.save_state()
        self.face = face
        if face not in self:
            self.add(face)

    def set_slant_face(self):
        eye_length = self.eye_length
        face = VGroup(
            Line(ORIGIN, UP * eye_length).shift((UP + LEFT) * (eye_length / 3)),
            Line(ORIGIN, UP * eye_length).shift((UP + RIGHT) * (eye_length / 3)),
            Line(LEFT * (self.head_radius / 3), RIGHT * (self.head_radius / 3), color=color).shift(
                DOWN * (self.head_radius / 5)),
        ).move_to(self.head.get_center() + self.facing * (self.head_radius / 6))
        face.save_state()
        self.face = face
        if face not in self:
            self.add(face)

    def set_frown_face(self):
        eye_length = self.eye_length
        face = VGroup(
            Line(ORIGIN, UP * eye_length).shift((UP + LEFT) * (eye_length / 3)),
            Line(ORIGIN, UP * eye_length).shift((UP + RIGHT) * (eye_length / 3)),
            Arc(radius=self.head_radius / 3, angle=PI / 3).rotate(-PI * 2 / 3).shift(
                DOWN * (self.head_radius / 2.5) + LEFT * (eye_length / 1.5))
        ).move_to(self.head.get_center() + self.facing * (self.head_radius / 6))
        face.save_state()
        self.face = face
        if face not in self:
            self.add(face)

    def set_surprised_face(self):
        eye_length = self.eye_length
        face = VGroup(
            Line(ORIGIN, UP * eye_length).shift((UP + LEFT) * (eye_length / 3)),
            Line(ORIGIN, UP * eye_length).shift((UP + RIGHT) * (eye_length / 3)),
            Arc(radius=self.head_radius / 3, angle=PI / 3).rotate(-PI * 2 / 3).shift(
                DOWN * (self.head_radius / 2.5) + LEFT * (eye_length / 1.5))
        ).move_to(self.head.get_center() + self.facing * (self.head_radius / 6))
        face.save_state()
        self.face = face
        if face not in self:
            self.add(face)

    def remove_face(self):
        self.remove(self.face)


class SmilingStickFigure(StickFigure):
    """
    A class that represents 2D Stickfigure
    The face consists of a circle, an arc, and two dots.
    """

    def __init__(self, height=2.0, color=RED, facing=ORIGIN):
        StickFigure.__init__(self, height, color)
        eye_length = (self.head_radius / 3)
        face = VGroup(
            Line(ORIGIN, UP * eye_length, color=color).shift((UP + LEFT) * (eye_length / 3)),
            Line(ORIGIN, UP * eye_length, color=color).shift((UP + RIGHT) * (eye_length / 3)),
            Arc(radius=self.head_radius / 3, angle=PI / 3, color=color).rotate(-PI*2/3).shift(DOWN * (self.head_radius / 2.5) + LEFT * (eye_length/ 1.5))
        ).move_to(self.head.get_center() + facing * (self.head_radius / 6))
        self.face = face.save_state()
        self.add(face)

class SlantedStickFigure(StickFigure):
    """
    A class that represents 2D Stickfigure
    The face consists of a circle, an arc, and two dots.
    """

    def __init__(self, height=2.0, color=RED, facing=ORIGIN):
        StickFigure.__init__(self, height=height, color=color)
        eye_length = (self.head_radius / 3)
        face = VGroup(
            Line(ORIGIN, UP * eye_length, color=color).shift((UP + LEFT) * (eye_length / 3)),
            Line(ORIGIN, UP * eye_length, color=color).shift((UP + RIGHT) * (eye_length / 3)),
            Line(LEFT * (self.head_radius / 3), RIGHT * (self.head_radius / 3), color=color).shift(DOWN * (self.head_radius / 5)),
        ).move_to(self.head.get_center() + facing * (self.head_radius / 6))
        self.face = face

        self.add(face).center()

    def talk_anim(self, run_time=1):
        talk_anim = []
        self.face[2].save_state()
        for i in range(int(np.ceil(run_time))):
            talk_anim.append(LaggedStart(
                self.face[2].animate.scale(0.5).scale_to_fit_height(self.head_radius / 6),
                self.face[2].animate.restore(),
                lag_ratio=1,
                run_time=run_time
            )
            )

        return LaggedStart(*talk_anim, lag_ratio=1)