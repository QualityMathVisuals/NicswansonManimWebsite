'''
Chapter 1: 1D Waves and Fourier Series
'''
from manimlib import *
from scipy.misc import derivative
from _2025.Vibrations.WavesV3 import *
from QMVlib import * 

class Chapter1Scene(InteractiveScene):
    def construct(self):
        t_tracker = ValueTracker(0)
        L = 10
        wave_config = WaveConfig(dampening=0.0, wave_speed=6, length=L, amplitude=2)
        points_A = [(0, 0)]
        points_A += [(2, 3)]
        points_A += [(L, 0)]
        N = 100
        guitar_string_gp = FourierGroup(
            num_harmonics=N,
            wave_function=points_A,
            wave_config=wave_config
        )
        guitar_string_real = guitar_string_gp.transverse_wave.save_state()
        flat_line = Line(start=guitar_string_real.get_start(), end=guitar_string_real.get_end()).save_state()
        end_points = VGroup(
            Dot(radius=1, z_index=1).move_to(guitar_string_real.get_start()).set_color(GREY_C),
            Dot(radius=1, z_index=1).move_to(guitar_string_real.get_end()).set_color(GREY_C),
        )
        finger_point = Dot(radius=0.1, z_index=1).move_to(end_points[0]).shift(RIGHT * 2).set_color(LIGHT_BROWN)
        fourier_guitar_str = guitar_string_gp.fourier_wave.save_state()
        fourier_guitar_str.attach_value_tracker(t_tracker)

        # Show Creation of string and end points.
        self.play(ShowCreation(flat_line))
        self.play(FlashyFadeIn(end_points[0]), end_points[0].animate.scale(0.15))
        self.play(FlashyFadeIn(end_points[1]), end_points[1].animate.scale(0.15))

        # Animate string being lifted
        pause = Text("| |").scale(5).set_color(GREY_A)
        amp_markup = MarkupText.MARKUP_ENTITY_DICT.pop('&')
        play_text = Text(R"&#9658;").scale(5).set_color(GREY_A)
        MarkupText.MARKUP_ENTITY_DICT['&'] = amp_markup
        self.play(FlashyFadeIn(finger_point))
        self.play(ReplacementTransform(flat_line, guitar_string_real), finger_point.animate.shift(UP * 3), run_time=2)
        flat_line.restore()
        self.play(FadeOut(finger_point), FadeIn(pause), run_time=0.2)
        self.wait(2)
        self.play(FadeOut(pause), FadeIn(play_text), t_tracker.animate.increment_value(0.2), run_time=0.2)
        self.replace(guitar_string_real, fourier_guitar_str)
        self.play(FadeOut(play_text), t_tracker.animate.increment_value(0.2), run_time=0.2)

        # Show guitar string release.
        self.play(t_tracker.animate.increment_value(10), run_time=10, rate_func=linear)

        # Write wave equation and show concavity.
        moral_wave_label = Text('Wave equation:')
        moral_wave_equation = Tex(
            R"""
            \frac{\partial ^2 y}{\partial t ^2} = \frac{\partial^2 y}{\partial x^2} 
            """,
            t2c={
                R"\frac{\partial ^2 y}{\partial t ^2}": YELLOW,
                R"\frac{\partial^2 y}{\partial x^2}": PINK,
                "y": WHITE
            }
        )
        moral_wave_text = VGroup(
            moral_wave_label,
            moral_wave_equation
        ).arrange(DOWN).shift(UP * 2)
        asteriks = Text("*boring physical constants have been omitted").scale(0.5).to_corner(DR)
        concavity_arrows = []
        velocity_arrows = []
        x_offset = L // 2
        for x in range(1, int(L)):
            def updater_a(arrow, x=x):
                start = ((x - x_offset) * RIGHT) + (UP * fourier_guitar_str.wave_function(x, t_tracker.get_value()))
                second_deriv = derivative(lambda X: fourier_guitar_str.wave_function(X, t_tracker.get_value()), x, n=2) / 1.5
                arrow.set_points_by_ends(start=start, end=start + UP * second_deriv)
            def updater_v(arrow, x=x):
                start = ((x - x_offset) * RIGHT) + (UP * fourier_guitar_str.wave_function(x, t_tracker.get_value()))
                deriv = derivative(lambda T: fourier_guitar_str.wave_function(x, T), t_tracker.get_value(), n=1, dx=0.001) / 10
                arrow.set_points_by_ends(start=start, end=start + UP * deriv)
            acc_arrow = Arrow(max_tip_length_to_length_ratio=0.3, fill_color=PINK, z_index=2)
            acc_arrow.add_updater(updater_a)
            concavity_arrows += [acc_arrow]
            vel_arrow = Arrow(max_tip_length_to_length_ratio=0.3, fill_color=GREEN_D, z_index=1)
            vel_arrow.add_updater(updater_v)
            velocity_arrows += [vel_arrow]
        self.add(*concavity_arrows)
        self.play(t_tracker.animate.increment_value(5), rate_func=linear, run_time=5)
        self.play(Write(moral_wave_text), t_tracker.animate.increment_value(2), rate_func=linear, run_time=2)
        self.play(moral_wave_text.animate.to_corner(UR), t_tracker.animate.increment_value(1), rate_func=linear, run_time=1)
        self.play(FadeIn(asteriks), t_tracker.animate.increment_value(2), rate_func=linear, run_time=2)
        self.play(t_tracker.animate.increment_value(1), rate_func=linear, run_time=1)

        # Add velocity arrows and slow down 
        self.add(*velocity_arrows)
        self.play(t_tracker.animate.increment_value(3), rate_func=linear, run_time=10)
        self.play(t_tracker.animate.increment_value(5), rate_func=linear, run_time=5)
        for arrow in concavity_arrows:
            arrow.clear_updaters()
        for arrow in velocity_arrows:
            arrow.clear_updaters()
        fourier_guitar_str.suspend_updating()
        self.play(FadeOut(asteriks), *[FadeOut(arrow) for arrow in concavity_arrows], *[FadeOut(arrow) for arrow in velocity_arrows], run_time=1)
        fourier_guitar_str.resume_updating()

        # Write the desired form of wave function, add a graph highlight inputs and outputs.
        isolate_tex = ["=", "y", "x", "t", "0", ","]
        self.play(FadeOut(fourier_guitar_str), run_time=0.5)
        t_tracker.set_value(0)
        self.play(FadeIn(fourier_guitar_str), run_time=0.5)
        desired_form = Tex(
            R"""
            y ( x , t ) = ??
            """,
            t2c={R"t": YELLOW, R"x": PINK,"y": WHITE},
            isolate=isolate_tex
        ).shift(DOWN * 2.5).save_state()
        axes = Axes(
            x_range = [0, 1],
            y_range = [-1, 1],
            width=L,
            height=6
        )
        axes.add_coordinate_labels([0, 1], [-1, 1])
        y_label = Tex("y",t2c={R"t": YELLOW, R"x": PINK,"y": WHITE}).shift(UP * 2.9 + LEFT * 4.5)
        x_label = Tex("x",t2c={R"t": YELLOW, R"x": PINK,"y": WHITE}).shift(UP * 0.4 + RIGHT * 5.3)
        t_label = Tex("t = ",t2c={R"t": YELLOW, R"x": PINK,"y": WHITE}).to_edge(UP)
        t_value = always_redraw(lambda: DecimalNumber(t_tracker.get_value(), num_decimal_places=1).next_to(t_label, RIGHT))
        dot_tracking = Dot(radius=0.1, fill_color=BLUE_C).move_to(RIGHT * (5 - 5) + UP * fourier_guitar_str.wave_function(5, t_tracker.get_value()))
        dot_tracking_info = always_redraw(lambda:
            VGroup(
                axes.get_line_from_axis_to_point(1, dot_tracking.get_center()),
                axes.get_line_from_axis_to_point(0, dot_tracking.get_center()),
                DecimalNumber((fourier_guitar_str.wave_function(dot_tracking.get_center()[0] + 5, t_tracker.get_value())) / 3, num_decimal_places=1).next_to(axes.get_y_axis().get_projection(dot_tracking.get_center()), LEFT),
                DecimalNumber((dot_tracking.get_center()[0] + 5) / L, num_decimal_places=1).next_to(axes.get_x_axis().get_projection(dot_tracking.get_center()), DOWN)
            )
        )
        self.play(Write(desired_form), run_time=1)
        self.play(DrawBorderThenFill(axes))
        self.play(Write(y_label))
        self.play(Write(x_label))
        t_value.suspend_updating()
        self.play(Write(t_label), Write(t_value))
        t_value.resume_updating()
        self.play(FadeIn(dot_tracking, scale=0.1))
        dot_tracking_info.suspend_updating()
        self.play(FadeIn(dot_tracking_info))
        dot_tracking_info.resume_updating()
        self.play(FlashAround(dot_tracking_info[2]))
        self.play(FlashAround(dot_tracking_info[3]))
        self.play(FlashAround(VGroup(t_label, t_value)))

        # Highlight inputs and outputs. Highlight t=0 graph.
        x_tracker = ValueTracker(5)
        dot_tracking.add_updater(lambda dot:
            dot.move_to(RIGHT * (x_tracker.get_value() - 5) + UP * fourier_guitar_str.wave_function(x_tracker.get_value(), t_tracker.get_value()))
        )
        yzero_tex = Tex(
            R"""
            y ( x , 0) = \begin{cases} 5x & x < 0.2 \\ \frac{-5}{4}x + \frac{5}{4} & x \ge 0.2 \end{cases}
            """,
            t2c={R"t": YELLOW, R"x": PINK,"y": WHITE},
            isolate=isolate_tex
        ).shift(DOWN * 2.5)
        self.play(x_tracker.animate.set_value(0), run_time=5)
        self.play(TransformMatchingStrings(desired_form, yzero_tex))
        self.play(x_tracker.animate.set_value(8), run_time=5)
        self.wait()

        # Fade in and out extra conditions, fade in function over time
        realism_str = Text("*totally very super realistic").scale(0.5).to_corner(DR)
        dot_tracking_info.suspend_updating()
        t_value.suspend_updating()
        desired_form.restore()
        self.play(FadeOut(VGroup(dot_tracking_info, dot_tracking, end_points, fourier_guitar_str, t_value, t_label, axes, x_label, y_label, yzero_tex)), FadeIn(desired_form))
        self.play(moral_wave_text.animate.center().shift(UP * 2.5 + RIGHT * 3), desired_form.animate.center().shift(UP * 2.5 + LEFT * 3))
        extra_conditions = [
            VGroup(
                Text('Fixed end points'),
                Tex(
                    R"""
                    y(0, t) = 0\\
                    y(1, t) = 0
                    """,
                    t2c={R"t": YELLOW, R"x": PINK,"y": WHITE}
                )
            ).arrange(DOWN).scale(0.8).shift(DOWN * 2 + LEFT * 2),
            VGroup(
                Text('Perfect reflection:'),
                Tex(
                    R"""
                    \frac{\partial y}{\partial x}(0, t) = \frac{\partial y}{\partial x}(1, t) = 0
                    """,
                    t2c={R"t": YELLOW, R"x": PINK,"y": WHITE}
                )
            ).arrange(DOWN).scale(0.8).shift(DOWN * 2 + RIGHT * 2),
        ]
        self.play(FadeIn(extra_conditions[0]), FadeIn(realism_str))
        self.play(FadeOut(extra_conditions[0]))
        self.wait()
        self.play(FadeIn(extra_conditions[1]))
        self.play(FadeOut(extra_conditions[1]), FadeOut(realism_str))
        desired_form_again = Tex(
            R"""
            y ( x , t ) =
            """,
            t2c={R"t": YELLOW, R"x": PINK,"y": WHITE},
            isolate=isolate_tex
        ).shift(DOWN * 2.8 + LEFT * 2).scale(1.3).save_state()
        solutions = [
            Tex(
                Rf"""
                \sin ({n} \pi x) \cos ({n} \pi t)
                """,
                t2c={R"t": YELLOW, R"x": PINK,"y": WHITE},
                isolate=isolate_tex
            ).scale(1.3).next_to(desired_form_again, RIGHT)
            for n in range(2, 14)
        ]
        one_sol = Tex(
                Rf"""
                \sin (\pi x) \cos (\pi t)
                """,
                t2c={R"t": YELLOW, R"x": PINK,"y": WHITE},
                isolate=isolate_tex
            ).scale(1.3).next_to(desired_form_again, RIGHT)
        solutions = [one_sol] + solutions
        self.play(TransformMatchingStrings(desired_form, desired_form_again), moral_wave_text.animate.center().to_edge(UP))
        self.play(Write(solutions[1]))
        self.wait()

        # Write out the product solution, then graph the n=2 mode.
        self.play(FlashAround(VGroup(desired_form_again, solutions[1])))
        self.play(FlashAround(moral_wave_text))
        self.wait()
        axes = Axes(
            x_range = [0, 1],
            y_range = [-1, 1],
            width=L,
            height=4
        )
        axes.add_coordinate_labels([0, 1], [-1, 1])
        self.play(FadeOut(moral_wave_text), DrawBorderThenFill(axes))
        self.play(LaggedStart(*[ShowCreation(txt) for txt in [t_label, t_value, x_label, y_label.shift(DOWN * 0.6)]]))
        standing_wave = StandingWave(mode=2, config=wave_config, color=QMV_BLUE_B)
        self.play(ShowCreation(standing_wave))
        self.wait()

        # Increase t_tracker the standing sine wave
        standing_wave.attach_value_tracker(t_tracker)
        t_value.resume_updating()
        self.play(t_tracker.animate.increment_value(10), rate_func=linear, run_time=10)
        t_value.suspend_updating()
        self.play(FadeOut(VGroup(t_label, t_value)))

        # Move the height of the amplitude up and down, change the mode of the graph, ending on a single mode, then back to 2.
        standing_wave.clear_updaters()
        more_standing_waves = [
            StandingWave(mode=n, config=wave_config, color=QMV_BLUE_A)
            for n in range(1, 14)
        ]
        self.play(standing_wave.animate.stretch(2, 1), rate_func=there_and_back, run_time=1.5)
        self.play(standing_wave.animate.stretch(0.2, 1), rate_func=there_and_back, run_time=1.5)
        self.play(ReplacementTransform(standing_wave, more_standing_waves[2]), TransformMatchingStrings(solutions[1], solutions[2], run_time=0.75))
        for n in range(2, 12):
            self.play(
                ReplacementTransform(more_standing_waves[n], more_standing_waves[n + 1]),
                TransformMatchingStrings(solutions[n], solutions[n + 1]),
                run_time=0.33
            )
        self.play(ReplacementTransform(more_standing_waves[-1], more_standing_waves[1]))

        # Create many standing waves at the same time, make the word dense appear and fade to point.
        general_sol = Tex(
            Rf"""
            \sin (n \pi x) \cos (n \pi t)
            """,
            t2c={R"t": YELLOW, R"x": PINK,"y": WHITE},
            isolate=isolate_tex + ["n"]
        ).scale(1.3).next_to(desired_form_again, RIGHT)
        general_sol["n"].set_color_by_gradient(QMV_BLUE_A)
        dense = Text(R"Dense").set_color_by_gradient(RED_D, PURPLE_A).scale(5)
        self.play(Uncreate(more_standing_waves[1]), t_tracker.animate.set_value(0), TransformMatchingStrings(solutions[-1], general_sol))
        self.play(FadeIn(dense), run_time=0.15)
        self.play(FadeOut(dense, scale=0.1), run_time=2)
        num_many_waves = 100
        color_grad = color_gradient([QMV_BLUE_A, QMV_PURPLE_A, QMV_BLUE_C, QMV_PURPLE_C], num_many_waves)
        colored_standing_waves = [
            StandingWave(mode=n, config=wave_config, color=color_grad[n - 1])
            for n in range(1, num_many_waves + 1)
        ]
        N_label = VGroup(Tex("N = ", t2c = {"N": QMV_PURPLE_C}), Integer(10)).arrange(RIGHT).next_to(t_value, RIGHT, buff=1)
        self.play(Write(N_label))
        self.play(
            LaggedStart(*[
                Succession(ShowCreation(wave), wave.animate.set_stroke(width=0.5))
                for wave in colored_standing_waves[:10]
            ]),
            run_time=5,
            rate_func=slow_into
        )
        N_label_again =  VGroup(Tex("N = ", t2c = {"N": QMV_PURPLE_C}), Integer(100)).arrange(RIGHT).move_to(N_label, aligned_edge=LEFT)
        self.play(TransformMatchingParts(N_label, N_label_again))
        self.play(
            LaggedStart(*[
                Succession(ShowCreation(wave), wave.animate.set_stroke(width=0.1))
                for wave in colored_standing_waves[10:]
            ]),
            run_time=10,
            rate_func=slow_into
        )

        # Uncreate the standing waves and show just two standing waves.
        self.play(
            LaggedStart(*[
                Uncreate(wave)
                for wave in colored_standing_waves
            ]),
            FadeOut(VGroup(desired_form_again, general_sol, N_label_again)),
            run_time=2
        )
        half_amp_config = copy.deepcopy(wave_config)
        half_amp_config.amplitude = 1
        third_amp_config = copy.deepcopy(wave_config)
        third_amp_config.amplitude = 2/3
        individual_standing_waves = [
            VGroup(
                StandingWave(mode=1, config=half_amp_config, color=QMV_BLUE_A),
                StandingWave(mode=3, config=half_amp_config, color=QMV_PURPLE_A),
            ),
            VGroup(
                StandingWave(mode=2, config=third_amp_config, color=QMV_BLUE_A),
                StandingWave(mode=7, config=wave_config, color=QMV_PURPLE_A),
            ),
            VGroup(
                StandingWave(mode=1, config=wave_config, color=QMV_BLUE_A),
                StandingWave(mode=2, config=half_amp_config, color=QMV_BLUE_C),
                StandingWave(mode=5, config=third_amp_config, color=QMV_PURPLE_A),
            ),
        ]
        t_tracker.set_value(0)
        t_value.resume_updating()
        self.play(LaggedStart(*[ShowCreation(wave) for wave in individual_standing_waves[0]], lag_ratio=0.5))
        self.play(Write(VGroup(t_label, t_value)))
        single_mode_eq = Tex(
            R"""
            \frac{1}{2} \sin (\pi x) \cos (\pi t)
            """,
            t2c={R"\pi": QMV_BLUE_A, "t": YELLOW, R"x": PINK,"y": WHITE},
            isolate=isolate_tex
        )
        tripple_mode_eq = Tex(
            R"""
            \frac{1}{2}\sin (3\pi x) \cos (3\pi t)
            """,
            t2c={R"3\pi": QMV_PURPLE_A, "t": YELLOW, R"x": PINK,"y": WHITE},
            isolate=isolate_tex
        )
        addition_symbol = Tex("+")
        summing_modes_tex = VGroup(single_mode_eq, addition_symbol, tripple_mode_eq).arrange(RIGHT).shift(3 * DOWN)
        self.play(Write(summing_modes_tex[0]))
        self.play(Write(summing_modes_tex[2]))
        self.wait()
        for wave in individual_standing_waves[0]:
            wave.attach_value_tracker(t_tracker)
        self.play(t_tracker.animate.increment_value(5), rate_func=linear, run_time=5)
        wave_sum = SummedWave(*individual_standing_waves[0], config=wave_config, t_initial=t_tracker.get_value()).set_color_by_gradient(QMV_BLUE_A, QMV_PURPLE_A)
        for wave_gp in individual_standing_waves:
            for wave in wave_gp:
                wave.suspend_updating()
        self.play(
            FadeIn(summing_modes_tex[1], scale=0.1),
            *[ReplacementTransform(
                wave,
                wave_sum
            ) for wave in individual_standing_waves[0]],
        )
        wave_sum.attach_value_tracker(t_tracker)
        wave_sum.add_updater(lambda m: m.set_color_by_gradient(QMV_BLUE_A, QMV_PURPLE_A))
        self.play(t_tracker.animate.increment_value(5), rate_func=linear, run_time=5)
        double_mode_eq = Tex(
            R"""
            \frac{1}{3}\sin (2\pi x) \cos (2\pi t)
            """,
            t2c={R"2\pi": QMV_BLUE_A, "t": YELLOW, R"x": PINK,"y": WHITE},
            isolate=isolate_tex
        )
        sev_mode_eq = Tex(
            R"""
            \sin (7\pi x) \cos (7\pi t)
            """,
            t2c={R"7\pi": QMV_PURPLE_A, "t": YELLOW, R"x": PINK,"y": WHITE},
            isolate=isolate_tex
        )
        add_copy = addition_symbol.copy()
        summing_modes_tex = VGroup(double_mode_eq, add_copy, sev_mode_eq).arrange(RIGHT).shift(3 * DOWN)
        wave_sum_prev = wave_sum
        wave_sum_prev.clear_updaters()
        wave_sum = SummedWave(*individual_standing_waves[1], config=wave_config, t_initial=t_tracker.get_value()).set_color_by_gradient(QMV_BLUE_A, QMV_PURPLE_A)
        self.play(
            TransformMatchingStrings(single_mode_eq, double_mode_eq),
            TransformMatchingStrings(tripple_mode_eq, sev_mode_eq),
            FadeOut(addition_symbol),
            ReplacementTransform(wave_sum_prev.copy(), individual_standing_waves[1][0]),
            ReplacementTransform(wave_sum_prev, individual_standing_waves[1][1]),
        )
        addition_symbol.move_to(add_copy)
        self.wait()
        self.play(
            FadeIn(addition_symbol, scale=0.1),
            *[ReplacementTransform(
                wave,
                wave_sum
            ) for wave in individual_standing_waves[1]],
        )
        wave_sum.attach_value_tracker(t_tracker)
        wave_sum.add_updater(lambda m: m.set_color_by_gradient(QMV_BLUE_A, QMV_PURPLE_A))
        self.play(t_tracker.animate.increment_value(5), rate_func=linear, run_time=5)
        final_sing_mode_eq = Tex(
            R"""
            \sin (\pi x) \cos (\pi t)
            """,
            t2c={R"\pi": QMV_BLUE_A, "t": YELLOW, R"x": PINK,"y": WHITE},
            isolate=isolate_tex
        )
        final_double_mode_eq = Tex(
            R"""
            \frac{1}{2} \sin (2\pi x) \cos (2\pi t)
            """,
            t2c={R"2\pi": QMV_BLUE_C, "t": YELLOW, R"x": PINK,"y": WHITE},
            isolate=isolate_tex
        )
        final_fifth_mode_eq = Tex(
            R"""
            \frac{1}{3} \sin (5\pi x) \cos (5\pi t)
            """,
            t2c={R"5\pi": QMV_PURPLE_A, "t": YELLOW, R"x": PINK,"y": WHITE},
            isolate=isolate_tex
        )
        add_copy = addition_symbol.copy()
        summing_modes_tex = VGroup(final_sing_mode_eq, add_copy, final_double_mode_eq, add_copy.copy(), final_fifth_mode_eq).arrange(RIGHT).shift(3 * DOWN).scale(0.9)
        wave_sum_prev = wave_sum
        wave_sum_prev.clear_updaters()
        wave_sum = SummedWave(*individual_standing_waves[2], config=wave_config, t_initial=0).set_color_by_gradient(QMV_BLUE_A, QMV_PURPLE_A)
        self.play(
            TransformMatchingStrings(double_mode_eq, final_sing_mode_eq),
            TransformMatchingStrings(sev_mode_eq, final_double_mode_eq),
            Write(final_fifth_mode_eq),
            FadeOut(addition_symbol),
            ReplacementTransform(wave_sum_prev.copy(), individual_standing_waves[2][0]),
            ReplacementTransform(wave_sum_prev.copy(), individual_standing_waves[2][1]),
            ReplacementTransform(wave_sum_prev, individual_standing_waves[2][2]),
            t_tracker.animate.set_value(0)
        )
        self.wait()
        self.play(
            FadeIn(summing_modes_tex[1], scale=0.1),
            FadeIn(summing_modes_tex[3], scale=0.1),
            *[ReplacementTransform(
                wave,
                wave_sum
            ) for wave in individual_standing_waves[2]],
        )
        wave_sum.attach_value_tracker(t_tracker)
        wave_sum.add_updater(lambda m: m.set_color_by_gradient(QMV_BLUE_A, QMV_PURPLE_A))
        self.play(t_tracker.animate.increment_value(5), rate_func=linear, run_time=5)
        wave_sum.clear_updaters()
        self.play(Uncreate(wave_sum), FadeOut(summing_modes_tex))

        # Show summing towards more complicated shapes, like a big square, a triangle, a square, a rectangle at the same time, a city skyline, a coffee mug
        N_value = ValueTracker(10)
        N_label = VGroup(Tex("N = ", t2c = {"N": QMV_PURPLE_C}), Integer(N_value.get_value())).arrange(RIGHT).next_to(t_value, RIGHT, buff=1)
        finite_sum_eq = Tex(
            R"""
            y(x, t) = \sum_{n = 1}^N A_n \sin (n\pi x) \cos (n\pi t)
            """,
            t2c={R"n\pi": QMV_BLUE_A, "t": YELLOW, R"x": PINK,"y": WHITE, "N": QMV_PURPLE_C, "A_n": RED_C},
            isolate=isolate_tex
        ).to_corner(DL, buff=1).shift(DOWN * 0.3)
        self.play(
            VGroup(t_label, t_value).animate.to_corner(DR, buff=1).shift(DOWN * 0.2 + LEFT * 0.6),
            VGroup(axes, x_label, y_label).animate.shift(UP)
        )
        self.play(
            t_tracker.animate.set_value(0),
            Write(N_label.next_to(t_label, UP, aligned_edge=LEFT, buff=0.5)),
            Write(finite_sum_eq)
        )
        N_label.add_updater(lambda m: m[1].set_value(N_value.get_value()))
        initial_condition_label = Tex(
            R"""
            y(x, 0) 
            """,
            t2c={R"n\pi": QMV_BLUE_A, "t": YELLOW, R"x": PINK,"y": WHITE, "N": QMV_PURPLE_C, "A_n": RED_C},
            isolate=isolate_tex
        ).next_to(axes, UP).shift(DOWN * 0.3 + RIGHT * 4)
        square_pts = [
            (0, 0),
            (2.95, 0),
            (3, 2),
            (7, 2),
            (7.05, 0),
            (L, 0)
        ]
        vis_config = WaveVisualizationConfig(int(N_value.get_value()), [QMV_BLUE_A, QMV_PURPLE_A])
        fourier_group = FourierGroup(
            num_harmonics=int(N_value.get_value()),
            wave_function=square_pts,
            wave_config=wave_config,
            vis_config=vis_config,
        )
        self.play(
            FadeIn(initial_condition_label),
            ShowCreation(fourier_group.transverse_wave.shift(UP))
        )
        self.wait()
        self.play(
            FadeOut(initial_condition_label, run_time=1),
            FadeOut(fourier_group.transverse_wave)
        )
        scaled_coeffs = ["{:0.2f}".format(fourier_group.harmonic_waves[n - 1].config.amplitude / 2) for n in range(1, int(N_value.get_value()) + 1)]
        intermediate_waves = [
            FourierWave(
                n, 
                square_pts,
                config=wave_config
            ).shift(UP)
            for n in range(1, int(N_value.get_value()) + 1)
        ]
        equations_of_sins = [
            Tex(
                Rf"""
                    {scaled_coeffs[n - 1]} \sin ({n}\pi x)
                """,
                t2c={Rf"{n}\pi": QMV_BLUE_A, "t": YELLOW, R"x": PINK,"y": WHITE, "N": QMV_PURPLE_C, scaled_coeffs[n - 1]: RED_C},
                isolate=isolate_tex
            ).next_to(axes, DOWN).shift(UP)
            for n in range(1, int(N_value.get_value()) + 1)
        ]
        equations_of_cosins = [
            Tex(
                Rf"""
                    \cos({n}\pi t)
                """,
                t2c={Rf"{n}\pi": QMV_BLUE_A, "t": YELLOW,},
                isolate=isolate_tex
            ).next_to(equations_of_sins[n - 1], RIGHT)
            for n in range(1, int(N_value.get_value()) + 1)
        ]
        self.play(ShowCreation(fourier_group.harmonic_waves[0].shift(UP)), Write(equations_of_sins[0]))
        self.play(FadeIn(equations_of_cosins[0], shift=LEFT))
        self.play(
                ReplacementTransform(fourier_group.harmonic_waves[0], intermediate_waves[0]),
                FadeOutToPoint(VGroup(equations_of_sins[0], equations_of_cosins[0]), finite_sum_eq.get_center()),
            )        
        for i in range(1, int(N_value.get_value())):
            self.play(ShowCreation(fourier_group.harmonic_waves[i].shift(UP)), Write(equations_of_sins[i]))
            self.play(FadeIn(equations_of_cosins[i], shift=LEFT))
            self.play(
                ReplacementTransform(fourier_group.harmonic_waves[i], intermediate_waves[i]),
                ReplacementTransform(intermediate_waves[i-1], intermediate_waves[i]),
                FadeOutToPoint(VGroup(equations_of_sins[i], equations_of_cosins[i]), finite_sum_eq.get_center()),
            )
        self.play(FadeOut(intermediate_waves[-1]))
        cake_points = [
            (0,0),
            (3, 0), (3.05, 1.5), (4, 1.5), (4.05, 2), (4.15, 2), (4.2, 1.5),
            (4.3, 1.5), (4.35, 2), (4.45, 2), (4.5, 1.5),
            (4.6, 1.5), (4.65, 2), (4.75, 2), (4.8, 1.5),
            (4.9, 1.5), (4.95, 2), (5.05, 2), (5.1, 1.5),
            (5.2, 1.5), (5.25, 2), (5.35, 2), (5.4, 1.5),
            (5.5, 1.5), (5.55, 2), (5.65, 2), (5.7, 1.5),
            (6.65, 1.5), (6.7, 0), (9.7, 0)
        ]
        cake_points = [(x * 10/9.7, y) for x, y in cake_points]
        cake_config = copy.deepcopy(wave_config)
        cake_config.resolution=50
        cake_N = 50
        vis_config = WaveVisualizationConfig(cake_N, [QMV_BLUE_A, QMV_PURPLE_A])
        fourier_group = FourierGroup(
            num_harmonics=cake_N,
            wave_function=cake_points,
            wave_config=cake_config,
            vis_config=vis_config,
        )
        self.play(
            FadeIn(initial_condition_label),
            ShowCreation(fourier_group.transverse_wave.shift(UP))
        )
        self.wait()
        self.play(
            FadeOut(initial_condition_label, run_time=1),
            FadeOut(fourier_group.transverse_wave)
        )
        self.play( N_value.animate.set_value(cake_N))
        self.play(Indicate(N_label, scale_factor=2))
        scaled_coeffs = ["{:0.2f}".format(fourier_group.harmonic_waves[n - 1].config.amplitude / 2) for n in range(1, int(N_value.get_value()) + 1)]
        intermediate_waves = [
            FourierWave(
                n, 
                cake_points,
                config=cake_config
            ).shift(UP)
            for n in range(1, int(N_value.get_value()) + 1)
        ]
        for wave in fourier_group.harmonic_waves:
            wave.shift(UP)
        for wave in intermediate_waves:
            wave.set_stroke(width=0.1)
        fourier_group.fourier_wave.shift(UP)
        self.play(
            LaggedStart(*[
                Succession(ShowCreation(wave), wave.animate.set_stroke(width=1))
                for wave in fourier_group.harmonic_waves
            ]),
            run_time=5,
            rate_func=linear
        )
        self.play(
            LaggedStart(
                ReplacementTransform(orig_wave, int_wave)
                for orig_wave, int_wave in zip(fourier_group.harmonic_waves, intermediate_waves)
            ),
            run_time=5,
            rate_func=slow_into
        )
        self.play(
            FadeOut(VGroup(*intermediate_waves)),
            FadeIn(fourier_group.fourier_wave)
        )
        self.wait()
        self.play(FadeOut(fourier_group.fourier_wave))
        city_skyline_points = [
            (0,0),
            (0.5, 0), (0.55, 2), (0.775, 2.3), (1, 2), (1.05, 0),
            (1.15, 0), (1.2, 3.7), (1.35, 4.7), (2, 4.7), (2.15, 3.7), (2.2, 0),
            (2.3, 0), (2.35, 1.5), (4, 1.5), (4.05, 0),
            (4.15, 0), (4.2, 6.5), (4.3, 6.5), (4.35, 7), (4.45, 7), (4.5, 9), (4.55, 9), (4.6, 7), (4.7, 7), (4.75, 6.5), (4.85, 6.5), (4.9, 0),
            (5, 0), (5.05, 1.4), (5.15, 1.4), (5.2, 3.5), (5.3, 3.5), (5.35, 4), (5.85, 4), (5.9, 3.5), (6, 3.5), (6.05, 1.4), (6.15, 1.4), (6.2, 0),
            (6.6, 0), (6.65, 5), (6.8, 5), (6.85, 0),
            (7, 0)
        ]
        city_skyline_points = [(x * 10/7, y * 2/8) for x, y in city_skyline_points]
        sky_line_config = copy.deepcopy(wave_config)
        sky_line_config.resolution=100
        sky_line_N = 75
        vis_config = WaveVisualizationConfig(sky_line_N, [QMV_BLUE_A, QMV_PURPLE_A])
        fourier_group = FourierGroup(
            num_harmonics=sky_line_N,
            wave_function=city_skyline_points,
            wave_config=sky_line_config,
            vis_config=vis_config,
        )
        self.play(
            FadeIn(initial_condition_label),
            ShowCreation(fourier_group.transverse_wave.shift(UP))
        )
        self.wait()
        self.play(
            FadeOut(initial_condition_label, run_time=1),
            FadeOut(fourier_group.transverse_wave)
        )
        self.play(N_value.animate.set_value(sky_line_N))
        self.play(Indicate(N_label, scale_factor=2))
        scaled_coeffs = ["{:0.2f}".format(fourier_group.harmonic_waves[n - 1].config.amplitude / 2) for n in range(1, int(N_value.get_value()) + 1)]
        intermediate_waves = [
            FourierWave(
                n, 
                city_skyline_points,
                config=sky_line_config
            ).shift(UP)
            for n in range(1, int(N_value.get_value()) + 1)
        ]
        for wave in fourier_group.harmonic_waves:
            wave.shift(UP)
        for wave in intermediate_waves:
            wave.set_stroke(width=0.1)
        fourier_group.fourier_wave.shift(UP)
        self.play(
            LaggedStart(*[
                Succession(ShowCreation(wave), wave.animate.set_stroke(width=0.3))
                for wave in fourier_group.harmonic_waves
            ]),
            run_time=5,
            rate_func=linear
        )
        self.play(
            LaggedStart(
                Succession(ReplacementTransform(orig_wave, int_wave))
                for orig_wave, int_wave in zip(fourier_group.harmonic_waves, intermediate_waves)
            ),
            run_time=5,
            rate_func=slow_into
        )
        self.play(
            FadeOut(VGroup(*intermediate_waves)),
            FadeIn(fourier_group.fourier_wave)
        )
        self.wait()
        self.play(FadeOut(fourier_group.fourier_wave))

        # Create guitar string real show 5 summing sine waves. Transform copy into guitar string while fading guitar string. Contrast real and modeled guitar string.
        points_A = [(0, 0)]
        points_A += [(2, 2)]
        points_A += [(L, 0)]
        N = 5
        vis_config = WaveVisualizationConfig(N, [QMV_BLUE_A, QMV_PURPLE_A])
        guitar_string_gp = FourierGroup(
            num_harmonics=N,
            wave_function=points_A,
            wave_config=wave_config,
            vis_config=vis_config,
        )
        self.play(
            FadeIn(initial_condition_label),
            ShowCreation(guitar_string_gp.transverse_wave.shift(UP))
        )
        self.play(
            guitar_string_gp.transverse_wave.animate.set_color(ORANGE),
            initial_condition_label.animate.set_color(ORANGE)
        )
        self.wait()
        self.play(
            FadeOut(initial_condition_label, run_time=1),
            FadeOut(guitar_string_gp.transverse_wave)
        )
        self.play(N_value.animate.set_value(N))
        self.play(Indicate(N_label, scale_factor=2))
        scaled_coeffs = ["{:0.2f}".format(guitar_string_gp.harmonic_waves[n - 1].config.amplitude / 2) for n in range(1, int(N_value.get_value()) + 1)]
        intermediate_waves = [
            FourierWave(
                n, 
                points_A,
                config=wave_config
            ).shift(UP)
            for n in range(1, int(N_value.get_value()) + 1)
        ]
        for wave in guitar_string_gp.harmonic_waves:
            wave.shift(UP)
            wave.save_state()
        guitar_string_gp.fourier_wave.shift(UP)
        equations_of_sins = [
            Tex(
                Rf"""
                    {scaled_coeffs[n - 1]} \sin ({n}\pi x)
                """,
                t2c={Rf"{n}\pi": QMV_BLUE_A, "t": YELLOW, R"x": PINK,"y": WHITE, "N": QMV_PURPLE_C, scaled_coeffs[n - 1]: RED_C},
                isolate=isolate_tex
            ).next_to(axes, DOWN).shift(UP)
            for n in range(1, int(N_value.get_value()) + 1)
        ]
        equations_of_cosins = [
            Tex(
                Rf"""
                    \cos({n}\pi t)
                """,
                t2c={Rf"{n}\pi": QMV_BLUE_A, "t": YELLOW,},
                isolate=isolate_tex
            ).next_to(equations_of_sins[n - 1], RIGHT)
            for n in range(1, int(N_value.get_value()) + 1)
        ]
        self.play(ShowCreation(guitar_string_gp.harmonic_waves[0]), Write(equations_of_sins[0]))
        self.play(FadeIn(equations_of_cosins[0], shift=LEFT))
        self.play(
                ReplacementTransform(guitar_string_gp.harmonic_waves[0], intermediate_waves[0]),
                FadeOutToPoint(VGroup(equations_of_sins[0], equations_of_cosins[0]), finite_sum_eq.get_center()),
            )
        for i in range(1, int(N_value.get_value())):
            self.play(ShowCreation(guitar_string_gp.harmonic_waves[i]), Write(equations_of_sins[i]))
            self.play(FadeIn(equations_of_cosins[i], shift=LEFT))
            self.play(
                ReplacementTransform(guitar_string_gp.harmonic_waves[i], intermediate_waves[i]),
                ReplacementTransform(intermediate_waves[i-1], intermediate_waves[i]),
                FadeOutToPoint(VGroup(equations_of_sins[i], equations_of_cosins[i]), finite_sum_eq.get_center()),
            )
        self.play(FadeIn(guitar_string_gp.transverse_wave))
        self.play(FadeOut(guitar_string_gp.transverse_wave))

        # Fade out guitar string. Show individually all standing waves moving over time, then all of them together, then at a fixed point in time sum to get the guitar string.
        self.play(ReplacementTransform(intermediate_waves[-1], guitar_string_gp.harmonic_waves[0]), *[wave.animate.restore() for wave in guitar_string_gp.harmonic_waves])
        for wave in guitar_string_gp.harmonic_waves:
            wave.attach_value_tracker(t_tracker)
            wave.add_updater(lambda m: m.shift(UP))
        self.wait()
        self.play(t_tracker.animate.increment_value(4), run_time=4, rate_func=linear)
        VGroup(*guitar_string_gp.harmonic_waves).suspend_updating()
        self.play(
            LaggedStart(
                TransformFromCopy(orig_wave, guitar_string_gp.fourier_wave)
                for orig_wave in guitar_string_gp.harmonic_waves
            ),
            run_time=1
        )
        VGroup(*guitar_string_gp.harmonic_waves).resume_updating()
        guitar_string_gp.fourier_wave.attach_value_tracker(t_tracker)
        guitar_string_gp.fourier_wave.add_updater(lambda m: m.shift(UP))
        self.play(t_tracker.animate.increment_value(3), run_time=3, rate_func=linear)
        self.play(*[wave.animate.set_stroke(opacity=0.4) for wave in guitar_string_gp.harmonic_waves], t_tracker.animate.increment_value(1), run_time=1, rate_func=linear)
        self.play(t_tracker.animate.increment_value(3), run_time=3, rate_func=linear)

        # Animate the guitar string moving, increase N
        self.play(FadeOut(finite_sum_eq), t_tracker.animate.set_value(0))
        most_important_copies = sorted(copy.deepcopy(guitar_string_gp.harmonic_waves), key=lambda wave: abs(wave.config.amplitude), reverse=True)[:5]
        copy_axes = axes.copy().scale(0.5, about_point=axes.get_center()).shift(DOWN * 3.5 + LEFT * 3.5)
        VGroup(*most_important_copies).clear_updaters().set_stroke(opacity=0.7).scale(0.5, about_point=axes.get_center()).shift(DOWN * 3.5 + LEFT * 3.5)
        VGroup(*guitar_string_gp.harmonic_waves).clear_updaters()
        self.play(
            TransformFromCopy(axes, copy_axes),
            *[
            TransformFromCopy(old_wave, new_wave) 
            for old_wave, new_wave in zip(guitar_string_gp.harmonic_waves, most_important_copies)
            ]
        )
        most_important_label = Text(R"Largest Contributors:").scale(0.7).next_to(copy_axes, DOWN, buff=0.0)
        self.play(Write(most_important_label))
        def get_final_anim_group(num_waves, example, config=wave_config):
            vis_config = WaveVisualizationConfig(num_waves, [QMV_BLUE_A, QMV_PURPLE_A])
            fourier_gp = FourierGroup(
                num_harmonics=num_waves,
                wave_function=example,
                wave_config=config,
                vis_config=vis_config,
            )
            sorted_prior = sorted(copy.deepcopy(fourier_gp.harmonic_waves), key=lambda wave: abs(wave.config.amplitude), reverse=True)[:5]
            standing_waves = VGroup(*fourier_gp.harmonic_waves).clear_updaters().shift(UP).set_stroke(opacity=0.5)
            fourier_wave = fourier_gp.fourier_wave.clear_updaters().shift(UP)
            fourier_wave.attach_value_tracker(t_tracker)
            fourier_wave.add_updater(lambda m: m.shift(UP))
            for wave in standing_waves:
                wave.attach_value_tracker(t_tracker)
            standing_waves.add_updater(lambda m: m.shift(UP))
            largest_contributors = VGroup(*sorted_prior).clear_updaters()
            largest_contributors.shift(UP).set_stroke(opacity=0.7).scale(0.5, about_point=axes.get_center()).shift(DOWN * 3.5 + LEFT * 3.5)
            colors_of_largest = color_gradient([QMV_BLUE_A, QMV_PURPLE_A], 5)
            for i, wave in enumerate(largest_contributors):
                wave.attach_value_tracker(t_tracker)
                wave.set_color(colors_of_largest[i])
            largest_contributors.add_updater(lambda m: m.shift(UP).scale(0.5, about_point=axes.get_center()).shift(DOWN * 3.5 + LEFT * 3.5))
            return fourier_wave, standing_waves, largest_contributors
        big_N = 50
        guitar_string_gp.fourier_wave.clear_updaters()
        fourier_wave, standing_waves, largest_contributors = get_final_anim_group(big_N, points_A)
        fourier_wave.suspend_updating(), standing_waves.suspend_updating(), largest_contributors.suspend_updating()
        self.play(
            ReplacementTransform(guitar_string_gp.fourier_wave, fourier_wave),
            FadeOut(guitar_string_gp.harmonic_waves),
            *[ReplacementTransform(old, new) for old, new in zip(most_important_copies, largest_contributors)],
            LaggedStart(*[ShowCreation(wave) for wave in standing_waves]),
            N_value.animate.set_value(big_N),
            run_time=10
        )
        self.remove(*guitar_string_gp.harmonic_waves, *most_important_copies)
        self.add(largest_contributors, standing_waves)
        fourier_wave.resume_updating()
        standing_waves.resume_updating()
        largest_contributors.resume_updating()
        self.play(t_tracker.animate.increment_value(5), rate_func=linear, run_time=5)

        # Highlight red the endpoints. Rewrite wave equation, add beta. Put to side, change the wave equation. Animate a little more.
        fourier_wave.suspend_updating(), standing_waves.suspend_updating(), largest_contributors.suspend_updating()
        self.play(FadeOut(VGroup(fourier_wave, standing_waves, largest_contributors)), t_tracker.animate.set_value(0))
        end_points[0].set_color(RED_C).scale(1/0.15).shift(UP)
        end_points[1].set_color(RED_C).scale(1/0.15).shift(UP)
        damped_config = copy.deepcopy(wave_config)
        damped_config.dampening = 0.15
        damped_config.wave_speed = wave_config.wave_speed * 2
        damped_fourier_wave, damped_standing_wave, damped_largest_contributors = get_final_anim_group(big_N, points_A, damped_config)
        damped_fourier_wave.suspend_updating(), damped_standing_wave.suspend_updating(), damped_largest_contributors.suspend_updating()
        beta_label = VGroup(Tex(R"\beta = ", t2c = {R"\beta": RED_C}), DecimalNumber(0.15)).arrange(RIGHT).next_to(N_label, UP).shift(UP * 0.13)
        self.play(FadeIn(VGroup(damped_fourier_wave, damped_standing_wave, damped_largest_contributors)))
        self.play(Write(beta_label))
        damped_fourier_wave.resume_updating(), damped_standing_wave.resume_updating(), damped_largest_contributors.resume_updating()
        self.play(t_tracker.animate.increment_value(1),run_time=1,rate_func=linear)
        self.play(
            FlashyFadeIn(end_points[0]), end_points[0].animate.scale(0.15),
            t_tracker.animate.increment_value(1),
            run_time=1,
            rate_func=linear
        )
        self.play(
            FlashyFadeIn(end_points[1]), end_points[1].animate.scale(0.15),
            t_tracker.animate.increment_value(1),
            run_time=1,
            rate_func=linear
        )
        self.play(t_tracker.animate.increment_value(1),run_time=1,rate_func=linear)
        damped_moral_wave_text = VGroup(
            Text("Damped Wave equation:"),
            Tex(
                R"""
                \frac{\partial ^2 y}{\partial t ^2} = \frac{\partial^2 y}{\partial x^2} - \beta \frac{\partial y}{\partial t}
                """,
                t2c={
                    R"\frac{\partial ^2 y}{\partial t ^2}": YELLOW,
                    R"\frac{\partial y}{\partial t}": YELLOW,
                    R"\frac{\partial^2 y}{\partial x^2}": PINK,
                    R"\beta": RED_C,
                    "y": WHITE
                }
            )
        ).arrange(DOWN)
        damped_moral_wave_text.center().next_to(copy_axes, RIGHT).shift(LEFT * 0.2)
        self.play(
            Write(damped_moral_wave_text, rate_func=smooth),
            t_tracker.animate.increment_value(2),
            run_time=2,
            rate_func=linear
        )
        self.play(t_tracker.animate.increment_value(3),run_time=3,rate_func=linear)
        self.play(
            FadeOut(damped_moral_wave_text),
            t_tracker.animate.increment_value(2),
            run_time=2,
            rate_func=linear
        )
        self.play(t_tracker.animate.increment_value(3),run_time=3,rate_func=linear)
        damped_fourier_wave.suspend_updating(), damped_standing_wave.suspend_updating(), damped_largest_contributors.suspend_updating()
        self.play(
            FadeOut(VGroup(damped_fourier_wave, end_points, damped_standing_wave, damped_largest_contributors)),
            t_tracker.animate.set_value(0),
            beta_label[1].animate.set_value(0.05),
            N_value.animate.set_value(1)
        )

        # Do final animations.
        # Cake
        high_res_config = WaveConfig(dampening=0.0, wave_speed=6, length=L, amplitude=2, resolution=100)
        N = 10
        cake_real = TransverseWave(cake_points, config=high_res_config).shift(UP)
        fourier_wave, standing_waves, largest_contributors = get_final_anim_group(N, cake_points, config=high_res_config)
        all_cake = VGroup(fourier_wave, standing_waves, largest_contributors).suspend_updating()
        self.play(ShowCreation(cake_real))
        self.play(FadeOut(cake_real))
        intermediate_waves = [
            FourierWave(
                n, 
                cake_points,
                config=high_res_config
            ).shift(UP)
            for n in range(1, N + 1)
        ]
        for wave in intermediate_waves:
            wave.set_stroke(width=0.5)
        self.play(
            LaggedStart(*[
                Succession(ShowCreation(wave), wave.animate.set_stroke(width=2))
                for wave in standing_waves
            ]),
            LaggedStart(
                ReplacementTransform(orig_wave.copy(), int_wave)
                for orig_wave, int_wave in zip(standing_waves, intermediate_waves)
            ),
            N_value.animate.set_value(N),
            run_time=8,
            rate_func=linear
        )
        self.play(
            LaggedStart(*[ReplacementTransform(standing_waves[wave.mode - 1].copy(), wave) for wave in largest_contributors])
        )
        self.play(
            FadeOut(VGroup(*intermediate_waves)),
            FadeIn(fourier_wave)
        )
        self.remove(*intermediate_waves, all_cake)
        self.add(all_cake)
        all_cake.resume_updating()
        self.play(t_tracker.animate.increment_value(10), rate_func=linear, run_time=20)
        all_cake.suspend_updating()
        self.play(FadeOut(all_cake), N_value.animate.set_value(1), t_tracker.animate.set_value(0))
        
        # Skyline
        N = 500
        skyline_real = TransverseWave(city_skyline_points, config=high_res_config).shift(UP)
        fourier_wave, standing_waves, largest_contributors = get_final_anim_group(N, city_skyline_points, config=high_res_config)
        all_skyline = VGroup(fourier_wave, standing_waves, largest_contributors).suspend_updating()
        self.play(ShowCreation(skyline_real))
        self.play(FadeOut(skyline_real))
        intermediate_waves = [
            FourierWave(
                n, 
                city_skyline_points,
                config=high_res_config
            ).shift(UP)
            for n in range(1, N + 1)
        ]
        for wave in intermediate_waves:
            wave.set_stroke(width=0.1)
        self.play(
            LaggedStart(*[
                Succession(ShowCreation(wave), wave.animate.set_stroke(width=2))
                for wave in standing_waves
            ]),
            LaggedStart(
                ReplacementTransform(orig_wave.copy(), int_wave)
                for orig_wave, int_wave in zip(standing_waves, intermediate_waves)
            ),
            N_value.animate.set_value(N),
            run_time=8,
            rate_func=slow_into
        )
        self.play(
            LaggedStart(*[ReplacementTransform(standing_waves[wave.mode - 1].copy(), wave) for wave in largest_contributors])
        )
        self.play(
            FadeOut(VGroup(*intermediate_waves)),
            FadeIn(fourier_wave)
        )
        self.remove(*intermediate_waves, all_skyline)
        self.add(all_skyline)
        all_skyline.resume_updating()
        self.play(t_tracker.animate.increment_value(15), rate_func=linear, run_time=30)
        all_skyline.suspend_updating()
        t_value.suspend_updating()
        N_label.suspend_updating()
        self.play(FadeOut(VGroup(all_skyline, axes, copy_axes, x_label, y_label, beta_label, N_label, t_value, t_value, most_important_label)))






