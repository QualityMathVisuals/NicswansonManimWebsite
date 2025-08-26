from manim import *
import subprocess
import itertools


def path_backwards_then_forwards(start_points, end_points, alpha):
    v = end_points - start_points
    cutoff = 0.2  # Represents 10% of the animation time for the backward motion
    if alpha <= cutoff:
        # Use rush_into: Starts very fast backward, then slows down sharply
        progress_back = alpha / cutoff
        eased_progress_back = rate_functions.rush_into(progress_back)
        return start_points - eased_progress_back * cutoff * v
    else:
        # Use ease_in_cubic: Starts slow forward, then accelerates
        start_of_forward = start_points - cutoff * v
        progress_fwd = (alpha - cutoff) / (1 - cutoff)
        eased_progress_fwd = rate_functions.smooth(progress_fwd)
        return start_of_forward + eased_progress_fwd * (end_points - start_of_forward)


class LatticePath(Scene):
    def __init__(self, M=4, N=3, **kwargs):
        self.M = M
        self.N = N

        super().__init__(**kwargs)

    def construct(self):

        M, N = self.M, self.N
        self.next_section(f"M{M}N{N}Scene 1- Lattice Paths", skip_animations=False)

        def lattice_grid():
            plane = NumberPlane(
                x_range=[0, M + 0.001],
                y_range=[0, N + 0.001],
                background_line_style={
                    "stroke_color": WHITE,
                    "stroke_width": 1,
                    "stroke_opacity": 0.5,
                },
                axis_config={
                    "include_numbers": False,
                    "font_size": 24,
                    "color": WHITE,
                    "stroke_opacity": 0,
                    "stroke_width": 0,
                },
                y_axis_config={"label_direction": LEFT},
            )
            dots = VGroup()
            for i in range(M + 1):
                for j in range(N + 1):
                    dot = Dot(point=plane.c2p(i, j), radius=0.05, color=LIGHT_GRAY)
                    dots.add(dot)

            return VGroup(plane, dots)

        grid = lattice_grid().save_state()

        # Generate all lattice path sequences (strings)
        total_length = M + N
        indices = range(total_length)
        lattice_path_sequences = [
            "".join("E" if i in set(east_indices) else "N" for i in indices)
            for east_indices in itertools.combinations(indices, M)
        ]

        gradient = color_gradient([BLUE, PURPLE, PINK], len(lattice_path_sequences))
        lattice_paths = []
        for i, sequence in enumerate(lattice_path_sequences):
            points = [grid[0].c2p(0, 0)]
            x, y = 0, 0
            for move in sequence:
                if move == "E":
                    x += 1
                else:  # move == 'N'
                    y += 1
                points.append(grid[0].c2p(x, y))

            path = VMobject(color=gradient[i], stroke_opacity=0.3)
            path.set_points_as_corners(points)
            lattice_paths.append(path)
            path.save_state()

        counter = Integer(0, font_size=36, color=WHITE).next_to(grid[0], DOWN)
        counter.add_updater(
            lambda m: m.set_value(
                len(
                    [path for path in lattice_paths if path.get_stroke_opacity() > 0.31]
                )
            )
        )

        self.add(grid, counter)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(Create(path), path.animate.set_stroke(opacity=0.8))
                    for i, path in enumerate(lattice_paths)
                ],
            ),
            run_time=M + N,
            # rate_func=rate_functions.rush_into
        )
        counter.clear_updaters()

        lattice_path_title = Text(
            rf"Lattice Paths",
            color=LIGHT_PINK,
            font_size=52,
        ).to_edge(UP, buff=0.3)
        self.play(
            Write(lattice_path_title),
            grid.animate.shift(UP * 1),
            counter.animate.shift(UP * 1),
            *[path.animate.shift(UP * 1) for path in lattice_paths],
        )

        self.wait()

        self.next_section(f"M{M}N{N}Scene 1.5- Lattice Paths Distribute", skip_animations=False)

        # Calculate grid layout parameters
        binom_coeff = len(lattice_paths)
        num_cols = int(np.ceil(binom_coeff**0.5625))
        num_rows = int(np.ceil(binom_coeff / num_cols))

        # # Adjust spacing and scaling based on scene dimensions
        total_grid_width = self.camera.frame_width - 1  # Leave margin
        total_grid_height = self.camera.frame_height - 2  # Leave margin top/bottom

        grid_cell_width = total_grid_width / num_cols
        grid_cell_height = total_grid_height / num_rows

        # # Calculate scale factor based on cell size and M, N dimensions
        available_width = grid_cell_width * 0.9
        available_height = grid_cell_height * 0.7
        scale_factor_x = available_width / M if M > 0 else available_width
        scale_factor_y = available_height / N if N > 0 else available_height
        small_scale_factor = min(scale_factor_x, scale_factor_y)

        # # Groups for animation elements
        small_grids = VGroup()
        target_paths = VGroup()
        index_labels = VGroup()
        all_labels_under_steps = VGroup()
        labeled_sequences = VGroup()  # Holds VGroups of final text sequences

        path_transforms = []
        counter_transforms = []
        label_transforms = []

        # Iterate through paths to create small grids, target paths, and labels
        for i, (path, sequence) in enumerate(
            zip(lattice_paths, lattice_path_sequences)
        ):
            row = i // num_cols
            col = i % num_cols

            # Calculate target center for the small grid cell
            target_center_x = -total_grid_width / 2 + (col + 0.5) * grid_cell_width
            target_center_y = total_grid_height / 2 - (row + 0.5) * grid_cell_height
            target_center = np.array([target_center_x, target_center_y, 0])

            # Create small grid
            small_grid = (
                lattice_grid()
                .scale(small_scale_factor)
                .move_to(target_center)
                .set_z_index(-1)
                .set_stroke(opacity=0.3)
            )
            small_grid[0].get_axes().set_stroke(opacity=0)
            small_grids.add(small_grid)

            # Create target path
            target_path = path.copy()
            # Scale relative to the origin (0,0) in the original grid's coordinates
            target_path.scale(small_scale_factor)
            target_path.move_to(target_center)
            target_paths.add(target_path)

            # Store the transform animation
            path_transforms.append(Transform(path, target_path))

            # --- Label Generation ---
            label_sequence = (
                MathTex(
                    *[char for char in sequence],
                    tex_to_color_map={"E": ORANGE, "N": GOLD_B},
                )
                .scale(small_scale_factor)
                .next_to(small_grid, DOWN, buff=0.1)
            )
            labeled_sequences.add(label_sequence)

            index_label = (
                Integer(i + 1)
                .scale(small_scale_factor)
                .next_to(small_grid, UP, buff=0.1)
            )
            index_labels.add(index_label)
            counter_transforms.append(TransformFromCopy(counter, index_label))

            labels_under_steps = VGroup()
            # Regenerate points based on sequence and small grid for accurate label positioning
            temp_points = [small_grid[0].c2p(0, 0)]
            tx, ty = 0, 0
            for move in sequence:
                if move == "E":
                    tx += 1
                else:
                    ty += 1
                temp_points.append(small_grid[0].c2p(tx, ty))
                path_points = np.array(temp_points)

            for j, move in enumerate(sequence):
                segment_start = path_points[j]
                segment_end = path_points[j + 1]
                mid_point = (segment_start + segment_end) / 2

                label = MathTex(move).scale(small_scale_factor)
                label.move_to(mid_point)
                if move == "E":
                    label.shift(DOWN * small_scale_factor * 0.3).set_color(ORANGE)
                else:
                    label.shift(LEFT * small_scale_factor * 0.3).set_color(GOLD_B)

                labels_under_steps.add(label)
                label_transforms.append(Transform(label, label_sequence[j]))

            all_labels_under_steps.add(labels_under_steps)

        self.wait()

        self.play(
            FadeOut(grid),
            *[DrawBorderThenFill(small_grid) for small_grid in small_grids],
            FadeOut(counter, rate_func=rate_functions.slow_into),
            AnimationGroup(*counter_transforms),
            AnimationGroup(*path_transforms),
            run_time=2,
        )

        self.wait(2)

        self.play(
            LaggedStart(
                *[Create(label) for label in all_labels_under_steps],
                lag_ratio=0.3,
                run_time=5,
            ),
            FadeOut(index_labels),
        )

        self.wait()

        self.play(
            LaggedStart(
                *label_transforms,
                lag_ratio=0.3,
                run_time=5,
            )
        )

        self.wait()

        # Clean up scene
        for label in all_labels_under_steps:
            self.remove(*label)
        self.remove(*lattice_paths)
        self.add(labeled_sequences, target_paths)

        self.play(FadeOut(lattice_path_title))
        self.wait()

        self.next_section(f"M{M}N{N}Scene 2- Binary Sequences", skip_animations=False)
        sequence_scale_factor = 1.5
        self.play(
            FadeOut(small_grids),
            FadeOut(target_paths),
            *[
                labeled_sequences[i]
                .animate.move_to(small_grids[i].get_center())
                .scale(sequence_scale_factor)
                for i in range(len(labeled_sequences))
            ],
        )

        binary_sequences = VGroup()
        sequence_transforms = []
        for i, labeled_sequence in enumerate(labeled_sequences):
            binary_sequence = VGroup()
            character_transforms = []
            for j, char in enumerate(labeled_sequence):
                if char.get_tex_string() == "E":
                    binary_sequence.add(
                        Integer(0)
                        .scale(small_scale_factor * sequence_scale_factor)
                        .match_style(char)
                        .move_to(char.get_center())
                    )
                else:  # char.get_tex_string() == "N"
                    binary_sequence.add(
                        Integer(1)
                        .scale(small_scale_factor * sequence_scale_factor)
                        .match_style(char)
                        .move_to(char.get_center())
                    )

                character_transforms.append(
                    ReplacementTransform(char, binary_sequence[-1])
                )
            sequence_transforms.append(
                LaggedStart(*character_transforms, lag_ratio=0.1)
            )
            binary_sequences.add(binary_sequence)

        binary_sequence_title = Text(
            rf"Binary sequences", color=LIGHT_PINK, font_size=52
        )
        binary_sequence_title.to_edge(UP, buff=0.3)
        counter.set_value(0).scale_to_fit_height(
            binary_sequence_title.height * 0.65
        ).next_to(binary_sequence_title, RIGHT, buff=1)
        self.play(Write(counter))
        dummy_tracker = ValueTracker(0)
        counter.add_updater(lambda m: m.set_value(dummy_tracker.get_value()))

        self.wait(0.5)
        self.play(ScaleInPlace(labeled_sequences[0], 1.5))
        self.play(sequence_transforms[0], dummy_tracker.animate.set_value(1))
        self.wait(0.5)
        self.play(ScaleInPlace(labeled_sequences[1], 1.5))
        self.play(sequence_transforms[1], dummy_tracker.animate.set_value(2))
        self.play(
            Write(binary_sequence_title),
            LaggedStart(*sequence_transforms[2:], lag_ratio=0.3, run_time=5),
            dummy_tracker.animate(rate_func=linear, run_time=5).set_value(binom_coeff),
        )
        counter.clear_updaters()
        self.play(
            LaggedStart(
                *[Indicate(binary_seq) for binary_seq in binary_sequences],
                lag_ratio=0.1,
            ),
            run_time=3,
        )
        self.play(Unwrite(counter))

        big_center_binary_sequences = [
            bin_seq.copy().move_to(ORIGIN).scale(3) for bin_seq in binary_sequences
        ]

        self.play(
            *[
                bin_seq.animate.set_opacity(0.5)
                for i, bin_seq in enumerate(binary_sequences)
                if i != 0
            ],
            ReplacementTransform(
                binary_sequences[0],
                big_center_binary_sequences[0],
                path_func=path_backwards_then_forwards,
            ),
            run_time=1.5,
        )
        self.play(binary_sequences[1].animate.set_opacity(1), run_time=0.3)
        self.play(
            ReplacementTransform(
                binary_sequences[1],
                big_center_binary_sequences[1],
                path_func=path_backwards_then_forwards,
            ),
            run_time=1.5,
        )
        self.remove(big_center_binary_sequences[0], big_center_binary_sequences[1])

        num_sequences_holding = ValueTracker(3)
        flicker_timer = ValueTracker(0.999)
        flicker_timer.add_updater(lambda m, dt: m.increment_value(dt))

        # Always cycle through each one every second
        def sequence_flickering():
            flicker_timer_normalized = flicker_timer.get_value() % 1
            current_index = int(
                flicker_timer_normalized * num_sequences_holding.get_value()
            )
            return big_center_binary_sequences[current_index]

        big_center_flickering_sequence = always_redraw(sequence_flickering)

        self.add(num_sequences_holding, flicker_timer, big_center_flickering_sequence)
        self.wait(2)
        self.play(binary_sequences[2].animate.set_opacity(1), run_time=0.3)
        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        *(
                            [
                                binary_sequences[i + 1]
                                .animate(run_time=0.3)
                                .set_opacity(1)
                            ]
                            if i + 1 < binom_coeff
                            else []
                        ),
                        Transform(
                            binary_sequences[i],
                            big_center_binary_sequences[i].copy(),
                            path_func=path_backwards_then_forwards,
                        ),
                    )
                    for i in range(2, binom_coeff)
                ]
            ),
            num_sequences_holding.animate.set_value(binom_coeff),
            run_time=3,
        )
        self.play(
            FadeOut(*binary_sequences),
            run_time=2,
        )
        self.wait(2)
        self.play(FadeOut(binary_sequence_title))
        self.wait()

        self.next_section(
            f"M{M}N{N}Scene 2.5- Universal Set Introduction", skip_animations=False
        )

        flicker_timer.clear_updaters()
        self.play(
            flicker_timer.animate.set_value(int(flicker_timer.get_value())),
            rate_func=slow_into,
            run_time=2,
        )
        self.remove(big_center_flickering_sequence)
        self.add(big_center_binary_sequences[0])
        universal_set = (
            MathTex(
                r"\{",
                *[
                    chr(ord("a") + i) + (", " if i < M + N - 1 else "")
                    for i in range(M + N)
                ],
                r"\}",
                substrings_to_isolate=", ",
            )
            .scale_to_fit_width(
                big_center_binary_sequences[0].width * 1 + (2.4 / (M + N + 2))
            )
            .next_to(big_center_binary_sequences[0], UP)
        ).save_state()
        self.play(Write(universal_set, run_time=2))
        self.play(
            VGroup(big_center_binary_sequences[0], universal_set).animate.to_edge(DOWN)
        )
        for i in range(M + N):
            char = universal_set[3 * i + 1]
            bin_num = big_center_binary_sequences[0][i]
            self.play(Indicate(VGroup(char, bin_num), run_time=0.3))

        for sequence in big_center_binary_sequences[1:]:
            sequence.to_edge(DOWN)

        self.next_section(f"M{M}N{N}Scene 3- Subsets", skip_animations=False)

        subsets_title = Text(rf"Subsets", color=LIGHT_PINK, font_size=52).to_edge(
            UP, buff=0.3
        )
        desired_width = MathTex(
            r"\{",
            *["a" + (", " if i < N - 1 else "") for i in range(N)],
            r"\}",
        ).width

        available_width = grid_cell_width * 0.8
        scale_factor_x = available_width / desired_width if M > 0 else available_width
        small_scale_factor = scale_factor_x

        all_one_mobs = []
        copy_up_subset_selections = []
        final_subset_selections = VGroup()
        copy_up_to_final_transforms = []
        for j in range(binom_coeff):
            # Pickup here.
            ones_in_sequence = VGroup()
            indicies_of_ones = []
            for i, digit in enumerate(big_center_binary_sequences[j]):
                if int(digit.get_value()) == 1:
                    ones_in_sequence.add(digit)
                    indicies_of_ones.append(i)

            all_one_mobs.append(ones_in_sequence)

            copied_up_selection = VGroup()
            for indx in indicies_of_ones:
                indx_in_universal_set = 1 + indx * 3
                copied_element = (
                    universal_set[indx_in_universal_set]
                    .copy()
                    .shift(UP)
                    .set_color(GOLD_B)
                )
                copied_up_selection.add(copied_element)

            copy_up_subset_selections.append(copied_up_selection)

            row = j // num_cols
            col = j % num_cols
            target_center_x = -total_grid_width / 2 + (col + 0.5) * grid_cell_width
            target_center_y = total_grid_height / 2 - (row + 0.5) * grid_cell_height
            target_center = np.array([target_center_x, target_center_y, 0])
            final_subset_selection = (
                MathTex(
                    r"\{",
                    *[
                        char.get_tex_string()
                        + (", " if i < len(copied_up_selection) - 1 else "")
                        for i, char in enumerate(copied_up_selection)
                    ],
                    r"\}",
                    substrings_to_isolate=", ",
                    tex_to_color_map={
                        **{
                            char.get_tex_string(): GOLD_B
                            for char in copied_up_selection
                        },
                        r"\{": gradient[j],
                        r"\}": gradient[j],
                        r",": gradient[j],
                    },
                )
                .scale(small_scale_factor)
                .move_to(target_center)
            )
            final_subset_selections.add(final_subset_selection)

            copy_up_to_final_transforms.append(
                LaggedStart(
                    *[
                        ReplacementTransform(
                            copied_up_selection[i],
                            final_subset_selection[3 * i + 1],
                        )
                        for i in range(len(copied_up_selection))
                    ],
                    FadeIn(
                        *[
                            elt
                            for k, elt in enumerate(final_subset_selection)
                            if k % 3 != 1
                        ]
                    ),
                )
            )

        final_subset_selections.save_state()

        # First two manually
        indication_animations = []
        for i in range(M + N):
            char = universal_set[3 * i + 1]
            bin_num = big_center_binary_sequences[0][i]
            if bin_num.get_value() != 1:
                continue
            indication_animations.append(Indicate(VGroup(char, bin_num)))
        self.play(
            LaggedStart(
                *indication_animations,
                lag_ratio=0.6,
            )
        )
        self.play(
            *[
                ReplacementTransform(one.copy(), copy_up_subset_selections[0][i])
                for i, one in enumerate(all_one_mobs[0])
            ],
        )
        self.wait()
        self.play(
            copy_up_to_final_transforms[0],
            *[
                FadeTransform(
                    old_bin, new_bin, replace_mobject_with_target_in_scene=True
                )
                for old_bin, new_bin in zip(
                    big_center_binary_sequences[0], big_center_binary_sequences[1]
                )
            ],
        )
        self.wait()
        indication_animations = []
        for i in range(M + N):
            char = universal_set[3 * i + 1]
            bin_num = big_center_binary_sequences[1][i]
            if bin_num.get_value() != 1:
                continue
            indication_animations.append(Indicate(VGroup(char, bin_num)))

        self.play(
            LaggedStart(
                *indication_animations,
                lag_ratio=0.6,
            )
        )
        self.play(
            *[
                ReplacementTransform(one.copy(), copy_up_subset_selections[1][i])
                for i, one in enumerate(all_one_mobs[1])
            ],
        )
        self.wait()
        self.play(
            Write(subsets_title),
            copy_up_to_final_transforms[1],
            *[
                FadeTransform(
                    old_bin, new_bin, replace_mobject_with_target_in_scene=True
                )
                for old_bin, new_bin in zip(
                    big_center_binary_sequences[1], big_center_binary_sequences[2]
                )
            ],
        )
        per_round_run_time = max(3 / (binom_coeff - 2), 0.3)
        for i in range(2, binom_coeff):
            self.play(
                *[
                    ReplacementTransform(one.copy(), copy_up_subset_selections[i][j])
                    for j, one in enumerate(all_one_mobs[i])
                ],
                run_time=per_round_run_time / 2,
            )
            if i != binom_coeff - 1:
                self.play(
                    copy_up_to_final_transforms[i],
                    *[
                        FadeTransform(
                            old_bin, new_bin, replace_mobject_with_target_in_scene=True
                        )
                        for old_bin, new_bin in zip(
                            big_center_binary_sequences[i],
                            big_center_binary_sequences[i + 1],
                        )
                    ],
                    run_time=per_round_run_time / 2,
                )
        self.play(copy_up_to_final_transforms[-1], run_time=per_round_run_time / 2)
        self.wait()
        self.play(
            FadeOut(big_center_binary_sequences[-1], subsets_title),
            universal_set.animate.to_edge(UP, buff=0.3),
            run_time=1.5,
        )
        self.wait()

        self.next_section(f"M{M}N{N}Scene 3.5- Splitting subsets", skip_animations=False)

        def placeholder_subset(num_elements):
            return MathTex(
                r"\{",
                *[
                    r"\underline{\hspace{7pt}}" + (", " if i < num_elements - 1 else "")
                    for i in range(num_elements)
                ],
                r"\}",
                substrings_to_isolate=", ",
            ).scale(small_scale_factor)

        n_placeholder = placeholder_subset(N)
        copy_unv_grp = (
            VGroup(n_placeholder, universal_set.copy())
            .arrange(RIGHT, buff=0.5)
            .to_edge(UP, buff=0.3)
        )
        universal_set.animate.move_to(copy_unv_grp[1])
        self.play(FadeIn(n_placeholder), universal_set.animate.move_to(copy_unv_grp[1]))

        indications_by_position = []
        for i in range(N):
            indications = [Indicate(n_placeholder[3 * i + 1])]
            indications.extend(
                [
                    Indicate(final_subset_selection[3 * i + 1])
                    for final_subset_selection in final_subset_selections
                ]
            )
            indications_by_position.append(AnimationGroup(*indications))

        self.play(
            LaggedStart(
                *indications_by_position,
                lag_ratio=1,
                run_time=N,
            )
        )

        original_subsets = [subset.copy() for subset in final_subset_selections]
        all_As = VGroup()
        reducing_transforms = []
        size_n_subsets = VGroup()
        size_nminus1_subsets = VGroup()
        for j, final_subset_selection in enumerate(final_subset_selections):
            contains_A = False
            for i in range(N):
                char = final_subset_selection[3 * i + 1]
                if char.get_tex_string() == "a":
                    all_As.add(char)
                    contains_A = True
                    break

            if contains_A:
                # Create a copy of the subset without 'a'
                reduced_subset = MathTex(
                    r"\{",
                    *[
                        final_subset_selection[3 * j + 1].get_tex_string()
                        + (", " if j < N - 1 else "")
                        for j in range(1, N)
                    ],
                    r"\}",
                    substrings_to_isolate=", ",
                    tex_to_color_map={
                        r"\{": gradient[j],
                        r"\}": gradient[j],
                        r",": gradient[j],
                    },
                ).scale(small_scale_factor)
                for j in range(N - 1):
                    reduced_subset[3 * j + 1].set_color(GOLD_B)
                reduced_subset.move_to(final_subset_selection.get_center())
                reducing_transforms.append(
                    TransformMatchingShapes(
                        VGroup(
                            *[
                                tex
                                for i, tex in enumerate(final_subset_selection)
                                if i != 1
                            ]
                        ),
                        reduced_subset,
                        replace_mobject_with_target_in_scene=True,
                    )
                )
                size_nminus1_subsets.add(reduced_subset)
            else:
                size_n_subsets.add(final_subset_selection)

        self.play(universal_set[1].animate.set_color(RED))
        self.play(all_As.animate.set_color(RED))
        self.play(FadeOut(universal_set[1], shift=UP), FadeOut(all_As, shift=UP))
        reduced_universal_set = (
            MathTex(
                r"\{",
                *[
                    chr(ord("a") + i) + (", " if i < M + N - 1 else "")
                    for i in range(1, M + N)
                ],
                r"\}",
                substrings_to_isolate=", ",
            )
            .move_to(copy_unv_grp[1])
            .scale(small_scale_factor)
        )
        self.play(
            *reducing_transforms,
            TransformMatchingShapes(
                VGroup(*[tex for i, tex in enumerate(universal_set) if i != 1]),
                reduced_universal_set,
                replace_mobject_with_target_in_scene=True,
            ),
        )
        self.wait()

        subset_scale_factor = 0.7

        move_to_clouds_shifts = []
        size_n_cloud_x_range = (-self.camera.frame_width / 2 + 1, -2)
        size_n_cloud_y_range = (
            -self.camera.frame_height / 2,
            self.camera.frame_height / 2 - 2.5,
        )
        subset_padding_x = size_n_subsets[0].width / 2
        subset_padding_y = size_n_subsets[0].height / 2
        for i, subset in enumerate(size_n_subsets):
            random_x = np.random.uniform(
                size_n_cloud_x_range[0] + subset_padding_x,
                size_n_cloud_x_range[1] - subset_padding_x,
            )  # Random x offset
            random_y = np.random.uniform(
                size_n_cloud_y_range[0] + subset_padding_y,
                size_n_cloud_y_range[1] + subset_padding_y,
            )  # Random y offset
            move_to_clouds_shifts.append(
                subset.animate.scale(subset_scale_factor).move_to(
                    [random_x, random_y, 0]
                )
            )

        size_nminus1_cloud_x_range = (2, self.camera.frame_width / 2 - 1)
        size_nminus1_cloud_y_range = (
            -self.camera.frame_height / 2,
            self.camera.frame_height / 2 - 2.5,
        )
        for i, subset in enumerate(size_nminus1_subsets):
            random_x = np.random.uniform(
                size_nminus1_cloud_x_range[0] + subset_padding_x,
                size_nminus1_cloud_x_range[1] - subset_padding_x,
            )  # Random x offset
            random_y = np.random.uniform(
                size_nminus1_cloud_y_range[0] + subset_padding_y,
                size_nminus1_cloud_y_range[1] + subset_padding_y,
            )
            move_to_clouds_shifts.append(
                subset.animate.scale(subset_scale_factor).move_to(
                    [random_x, random_y, 0]
                )
            )

        group_borders = VGroup(
            RoundedRectangle(
                corner_radius=0.5,
                width=size_n_cloud_x_range[1] - size_n_cloud_x_range[0],
                height=size_n_cloud_y_range[1] - size_n_cloud_y_range[0],
                color=BLUE,
            ).align_to(RIGHT * size_n_cloud_x_range[1], RIGHT),
            RoundedRectangle(
                corner_radius=0.5,
                width=size_nminus1_cloud_x_range[1] - size_nminus1_cloud_x_range[0],
                height=size_nminus1_cloud_y_range[1] - size_nminus1_cloud_y_range[0],
                color=BLUE,
            ).align_to(RIGHT * size_nminus1_cloud_x_range[0], LEFT),
        )
        counter_n = Integer(0).scale(small_scale_factor * subset_scale_factor)
        counter_nminus1 = Integer(0).scale(small_scale_factor * subset_scale_factor)
        group_labels = VGroup(
            VGroup(counter_n, placeholder_subset(N).scale(subset_scale_factor))
            .arrange(RIGHT, buff=0.5)
            .next_to(group_borders[0], UP, buff=0.1)
            .shift(LEFT * 0.3),
            VGroup(
                counter_nminus1, placeholder_subset(N - 1).scale(subset_scale_factor)
            )
            .arrange(RIGHT, buff=0.5)
            .next_to(group_borders[1], UP, buff=0.1)
            .shift(LEFT * 0.3),
        )

        self.play(
            reduced_universal_set.animate.center().to_edge(UP, buff=0.3),
            FadeOut(n_placeholder),
            LaggedStart(*move_to_clouds_shifts),
            run_time=2,
        )

        self.wait()
        self.play(Create(group_borders))
        self.play(Write(group_labels))
        self.wait(2)

        # Updaters for counters to track number of subsets with low opacity
        counter_n.add_updater(
            lambda m: m.set_value(
                len(
                    [
                        subset
                        for subset in size_n_subsets
                        if subset.get_center()[1] > group_borders.get_edge_center(UP)[1]
                    ]
                )
            )
        )
        counter_nminus1.add_updater(
            lambda m: m.set_value(
                len(
                    [
                        subset
                        for subset in size_nminus1_subsets
                        if subset.get_center()[1] > group_borders.get_edge_center(UP)[1]
                    ]
                )
            )
        )

        self.play(
            LaggedStart(
                *[
                    subset.animate(path_func=path_backwards_then_forwards)
                    .set_opacity(0)
                    .move_to(group_labels[0])
                    for subset in size_n_subsets
                ],
                run_time=N + M,
                lag_ratio=0.1,
            ),
            LaggedStart(
                *[
                    subset.animate(path_func=path_backwards_then_forwards)
                    .set_opacity(0)
                    .move_to(group_labels[1])
                    for subset in size_nminus1_subsets
                ],
                run_time=N + M,
                lag_ratio=0.1,
            ),
        )

        counter_n.clear_updaters()
        counter_nminus1.clear_updaters()

        self.wait()
        self.play(
            group_borders.animate.stretch_to_fit_height(0),
        )
        self.play(ShrinkToCenter(group_borders[0]), ShrinkToCenter(group_borders[1]))
        self.remove(group_borders)

        n_set_placeholder = (
            group_labels[0][1]
            .copy()
            .scale(1 / (subset_scale_factor * small_scale_factor))
        )
        nm1_set_placeholder = (
            group_labels[1][1]
            .copy()
            .scale(1 / (subset_scale_factor * small_scale_factor))
        )
        new_n_set_placeholder = n_set_placeholder.copy()
        addition_sets_line = (
            VGroup(
                n_set_placeholder,
                MathTex(" + "),
                nm1_set_placeholder,
                MathTex(" = "),
                new_n_set_placeholder,
            )
            .arrange(RIGHT, buff=0.5)
            .center()
        )

        self.play(
            reduced_universal_set.animate.scale(
                1 / (subset_scale_factor * small_scale_factor * 1.3)
            )
            .move_to(VGroup(addition_sets_line[:2]).get_center())
            .next_to(VGroup(addition_sets_line[:2]), UP, buff=0.5),
        )

        universal_set.scale_to_fit_width(reduced_universal_set.width + 0.1).next_to(
            addition_sets_line[4], UP, buff=0.5
        ).set_color(WHITE)
        universal_set[1].set_opacity(0).set_color(RED).shift(UP)

        self.play(
            TransformMatchingShapes(
                reduced_universal_set.copy(),
                universal_set,
                replace_mobject_with_target_in_scene=True,
            )
        )
        self.play(universal_set[1].animate.set_opacity(1).set_color(WHITE).shift(DOWN))

        self.play(
            LaggedStart(
                ReplacementTransform(group_labels[0][1], n_set_placeholder),
                Write(addition_sets_line[1]),
                ReplacementTransform(group_labels[1][1], nm1_set_placeholder),
                Write(addition_sets_line[3]),
                Write(addition_sets_line[4]),
                run_time=2,
                lag_ratio=0.1,
            )
        )

        total_count = (
            Integer(0)
            .match_style(counter_n)
            .next_to(addition_sets_line[4], DOWN, buff=0.5)
        )
        self.play(
            LaggedStart(
                counter_n.animate.scale(1 / (subset_scale_factor * small_scale_factor))
                .next_to(addition_sets_line[0], DOWN, buff=0.5)
                .set_value(0),
                counter_nminus1.animate.scale(
                    1 / (subset_scale_factor * small_scale_factor)
                )
                .next_to(addition_sets_line[2], DOWN, buff=0.5)
                .set_value(0),
                Write(total_count),
                run_time=2,
                lag_ratio=0.1,
            )
        )

        deal_to_n_reduced = []
        deal_to_nm1_reduced = []
        deal_to_n = []
        copied_n_subset_gps = []
        copied_nm1_subset_gps = []
        copied_original_subset_gps = []
        for i, subset in enumerate(size_n_subsets):
            subset.scale_to_fit_width(addition_sets_line[0].width).set_opacity(0)
            transform_animation = []
            current_subset_char_indx = 0
            copied_gp = VGroup()
            for j in range(M + N - 1):
                if current_subset_char_indx >= N:
                    break
                if (
                    reduced_universal_set[3 * j + 1].get_tex_string()
                    == subset[3 * current_subset_char_indx + 1].get_tex_string()
                ):
                    subset[3 * current_subset_char_indx + 1].move_to(
                        addition_sets_line[0][
                            3 * current_subset_char_indx + 1
                        ].get_center()
                    )
                    copied_char = (
                        reduced_universal_set[3 * j + 1].copy().set_color(GOLD_B)
                    )
                    copied_gp.add(copied_char)
                    transform_animation.append(
                        Transform(
                            copied_char,
                            subset[3 * current_subset_char_indx + 1],
                        )
                    )
                    current_subset_char_indx += 1
            copied_n_subset_gps.append(copied_gp)
            deal_to_n_reduced.append(AnimationGroup(*transform_animation))

        for i, subset in enumerate(size_nminus1_subsets):
            subset.scale_to_fit_width(addition_sets_line[2].width).set_opacity(0)
            transform_animation = []
            current_subset_char_indx = 0
            copied_gp = VGroup()

            for j in range(M + N - 1):
                if current_subset_char_indx >= N - 1:
                    break
                if (
                    reduced_universal_set[3 * j + 1].get_tex_string()
                    == subset[3 * current_subset_char_indx + 1].get_tex_string()
                ):
                    subset[3 * current_subset_char_indx + 1].move_to(
                        addition_sets_line[2][
                            3 * current_subset_char_indx + 1
                        ].get_center()
                    )
                    copied_char = (
                        reduced_universal_set[3 * j + 1].copy().set_color(GOLD_B)
                    )
                    copied_gp.add(copied_char)
                    transform_animation.append(
                        Transform(
                            copied_char,
                            subset[3 * current_subset_char_indx + 1],
                        )
                    )
                    current_subset_char_indx += 1

            copied_nm1_subset_gps.append(copied_gp)
            deal_to_nm1_reduced.append(AnimationGroup(*transform_animation))

        for i, subset in enumerate(original_subsets):
            subset.scale_to_fit_width(addition_sets_line[4].width).set_opacity(0)
            transform_animation = []
            current_subset_char_indx = 0
            copied_gp = VGroup()

            for j in range(M + N):
                if current_subset_char_indx >= N:
                    break
                if (
                    universal_set[3 * j + 1].get_tex_string()
                    == subset[3 * current_subset_char_indx + 1].get_tex_string()
                ):
                    subset[3 * current_subset_char_indx + 1].move_to(
                        addition_sets_line[4][
                            3 * current_subset_char_indx + 1
                        ].get_center()
                    )
                    copied_char = universal_set[3 * j + 1].copy().set_color(GOLD_B)
                    copied_gp.add(copied_char)
                    transform_animation.append(
                        Transform(
                            copied_char,
                            subset[3 * current_subset_char_indx + 1],
                        )
                    )
                    current_subset_char_indx += 1

            copied_original_subset_gps.append(copied_gp)
            deal_to_n.append(AnimationGroup(*transform_animation))

        counter_n.add_updater(
            lambda m: m.set_value(
                len(
                    [
                        subset
                        for subset in copied_n_subset_gps
                        if subset.get_center()[1] < 0.1
                    ]
                )
            )
        )
        counter_nminus1.add_updater(
            lambda m: m.set_value(
                len(
                    [
                        subset
                        for subset in copied_nm1_subset_gps
                        if subset.get_center()[1] < 0.1
                    ]
                )
            )
        )
        total_count.add_updater(
            lambda m: m.set_value(
                len(
                    [
                        subset
                        for subset in copied_original_subset_gps
                        if subset.get_center()[1] < 0.1
                    ]
                )
            )
        )

        self.play(
            LaggedStart(
                *deal_to_n_reduced,
                lag_ratio=0.1,
                run_time=3,
            ),
        )
        self.play(
            LaggedStart(
                *deal_to_nm1_reduced,
                lag_ratio=0.1,
                run_time=3,
            ),
        )
        self.play(
            LaggedStart(*deal_to_n, lag_ratio=0.1, run_time=5),
        )
        counter_n.clear_updaters()
        counter_nminus1.clear_updaters()
        total_count.clear_updaters()

        self.wait()
        self.play(Circumscribe(total_count))
        self.wait()

        self.remove(
            *size_n_subsets,
            *size_nminus1_subsets,
            *original_subsets,
            *copied_nm1_subset_gps,
            *copied_n_subset_gps,
            *copied_original_subset_gps,
        ),

        all_labels = VGroup(
            reduced_universal_set,
            universal_set,
            *addition_sets_line,
            counter_n,
            counter_nminus1,
        )

        self.next_section(
            f"M{M}N{N}Scene 4-Setting Up Pascal’s Triangle ", skip_animations=False
        )

        pascal_triangle_title = Tex(
            "Pascal's Triangle",
            color=LIGHT_PINK,
            font_size=52,
        ).to_edge(UP, buff=0.1)

        grid_lengths = (M + N + 1, 51)

        def get_faded_zero_mob():
            return Text("0", font="monospace").set_opacity(0.3).scale(0.5)

        first_zero_row = (
            VGroup(*[get_faded_zero_mob() for _ in range(grid_lengths[1])])
            .arrange(RIGHT, buff=0.3)
            .center()
            .next_to(pascal_triangle_title, DOWN, buff=0.3)
        )

        middle_top_index = grid_lengths[1] // 2
        first_one = (
            Text("1", font="monospace")
            .scale(0.5)
            .to_edge(UP)
            .move_to(first_zero_row[middle_top_index].get_center())
        )
        complete_first_row = first_zero_row.copy()
        complete_first_row[middle_top_index] = first_one

        zeros_by_row = [first_zero_row]
        pascal_mobs = {}
        pascal_mobs[(0, middle_top_index)] = first_one  # Store the actual Text mobject
        mobs_by_row = [complete_first_row]

        prev_row_values = [0, 1]  # Pad left with 0 always
        prev_row_left_index = middle_top_index - 1
        for r in range(1, grid_lengths[0]):
            row_values = [0]

            first_zero_mobs = [get_faded_zero_mob() for _ in range(prev_row_left_index)]
            last_zero_mobs = [get_faded_zero_mob() for _ in first_zero_mobs]
            middle_nonzero_mobs = []

            for i, val in enumerate(prev_row_values):
                current_index = prev_row_left_index + i

                val_right = (
                    prev_row_values[i + 1] if i + 1 < len(prev_row_values) else 0
                )
                new_val = val + val_right
                row_values.append(new_val)

                # Create the new number mobject
                non_zero_mob = Text(str(new_val), font="monospace").scale(0.5)
                middle_nonzero_mobs.append(non_zero_mob)
                pascal_mobs[(r, current_index)] = non_zero_mob

            complete_row_mobs = (
                VGroup(*first_zero_mobs, *middle_nonzero_mobs, *last_zero_mobs)
                .arrange(RIGHT, buff=0.3)
                .center()
                .next_to(mobs_by_row[-1], DOWN, buff=0.3)
            )
            mobs_by_row.append(complete_row_mobs)
            zero_row_mobs = complete_row_mobs.copy()
            for i in range(len(prev_row_values) + 1):
                zero_row_mobs[prev_row_left_index + i] = get_faded_zero_mob().move_to(
                    zero_row_mobs[prev_row_left_index + i].get_center()
                )

            zeros_by_row.append(zero_row_mobs)
            prev_row_left_index = prev_row_left_index - 1
            prev_row_values = row_values

        self.play(
            LaggedStart(*[Write(zero_row) for zero_row in zeros_by_row]),
            all_labels.animate.scale(0.7).to_edge(DOWN, buff=0.3),
            FadeOut(total_count, scale=0.5),
        )
        total_count.scale(0.7)
        self.wait()

        self.next_section(f"M{M}N{N} Scene 4.5-Pascal’s Triangle ", skip_animations=False)

        self.play(
            FadeIn(first_one, scale=2),
            FadeOut(first_zero_row[middle_top_index], scale=0.5),
        )
        self.wait()
        # Create animations
        row_by_row_animations = []
        padded_zeros = VGroup(first_zero_row).remove(first_one)
        pascal_by_row = [VGroup(first_one)]
        gradient_lines_by_row = VGroup()
        prev_row_length = 2  # Pad left with 0 always
        prev_row_left_index = middle_top_index - 1
        for r in range(1, grid_lengths[0]):
            padded_zeros.add(zeros_by_row[r])
            row_animations = []
            rows_gradient_lines = VGroup()
            row_mobs = VGroup()
            for i in range(prev_row_length):
                current_index = prev_row_left_index + i
                current_mob = pascal_mobs[(r, current_index)]
                prev_row_mob_indices = [
                    (r - 1, current_index),
                    (r - 1, current_index + 1),
                ]
                prev_row_mobs = []
                for coord in prev_row_mob_indices:
                    if coord not in pascal_mobs:
                        prev_row_mobs.append(zeros_by_row[coord[0]][coord[1]])
                    else:
                        prev_row_mobs.append(pascal_mobs[coord])

                gradient_lines = VGroup(
                    Line(
                        prev_row_mobs[0].get_center(),
                        current_mob.get_center(),
                        stroke_color=[BLUE, PINK],
                        buff=0.15,
                    ),
                    Line(
                        prev_row_mobs[1].get_center(),
                        current_mob.get_center(),
                        stroke_color=[BLUE, PINK],
                        buff=0.15,
                    ),
                )

                padded_zeros.remove(zeros_by_row[r][current_index])
                rows_gradient_lines.add(gradient_lines)
                row_animations.append(
                    Succession(
                        Create(gradient_lines),
                        ReplacementTransform(
                            zeros_by_row[r][current_index], current_mob
                        ),
                        FadeOut(gradient_lines),
                    )
                )
                row_mobs.add(current_mob)

            row_by_row_animations.append(row_animations)
            pascal_by_row.append(row_mobs)
            gradient_lines_by_row.add(rows_gradient_lines)
            prev_row_left_index = prev_row_left_index - 1
            prev_row_length = prev_row_length + 1

        current_row = 0
        while current_row < grid_lengths[0] - 1 and current_row < 3:
            wait_time = 0
            if current_row == 0:
                run_time = 2
                wait_time = 1
                lag_ratio = 1
            elif current_row == 1:
                run_time = 2
                lag_ratio = 1
                wait_time = 1
            elif current_row == 2:
                run_time = 2
                lag_ratio = 0.8
                wait_time = 1
            self.play(
                LaggedStart(
                    *row_by_row_animations[current_row],
                    lag_ratio=lag_ratio,
                    run_time=run_time,
                )
            )
            if wait_time != 0:
                self.wait(wait_time)
            current_row += 1

        if current_row < grid_lengths[0] - 1:
            remaining_animations = []
            for row_animations in row_by_row_animations[current_row:]:
                remaining_animations.extend(row_animations)
            self.play(
                Write(pascal_triangle_title),
                LaggedStart(
                    *remaining_animations,
                    lag_ratio=0.5,
                    run_time=grid_lengths[0] - 1 - current_row,
                ),
            )
        else:
            self.play(Write(pascal_triangle_title))

        highlighted_background_for_last_row = BackgroundRectangle(
            mobs_by_row[-1], fill_color=WHITE, fill_opacity=0.15, buff=0.1
        )

        highlighted_background_for_second_last_row = BackgroundRectangle(
            mobs_by_row[-2], fill_color=WHITE, fill_opacity=0.15, buff=0.1
        )

        self.wait(2)

        self.play(
            LaggedStart(
                FadeIn(highlighted_background_for_second_last_row, scale=4),
                FadeOut(padded_zeros),
            )
        )

        n_reduced_entry = pascal_mobs[(M + N - 1, middle_top_index - (M + N - 1) + N)]
        nm1_reduced_entry = pascal_mobs[
            (M + N - 1, middle_top_index - (M + N - 1) + (N - 1))
        ]
        binom_entry = pascal_mobs[(M + N, middle_top_index - (M + N) + N)]

        placed_counters = (
            VGroup(counter_nminus1.copy(), counter_n.copy())
            .arrange(RIGHT, buff=0.5)
            .next_to(pascal_by_row[-2], LEFT, buff=1)
        )

        self.wait()
        self.play(
            LaggedStart(
                ReplacementTransform(counter_n, placed_counters[0]),
                ReplacementTransform(counter_nminus1, placed_counters[1]),
                reduced_universal_set.animate.next_to(pascal_by_row[-2], RIGHT, buff=1),
                addition_sets_line[2]
                .animate.scale(0.5)
                .next_to(placed_counters[0], UP, buff=0.3),
                addition_sets_line[0]
                .animate.scale(0.5)
                .next_to(placed_counters[1], UP, buff=0.3),
                FadeOut(addition_sets_line[1], addition_sets_line[3], scale=0.5),
            ),
            run_time=2,
        )
        self.wait(0.5)
        self.play(Circumscribe(n_reduced_entry), Circumscribe(nm1_reduced_entry))
        self.wait(0.5)
        self.play(
            FadeOut(highlighted_background_for_second_last_row, scale=0.25),
            FadeIn(highlighted_background_for_last_row, scale=4),
        )

        total_count.next_to(pascal_by_row[-1], LEFT, buff=1)
        copied_gradient_lines = VGroup(
            Line(
                placed_counters[0].get_center(),
                total_count.get_center(),
                stroke_color=[BLUE, PINK],
                buff=0.15,
            ),
            Line(
                placed_counters[1].get_center(),
                total_count.get_center(),
                stroke_color=[BLUE, PINK],
                buff=0.15,
            ),
        )

        self.wait()
        self.play(
            LaggedStart(
                universal_set.animate.next_to(pascal_by_row[-1], RIGHT, buff=1),
                addition_sets_line[4]
                .animate.scale(0.5)
                .next_to(total_count, DOWN, buff=0.3),
            )
        )
        self.wait(0.5)
        self.play(
            Succession(
                Create(copied_gradient_lines),
                Write(total_count),
                AnimationGroup(
                    Circumscribe(binom_entry), FadeOut(copied_gradient_lines)
                ),
            ),
            run_time=2,
        )

        self.wait(2)
        self.play(
            LaggedStart(
                FadeOut(
                    addition_sets_line[0], addition_sets_line[2], addition_sets_line[4]
                ),
                FadeOut(reduced_universal_set),
                FadeOut(universal_set),
                FadeOut(placed_counters),
                FadeOut(total_count),
                FadeOut(pascal_triangle_title),
                FadeOut(highlighted_background_for_last_row),
                run_time=2,
            )
        )

        self.play(FadeOut(*pascal_by_row))

        self.next_section(f"M{M}N{N}Scene 5-Concluding Bijections", skip_animations=False)

        final_lattice_paths = (
            VGroup(grid.restore(), *[path.restore() for path in lattice_paths])
            .scale(0.7)
            .center()
            .to_edge(UP)
        )
        lattice_path_animation = LaggedStart(
            *[Create(path) for path in lattice_paths], lag_ratio=0.5
        )

        final_sequences = big_center_flickering_sequence

        for seq in big_center_binary_sequences:
            seq.center().to_edge(RIGHT).shift(UP * 0.5)

        final_pascal = (
            VGroup(
                [row[len(row) // 2 - 10 : len(row) // 2 + 10] for row in mobs_by_row]
            )
            .scale(0.5)
            .center()
            .to_edge(LEFT)
            .shift(UP * 0.5)
        )
        gradient_lines_by_row.scale(0.5).move_to(final_pascal.get_center())
        gradient_lines_animation = LaggedStart(
            *[
                Succession(
                    Create(lines),
                    FadeOut(lines),
                )
                for row in gradient_lines_by_row
                for lines in row
            ],
            lag_ratio=0.5,
        )

        final_subsets = (
            final_subset_selections.restore().scale(0.5).center().to_edge(DOWN)
        )

        self.play(Create(final_lattice_paths[0]))
        self.wait(0.5)
        self.play(Write(final_sequences))
        self.wait(0.5)
        self.play(Create(final_subsets))
        self.wait(0.5)
        self.play(Create(final_pascal))
        self.wait(0.5)
        flicker_timer.add_updater(lambda m, dt: m.increment_value(dt * 0.1))
        self.add(flicker_timer)
        self.play(gradient_lines_animation, lattice_path_animation, run_time=20)


class TempScene(Scene):
    def construct(self):
        pass


manim_configuration = {
    "quality": "high_quality",
    "preview": False,
    "output_file": "BinomialCoeffientsBijections",
    "disable_caching": False,
    "write_to_movie": True,
    "show_file_in_browser": False,
    "max_files_cached": 1000,
    "save_sections": True,
}

if __name__ == "__main__":
    # for m in range(1, 6):
    #     for n in range(1, 6):
    for m in range(1, 6):
        for n in range(2, 6):
            if m + n >= 4:
                # Update output file name for each combination
                manim_configuration["output_file"] = f"BinomialCoeffientsBijections_M{m}_N{n}"
                with tempconfig(manim_configuration):
                    np.random.seed(2)  # For reproducibility
                    print(f"Rendering M={m}, N={n}")
                    scene = LatticePath(m, n)
                    scene.render()


