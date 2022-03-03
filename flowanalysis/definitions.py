import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy
from matplotlib_venn import venn3, venn3_circles
from plotly.subplots import make_subplots

plt.rcParams.update({"text.usetex": True})
plt.rcParams["text.latex.preamble"] = r"\usepackage{graphicx}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


def header_clipping(experiment, cd45="+", check=False):

    with open(f"{experiment} priming/NTW-CD45{cd45}.csv") as oldfile:
        oldcsv = csv.reader(oldfile)

        with open(f"{experiment}{cd45}.csv", "w") as newfile:
            newcsv = csv.writer(newfile)

            header = True
            for row in oldcsv:
                if header:
                    new_names = []
                    for current_name in row:
                        new_names.append(current_name.rstrip())
                    newcsv.writerow(new_names)
                    header = False
                else:
                    newcsv.writerow(row)
            if check:
                print(new_names)


def time_name_list(experiment, headers, cd45="+"):

    names = []

    with open(f"{experiment}{cd45}.csv", "r") as file:
        csvfile = csv.reader(file)

        header = True
        for row in csvfile:
            if header:
                time_name = row.index(headers[1])
                header = False
            else:
                if row[time_name].rstrip() not in names:
                    names.append(row[time_name].rstrip())
    return names


def timepoint_extraction(
    experiment,
    organ,
    indices,
    time_names,
    time_name_index,
    normalise,
    double_count,
    cd45="+",
):
    # RETURN A THIRD LIST WITH NEGATIVE DATA
    data = []
    total_data = []

    with open(f"{experiment}{cd45}.csv", "r") as file:
        csvfile = csv.reader(file)

        header = True
        for row in csvfile:
            if header:
                header = False
                continue
            elif (
                row[indices[0]].rstrip().lower() == organ.lower()
                and row[indices[1]].rstrip().lower()
                == time_names[time_name_index].rstrip().lower()
            ):
                values = [int(float(row[index])) for index in indices[2:9]]
                # neg_values = [int(float(row[indices[9]])) - sum(values)]
                if double_count:
                    values[2] -= values[6]
                    values[4] -= values[6]
                    values[5] -= values[6]
                    values[0] -= values[2] + values[4] + values[6]
                    values[1] -= values[2] + values[5] + values[6]
                    values[3] -= values[4] + values[5] + values[6]
                if normalise:
                    if values[0] > 0:
                        values = [current_value / values[0] for current_value in values]
                    else:
                        print(
                            f"Sample for {time_names[time_name_index]} could not be normalised."
                        )
                data.append(tuple(values))
                # total_data.append(tuple(neg_values))
                total_data.append(tuple([int(float(row[indices[9]]))]))
    return data, total_data


def data_extraction(
    experiment,
    organ,
    headers,
    time_names,
    normalise=False,
    double_count=False,
    cd45="+",
    timepoints=None,
):

    data = []
    neg_data = []

    if timepoints is None:
        num_timepoints = len(time_names)
    else:
        num_timepoints = timepoints

    with open(f"{experiment}{cd45}.csv", "r") as file:
        csvfile = csv.reader(file)
        row = next(csvfile)

        indices = [
            row.index(headers[0]),  # Tissue
            row.index(headers[1]),  # time_name
            row.index(headers[2]),  # WT Single
            row.index(headers[3]),  # T8A Single
            row.index(headers[4]),  # WT T8A Double
            row.index(headers[5]),  # N3A Single
            row.index(headers[6]),  # WT N3A Double
            row.index(headers[7]),  # T8A N3A Double
            row.index(headers[8]),  # Triple
            row.index(headers[9]),  # Negative
        ]

    for current_time_name in range(num_timepoints):
        current_col, neg_current_col = timepoint_extraction(
            experiment,
            organ,
            indices,
            time_names,
            current_time_name,
            normalise,
            double_count,
            cd45=cd45,
        )
        for _ in range(len(current_col) - len(data)):
            previous = []
            neg_previous = []
            for _ in range(current_time_name):
                previous.append((-1, 0, 0, 0, 0, 0, 0))
                neg_previous.append((-1,))
            data.append(previous)
            neg_data.append(neg_previous)
        if current_time_name > 0:
            for row in data:
                row.append((-1, 0, 0, 0, 0, 0, 0))
            for row in neg_data:
                row.append((-1,))
        for current_mouse in range(len(current_col)):
            if current_time_name > 0:
                data[current_mouse][-1] = current_col[current_mouse]
                neg_data[current_mouse][-1] = neg_current_col[current_mouse]
            else:
                data[current_mouse].append(current_col[current_mouse])
                neg_data[current_mouse].append(neg_current_col[current_mouse])
    return data, neg_data


def population_means(populations, normalised=False):

    means = []
    rows = len(populations)
    cols = len(populations[0])

    for current_col in range(cols):

        actual_mice = rows
        for current_row in range(rows):
            if populations[current_row][current_col] == (
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
            ) or populations[current_row][current_col] == (-1,):
                actual_mice -= 1
            if normalised and int(populations[current_row][current_col][0]) != 1:
                actual_mice -= 1

        if actual_mice == 0:
            means.append((-1, 0, 0, 0, 0, 0, 0))
            continue

        current_mean = []
        for element in range(len(populations[0][current_col])):
            value = 0
            for current_row in range(rows):
                if not normalised and (
                    populations[current_row][current_col] != (-1, 0, 0, 0, 0, 0, 0)
                    or populations[current_row][current_col] != (-1,)
                ):
                    value += populations[current_row][current_col][element]
                if normalised and int(populations[current_row][current_col][0]) == 1:
                    value += populations[current_row][current_col][element]
            current_mean.append(round(value / actual_mice, 2))
        means.append(tuple(current_mean))
    return means


def venn_plots(
    num_timepoints,
    populations,
    experiments,
    tetramers,
    title=None,
    file_name=None,
    normalised=False,
    extension=None,
    show=False,
):

    max_mice = len(populations)
    means = population_means(populations, normalised=normalised)

    if title is not None:
        fig = plt.figure(
            constrained_layout=True,
            figsize=(16 * num_timepoints, (16 * (max_mice + 1)) + 4),
        )
        fig.suptitle("\n" + title + "\n", color="k", fontsize=90)
    else:
        fig = plt.figure(
            constrained_layout=True, figsize=(16 * num_timepoints, 16 * (max_mice + 1))
        )
        fig.suptitle("-", color="w", fontsize=60)
    col_figs = fig.subfigures(1, num_timepoints, wspace=0)
    row_figs = []
    for current_col in range(len(col_figs)):
        col_figs[current_col].suptitle(experiments[current_col], fontsize=80)
        row_figs.append(col_figs[current_col].subfigures(max_mice + 1, 1, hspace=0))
    fig_list = np.empty((max_mice + 1, num_timepoints), dtype=object)
    colours = [
        "#F7F4B6",
        "#A6DCF8",
        "#D66FAB",
        "#9BA3AB",
        "#86C665",
        "#BD7343",
        "#D6C6E1",
    ]
    patches = ["100", "010", "001", "111", "110", "101", "011"]

    for col in range(num_timepoints):
        for row in range(max_mice + 1):
            fig_list[row][col] = row_figs[col][row].subplots(1)

            if row < max_mice and (
                populations[row][col] == (-1, 0, 0, 0, 0, 0, 0)
                or populations[row][col] == (0, 0, 0, 0, 0, 0, 0)
            ):
                fig_list[row][col].axis("off")
                if populations[row][col] == (0, 0, 0, 0, 0, 0, 0):
                    fig_list[row][col].set_title(
                        f"Mouse {row + 1}", fontsize=65, color="grey"
                    )
                continue

            if row != max_mice:
                current = venn3(
                    subsets=populations[row][col],
                    set_labels=tetramers,
                    ax=fig_list[row][col],
                )
                fig_list[row][col].set_title(
                    f"Mouse {row + 1}", fontsize=65, color="grey"
                )
            elif means[col] != (-1, 0, 0, 0, 0, 0, 0):
                current = venn3(
                    subsets=means[col], set_labels=tetramers, ax=fig_list[row][col]
                )
                fig_list[row][col].set_title("Mean", fontsize=65, color="grey")
            else:
                fig_list[row][col].axis("off")

            for text in current.set_labels:
                text.set_fontsize(60)
            for text in current.subset_labels:
                if text is not None:
                    text.set_fontsize(50)

            for patch, colour in zip(patches, colours):
                try:
                    current.get_patch_by_id(patch).set_color(colour)
                except AttributeError:
                    pass

    if extension is None:
        extension = "pdf"
    if file_name is not None:
        fig.savefig(f"{file_name}.{extension}")
    if not show:
        plt.close("all")


def slope_plots(
    title,
    means,
    neg_means,
    times,
    patch_names,
    patch_indices,
    file_name=None,
    extension=None,
    zeroline=False,
    show=False,
):

    height = 14
    suptitle_size = 80
    title_size = 70
    label_size = 60
    tick_size = 50
    decimals = 2

    fig = plt.figure(constrained_layout=True, figsize=(8 * height, 3 * height))
    fig.suptitle("\n" + title + "\n", color="k", fontsize=suptitle_size)

    max_y = max([max(values) for values in means]) * 2

    col_figs = fig.subfigures(3, 8, wspace=0.05, hspace=0.05)
    fig_list = np.empty((3, 8), dtype=object)

    for current_row in range(len(col_figs)):
        for current_col in range(7):
            if current_row == 0:
                col_figs[current_row][current_col].suptitle(
                    patch_names[current_col], fontsize=title_size
                )
            fig_list[current_row][current_col] = col_figs[current_row][
                current_col
            ].subplots(1)
            current_patch = patch_indices[current_col]

            fig_list[current_row][current_col].set_yscale("symlog", linthresh=100)
            fig_list[current_row][current_col].set_ylim(-50, max_y)
            fig_list[current_row][current_col].set_xlim(5, 105)
            fig_list[current_row][current_col].tick_params(
                width=3, length=10, labelsize=tick_size
            )

            if current_col == 0:
                fig_list[current_row][current_col].set_ylabel(
                    f"{patch_names[current_row]} challenge (\\# of cells)",
                    fontsize=label_size,
                )
            else:
                fig_list[current_row][current_col].set_yticklabels([])

            fig_list[current_row][current_col].set_xticks(times)
            if current_row == len(col_figs) - 1:
                fig_list[current_row][current_col].set_xticklabels(times)
                fig_list[current_row][current_col].set_xlabel(
                    "Days post infection", fontsize=label_size
                )
            else:
                fig_list[current_row][current_col].set_xticklabels([])

            fig_list[current_row][current_col].plot(
                times[:2], [value[current_patch] for value in means[:2]], "-", color="k"
            )
            fig_list[current_row][current_col].plot(
                times[1:],
                [means[1][current_patch], means[current_row + 2][current_patch]],
                "-",
                color="k",
            )

            fig_list[current_row][current_col].plot(
                times[:2],
                [value[current_patch] for value in means[:2]],
                "D",
                ms=40,
                color="teal",
            )
            fig_list[current_row][current_col].plot(
                times[2],
                [means[current_row + 2][current_patch]],
                "D",
                ms=40,
                color="teal",
            )

            slope_1 = (means[1][current_patch] - means[0][current_patch]) / (
                times[1] - times[0]
            )
            y1 = (means[1][current_patch] + means[0][current_patch]) / 2
            slope_2 = (
                means[current_row + 2][current_patch] - means[1][current_patch]
            ) / (times[2] - times[1])
            y2 = (means[current_row + 2][current_patch] + means[1][current_patch]) / 8

            fig_list[current_row][current_col].text(
                40,
                y1,
                str(round(slope_1, decimals)),
                fontsize=label_size,
                bbox=dict(edgecolor="w", facecolor="w", alpha=1),
            )
            fig_list[current_row][current_col].text(
                80,
                y2,
                str(round(slope_2, decimals)),
                fontsize=label_size,
                bbox=dict(edgecolor="w", facecolor="w", alpha=1),
            )

            if zeroline:
                fig_list[current_row][current_col].axhline(
                    y=0, color="teal", linestyle="-"
                )

    for current_row in range(len(col_figs)):
        if current_row == 0:
            col_figs[current_row][7].suptitle(patch_names[7], fontsize=title_size)
        fig_list[current_row][7] = col_figs[current_row][7].subplots(1)
        fig_list[current_row][7].yaxis.tick_right()

        fig_list[current_row][7].set_yscale("symlog", linthresh=100)
        fig_list[current_row][7].set_ylim(
            min([value[0] for value in neg_means]) / 2,
            max([value[0] for value in neg_means]) * 2,
        )
        fig_list[current_row][7].set_xlim(5, 105)
        fig_list[current_row][7].tick_params(width=3, length=10, labelsize=tick_size)

        fig_list[current_row][7].set_xticks(times)
        if current_row == len(col_figs) - 1:
            fig_list[current_row][7].set_xticklabels(times)
            fig_list[current_row][7].set_xlabel(
                "Days post infection", fontsize=label_size
            )
        else:
            fig_list[current_row][7].set_xticklabels([])

        fig_list[current_row][7].plot(
            times[:2], [value[0] for value in neg_means[:2]], "-", color="k"
        )
        fig_list[current_row][7].plot(
            times[1:], [neg_means[1][0], neg_means[current_row + 2][0]], "-", color="k"
        )

        fig_list[current_row][7].plot(
            times[:2], [value[0] for value in neg_means[:2]], "D", ms=40, color="teal"
        )
        fig_list[current_row][7].plot(
            times[2], [neg_means[current_row + 2][0]], "D", ms=40, color="teal"
        )

        slope_1 = (neg_means[1][0] - neg_means[0][0]) / (times[1] - times[0])
        y1 = (neg_means[1][0] + neg_means[0][0]) / 2
        slope_2 = (neg_means[current_row + 2][0] - neg_means[1][0]) / (
            times[2] - times[1]
        )
        y2 = (neg_means[current_row + 2][0] + neg_means[1][0]) / 3

        fig_list[current_row][7].text(
            40,
            y1,
            str(round(slope_1, decimals)),
            fontsize=label_size,
            bbox=dict(edgecolor="w", facecolor="w", alpha=1),
        )
        fig_list[current_row][7].text(
            70,
            y2,
            str(round(slope_2, decimals)),
            fontsize=label_size,
            bbox=dict(edgecolor="w", facecolor="w", alpha=1),
        )

    if extension is None:
        extension = "pdf"
    if file_name is not None:
        fig.savefig(f"{file_name}.{extension}")
    if not show:
        plt.close("all")


def old_dataframe(data, priming, time, organ, cd45, columns, patches, timepoints):

    organised_data = []

    for current_infection in range(len(data)):
        for current_row in range(len(data[current_infection])):
            for current_col in range(len(data[current_infection][current_row])):
                new_item = [
                    organ[current_infection],
                    cd45[current_infection],
                    priming[current_infection],
                    time[current_row],
                    data[current_infection][current_row][current_col],
                    patches[current_col],
                    timepoints[current_row],
                ]

                organised_data.append(new_item[:])

    return pd.DataFrame(organised_data, columns=columns)


def plot_dataframe(data, priming, time, organ, cd45, columns, patches, timepoints):

    positive_data = [population_means(data[i]) for i in range(3)]

    organised_data = []

    for current_infection in range(len(positive_data)):
        for current_row in range(len(positive_data[current_infection])):
            for current_col in range(
                len(positive_data[current_infection][current_row])
            ):
                new_item = [
                    organ[current_infection],
                    cd45[current_infection],
                    priming[current_infection],
                    time[current_row],
                    positive_data[current_infection][current_row][current_col],
                    patches[current_col],
                    timepoints[current_row],
                ]

                organised_data.append(new_item[:])

    initial_dataframe = pd.DataFrame(organised_data, columns=columns)

    total_data = [population_means(data[i + 3]) for i in range(3)]

    organised_data = []

    for current_infection in range(3):
        for current_row in range(len(total_data[current_infection])):
            # for current_col in range(len(data[current_infection][current_row])):
            new_item = [
                organ[current_infection],
                cd45[current_infection],
                priming[current_infection],
                time[current_row],
                total_data[current_infection][current_row][0],
                "Total",
                timepoints[current_row],
            ]

            # if current_row%2 == 0:
            #    new_item.append('Triple negative')
            # else:
            #    new_item.append('Total')

            organised_data.append(new_item[:])

    total_dataframe = pd.DataFrame(organised_data, columns=columns)

    return initial_dataframe.append(total_dataframe, ignore_index=True)


def separate_data(data, primary, challenge, tetramer):
    return data[
        (data.Priming == primary)
        & (data.Tetramer == tetramer)
        & (
            (data.Timepoint == "Primary")
            | (data.Timepoint == "Memory")
            | (data.Timepoint == challenge)
        )
    ]


# SEPARATE INTO ONE LINE FOR FIRST CONTRACTION AND 3 LINES FOR EXPANSIONS [FUNCTION NEEDS UPDATING] (DOWN)


def data_update(data, tetramer):

    tetramer_list = ["WT", "T8A", "N3A"]
    updated_data = []

    for primary in tetramer_list:
        for challenge in tetramer_list:
            updated_data.append(separate_data(data, primary, challenge, tetramer).Cells)

    return updated_data
