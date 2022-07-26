import csv
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib_venn import venn3, venn3_circles
from plotly.subplots import make_subplots
from scipy import stats

plt.rcParams.update({"text.usetex": True})
plt.rcParams["text.latex.preamble"] = r"\usepackage{graphicx}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


class Mouse:
    """Class to represent the (tetramer) populations of cells from a mouse"""

    def __init__(
        self,
        wt=-1,
        t8a=0,
        n3a=0,
        wt_t8a=0,
        wt_n3a=0,
        t8a_n3a=0,
        triple_positive=0,
        triple_negative=0,
    ):
        """
        Constructor for the ``Mouse`` class.

        Parameters
        ----------
        wt : int
            Number of WT single positive cells.
        t8a : int
            Number of T8A single positive cells.
        n3a : int
            Number of N3A single positive cells.
        wt_t8a : int
            Number of WT+T8A double positive cells.
        wt_n3a : int
            Number of WT+N3A double positive cells.
        t8a_n3a : int
            Number of T8A+N3A double positive cells.
        triple_positive : int
            Number of triple positive cells.
        triple_negative : int
            Number of triple negative cells.
        """
        self.wt = wt
        self.t8a = t8a
        self.n3a = n3a
        self.wt_t8a = wt_t8a
        self.wt_n3a = wt_n3a
        self.t8a_n3a = t8a_n3a
        self.triple_positive = triple_positive
        self.triple_negative = triple_negative

    def __round__(self, n=None):
        return tuple([round(value, n) for value in self.cell_summary(complete=True)])

    def __bool__(self):
        if self.wt == -1:
            return False
        else:
            return True

    def __repr__(self):
        if self:
            return f"{self.cell_summary()} ({self.triple_negative})"
        else:
            return "Empty mouse"

    def cell_summary(self, venn=True, complete=False, ints=False):
        """
        Summarises the populations of tetramer positive cells.

        Parameters
        ----------
        venn : bool
            If True returns populations formatted for the venn3 package.
        complete : bool
            If False the triple negative population is removed.
        ints : bool
            If True returns a tuple of integers.

        Returns
        -------
        populations : tuple
            Tuple of populations of CD8 positive cells
        """
        populations = [
            self.wt,
            self.t8a,
            self.n3a,
            self.wt_t8a,
            self.wt_n3a,
            self.t8a_n3a,
            self.triple_positive,
            self.triple_negative,
        ]

        if venn:
            populations[2], populations[3] = populations[3], populations[2]
        if not complete:
            populations.pop()
        if ints:
            populations = [int(value) for value in populations]

        return tuple(populations)

    def max_cells(self, complete=False, ints=False):
        """
        Returns the maximum number of cells between populations.

        Parameters
        ----------
        complete : bool
            If False the triple negative population is removed.
        ints : bool
            if True returns an integer.

        Returns
        -------
        float, int
            Maximum number of cells between populations.

        """
        return max(self.cell_summary(complete=complete, ints=ints))

    def total_cells(self):
        """
        Calculates the total nuber of CD8 positive cells in the mouse

        Returns
        -------
        total_cells : int
            Total number of CD8 positive cells in the mouse
        """
        return sum(self.cell_summary(venn=False, complete=True))

    def no_plot(self):
        """
        Checks if a plot is required for the mouse.

        Returns
        -------
        bool
            True if there are positive populations, False otherwise.
        """
        if self.cell_summary() == (0, 0, 0, 0, 0, 0, 0):
            return True
        return False

    def frequency(self, venn=True, complete=False, digits=10):
        """
        Returns a tuple of the frequencies of each population with respect to CD8 positive cells.

        Parameters
        ----------
        venn : bool
            If True returns populations formatted for the venn3 package.
        complete : bool
            If False the triple negative population is removed.
        digits : int
            Number of decimal places to be rounded to.

        Returns
        -------
        tuple[float]
            Tuple of the frequencies of each population.
        """
        return tuple(
            [
                round(value / self.total_cells(), digits)
                for value in self.cell_summary(venn=venn, complete=complete)
            ]
        )


class Timepoint:
    """Class to represent a timepoint consisting of objects of the `Mouse` class"""

    def __init__(self):
        """
        Constructor for the ``Timepoint`` class.
        """
        self._mice = []
        self._num_mice = 0
        self._num_empty_mice = 0

    def __round__(self, n=None):
        return [round(mouse, n) for mouse in self._mice]

    def __len__(self):
        return self._num_mice + self._num_empty_mice

    def __repr__(self):
        return f"Timepoint with {self._num_mice} mice"

    def _add_mouse(self, mouse):
        """
        Adds ``mouse`` to the list of mice in the timepoint, and increases the number of mice.

        Parameters
        ----------
        mouse : Mouse
            Mouse to be added to the timepoint.
        """
        self._mice.append(mouse)
        self._num_mice += 1

    def mouse_list(self):
        """
        Returns the list of mice in the timepoint.

        Returns
        -------
        list[Mouse]
            List of mice in the timepoint.
        """
        return self._mice

    def num_mice(self):
        """
        Returns the number of mice in the timepoint.

        Returns
        -------
        int
            Number of mice in the timepoint
        """
        return self._num_mice

    def total_mice(self):
        """
        Returns the number of mice in the timepoint, including empty mice.

        Returns
        -------
        int
            Total number of mice in the timepoint.
        """
        return sum(self.mouse_summary())

    def mouse_summary(self):
        """
        Returns a summary of the mice in the timepoint.

        Returns
        -------
        tuple[int]
            Tuple of the number of actual mice and the number of empty mice.
        """
        return self._num_mice, self._num_empty_mice

    def add_mice(self, mice):
        """
        Adds a list of mice to the timepoint.

        Parameters
        ----------
        mice : list[Mouse]
        """
        for current in mice:
            self._add_mouse(current)

    def fill_empty_mice(self, number):
        """
        Fills the timepoint with empty mice so that the total number of mice is ``number``.

        Parameters
        ----------
        number : int
            Number of total mice wanted.
        """
        for _ in range(number - self.total_mice()):
            self._mice.append(Mouse())
        self._num_empty_mice += number - self.total_mice()

    def mean(self, venn=True, complete=False, frequency=True, digits=2):
        """
        ``Mouse``

        Parameters
        ----------
        venn : bool
            If True returns populations formatted for the venn3 package.
        complete : bool
            If False the triple negative population is removed.
        frequency : bool
            If True the frequency with respect to CD8 positive cells is calculated.
        digits : int
            Number of decimal places to be rounded to.

        Returns
        -------
        mean_values: tuple[float]
            Mean value of the populations in the timepoint.
        """
        mean_values = [0.0] * 8

        for mouse in self._mice:
            if mouse:
                if frequency:
                    current_values = mouse.frequency(venn=venn, complete=complete)
                else:
                    current_values = mouse.cell_summary(venn=venn, complete=complete)
                for population, value in enumerate(current_values):
                    mean_values[population] += value

        mean_values = [round(value / self._num_mice, digits) for value in mean_values]

        if not complete:
            mean_values.pop()

        return tuple(mean_values)

    def frequency(self, venn=True, complete=False, digits=10):
        """
        Returns a list of tuples containing the frequencies of each population for each mouse in the timepoint.

        Parameters
        ----------
        venn : bool
            If True returns populations formatted for the venn3 package.
        complete : bool
            If False the triple negative population is removed.
        digits : int
            Number of decimal places to be rounded to.

        Returns
        -------
        list[tuple[float]]
            List of frequencies for each mouse in the timepoint.
        """
        return [
            mouse.frequency(venn, complete, digits)
            if mouse
            else mouse.cell_summary_venn()
            for mouse in self._mice
        ]


class Experiment:
    """Class to represent and experiment consisting of objects of the `Timepoint` class"""

    def __init__(self, name):
        """
        Constructor for the ``Experiment`` class.

        Parameters
        ----------
        name : str
            Name of the experiment.
        """
        self.name = name
        self._timepoints = {}
        self._num_timepoints = 0
        self._shape = [0, 0]

    def __round__(self, n=None):
        return [round(timepoint, n) for timepoint in self._timepoints]

    def __len__(self):
        return self._num_timepoints

    def __repr__(self):
        return f"Experiment with {self._num_timepoints} timepoints {self.timepoint_names()}, with {[timepoint.mouse_summary() for timepoint in self.timepoints()]} mice each"

    def _add_timepoint(self, timepoint, timepoint_name):
        """
        Add ``timepoint`` under the name ``timepoint_name`` to the dictionary of timepoints.

        Updates the shape of the experiment and normalises the length of the timepoints.

        Parameters
        ----------
        timepoint : Timepoint
            Timepoint to be added to the experiment.
        timepoint_name : str
            Name of the timepoint to be added.
        """
        self._timepoints[timepoint_name] = timepoint

        self._num_timepoints += 1
        self._shape[1] += 1
        self._normalise_length()
        self._shape[0] = max(self.mouse_numbers())

    def _normalise_length(self):
        """
        Normalises the length of all timepoints to the length of largest timepoint.
        """
        max_mice = max(self.mouse_numbers())

        for name, timepoint in self._timepoints.items():
            if timepoint.total_mice() != max_mice:
                timepoint.fill_empty_mice(max_mice)

    def change_names(self, name_list):
        """
        Change the names of the timepoints.

        Parameters
        ----------
        name_list : list[str]
            Ordered list of new names for the timepoints.
        """
        self._timepoints = dict(zip(name_list, self._timepoints.values()))

    def shape(self):
        """
        Returns the shape of the experiment (mice, timepoints).

        Returns
        -------
        tuple[int]
            Shape of the experiment.
        """
        return tuple(self._shape)

    def num_timepoints(self):
        return self._num_timepoints

    def add_timepoints(self, timepoints, timepoint_names):
        """
        Adds a list of timepoints and their names to the experiment.

        Parameters
        ----------
        timepoints : list[Timepoint]
            List of ``Timepoint`` objects to be added.
        timepoint_names : list[str]
            List of names of all ``Timepoint`` objects to be added.

        Returns
        -------

        """
        for point, name in zip(timepoints, timepoint_names):
            self._add_timepoint(point, name)

    def timepoint_names(self):
        """
        Returns a list of the names of all timepoints in the experiment.

        Returns
        -------
        list[str]
            Names of timepoints in the experiment.
        """
        return list(self._timepoints.keys())

    def timepoints(self):
        """
        Returns a list of timepoints in the experiment.

        Returns
        -------
        list[Timepoint]
            List of timepoints in the experiment
        """
        return list(self._timepoints.values())

    def mouse_numbers(self):
        """
        Returns a list of the total number of mice in each timepoint.

        Returns
        -------
        list[int]
            Total number of mice in each timepoint.
        """
        return [timepoint.total_mice() for timepoint in self.timepoints()]

    def frequency(self, venn=True, complete=False, digits=10):
        """

        Parameters
        ----------
        venn : bool
            If True returns populations formatted for the venn3 package.
        complete : bool
            If False the triple negative population is removed.
        digits : int
            Number of decimal places to be rounded to.

        Returns
        -------
        list[list[tuple[float]]]
            List of frequencies for each mouse in each timepoint.
        """
        return [
            timepoint.frequency(venn, complete, digits)
            for timepoint in self.timepoints()
        ]

    def mean(self, venn=True, complete=False, frequency=False, digits=2):
        """

        Parameters
        ----------
        venn : bool
            If True returns populations formatted for the venn3 package.
        complete : bool
            If False the triple negative population is removed.
        frequency : bool
            If True the frequency with respect to CD8 positive cells is calculated.
        digits : int
            Number of decimal places to be rounded to.

        Returns
        -------
        list[tuple[float]]
            List of mean values for each timepoint in the experiment.
        """
        return [
            timepoint.mean(
                venn=venn, complete=complete, frequency=frequency, digits=digits
            )
            for timepoint in self.timepoints()
        ]

    def venn_plot(
        self, file_name, mean_only=False, frequency=True, labels=True, digits=2
    ):
        """
        Generate and save as a PDF a venn diagram plot of the experiment.

        Parameters
        ----------
        file_name : str
            Name of the file to save the plot to.
        mean_only : bool
            If True only the mean values will be plotted.
        frequency : bool
            If True the frequency with respect to CD8 positive cells is calculated.
        labels : bool
            If False the number labels for each subset will not be plotted.
        digits : int
            Number of decimal places to be rounded to.
        """

        height = self._shape[0] + 1
        if mean_only:
            height = 1

        if frequency:
            digits = 10

        fig = plt.figure(
            constrained_layout=True,
            figsize=(16 * self._shape[1], (16 * height) + 4),
        )
        fig.suptitle("\n" + self.name + "\n", color="k", fontsize=90)

        col_figs = fig.subfigures(1, self._shape[1], wspace=0)
        row_figs = []

        for current_col, _ in enumerate(col_figs):
            col_figs[current_col].suptitle(
                self.timepoint_names()[current_col], fontsize=80
            )
            row_figs.append(col_figs[current_col].subfigures(height, 1, hspace=0))
        fig_list = np.empty((height, self._shape[1]), dtype=object)

        for current_col, current_timepoint in enumerate(self.timepoints()):
            if not mean_only:
                for current_row, mouse in enumerate(current_timepoint.mouse_list()):
                    if mouse:
                        fig_list[current_row][current_col] = row_figs[current_col][
                            current_row
                        ].subplots(1)
                        fig_list[current_row][current_col].set_title(
                            f"Mouse {current_row + 1}", fontsize=65, color="grey"
                        )
                        if mouse.no_plot():
                            fig_list[current_row][current_col].axis("off")
                            continue

                        if frequency:
                            current_data = mouse.frequency(digits=digits)
                        else:
                            current_data = mouse.cell_summary(ints=True)

                        current_plot = venn3(
                            subsets=current_data,
                            set_labels=["WT", "T8A", "N3A"],
                            ax=fig_list[current_row][current_col],
                        )
                        _venn_plot_options(current_plot, labels, 60, 50)
            if mean_only:
                fig_list[current_col] = row_figs[current_col].subplots(1)
                fig_list[current_col].set_title("Mean", fontsize=65, color="grey")
            else:
                fig_list[-1][current_col] = row_figs[current_col][-1].subplots(1)
                fig_list[-1][current_col].set_title("Mean", fontsize=65, color="grey")
            means_plot = venn3(
                subsets=current_timepoint.mean(frequency=frequency, digits=digits),
                set_labels=["WT", "T8A", "N3A"],
                ax=fig_list[-1][current_col],
            )
            _venn_plot_options(means_plot, labels, 60, 50)

        fig.savefig(f"{file_name}.pdf")
        plt.close("all")

    def slope_plot(self, filename, zeroline=True):
        """
        Generate and save as a PDF a plot of all the slopes from the primary to memory, and memory to challenge timepoints.

        Parameters
        ----------
        filename : str
            Name of the file to save the plot to.
        zeroline : bool
            If True the line y=0 is plotted on all graphs except the triple negative one.
        """

        height = 14
        sup_title_size = 80
        title_size = 70

        fig = plt.figure(constrained_layout=True, figsize=(8 * height, 3 * height))
        fig.suptitle("\n" + self.name + "\n", color="k", fontsize=sup_title_size)

        max_y = max([max(value) * 5 for value in self.mean()])
        min_y_neg = min([max(value) / 2 for value in self.mean(complete=True)])
        max_y_neg = max([max(value) * 2 for value in self.mean(complete=True)])

        means = self.mean(venn=False, complete=True)

        col_figs = fig.subfigures(3, 8, wspace=0.05, hspace=0.05)
        fig_list = _slope_plot_array(
            col_figs, [max_y, min_y_neg, max_y_neg], zeroline, title_size
        )

        for current_row, _ in enumerate(col_figs):
            for current_col in range(7):
                _slope_plot(
                    fig_list[current_row, current_col],
                    means,
                    current_row,
                    current_col,
                    title_size,
                )
            _slope_plot(fig_list[current_row, -1], means, current_row, -1, title_size)

        fig.savefig(f"{filename}.pdf")
        plt.close("all")


def _venn_plot_options(ax, labels, label_size, number_size):
    """
    Set the parameters of ``ax`` to the desired values.

    Parameters
    ----------
    ax : matplotlib_venn._common.VennDiagram
        VennDiagram for which the parameters will be set.
    labels : bool
        If False the number labels for each subset will not be plotted.
    label_size : int
        Font size of the set labels.
    number_size : int
        Font size of the number labels.
    """

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

    if not labels:
        for i, _ in enumerate(ax.subset_labels):
            try:
                ax.subset_labels[i].set_visible(False)
            except AttributeError:
                pass
    for text in ax.set_labels:
        text.set_fontsize(label_size)
    for text in ax.subset_labels:
        if text is not None:
            text.set_fontsize(number_size)
    for patch, colour in zip(patches, colours):
        try:
            ax.get_patch_by_id(patch).set_color(colour)
        except AttributeError:
            pass


def _slope_plot_array(ax, y_lims, zeroline, fontsize):
    """
    Creates the array of subplots from the array of subfigures for the slope plots.

    Parameters
    ----------
    ax : numpy.ndarray
        Array of matplotlib.figure.SubFigure objects.
    y_lims : list[float]
        List of the y axis maximum of the tetramer positive plots, the minimum and the maximum of the triple negative plot.
    zeroline : bool
        If True the line y=0 is plotted on all graphs except the triple negative one.
    fontsize : int
        Font size of the plot.

    Returns
    -------
    fig_list : numpy.ndarray
        Array of subplots for the slope plots.
    """

    patch_names = [
        "WT",
        "T8A",
        "N3A",
        "WT+T8A",
        "WT+N3A",
        "T8A+N3A",
        "Triple positive",
        "Triple negative",
    ]
    times = [10, 70, 90]

    fig_list = np.empty((3, 8), dtype=object)

    for row, current_row in enumerate(ax):
        for col, _ in enumerate(current_row):
            fig_list[row, col] = ax[row, col].subplots(1)

            if col < len(current_row) - 1:
                fig_list[row, col].set_ylim(-50, y_lims[0])
            else:
                fig_list[row, col].set_ylim(y_lims[1], y_lims[2])
                fig_list[row, col].yaxis.tick_right()
            fig_list[row, col].set_yscale("symlog", linthresh=100)
            fig_list[row, col].set_xlim(5, 105)
            fig_list[row, col].tick_params(width=3, length=10, labelsize=fontsize - 20)

            if col == 0:
                fig_list[row, col].set_ylabel(
                    f"{patch_names[row]} challenge (\\# of cells)",
                    fontsize=fontsize - 10,
                )
            elif col < len(ax[0]) - 1:
                fig_list[row, col].set_yticklabels([])

            fig_list[row, col].set_xticks(times)
            if row == len(ax) - 1:
                fig_list[row, col].set_xticklabels(times)
                fig_list[row, col].set_xlabel(
                    "Days post infection", fontsize=fontsize - 10
                )
            else:
                fig_list[row, col].set_xticklabels([])

            if zeroline and col < len(ax[0]) - 1:
                fig_list[row, col].axhline(y=0, color="teal", linestyle="-")

    for index, current in enumerate(ax[0, :]):
        current.suptitle(patch_names[index], fontsize=fontsize)

    return fig_list


def _slope_plot(ax, means, row, col, fontsize, digits=2):
    """
    Plot the current slope plot on ``ax``.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Subplot in which the plot will be drawn.
    means : list[tuple[float]]
        List of mean values for each timepoint in the experiment.
    row : int
        Row of the position of the plot.
    col : int
        Column of the position of the plot.
    fontsize : int
        Font size of the plot.
    digits : int
        Number of decimal places to be rounded to.
    """

    times = [10, 70, 90]

    ax.plot(
        times[:2],
        [value[col] for value in means[:2]],
        "-",
        color="k",
    )
    ax.plot(
        times[1:],
        [means[1][col], means[row + 2][col]],
        "-",
        color="k",
    )

    ax.plot(
        times[:2],
        [value[col] for value in means[:2]],
        "D",
        ms=40,
        color="teal",
    )
    ax.plot(
        times[2],
        [means[row + 2][col]],
        "D",
        ms=40,
        color="teal",
    )

    slope_1 = (means[1][col] - means[0][col]) / (times[1] - times[0])
    y1 = (means[1][col] + means[0][col]) / 2
    slope_2 = (means[row + 2][col] - means[1][col]) / (times[2] - times[1])
    if col != -1:
        y2 = max(means[row + 2][col], means[1][col]) * 2
    else:
        y2 = (
            max(means[row + 2][col], means[1][col]) * 1.25
        )  # (means[row + 2][col] + means[1][col]) / 2

    ax.text(
        40,
        y1,
        str(round(slope_1, digits)),
        fontsize=fontsize - 10,
        ha="center",
        bbox=dict(edgecolor="w", facecolor="w", alpha=1),
    )
    ax.text(
        80,
        y2,
        str(round(slope_2, digits)),
        fontsize=fontsize - 10,
        ha="center",
        bbox=dict(edgecolor="w", facecolor="w", alpha=1),
    )


def header_clipping(experiment, cd45="+", filename=None, check=False):

    if filename is None:
        filename = "NTW-CD45"
        newname = f"{experiment}{cd45}.csv"
    else:
        newname = f"{experiment}-{filename}{cd45}.csv"

    with open(f"{experiment} priming/{filename}{cd45}.csv") as old_file:
        old_csv = csv.reader(old_file)

        with open(newname, "w") as newfile:
            new_csv = csv.writer(newfile)

            header = True
            for row in old_csv:
                if header:
                    new_names = []
                    for current_name in row:
                        new_names.append(current_name.rstrip())
                    new_csv.writerow(new_names)
                    header = False
                else:
                    new_csv.writerow(row)
            if check:
                print(new_names)


def time_name_list(experiment, headers, cd45="+", filename=None):

    if filename is None:
        filename = f"{experiment}{cd45}.csv"
    else:
        filename = f"{experiment}-{filename}{cd45}.csv"

    names = []

    with open(filename, "r") as file:
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


def _timepoint_extraction_challenge(  # RETURNS TIMEPOINT
    organ,
    indices,
    time_names,
    time_name_index,
    filename,
):

    current_timepoint = Timepoint()
    timepoint_mice = []
    # data = []
    # neg_data = []

    with open(filename, "r") as file:
        csvfile = csv.reader(file)
        next(csvfile)

        for row in csvfile:
            if (
                row[indices[0]].rstrip().lower() == organ.lower()
                and row[indices[1]].rstrip().lower()
                == time_names[time_name_index].rstrip().lower()
            ):
                values = [float(row[index]) for index in indices[2:9]]
                values.append(float(row[indices[9]]) - sum(values))
                timepoint_mice.append(Mouse(*values))

    current_timepoint.add_mice(timepoint_mice)
    return current_timepoint
    #             data.append(tuple(values))
    #             neg_data.append(tuple(neg_values))
    # return data, neg_data


def _timepoint_extraction_naive(  # RETURNS TIMEPOINT
    organ,
    indices,
    time_names,
    time_name_index,
    column,
    filename,
):

    non_zero_positions = {"WT": (0, 2, 4, 6), "T8A": (1, 2, 5, 6), "N3A": (3, 4, 5, 6)}
    # data = []
    current_timepoint = Timepoint()
    timepoint_mice = []

    with open(filename, "r") as file:
        csvfile = csv.reader(file)
        next(csvfile)

        for row in csvfile:
            if (
                row[indices[0]].rstrip().lower() == organ.lower()
                and row[indices[1]].rstrip().lower()
                == time_names[time_name_index].rstrip().lower()
            ):
                values = [0 for _ in range(7)]
                for index, value_index in zip(non_zero_positions[column], indices[2:6]):
                    values[index] = float(row[value_index])
                timepoint_mice.append(Mouse(*values))
    current_timepoint.add_mice(timepoint_mice)
    return current_timepoint
    #             data.append(tuple(values))
    # return data, None


def timepoint_extraction(
    organ,
    indices,
    time_names,
    time_name_index,
    data_type,
    column,
    filename,
):

    if data_type is None:
        return _timepoint_extraction_challenge(
            organ,
            indices,
            time_names,
            time_name_index,
            filename,
        )
    elif data_type == "naive" and column is not None:
        return _timepoint_extraction_naive(
            organ,
            indices,
            time_names,
            time_name_index,
            column,
            filename,
        )


def _column_index(filename, headers, data_type=None):

    if data_type is None:
        with open(filename, "r") as file:
            csvfile = csv.reader(file)
            row = next(csvfile)

            indices = [
                row.index(headers[0]),  # Tissue
                row.index(headers[1]),  # time_name
                row.index(headers[2]),  # WT Single
                row.index(headers[3]),  # T8A Single
                row.index(headers[4]),  # N3A Single
                row.index(headers[5]),  # WT T8A Double
                row.index(headers[6]),  # WT N3A Double
                row.index(headers[7]),  # T8A N3A Double
                row.index(headers[8]),  # Triple
                row.index(headers[9]),  # Negative
            ]
    elif data_type == "naive":
        with open(filename, "r") as file:
            csvfile = csv.reader(file)
            row = next(csvfile)

            indices = [
                row.index(headers[0]),  # Tissue
                row.index(headers[1]),  # column_name
                row.index(headers[2]),  # Single
                row.index(headers[3]),  # First Double
                row.index(headers[4]),  # Second Double
                row.index(headers[5]),  # Triple
                row.index(headers[6]),  # Negative
            ]

    return indices


def data_extraction(  # RETURNS EXPERIMENT
    experiment,
    organ,
    headers,
    time_names,
    standard_names=None,
    cd45=None,
    timepoints=None,
    data_type=None,
    column=None,
    filename=None,
):

    if cd45 is None:
        cd45 = "+"

    if cd45 == "+":
        cd45_name = "circulating"
    else:
        cd45_name = "resident"

    if filename is None:
        filename = f"{experiment}{cd45}.csv"
    else:
        filename = f"{experiment}-{filename}{cd45}.csv"

    current_experiment = Experiment(
        " ".join([organ[0].upper() + organ[1:], cd45_name, "--", experiment, "primary"])
    )
    experiment_timepoints = []

    if timepoints is None:
        num_timepoints = len(time_names)
    else:
        num_timepoints = timepoints

    indices = _column_index(filename, headers, data_type=data_type)

    for current_time_name in range(num_timepoints):
        current_timepoint = timepoint_extraction(
            organ,
            indices,
            time_names,
            current_time_name,
            data_type,
            column,
            filename,
        )
        experiment_timepoints.append(current_timepoint)

    current_experiment.add_timepoints(
        experiment_timepoints, time_names[:num_timepoints]
    )

    if standard_names is not None:
        current_experiment.change_names(standard_names)

    return current_experiment


def combine_naive_data(data_list):

    combined_data = []

    for current_row, _ in enumerate(data_list[0]):
        row_values = []
        for current_col, _ in enumerate(data_list):
            row_values.append(data_list[current_col][current_row][0])
        combined_data.append(deepcopy(row_values))

    return combined_data


def population_means(populations, normalised=False, ignore=None, decimals=None):

    if ignore is None:
        ignore = []

    if decimals is None:
        decimals = 2

    means = []
    actual_mice_list = []
    empty_row = (-1, 0, 0, 0, 0, 0, 0)
    rows = len(populations)
    cols = len(populations[0])
    actual_mice = 0

    for current_col in range(cols):

        actual_mice = rows
        for current_row in range(rows):
            if (
                populations[current_row][current_col] == empty_row
                or populations[current_row][current_col] == (-1,)
                or current_row in ignore
            ):
                actual_mice -= 1
            if normalised and int(populations[current_row][current_col][0]) != 1:
                actual_mice -= 1

        if actual_mice == 0:
            means.append((-1, 0, 0, 0, 0, 0, 0))
            continue

        current_mean = []
        for element, _ in enumerate(populations[0][current_col]):
            value = 0
            for current_row in range(rows):
                if current_row in ignore:
                    continue
                if not normalised and (
                    populations[current_row][current_col] != (-1, 0, 0, 0, 0, 0, 0)
                    or populations[current_row][current_col] != (-1,)
                ):
                    value += populations[current_row][current_col][element]
                if normalised and int(populations[current_row][current_col][0]) == 1:
                    value += populations[current_row][current_col][element]
            current_mean.append(round(value / actual_mice, decimals))
        means.append(tuple(current_mean))
        actual_mice_list.append(actual_mice)

    if ignore == [] or (
        all(num_mice == actual_mice_list[0] for num_mice in actual_mice_list)
        and ignore is not None
    ):
        return means, actual_mice
    else:
        print("Incomplete dataset, try using a different restriction.")
        return [], -1


def _round_populations(populations, decimals=None):

    if decimals is None:
        decimals = 2

    rounded_data = []

    for mouse, mouse_data in enumerate(populations):
        rounded_mouse = []
        for column in mouse_data:
            rounded_mouse.append(
                tuple([round(current_value, decimals) for current_value in column])
            )
        rounded_data.append(deepcopy(rounded_mouse))

    return rounded_data


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
    sup_title_size = 80
    title_size = 70
    label_size = 60
    tick_size = 50
    decimals = 2

    fig = plt.figure(constrained_layout=True, figsize=(8 * height, 3 * height))
    fig.suptitle("\n" + title + "\n", color="k", fontsize=sup_title_size)

    max_y = max([max(values) for values in means]) * 2

    col_figs = fig.subfigures(3, 8, wspace=0.05, hspace=0.05)
    fig_list = np.empty((3, 8), dtype=object)

    for current_row, _ in enumerate(col_figs):
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

    for current_row, _ in enumerate(col_figs):
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


def _old_dataframe(data, priming, time, organ, cd45, columns, patches, timepoints):

    organised_data = []

    for current_infection, _ in enumerate(data):
        for current_row, _ in enumerate(data[current_infection]):
            for current_col, _ in enumerate(data[current_infection][current_row]):
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

    positive_data = [population_means(data[i])[0] for i in range(3)]

    organised_data = []

    for current_infection, _ in enumerate(positive_data):
        for current_row, _ in enumerate(positive_data[current_infection]):
            for current_col, _ in enumerate(
                positive_data[current_infection][current_row]
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

                organised_data.append(deepcopy(new_item))

    initial_dataframe = pd.DataFrame(organised_data, columns=columns)

    total_data = [population_means(data[i + 3])[0] for i in range(3)]

    organised_data = []

    for current_infection in range(3):
        for current_row, _ in enumerate(total_data[current_infection]):
            # for current_col, _ in enumerate(data[current_infection][current_row]):
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

            organised_data.append(deepcopy(new_item))

    total_dataframe = pd.DataFrame(organised_data, columns=columns)

    return initial_dataframe.append(total_dataframe, ignore_index=True)


def stats_dataframe_infection(full_data, columns, complete=False):
    """
    Generates a data frame containing the data from `full_data` with column names given by `columns`.

    Parameters
    ----------
    full_data : tuple
        Tuple containing the list of positive stained cells, and the list of negative cells.
    columns : list[str]
        List of names of the columns of the data.
    complete : bool
        Consider only challenge timepoint data if False.

    Returns
    -------
    pd.DataFrame
        Data frame containing the organised extracted data.
    """

    if not complete:
        challenge_timepoints = {2: "WT", 3: "T8A", 4: "N3A"}
    else:
        challenge_timepoints = {0: "Primary", 1: "Memory", 2: "WT", 3: "T8A", 4: "N3A"}

    organised_data = []

    data, data_neg = full_data

    for current_mouse, current_data in enumerate(data):
        for index, column_data in enumerate(current_data):

            if index not in challenge_timepoints:
                continue

            current_values = []

            for cell_numbers in column_data:
                if cell_numbers == -1:
                    break
                else:
                    current_values.append(cell_numbers)
            else:
                current_values.append(data_neg[current_mouse][index][0])
                current_values.insert(0, challenge_timepoints[index])
                organised_data.append(deepcopy(current_values))

    organised_dataframe = pd.DataFrame(organised_data, columns=columns)

    return organised_dataframe


def stats_dataframe_naive(data, columns):
    """

    Parameters
    ----------
    data : tuple
    columns : list[str]

    Returns
    -------
    pd.DataFrame
        Data frame containing the organised extracted data.
    """

    non_zero_positions = {"WT": (0, 2, 4, 6), "T8A": (1, 2, 5, 6), "N3A": (3, 4, 5, 6)}
    column_indexes = {0: "WT", 1: "T8A", 2: "N3A"}

    organised_data = []

    for current_mouse, current_data in enumerate(data):
        organised_data.append([])
        for column_index, column_data in enumerate(current_data):
            for index, cell_numbers in enumerate(column_data):
                if index in non_zero_positions[column_indexes[column_index]]:
                    organised_data[current_mouse].append(cell_numbers)

    organised_dataframe = pd.DataFrame(organised_data, columns=columns)

    return organised_dataframe


def pairs_stats(x, y, comparisons=1, corr_position=0, corr_total=1, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            # corr, pvalue = stats.pearsonr(x, y)
            corr, pvalue = stats.spearmanr(x, y)
            corr_size = 15 + (1 - pvalue) * 15

            ast = ""
            if pvalue <= 0.01 / comparisons:
                ast = "{\\ast}{\\ast}"
            elif pvalue <= 0.05 / comparisons:
                ast = "{\\ast}"

            corr_text = f"${corr:.3f}{ast}$"
        except stats.PearsonRConstantInputWarning:
            corr_size = 30
            corr_text = "$\\textrm{--}$"
        except stats.SpearmanRConstantInputWarning:
            corr_size = 30
            corr_text = "$\\textrm{--}$"

    ax = plt.gca()
    ax.set_axis_off()
    ax.annotate(
        corr_text,
        xy=(0.5, (corr_total - corr_position) * (1 / (corr_total + 1))),
        xycoords=ax.transAxes,
        ha="center",
        fontsize=corr_size,
    )


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
