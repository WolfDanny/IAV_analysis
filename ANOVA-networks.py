import numpy as np
import pandas as pd

infections = ["WT", "T8A", "N3A"]
timepoints = ["Primary", "Memory", "WT", "T8A", "N3A"]
organs = ["Spleen", "Lungs"]
cd45 = ["Circulating", "Resident"]
populations = {
    "WT": 0,
    "T8A": 1,
    "N3A": 2,
    "WT+T8A": 3,
    "WT+N3A": 4,
    "T8A+N3A": 5,
    "TP": 6,
    "TN": 7,
}
standalone = False
frequencies = True


def coordinate_loop(nodes):
    """
    Generates the LaTeX commands to place ``nodes`` evenly spaced coordinates on a circle of radius r.

    Parameters
    ----------
    nodes : int
        Number of coordinates to be placed.

    Returns
    -------
    str
        LaTeX commands to generate evenly spaced nodes in a circle of radius r.
    """

    return f"\\foreach \\i in {{0,...,{nodes}}}{{\n\t\\coordinate (c\\i) at (\\i*360/{nodes}:\\ranova);\n}}\n"


def node_loop(nodes, edges, numbered=False):
    """
    Generates the LaTeX commands to draw ``nodes`` nodes in a set of pre-existing coordinates.

    Parameters
    ----------
    nodes : int
        Number of nodes to be drawn.
    edges : list[tuple[int]]
        Edges in the figure.
    numbered : bool
        Numerical labeling of the nodes.

    Returns
    -------
    str
        LaTeX commands to draw nodes in a set of coordinates.
    """

    edges = [a for (a, b) in edges]

    if not numbered:
        return f"\\foreach \\i in {{0,...,{nodes}}}{{\n\t\\node (p\\i) at (c\\i)[ANOVAS]{{}};\n}}\n\n"
    else:
        node_names = []
        large_node_names = []
        for i in range(nodes):
            if i not in edges:
                node_names.append(f"{i}/{i+1}")
            else:
                large_node_names.append(f"{i}/{i+1}")
        node_names = ", ".join(node_names)

        nodes_string = [
            f"\\foreach \\i/\\j in {{{node_names}}}{{\n\t\\node (p\\i) at (c\\i)[ANOVAS]{{$\\j$}};\n}}\n"
        ]
        if large_node_names:
            large_node_names = ", ".join(large_node_names)
            nodes_string.append(
                f"\\foreach \\i/\\j in {{{large_node_names}}}{{\n\t\\node (p\\i) at (c\\i)[ANOVAL]{{$\\j$}};\n}}\n\n"
            )
        else:
            nodes_string.append("\n")

        return "".join(nodes_string)


def edge_loop(edges, dashed=False):
    """
    Generates the LaTeX commands to draw the edges specified in ``edges``.

    Parameters
    ----------
    edges : list[tuple[int]]
        List of edges to be drawn.
    dashed : bool
        Dashed edges.

    Returns
    -------
    str
        LaTeX commands to draw the specified edges.
    """

    edges = [f"{a}/{b}" for (a, b) in edges]
    comma = ", "

    if dashed:
        return f"\\foreach \\i/\\j in {{{comma.join(edges)}}}{{\n\t\\draw[preaction={{draw, line width=3pt, white}}, thick, dashed, black!40] (c\\i) -- (c\\j);\n}}\n"
    else:
        return f"\\foreach \\i/\\j in {{{comma.join(edges)}}}{{\n\t\\draw[preaction={{draw, line width=3pt, white}}, thick] (c\\i) -- (c\\j);\n}}\n"


def tikz_legend(legend_populations):
    """
    Generates the LaTeX commands to draw the legend of the TikZ figure.

    Parameters
    ----------
    legend_populations : dict
        Dictionary of population names (keys) and indexes (values).

    Returns
    -------
    str
        LaTeX commands to draw the legend of the figure.
    """

    legend = [
        "\n\\multirow{3}{*}[3.5em]{ % [3.5em] is the fixup parameter for vertical alignment. # of units moved upwards\n",
        "\\scalebox{0.8}{\n\\begin{tikzpicture}\n\\matrix [row sep=5pt, draw=black, fill=black!5, rounded corners=15pt, thick] {\n",
    ]
    for legend_population, legend_index in legend_populations.items():
        legend.append(
            f"\t\\node[cell, label=right:{legend_population}]"
            + "{"
            + f"${legend_index+1}$"
            + "};\\\\\n"
        )
    legend.append("};\n\\end{tikzpicture}\n}\n}\n")

    return "".join(legend)


for organ in organs:
    for residency in cd45:
        with open(f"ANOVA/{organ}-{residency[:3]}.tex", "w") as outfile:

            experiment_name = f"{organ} {residency.lower()}"
            line = "=" * 70
            outfile.write(f"\n\n%{line}\n")
            outfile.write(f"%{line}\n")
            outfile.write(f"%{experiment_name : ^69}\n\n")
            outfile.write(
                "%Make sure to include these lines in the preamble of your TeX document\n\n"
            )
            outfile.write("%\\usepackage{tikz}\n")
            outfile.write("%\\pgfmathsetmacro\\ranova{1.5}\n")
            outfile.write("%\\usepackage{multirow}\n")
            outfile.write("%\\definecolor{ANOVAGreen}{RGB}{121,162,40}\n")
            outfile.write(
                "%\\tikzstyle{ANOVAS}=[circle,draw=ANOVAGreen,fill=ANOVAGreen!30,very thick,minimum size = 5pt]\n"
            )
            outfile.write(
                "%\\tikzstyle{ANOVAL}=[circle,draw=ANOVAGreen,fill=ANOVAGreen!30,very thick,minimum size = 25pt]\n"
            )
            outfile.write(f"%{line}\n")
            outfile.write(f"%{line}\n")
            outfile.write("\n\n")

            if standalone:
                outfile.write("\\begin{figure}\n")

            outfile.write(
                "\\centering\n\\begin{tabular}{ccc|c|cccc}\n\n\\multirow{3}{*}{\\rotatebox{90}{\Large Primary infection}} &\n"
            )

            for primary_index, primary in enumerate(infections):
                if primary_index != 0:
                    outfile.write("\n & ")
                outfile.write("\\rotatebox{90}{\\phantom{nnnn}" + primary + "} &\n")
                for timepoint_index, timepoint in enumerate(timepoints):

                    if timepoint_index < 2:
                        filename = f"ANOVA/{primary}/Tukey-{timepoint[0]}-{organ[0]}-{residency[:3]}"
                    else:
                        filename = f"ANOVA/{primary}/Tukey-{timepoint}-{organ[0]}-{residency[:3]}"

                    if frequencies:
                        filename = "".join([filename, "-F.csv"])
                    else:
                        filename = "".join([filename, ".csv"])

                    try:
                        with open(filename) as file:
                            current_datafame = pd.read_csv(file)

                        # Reading and standardising names
                        current_names = [
                            name.split("-") for name in current_datafame.iloc[:, 0]
                        ]
                        for row, current_pair in enumerate(current_names):
                            for col, current_population in enumerate(current_pair):
                                current_names[row][col] = current_population.replace(
                                    ".", "+"
                                )

                        # Reading data
                        current_data = current_datafame.iloc[:, 1:].to_numpy()

                        # Identifying significant differences
                        significant_differences = []
                        triple_negative_differences = []
                        for index, values in enumerate(current_data):
                            if values[-1] < 0.05:
                                current_nodes = [
                                    populations[current_names[index][i]]
                                    for i in range(2)
                                ]
                                if 7 in current_nodes:
                                    triple_negative_differences.append(
                                        (current_nodes[0], current_nodes[1])
                                    )
                                else:
                                    significant_differences.append(
                                        (current_nodes[0], current_nodes[1])
                                    )

                        outfile.write(
                            "\n\\scalebox{0.6}{ % " + f"{primary} -- {timepoint}\n"
                        )
                        outfile.write("\\begin{tikzpicture}\n\n")
                        outfile.write(coordinate_loop(len(populations)))
                        outfile.write(
                            edge_loop(triple_negative_differences, dashed=True)
                        )
                        if significant_differences:
                            outfile.write(edge_loop(significant_differences))
                        outfile.write(
                            node_loop(
                                len(populations), significant_differences, numbered=True
                            )
                        )
                        outfile.write("\\end{tikzpicture}\n}\n&\n\n")
                        if timepoint_index == 4:
                            if primary_index == 0:
                                outfile.write(tikz_legend(populations))
                            outfile.write("\\\\\n")

                    except FileNotFoundError:
                        print(
                            f"File for {primary} {timepoint} {organ} {residency} NOT found"
                        )
                        pass

            outfile.write(
                "\n & & Primary & Memory & WT challenge & T8A challenge & N3A challenge & \\\\\n"
            )
            outfile.write(" & & \\multicolumn{5}{c}{\\Large Timepoint} & \\\\\n")
            outfile.write("\\end{tabular}\n")
            if standalone:
                outfile.write("\\end{figure}\n")
