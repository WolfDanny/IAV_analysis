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
standalone = True


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

    return f"\\foreach \\i in {{0,...,{nodes}}}{{\n\t\\coordinate (c\\i) at (\\i*360/{nodes}:\\ranova);\n}}\n\n"


def node_loop(nodes, numbered=False):
    """
    Generates the LaTeX commands to draw ``nodes`` nodes in a set of pre-existing coordinates.

    Parameters
    ----------
    nodes : int
        Number of nodes to be drawn.
    numbered : bool
        Numerical labeling of the nodes.

    Returns
    -------
    str
        LaTeX commands to draw nodes in a set of coordinates.
    """

    if not numbered:
        return f"\\foreach \\i in {{0,...,{nodes}}}{{\n\t\\node (p\\i) at (c\\i)[cell]{{}};\n}}\n\n"
    else:
        node_names = []
        for i in range(nodes):
            node_names.append(f"{i}/{i+1}")
        node_names = ", ".join(node_names)

        return f"\\foreach \\i/\\j in {{{node_names}}}{{\n\t\\node (p\\i) at (c\\i)[cell]{{$\\j$}};\n}}\n\n"


def edge_loop(edges, dashed=False):
    """
    Generates the LaTeX commands to draw the edges specified in ``edges``.

    Parameters
    ----------
    edges : list
        List of edges to be drawn.
    dashed : bool
        Dashed edges.

    Returns
    -------
    str
        LaTeX commands to draw the specified edges.
    """

    comma = ", "

    if dashed:
        return f"\\foreach \\i/\\j in {{{comma.join(edges)}}}{{\n\t\\draw[preaction={{draw, line width=3pt, white}}, thick, dashed, black!40] (c\\i) -- (c\\j);\n}}\n"
    else:
        return f"\\foreach \\i/\\j in {{{comma.join(edges)}}}{{\n\t\\draw[preaction={{draw, line width=3pt, white}}, thick] (c\\i) -- (c\\j);\n}}\n"


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
                outfile.write("\\rotatebox{90}{\phantom{nnnn}" + primary + "} &\n")
                for timepoint_index, timepoint in enumerate(timepoints):

                    if timepoint_index < 2:
                        filename = f"ANOVA/{primary}/Tukey-{timepoint[0]}-{organ[0]}-{residency[:3]}.csv"
                    else:
                        filename = f"ANOVA/{primary}/Tukey-{timepoint}-{organ[0]}-{residency[:3]}.csv"

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
                                        f"{current_nodes[0]}/{current_nodes[1]}"
                                    )
                                else:
                                    significant_differences.append(
                                        f"{current_nodes[0]}/{current_nodes[1]}"
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
                        outfile.write(node_loop(len(populations), numbered=True))
                        outfile.write("\\end{tikzpicture}\n}\n&\n\n")
                        if timepoint_index == 4:
                            if primary_index == 0:
                                outfile.write(
                                    "\n\\multirow{3}{*}[3.5em]{ % [3.5em] is the fixup parameter for vertical alignment. # of units moved upwards\n"
                                )
                                outfile.write(
                                    "\\scalebox{0.8}{\n\\begin{tikzpicture}\n\\matrix [row sep=5pt, draw=black, fill=black!5, rounded corners=15pt, thick] {\n"
                                )
                                for population, index in populations.items():
                                    outfile.write(
                                        f"\t\\node[cell, label=right:{population}]"
                                        + "{"
                                        + f"${index+1}$"
                                        + "};\\\\\n"
                                    )
                                outfile.write("};\n\end{tikzpicture}\n}\n}\n")
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
