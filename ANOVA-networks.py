import numpy as np
import pandas as pd

infections = ["WT", "T8A", "N3A"]
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

    return (
        "\\foreach \\i in {0,...,"
        + f"{nodes}"
        + "}{\n\t\\coordinate (c\\i) at (\\i*360/"
        + f"{nodes}"
        + ":\\r);\n}\n\n"
    )


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
        return (
            "\\foreach \\i in {0,...,"
            + f"{nodes}"
            + "}{\n\t\\node (p\\i) at (c\\i)[cell]{};\n}\n\n"
        )
    else:
        node_names = []
        for i in range(nodes):
            node_names.append(f"{i}/{i+1}")
        node_names = ", ".join(node_names)

        return (
            "\\foreach \\i/\\j in {"
            + f"{node_names}"
            + "}{\n\t\\node (p\\i) at (c\\i)[cell]{$\\j$};\n}\n\n"
        )


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

    if dashed:
        return (
            "\\foreach \\i/\\j in {"
            + ", ".join(edges)
            + "}{\n\t\\draw[preaction={draw, line width=3pt, white}, thick, dashed] (c\\i) -- (c\\j);\n}\n"
        )
    else:
        return (
            "\\foreach \\i/\\j in {"
            + ", ".join(edges)
            + "}{\n\t\\draw[preaction={draw, line width=3pt, white}, thick] (c\\i) -- (c\\j);\n}\n"
        )


with open("ANOVA/anova-tikz.tex", "w") as outfile:
    outfile.write(
        "Make sure to include these lines in the preamble of your TeX document\n\n"
    )
    outfile.write("\\usepackage{tikz}\n")
    outfile.write("\\pgfmathsetmacro\\r{1.5}\n")
    outfile.write("\n\n")

    for _, priming in enumerate(infections):
        for _, challenge in enumerate(infections):
            filename = f"ANOVA/Tukey-{priming}-{challenge}.csv"

            try:
                with open(filename) as file:
                    current_datafame = pd.read_csv(file)

                # Reading and standardising names
                current_names = [
                    name.split("-") for name in current_datafame.iloc[:, 0]
                ]
                for row, current_pair in enumerate(current_names):
                    for col, current_population in enumerate(current_pair):
                        current_names[row][col] = current_population.replace(".", "+")

                # Reading data
                current_data = current_datafame.iloc[:, 1:].to_numpy()

                # Identifying significant differences
                significant_differences = []
                triple_negative_differences = []
                for index, values in enumerate(current_data):
                    if values[-1] < 0.05:
                        current_nodes = [
                            populations[current_names[index][i]] for i in range(2)
                        ]
                        if 7 in current_nodes:
                            triple_negative_differences.append(
                                f"{current_nodes[0]}/{current_nodes[1]}"
                            )
                        else:
                            significant_differences.append(
                                f"{current_nodes[0]}/{current_nodes[1]}"
                            )

                outfile.write("\n\n========================================\n")
                outfile.write(f"{priming : ^20}{challenge : ^20}\n")
                outfile.write("========================================\n\n")
                outfile.write("\\begin{tikzpicture}\n\n")
                outfile.write(coordinate_loop(len(populations)))
                outfile.write(edge_loop(triple_negative_differences, dashed=True))
                if significant_differences:
                    outfile.write(edge_loop(significant_differences))
                outfile.write(node_loop(len(populations), numbered=True))
                outfile.write("\\end{tikzpicture}\n\n")

                print(f"File for {priming}-{challenge} found")
            except FileNotFoundError:
                print(f"File for {priming}-{challenge} NOT found")
                pass
