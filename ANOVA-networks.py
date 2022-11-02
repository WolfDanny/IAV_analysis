import numpy as np
import pandas as pd

from flowanalysis.definitions import (
    coordinate_loop,
    edge_loop,
    node_loop,
    tex_requirements,
    tikz_legend,
)

infections = ["WT", "T8A", "N3A"]
timepoints = ["Primary", "Memory", "WT", "T8A", "N3A"]
organs = ["Spleen", "Lungs"]
cd45 = ["Circulating", "Resident"]
populations = {
    "WT": 0,
    "T8A": 1,
    "N3A": 2,
    "WT-T8A": 3,
    "WT-N3A": 4,
    "T8A-N3A": 5,
    "TP": 6,
    "TN": 7,
}
standalone = False
frequencies = True

for organ in organs:
    for residency in cd45:
        with open(f"ANOVA/{organ}-{residency[:3]}.tex", "w") as outfile:

            experiment_name = f"{organ} {residency.lower()}"

            tex_requirements(outfile, experiment_name)

            if standalone:
                outfile.write("\\begin{figure}\n")

            outfile.write(
                "\\centering\n\\begin{tabular}{ccc|c|cccc}\n\n\\multirow{3}{*}{\\rotatebox{90}{\Large Primary infection}} &\n"
            )

            for primary_index, primary in enumerate(infections):
                if primary_index != 0:
                    outfile.write("\n & ")
                outfile.write("\\rotatebox{90}{\\phantom{nnn}" + primary + "} &\n")
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
                                    ".", "-"
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
