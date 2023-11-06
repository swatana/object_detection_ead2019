import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from size_histogram import generate_colors


def average_feature(feature, image_dir):
    classes = [
        "artifact",
        "blur",
        "bubbles",
        "contrast",
        "instrument",
        "saturation",
        "specularity",
    ]
    classes = sorted(classes)
    array = {}

    for a in classes:
        dir_path = os.path.join(image_dir, a)
        results = []
        for file_path in glob.glob(os.path.join(dir_path, "*.sig")):
            result = []
            with open(file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                if line.find(feature) != -1:
                    result.append(float(line.split()[0]))

            results.append(result)
        results = np.array(results)
        array[dir_path] = np.mean(results, 0)

    color = generate_colors(len(classes))
    fig = plt.figure(figsize=(15, 8))
    ax = []
    for a, i in zip(classes, range(len(classes))):
        ax.append(fig.add_subplot(2, 4, i + 1))

    for a, i in zip(classes, range(len(classes))):
        ax[i].bar(range(len(array[a])), array[a], color=color[i])
        ax[i].set_title(a)

    fig.supxlabel(feature.replace(" ()", ""), fontsize=20)
    fig.supylabel("Frequency", fontsize=20)
    plt.subplots_adjust(right=0.93)
    plt.subplots_adjust(left=0.07)
    fig.savefig(feature.replace(" ()", ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--feature",
        type=str,
        required=True,
        help="feature name. example:Gini Coefficient ()",
    )
    parser.add_argument(
        "-r",
        "--result_dir",
        type=str,
        required=True,
        help="path to result directory of wnd-chrm",
    )

    args = vars(parser.parse_args())
    average_feature(args["feature"], args["result_dir"])
