import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import colorsys


def generate_colors(class_num):
    hsv_tuples = [(x / class_num, 1.0, 1.0) for x in range(class_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (float(x[0] * 1), float(x[1] * 1), float(x[2] * 1)), colors)
    )
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors


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
color = generate_colors(len(classes))
with open("feature.txt", "r") as f:
    features = f.readlines()
for i in range(len(features)):
    features[i] = features[i].replace("\n", "")
for feature in features:
    array = {}
    for dir_path in classes:
        results = []
        for file_path in glob.glob(os.path.join("./" + dir_path, "*.sig")):
            image_name = os.path.splitext(os.path.basename(file_path))[0]
            with open(file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                if line.find(feature) != -1:
                    results.append(float(line.split()[0]))

        array[dir_path] = results
    fig = plt.figure(figsize=(15, 8))
    ax = []
    for a, i in zip(classes, range(len(classes))):
        ax.append(fig.add_subplot(2, 4, i + 1))
    for a, i in zip(classes, range(len(classes))):
        ax[i].hist(array[a], color=color[i])
        ax[i].set_title(a)

    fig.supxlabel(feature.replace(" ()", ""), fontsize=20)
    fig.supylabel("Frequency", fontsize=20)
    plt.subplots_adjust(right=0.93)
    plt.subplots_adjust(left=0.07)
    save_dir = "figs/" + feature[0 : feature.index("()")-1].replace(" ","")+"/"
    os.makedirs(save_dir,exist_ok=True)
    fig.savefig(save_dir + feature.replace(" ()", ""))
    plt.close()
