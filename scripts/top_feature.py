import glob
import os
import numpy as np
import shutil

feature = "Fractal Features () "
wnd_path = "wnd-chrm-data"

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

for c in classes:
    results = []
    path = os.path.join(wnd_path, c)
    for file_path in glob.glob(os.path.join(path, "*.sig")):
        image_name = os.path.splitext(os.path.basename(file_path))[0]
        result = []
        with open(file_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.find(feature) != -1:
                result.append(float(line.split()[0]))

        results.append(result)
    results = np.array(results)
    array[c] = np.mean(results, 0)

for c in classes:
    path = os.path.join(wnd_path, c)
    i = str(np.argmax(array[c]))
    feature = feature + "[" + i + "]"
    results = []
    for file_path in glob.glob(os.path.join(path, "*.sig")):
        image_name = os.path.splitext(os.path.basename(file_path))[0]

        with open(file_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.find(feature) != -1:
                result = [file_path, float(line.split()[0])]

        results.append(result)
    results.sort(reverse=True, key=lambda x: x[1])
    for n in range(3):
        file_name = results[n][0][: results[n][0].index(".") - 2]
        shutil.copy(os.path.join(file_name + ".tiff"), os.path.join(wnd_path,"images/" + c + str(n) + ".tiff"))
    print(results)
