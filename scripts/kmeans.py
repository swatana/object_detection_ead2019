import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


data = {
    "artifact": 11.78,
    "blur": 17.14,
    "bubbles": 3.67,
    "contrast": 19.3,
    "instrument": 32.84,
    "saturation": 39.43,
    "specularity": 17.07
}
km = KMeans(n_clusters=2, random_state=42)
array= np.array(list(data.values())).reshape(-1,1)
label = km.fit_predict(array)
colors=["red","blue"]
color=[colors[a] for a in label]
fig, ax = plt.subplots(figsize = (12,10))
ax.bar(data.keys(), data.values(),color=color)
plt.ylabel('k-means (k=2)', fontsize=20)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, fontsize=15)
fig.savefig("figs/k-means")