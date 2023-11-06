from scipy import stats
data = {
    "artifact": 11.78,
    "blur": 17.14,
    "bubbles": 3.67,
    "contrast": 19.3,
    "instrument": 32.84,
    "saturation": 39.43,
    "specularity": 17.07
}
A = [data["instrument"],data["saturation"]]
B = [data["artifact"],data["blur"],data["bubbles"],data["contrast"],data["specularity"]]
print(stats.mannwhitneyu(A, B, alternative='greater'))