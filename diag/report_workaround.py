import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["text.hinting"] = sys.argv[1]
fig = plt.figure(figsize=(2, 1))
fig.text(0.1, 0.5, "yield", fontfamily="Helvetica", fontsize=5)
fig.savefig("out.png", dpi=72)
print("saved with text.hinting =", sys.argv[1])
