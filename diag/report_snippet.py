import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(2, 1))
fig.text(0.1, 0.5, "yield", fontfamily="Helvetica", fontsize=5)
fig.savefig("out.png", dpi=72)
