import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

plt.plot([1, 2, 3])
plt.savefig("remotely_fig.png")