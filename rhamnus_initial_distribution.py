import data
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

biomasses, *_ = data.get_data()
data = []
for label in biomasses.keys():
    if label[-1] == "R":
        biom = biomasses[label][0]
        data.append(biom)

x = np.array(data)
x = x[~np.isnan(x)]

print(np.mean(x))
plt.hist(x)
plt.show()