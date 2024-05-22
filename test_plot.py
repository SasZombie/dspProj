from matplotlib import pyplot as plt

data = [1, 2, 3, 4, 5]
data2 = [1, 2, 3, 4, 5]


fig, axes = plt.subplots(2, 1)

axes[0].plot(data)
axes[1].plot(data2)

plt.show()