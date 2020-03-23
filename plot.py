import numpy as np
import matplotlib.pyplot as plt

def plot_fig(data, plot_data, label_x, label_y, title, cat):

	for it in data:
		y = data[it]
		y_mean = y.mean(axis=0)
		y_std = y.std(axis=0)

		t_cons = 2.064# for dof=24 or number of samples=25
		y_lower = y_mean - t_cons*y_std/np.sqrt(len(y))
		y_upper = y_mean + t_cons*y_std/np.sqrt(len(y))

		plt.plot(range(len(y_mean)), y_mean, lw = 1, alpha = 1, label=plot_data["label"][it], color = plot_data["color"][it])
		plt.fill_between(range(len(y_mean)), y_lower, y_upper, alpha = 0.4, color = plot_data["color"][it])

	if cat == "incorrect":
		axes = plt.gca()
		axes.set_ylim([0.25,0.6])

	plt.title(title)
	plt.xlabel(label_x)
	plt.ylabel(label_y)

	plt.legend(loc = 'best')
	plt.show()