# plot.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import matplotlib.pyplot as plt

def plot_edits(history):
	"""Plot the location of edits across the procedure.
	
	This function takes in the history dictionary produced by the Ledidi object
	and plot the position of the edits as they were proposed during the editing
	procedure. This plot allows you to see when certain edits were proposed.


	Parameters
	----------
	history: dict
		The history object produced by Ledidi. To get it, pass in
		`return_history=True` to the initialization of Ledidi.
	"""
	
	ax = plt.gca()
	b = history['batch_size']
		
	for i, edits in enumerate(history['edits']):
		y, _, x = edits
		x = x.numpy(force=True)
		y = y.numpy(force=True)
		plt.scatter(x, y+i*b, s=0.05, c='0.6')

	ylim = len(history['edits']*b)
	yticks = numpy.array(ax.get_yticks())
	
	plt.xlim(0, X.shape[-1])
	plt.yticks(yticks, yticks // b)
	plt.ylim(ylim, 0)
	plt.xlabel("Position", fontsize=12)
	plt.ylabel("Iteration", fontsize=12)
