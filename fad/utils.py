import numpy as np
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
from itertools import combinations, permutations, product

def load_files(files, keys = None):
	"""
	Loads several feature files ans loads them into a dictionary of features
	"""
	if isinstance(files, str): files = [files]
	
	t0, srate = None, None
	features = {}
	for feat_file in files:

		new_features = np.loadtxt(feat_file, delimiter = ',')
		new_times, new_features = new_features[:,0], new_features[:,1:]
		
		if not t0: t0 = new_times[0]
		if not srate: srate = 1./(new_times[1]-new_times[0])
		
		assert np.isclose(new_times[0], t0), "Different files do not start from the same times"
		assert np.isclose(srate, 1./(new_times[1]-new_times[0])), "Different files do not have the same sample rate"

		with open(feat_file, 'r') as f:
			header = f.readline()[1:-1]
		headers = header.split('|')[1:]

		for i, h in enumerate(headers):
			if keys:
				if h not in keys: continue
			features[h] = TimeSeries(new_features[:,i], times = new_times)
	
	#min_len = np.min([len(v) for v in features])
	#for k in range(len(features)):
	#	assert np.isclose(features[k].sample_rate.value, srate), "The features don't have the same sample rate!"
	#	features[k].times = features[k].times[:min_len]
	#	features[k] = features[k][:min_len]
	return features

def plot_features(feat_aggregator_list, make_hist = False, savefile = None, labels = None, figsize = None):

	if not isinstance(feat_aggregator_list, (list,tuple)):
		feat_aggregator_list = [feat_aggregator_list]

	fig, axes = plt.subplots(len(feat_aggregator_list[0].features), 1, sharex = not make_hist, figsize = figsize)

	for i, feat_aggregator in enumerate(feat_aggregator_list):
		for (k, v), ax in zip(feat_aggregator.features.items(), axes):
			if make_hist:
				ax.hist(v, bins = int(np.sqrt(len(v))), histtype = 'step', density = True, label = labels[i] if labels else None)
				ax.set_yscale('log')
			else:
				ax.plot(v.times[:len(v)], v, label = labels[i] if labels else None)
			if not make_hist: ax.set_ylabel(k)
			ax.set_title(k)
			if labels and i==len(feat_aggregator_list)-1: ax.legend()
	
	if not make_hist: ax.set_xlabel('GPS time')
	plt.tight_layout()
	if savefile: plt.savefig(savefile)
	#plt.show()

def plot_scattered_features(feat_aggregator_list, savefile = None, labels = None, ax = None):
	if not isinstance(feat_aggregator_list, (list, tuple)):
		feat_aggregator_list = [feat_aggregator_list]
	matrix_list = [fa.feature_matrix for fa in feat_aggregator_list]
	

	_, D = matrix_list[0].shape

	fsize = 4*D-1
	size_template = 2
	
	fs = 10
	fig, axes = plt.subplots(D-1, D-1, figsize = (fsize, fsize), layout='constrained')
		
	if D-1 == 1:
		axes = np.array([[axes]])
	for i,j in permutations(range(D-1), 2):
		if i<j:	axes[i,j].remove()

			#Plot the matrix
	for ax_ in combinations(range(D), 2):
		currentAxis = axes[ax_[1]-1, ax_[0]]
		ax_ = list(ax_)
		
		for ii, (matrix, c) in enumerate(zip(matrix_list, plt.rcParams['axes.prop_cycle'])):
			np.random.seed(0)
			if matrix.shape[0] >500000: ids_ = np.random.choice(matrix.shape[0], 500000, replace = False)
			else: ids_ = range(matrix.shape[0])
			
			currentAxis.scatter(matrix[ids_,ax_[0]], matrix[ids_,ax_[1]],
				s = size_template, marker = 'o', c= c['color'], alpha = 0.3,
				label = labels[ii] if labels else None)
	
		if ax_[0] == 0:
			currentAxis.set_ylabel(feat_aggregator_list[-1].feature_labels[ax_[1]], fontsize = fs)
		else:
			currentAxis.set_yticks([])
		if ax_[1] == D-1:
			currentAxis.set_xlabel(feat_aggregator_list[-1].feature_labels[ax_[0]], fontsize = fs)
		else:
			currentAxis.set_xticks([])
			
		currentAxis.tick_params(axis='x', labelsize=fs)
		currentAxis.tick_params(axis='y', labelsize=fs)
	if labels: currentAxis.legend()
				#Plotting the hist
	n_bins = 1000
	for i, ax in enumerate(axes):
		currentAxis = axes[i,i]
		ax_histy = currentAxis.inset_axes([1.05, 0, 0.25, 1], sharey = currentAxis)
		for ii, (matrix, c) in enumerate(zip(matrix_list, plt.rcParams['axes.prop_cycle'])):
			ax_histy.hist(matrix[:,i+1], orientation='horizontal', bins = n_bins, histtype = 'step', density = True, color = c['color'])
		ax_histy.tick_params(axis="y", labelleft=False)

		if i == 0:
			ax_histx = currentAxis.inset_axes([0, 1.05, 1, 0.25], sharex = currentAxis)
			for ii, (matrix, c) in enumerate(zip(matrix_list, plt.rcParams['axes.prop_cycle'])):
				ax_histx.hist(matrix[:,0], bins = n_bins, histtype = 'step', density = True, color = c['color'])
			ax_histx.tick_params(axis="x", labelbottom=False)
	
	if isinstance(savefile, str):
		plt.savefig(savefile)

	return
	
