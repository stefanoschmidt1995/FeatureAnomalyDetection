import numpy as np
import warnings
from gwpy.timeseries import TimeSeries
from gwpy.segments import Segment
from gwpy.time import to_gps
from gwdatafind import find_urls
import os
import matplotlib.pyplot as plt
import sys

import shutil

from memspectrum import MESA
from tqdm import tqdm

from .utils import load_files
from .generator import feature_generator
from .loader import strain_loader

##################################################

def concatenate_ts(a, b):
	assert np.isclose(a.dt, b.dt), "Series are not compatible!"
	
		#a is always the first ts
	if a.span[0]>b.span[0]: a,b = b,a
		#making a copy: this prevents an obscure error in gwpy (maybe it's a bad idea?)
	a = a.copy()
	a.dt = b.dt

		#Preprocess b (if it's the case)
	if b.span[0] - a.span[1] < 4*b.dt.value:
		times_b = np.arange(a.span[1], b.span[1], b.dt.value)
		b_interp = np.interp(times_b, b.times.value[:len(b)], b.value, left = None, right = None)
		b = TimeSeries(b_interp, t0 = float(times_b[0]), dt = b.dt.value)
		#concat_ts = a.append(b_interp, inplace=False, gap='raise', resize=True)
		if len(b)==0:
			warnings.warn("The timeseries given has zero length after interpolation...")
			return a
	
	concat_ts = a.append(b, inplace=True, pad=np.nan, gap='pad', resize=True)

	#plt.figure()
	#plt.plot(a.times, a, label = 'a')
	#plt.plot(b.times, b, label = 'b')
	#plt.plot(concat_ts.times, concat_ts+0.001, label = 'concat')
	#plt.legend()
	#plt.show()
	return concat_ts
	

##################################################

class feature_aggregator():
	def __init__(self, feature_list, strain_loader, srate, verbose = False):
		for i, f in enumerate(feature_list):
			assert isinstance(f, feature_generator)
			if f.name is None: f.name = 'feature_{}'.format(i)
		self.feature_generator_list = feature_list

		self.features = {f.name: None for f in self.feature_generator_list}
		self.srate = srate
		self.strain_loader = strain_loader
		self.verbose = verbose

	@property
	def times(self):
		self.set_equal_length()
		h = list(self.features.keys())[0]
		return self.features[h].times.value

	@property
	def t0(self):
		h = list(self.features.keys())[0]
		return self.features[h].t0.value

	@property
	def feature_matrix(self):
		return np.stack([v.value for _, v in self.features.items()], axis = 1)
	
	@property
	def feature_labels(self):
		return list(self.features.keys())
	
	@classmethod
	def from_files(cls, feature_files, keys = None, verbose = False):
		"""
		Given a list of files, it reads them and initializes the object with the features stored there.
		The feature list will be filled with abstract classes, hence an object instatiated in this way cannot be used to generate features with `compute_features`.
		The strain loader is also set to `None`.
		"""
		feat_dict = load_files(feature_files, keys)
		return cls.from_dict(feat_dict, verbose)
	
	@classmethod
	def from_dict(cls, feat_dict, verbose = False):
		"""
		Given a dictionary, it fills the feature aggregator with the features in the dictionary.
		The feature list will be filled with abstract classes, hence an object instatiated in this way cannot be used to generate features with `compute_features`.
		The strain loader is also set to `None`.
		"""
		srate = None
		length = 0
		
		for k in feat_dict.keys():
			assert isinstance(feat_dict[k], TimeSeries)
			if not srate:
				srate = feat_dict[k].sample_rate.value
			else:
				assert srate == feat_dict[k].sample_rate.value, "The features in the input dictionary must have the same sample rate"

		feat_gen_list = [feature_generator(None, None, feature_name = k) for k in feat_dict.keys()]

		new_obj = cls(feat_gen_list, strain_loader = None, srate = srate, verbose = verbose)
		new_obj.features = feat_dict
		new_obj.set_equal_length()

		return new_obj
	
		#Uncomment it and run with `kernprof -l -v script.py`
	#@profile
	def compute_features(self):
		"""
		Loops over the strain urls, loads the data and computes the features.
		"""
		assert self.strain_loader is not None, "The strain loader must be given to compute the features"
		self.strain_loader.sort_segments()
		N = len(self.strain_loader.white_strain_segments)
		for white_ts in tqdm(self.strain_loader, total = N, disable = not self.verbose, desc = 'Computing features over batches'):
			for feat_name, feat in zip(self.features.keys(), self.feature_generator_list):
				new_features = feat(white_ts)
				
				if self.strain_loader.overlap:
					n_cut = int(self.strain_loader.overlap*new_features.sample_rate.value)
					new_features = TimeSeries(np.array(new_features[:-n_cut]), t0 = new_features.t0, sample_rate = new_features.sample_rate)
				
					#If there are nans in new_features, resample will mess up and return all nans! 
				if new_features.sample_rate.value != self.srate:
					new_features = new_features.resample(self.srate)

					#This is if we want the trigger to be at the center of the time window (not necessarly a good idea)
				#new_features.t0 = white_ts.t0.value + 1/(2*self.srate)

				if self.features[feat_name] is None:
					self.features[feat_name] = new_features
				else:
					self.features[feat_name] = concatenate_ts(self.features[feat_name], new_features)
					del new_features
			#making sure that all the feature timeseries have the same length and time grid
			self.set_equal_length()
		
	def set_equal_length(self):
		"""
		Makes sure that all the features are of the same length
		"""
		min_len = np.min([len(v) for _, v in self.features.items()])
		for k, v in self.features.items():
			assert np.isclose(self.features[k].sample_rate.value, self.srate), "The features don't have the same sample rate!"
			t0 = self.features[k].times[0]
			self.features[k] = TimeSeries(self.features[k][:min_len], t0 = t0, sample_rate = self.srate)			

			#self.features[k].times = self.features[k].times[:min_len]
			#self.features[k] = self.features[k][:min_len]
		
		
	def save_features(self, filename):
		"""
		Save the features to file with columns `GPS | feature 1 | feature 2 | ...`
		"""
		self.set_equal_length()
		to_stack = []
		for k, v in self.features.items():
			if len(to_stack) ==0: to_stack.append(v.times.value[:len(v)])
			to_stack.append(v.value)
		to_save = np.stack(to_stack, axis = 1)
		
		header =  '|'.join(['times', *list(self.features.keys())])
		
		np.savetxt(filename, to_save, header = header, delimiter = ',')
	
	def load_features(self, filename):
		"""
		Loads the features from file.
		"""
		features = np.loadtxt(filename, delimiter = ',')
		times, features = features[:,0], features[:,1:]
		assert features.shape[1] == len(self.features), "The given input features are not of the same number of the features of the current object"
		assert np.all(np.isclose(np.diff(times),1/self.srate)), "The given input features do not have the same sample rate of the current object"

		with open(filename, 'r') as f:
			header = f.readline()[1:-1]
		headers = header.split('|')[1:]

		for i, h in enumerate(headers):
			if h in self.features.keys():
				self.features[h] = TimeSeries(features[:,i], t0 = times[0], sample_rate = self.srate)
			else:
				raise ValueError("The feature '{}' loaded from file is not a feature of this feature aggregator!".format(h))




















