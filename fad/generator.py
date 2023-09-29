import numpy as np
import warnings
from gwpy.timeseries import TimeSeries
from gwpy.signal import qtransform
import scipy.stats 
from tqdm import tqdm
import json

###################################################

class feature_generator():
	def __init__(self, srate, seg_length, stride=None, filename = None, feature_name = None, store_features = True, verbose = False):
		"""
		Initialize the class.
		seg_length and stride are the number of samples
		"""
		if not stride: stride = seg_length
		self.stride = stride
		self.seg_length = seg_length
		self.input_srate = srate #sample rate of the input timeseries
		self.verbose = verbose
	
		self.features = None
		if isinstance(feature_name, str): self.name = feature_name
		self.store_features = store_features
		
		self.old_val = np.nan
		
		if filename: self.load_features(filename)
	
	@property
	def output_srate(self):
		return self.input_srate/self.stride
	
	@classmethod
	def load(cls, filename):
			#Trying to use the header to extract the relevant parameters
		with open(filename, 'r') as f:
			header_str = f.readline()[1:-1]

		try:
			header_dict = json.JSONDecoder().decode(header_str)

			stride = header_dict['stride']
			seg_length = header_dict['seg_length']
			input_srate = header_dict['input_srate']
			name = header_dict['name']
			verbose = header_dict['verbose']
			store_features = header_dict['store_features']
		except (KeyError, json.decoder.JSONDecodeError):
			raise ValueError("Wrong header: unable to load the class from it!")
		
		return cls(input_srate, seg_length, stride, filename, name, store_features, verbose)
	
	def get_ids(self, N):
		"""
		Returns the ranges of indices of the input for each of the stretch of data over which the feature is computed.
		"""
		N = int(N)
		ids_ = [slice(i, i+self.seg_length) for i in range(0, N, self.stride)]
		#ids_ = [slice(i, i+self.seg_length) for i in range(0, N-self.seg_length, self.stride)]
		return ids_
	
	def save_features(self, filename):
		"""
		Saves the features in the format given by the extension of filename.
		See `gwpy.timeseries.TimeSeries.write <https://gwpy.github.io/docs/stable/api/gwpy.timeseries.TimeSeries/#gwpy.timeseries.TimeSeries.write>`_ for information about the available formats.
		
		Input
		-----
			filename: str
			Name of the file to save the features at
		"""
		header_dict = {
			'stride': self.stride,
			'seg_length': self.seg_length,
			'input_srate': self.input_srate,
			'name': self.name,
			'verbose': self.verbose,
			'store_features': self.store_features
		}
		header_str = json.JSONEncoder().encode(header_dict)
	
		if self.features is None:
			warnings.warn("Empty features, cannot save them!")
			return
		self.features.write(filename, header = header_str)
		
	
	def load_features(self, filename):
		"""
		Loads the features as a gwpy Timeseries and stores it
		
		Input
		-----
			filename: str
			Name of the file to save the features at
		"""
		self.features = TimeSeries.read(filename)
	
	def __get_feature(self, input_ts):
		#print(input_ts.t0, len(input_ts))
		if len(input_ts) < self.seg_length: return self.old_val
		try:
			val = self.get_feature(input_ts)
			assert not np.isnan(val)
		except (IndexError, ValueError, RuntimeError, AssertionError):
			return self.old_val
		self.old_val = val
		return val

	
	def __call__(self, white_strain = None):
		"""
		Given some whitened strain, it computes the features and adds them to the features stored internally.
		It calls :func:`feature_generator.compute_features`
		"""
		if white_strain is None: return self.features
		
		if not isinstance(white_strain, TimeSeries):
			white_strain = TimeSeries(white_strain, srate = self.input_srate, t0 = 0.)
		if white_strain.sample_rate.value != self.input_srate:
			white_strain = white_strain.resample(self.input_srate)

		ids = self.get_ids(len(white_strain))
		id_iter = tqdm(ids, desc = 'Computing feature "{}"'.format(self.name), disable = not self.verbose)
		
		new_features = [self.__get_feature(white_strain[ids_]) for ids_ in id_iter]
		
		#	It might be better to set t0 = white_strain.t0 + 1/(2*self.output_srate)
		#		But this will heavily mess up the feature aggregator, since the feature won't be evaluated on the same time grid anymore
		#		On the other hand, this is easy to interprete: the trigger will be always before the actual anomaly: easy to communicate and understand!
		new_features = TimeSeries(new_features,
			sample_rate = self.output_srate, t0 = white_strain.t0.value, name = self.name)
		
		if self.store_features:
			if self.features is None:
				self.features = new_features
			else:
				#TODO: replace this with concatenate_ts?
				#	it interpolates, which is probably a waste of computational power
				#	but it's more robust
				self.features.append(new_features, inplace=True, pad=np.nan, gap='pad', resize=True)
		return new_features
	
	def get_feature(self, white_strain):
		raise NotImplementedError

class variance_feature(feature_generator):
	name = 'variance'
	
	def get_feature(self, input_ts):
		if len(input_ts)<10: return np.nan
		return np.log10(np.var(input_ts))

class max_feature(feature_generator):
	name = 'max'
	
	def get_feature(self, input_ts):
		if len(input_ts)<10: return np.nan
		return np.log10(np.max(np.abs(input_ts)))

class KL_feature(feature_generator):
	name = 'KL'

	#@profile	
	def get_feature(self, input_ts):
		input_ts = np.asarray(input_ts)
		white_data = np.sort(input_ts)/np.std(input_ts)

		std = 1#np.std(white_data)
		sigma = 0.2

		kde = (white_data[:, None]-white_data)/std #(D-trim, D-trim)
		kde = np.sum(np.exp(-0.5*np.square(kde/sigma)), axis = -1)/(sigma*std*np.sqrt(2*np.pi)*len(white_data)) 
		KL_dist = (np.log(kde) - scipy.stats.norm(scale = std).logpdf(white_data)).mean()
		
		if False:
			import matplotlib.pyplot as plt
			#df, loc, scale = scipy.stats.t.fit(input_ts, loc = 1., scale = 1.)
			plt.plot(white_data, kde, label = 'True')
			plt.plot(white_data, scipy.stats.norm(scale = std).pdf(white_data), label = 'Gaussian')
			plt.hist(white_data, bins = 100, histtype = 'step', density = True, label = 'Hist')
			#plt.plot(white_data, scipy.stats.t(scale = scale, loc = loc, df = df).pdf(white_data), label = 't-student')
			plt.yscale('log')
			plt.legend()
			plt.show()
	
		return np.log10(KL_dist)

class FD_feature(feature_generator):
	name = 'FD'
	
	def get_feature(self, input_ts, dec = 32, step = 1):
		"""
		Computes the Fractal Dimension using the VAR method 
		and `pyramid summing` for single-precision (float32) data.
		For double-precision (float64) data use compute_fd_parallel/compute_fd_serial.
		"""
		input_ts = np.asarray(input_ts)
		N = len(input_ts)
		if N <10: return np.nan
		k_n = np.arange(1, N//(2*dec), step, np.int64)
		if len(k_n) == 0: k_n = [1]
		n_max = len(k_n)

		f = np.asarray(input_ts)

		V_i = np.empty(shape=(n_max), dtype=np.float32)
		
		ub = np.empty(shape=(N-2*k_n[0], 2), dtype=np.float32) # current iteration
		for i in range(0, N-2*k_n[0]):
			ub[i,0] = np.max(f[i:i+2*k_n[0]+1])
			ub[i,1] = np.min(f[i:i+2*k_n[0]+1])
		V_i[0] = np.mean(ub[:,0]-ub[:,1])

		for n in range(1,n_max):
			d = k_n[n] - k_n[n-1]
			for i in range(0, N-2*k_n[n]):
				ub[i,0] = max(ub[i,0], ub[i+2*d,0])
				ub[i,1] = min(ub[i,1], ub[i+2*d,1])
			V_i[n] = np.mean(ub[:N-2*k_n[n],0]-ub[:N-2*k_n[n],1])

		X = np.log(k_n)
		X_m = X - np.mean(X)
		Y = np.log(V_i)
		Y_m = Y - np.mean(Y)
		FD = 2 - np.sum((X_m)*(Y_m))/np.sum((X_m)**2)
		return FD

class far_qtranform_feature(feature_generator):
	name = 'far-q-transform'
	
	def get_feature(self, input_ts):
		qgram, far = qtransform.q_scan(input_ts)
		return np.log10(far+1e-30)
		#return far

class student_t_feature(feature_generator):
	name = 'KL-t-student'
	
	def get_feature(self, input_ts):
		white_data = (input_ts-np.mean(input_ts))/np.std(input_ts)
		df, loc, scale = scipy.stats.t.fit(white_data, floc = 0., fscale = 1.)
		return 1/df
	
class gen_hyperbolic_feature(feature_generator):
	name = 'gen-hyperbolic'
	
	def get_feature(self, input_ts):
		#FIXME: this shit is in general a good idea since it manages to describe well the data. But it is too slow to be feasible!
		
		p, a, b, loc, scale = scipy.stats.genhyperbolic.fit(input_ts, floc = 0., fscale = 1.)

		alpha, beta = a/scale, b/scale
		delta, mu = scale, loc
		
		xi = 1/np.sqrt(1+delta*np.sqrt(alpha**2-beta**2))
		chi = xi * beta/alpha

		#print('##')
		#print(p, a, b, loc, scale)
		#print(xi, chi)

		return xi










	
