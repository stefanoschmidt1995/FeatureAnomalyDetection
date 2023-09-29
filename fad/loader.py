import numpy as np
import matplotlib.pyplot as plt
import warnings
from gwpy.timeseries import TimeSeries
from gwpy.segments import Segment
from gwpy.time import to_gps
from gwdatafind import find_urls
import os
import matplotlib.pyplot as plt
import sys

from ast import literal_eval as make_tuple

import shutil

from memspectrum import MESA
from tqdm import tqdm

from .utils import load_files

#################################################################

#TODO: everything MESA-related must be done with the MESA segment class
#		- I/O operations
#		- whitening

class MESA_segment(MESA):
	"""
	Representation of a MESA segment.
	A MESA segment is GPS segment paired with a MESA obj, representing the PSD estimated in the given segment
	"""
	def __init__(self, filename = None, GPS_start = None, GPS_stop = None, duration = None):
		self.segment = None
		if not filename:
			assert GPS_start and (GPS_stop or duration), "If filename is not given, GPS_start and one between GPS_end and duration must be provided"
			if not GPS_stop:
				GPS_stop = GPS_start + duration
			self.segment = Segment(GPS_start, GPS_stop)
		super().__init__(filename = filename)
	
	@property
	def duration(self):
		if self.segment is not None: return self.segment.end - self.segment.start

	@property
	def start(self):
		if self.segment is not None: return self.segment.start

	@property
	def stop(self):
		if self.segment is not None: return self.segment.end
	
	def get_interp_coeff(self, times):
		p_coeff = np.square((times-(self.start+self.stop)/2.)/self.duration)
		return np.exp(-0.5*p_coeff)+1e-10
	
	def save(self,filename):
		"""
		Save the MESA segment to file (if the spectral density analysis is already performed).
		The output file can be used to load the class with method load()
		File is a 1D array with the format: [GPS_start, GPS_stop, P, a_k].
		
		Parameters
		----------
		filename: str	  
			Name of the file to save the data at
		"""
		if self.P is None or self.a_k is None:
			raise RuntimeError("PSD analysis is not performed yet: unable to save the model. You should call solve() before saving to file") 
		if self.segment is None:
			raise RuntimeError("Segment not specified, unable to save the MESA segment")
		
		to_save = [self.segment.start, self.segment.end, self.P, *self.a_k]
		np.savetxt(filename, to_save, header = "MESA segment | GPS_start, GPS_stop, P, a_k \n(1,1,1,{})".format(len(self.a_k)))

	def load(self,filename):
		"""
		Load the class from a given file. The file shall be the same format produced by save().
		
		Parameters
		----------
		filename: str	  
			Name of the file to load the data from
		"""
		data = np.loadtxt(filename)
		
		if data.ndim != 1:
			raise ValueError("Wrong format for the file: unable to load the model")
		
			#reading the first line
		with open(filename) as f:
			_, first_line = f.readline(), f.readline()

		shapes = make_tuple(first_line.translate({ord('#'): None, ord(' '): None}))
			#checking for the header
		if not isinstance(shapes, tuple):
			if len(shapes) != 4 or not np.all([isinstance(s, int) for s in shapes]):
				raise ValueError("Wrong format for the header: unable to load the model")

			#assigning values
		start, stop, self.P, self.a_k = np.split(data, np.cumsum(shapes)[:-1])
		self.segment = Segment(start, stop)
	

class gaussian_noise_loader():
	def __init__(self, segment_length, N_segments, srate = 4096., seed = 0, t0 = 0.):
		self.rnd_gen = np.random.default_rng(seed = seed)
		self.segment_length = segment_length
		self.srate = srate
		self.N_segments = N_segments
		self.t0 = t0
		self.white_strain_segments = []
		
		dt = 1/self.srate
		N_step = int(self.segment_length*self.srate)+1
		
		for _ in range(self.N_segments):
			self.white_strain_segments.append(TimeSeries(self.rnd_gen.normal(0,1, size = (N_step,)), t0 = t0, sample_rate = self.srate))
			t0 += (N_step)/self.srate
	
	def sort_segments(self):
		return
	
	def __iter__(self):
		"Iterate over batches of gaussian white noise"
		for seg in self.white_strain_segments:
			yield seg

class url_info():
	def __init__(self, url, read_kwargs):
		self.url = url
		self.read_kwargs = read_kwargs

	@property
	def info_dict(self):
		return {'source': self.url, **self.read_kwargs}

class strain_loader():

	def __init__(self, segment_length, overlap = 0., srate = 4096., verbose = False):
		self.strain_urls = []
		self.srate = srate
		self.segment_length = segment_length
		self.overlap = overlap
		self.strain_segments = []
		self.white_strain_segments = []
		self.mesa_segments = []
		self.trim = 3000 
		self.verbose = verbose
		self.ifo = None
		return
	
	def _get_batch_ids(self, N):
		"Given a length of a data stretch, computes the ids for each of the psd batch"
		ids_ = []
		i = 0
		N_points = int((self.segment_length)*self.srate)+1
		N_step = int((self.segment_length-self.overlap)*self.srate)+1
		
		for i in range(0, N, N_step):
			start = max(i-self.trim, 0)
			if start == 0: stop = min(start+N_points+self.trim, N)
			else: stop = min(start+N_points+2*self.trim, N)
			ids_.append(slice(start, stop))

		return ids_		
	
	def _update_ifo_str(self, filename):
		ifo_attempt = os.path.basename(filename)[0]
		if ifo_attempt in ['H', 'V', 'L']:
			ifo = ifo_attempt+'1'
			if self.ifo: assert self.ifo == ifo, "The strain given comes from different ifos!"
			else: self.ifo = ifo
		return
	
	def save_zip(self, basefile, compress_format = 'tar', save_mesa = True):
		"""
		Save the batches to a zip file.
		"""
		extension = basefile.split('.')[-1]
		if extension in ['zip', 'tar', 'gztar', 'bztar', 'xztar']:
			basefile = basefile[:-(len(extension)+1)]
			compress_format = extension
		
		random_number = np.random.randint(np.iinfo(int).max)
		tmp_dir = '{}/{}'.format(os.path.dirname(basefile+'.'+compress_format), '.tmp_batches_{}'.format(random_number))
		if self.verbose: print("Saving temporary batches to: ", tmp_dir)

		self.save_batches(tmp_dir, save_mesa)
		
		shutil.make_archive(basefile, compress_format, root_dir = tmp_dir)
		
		shutil.rmtree(tmp_dir)

		#print('basefile ', basefile)
		#print('tmp dir ', tmp_dir)		

		return
	
	def load_zip(self, filename, compress_format = 'tar'):
		"""
		Loads the batches from compressed format.
		"""
		random_number = np.random.randint(np.iinfo(int).max)
		tmp_dir = '{}/{}'.format(os.path.dirname(filename), '.tmp_batches_{}'.format(random_number))
		os.makedirs(tmp_dir)
		if self.verbose: print("Temporarly unpacking batches to: ", tmp_dir)
		
		shutil.unpack_archive(filename, extract_dir = tmp_dir, format = compress_format)
		
		self.load_batches(tmp_dir)
		
		shutil.rmtree(tmp_dir)
		
		return
	
	def save_batches(self, savefolder, save_mesa = True):
		"""
		Save the batches of white noise to file in the given folder. Also stores the mesa objects, just in case.
		"""
		if len(self.white_strain_segments)==0:
			warnings.warn('Unable to save batches, no data were given!')
			return
		
		os.makedirs(savefolder, exist_ok = True)
		if not savefolder.endswith('/'): savefolder = savefolder+'/'

		if self.ifo: basefilename = savefolder+self.ifo+'_WHITE_STRAIN_BATCH_{}Hz-{}-{}.csv'
		else: basefilename = savefolder+'WHITE_STRAIN_BATCH_{}Hz-{}-{}.csv'
		
		if self.ifo: mesa_basefilename = savefolder+self.ifo+'_MESA_OBJ-{}-{}.dat'
		else: mesa_basefilename = savefolder+'MESA_OBJ-{}-{}.csv'
		
		for segment in self.white_strain_segments:
			start, stop = segment.times[[0, -1]].value
			segment.write(basefilename.format(int(self.srate), int(start), int(stop-start)))

		if save_mesa:
			for m_seg in self.mesa_segments:
				m_seg.save(mesa_basefilename.format(int(m_seg.start), int(m_seg.duration)))
		return
	
	def add_mesa_segment(self, filename):
		"Adds a MESA segment to the strain loader"
		if isinstance(filename, str): filename = [filename]
		assert isinstance(filename, (list, tuple, np.ndarray)), "Filename must be a string or a list/tuple"
		for f in filename:
			self.mesa_segments.append(MESA_segment(filename = f))
		return
	
	def load_batches(self, loadfolder):
		"""
		Loads the batches saved by save_batches and makes them available in the iterator.
		**It doesn't load (yet) the MESA objects**
		"""
		#TODO: implement loading of MESA, by matching segments name
		if not loadfolder.endswith('/'): loadfolder = loadfolder+'/'
		f_list = os.listdir(loadfolder)
		#f_list.sort()
		
		for f in tqdm(f_list, disable = not self.verbose, desc = 'Loading whitened batches from file'):
			if f.find('MESA')>-1:
				self.mesa_segments.append(MESA_segment(filename = f))
			else:
				try:
					new_batch = TimeSeries.read(loadfolder+f)
				except (IndexError, IOError):
					continue
				if self.srate is None: self.srate = new_batch.sample_rate.value
				if self.segment_length is None: self.segment_length = len(new_batch)/self.srate
				
				assert self.srate == new_batch.sample_rate.value, 'The given batch has sample rate {}, which is different from the sample rate {} Hz set at initialization.'.format(new_batch.sample_rate.value, self.srate)

				self.white_strain_segments.append(new_batch)

		self.sort_segments()
			#It is dangerous to infer the overlap here: it should be passed trhough the dag
		if self.overlap is None:
			overlap = self.white_strain_segments[0].span[1] - self.white_strain_segments[1].span[0]
			if overlap>0: self.overlap = overlap
	
	def sort_segments(self):
		"Sort all the segments according to their initial time"
		if len(self.white_strain_segments)==0: return
		ids_ = np.argsort([s.t0.value for s in self.white_strain_segments])
		ids_mesa = np.argsort([m.start for m in self.mesa_segments])
		
		sorted_white_strain_segments, sorted_mesa = [], []
		
		for id_ in ids_:
			sorted_white_strain_segments.append(self.white_strain_segments[id_])
		if len(ids_mesa)>0:
			for id_ in ids_mesa:
				sorted_mesa.append(self.mesa_segments[id_])
		del self.white_strain_segments
		del self.mesa_segments
		
		self.white_strain_segments = sorted_white_strain_segments
		self.mesa_segments = sorted_mesa
		
	
	def __iter__(self):
		"Iterate over batches of whitened noise"
		if len(self.white_strain_segments)==0:
			warnings.warn('Cannot iterate over the strain_loader: no white data were given!')
			return
		for seg in self.white_strain_segments:
			yield seg

	
	def fetch_strain(self, filename = None, GPS_start = None, GPS_stop = None, channel = None, frame = None, host = 'datafind.ligo.org:443', format = None):
		"""
		Loads some strain data, using gwdatafind pacakge and loads them into the strain variable. Can also load the data from a given filename (must be recognisable by gwpy.
		"""
		
		if not (self.segment_length or self.srate):
			raise ValueError("Segment lenght and sample rate must be set before fetching the strain")
		
		#channel = 'L1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01', frame = 'L1_HOFT_CLEAN_SUB60HZ_C01'
		if filename:
			if isinstance(filename, str): urls = [filename]
			else: urls = filename
			new_kwargs = {'format': format}
			if channel:	new_kwargs['channel'] = channel
			if GPS_start:	new_kwargs['start'] = GPS_start
			if GPS_stop:	new_kwargs['end'] = GPS_stop
			new_kwargs = [new_kwargs for _ in urls]
		elif (GPS_start is not None) and (GPS_stop is not None):
			urls = find_urls(frame[0], frame, to_gps(GPS_start), to_gps(GPS_stop), host= host)
			#self.strain = TimeSeries.read(urls, channel = channel, start = GPS_start, end = GPS_stop,  format = format)
			new_kwargs = []
			for url in urls:
				try:
					start = int(url.split('-')[-2])
					end = start + int(url.split('-')[-1].split('.')[0])
				except ValueError:
					raise ValueError("The filename does not provide information on the interval spanned by the frame! Unable to continue")
				if start < GPS_start: start = GPS_start
				if end > GPS_stop: end = GPS_stop
				new_kwargs.append({'channel': channel, 'format':format, 'start': start, 'end': end})
			#raise NotImplementedError
		else:
			raise ValueError("One argument between filename and GPS time must be provided")
		
		self.strain_urls.extend([url_info(u,k) for u, k in zip(urls, new_kwargs)])

		for ui in self.strain_urls:
			self._update_ifo_str(ui.url)
			strain_ts = TimeSeries.read(**ui.info_dict)
			if strain_ts.sample_rate.value != self.srate: strain_ts = strain_ts.resample(self.srate)
			self.strain_segments.append(strain_ts)

		return

	def q_transform(self, GPS_start, GPS_stop, **kwargs):
		"""
		Returns a gwpy spectrogram in the given time range.
		"""
		#Looking for the right segment
		id_white_batch = None
		for id_, w_b in enumerate(self.white_strain_segments):
			s = Segment((w_b.t0.value, w_b.times[-1].value))
			if GPS_start in s:
				id_white_batch = id_
				if GPS_stop not in s:
					msg = "The given segment for spectrogram [{},{}] is not in a single white strain batch.\nThe segment [{},{}] will be plotted instead".format(GPS_start, GPS_stop, GPS_start, s.end)
					warnings.warn(msg)
					GPS_stop = s.end
		if id_white_batch is None:
			raise ValueError("Unable to find the [{},{}]: is it loaded?".format(GPS_start, GPS_stop))
		
			#Plotting
		kwargs['whiten'] = False
		q_scan = self.white_strain_segments[id_white_batch].q_transform(**kwargs)
		q_scan = q_scan.crop(GPS_start, GPS_stop)
		return q_scan
	
	
	def plot_q_transform(self, GPS_start, GPS_stop, savefile = None, ax = None, **kwargs):
		q_scan = self.q_transform(GPS_start, GPS_stop, **kwargs)
		if ax is None: fig, ax = plt.subplots(dpi=120)
		ax.imshow(q_scan)
		ax.set_yscale('log', base=2)
		ax.set_xscale('linear')
		ax.set_ylabel('Frequency (Hz)', fontsize=14)
		ax.set_xlabel('Time (s)', labelpad=0.1,  fontsize=14)
		ax.tick_params(axis ='both', which ='major', labelsize = 14)
		cb = ax.colorbar(label='Normalized energy',clim=[0, 25.5])

		if savefile: plt.savefig(savefile)
		return

	def estimate_PSD(self, **kwargs):
		"""
		Estimate the PSD in batches on the loaded strain (in self.strain_segments)
		"""
		if len(self.strain_segments) ==0:
			raise ValueError("No strain segments found: have you loaded them with fetch strain?")

		for strain_ts in self.strain_segments:
			ids_batches = self._get_batch_ids(len(strain_ts))
			for id_ in tqdm(ids_batches, disable = not self.verbose, desc = 'Estimating PSD on batches'):
				m = MESA_segment(GPS_start = strain_ts.times[id_][0].value, GPS_stop = strain_ts.times[id_][-1].value)
				m.solve(strain_ts[id_], **kwargs)
				self.mesa_segments.append(m)
		return

	def whiten_batch(self, **kwargs):
		"""
		It whitens the data in batches by estimating the PSD in each batch.
		"""
		for strain_ts in self.strain_segments:
			ids_batches = self._get_batch_ids(len(strain_ts))
			for id_ in tqdm(ids_batches, disable = not self.verbose, desc = 'Estimating PSD on batches'):
				m = MESA_segment(GPS_start = strain_ts.times[id_][0].value, GPS_stop = strain_ts.times[id_][-1].value)
				m.solve(strain_ts[id_], **kwargs)
				white_batch = m.whiten(strain_ts[id_], trim = self.trim)
				#white_batch = white_batch / np.sqrt(np.var(white_batch))
				white_batch = TimeSeries(white_batch,
					times = strain_ts.times[id_.start+self.trim: id_.stop-self.trim])
				self.mesa_segments.append(m)
				self.white_strain_segments.append(white_batch)
		return

	def whiten_interpolation(self, **kwargs):
		"""
		Whiten interpolation :D
		"""
		fig, ax = plt.subplots(1,1)
		fig_hist, ax_hist = plt.subplots(1,1)
		import scipy.stats
		#raise NotImplementedError
		for strain_ts in self.strain_segments:
			norm_factor, white_batch = 0, 0
			times = strain_ts.times[self.trim: -self.trim].value
			for i, ms in enumerate(tqdm(self.mesa_segments, disable = not self.verbose, desc = 'Whitening over different PSDs')):
				w = ms.get_interp_coeff(times)
				white_batch += ms.whiten(strain_ts, trim = self.trim)*w
				norm_factor += w**2
				print(np.var(ms.whiten(strain_ts.value, trim = self.trim)))
				
				ax.plot(times, w)
				
				x = np.sort(np.array(ms.whiten(strain_ts, trim = self.trim)))
				ax_hist.hist(x, density = True, histtype = 'step', bins = int(np.sqrt(len(x))))
				if i ==0: ax_hist.plot(x, scipy.stats.norm(scale = 1).pdf(x))
				ax_hist.set_yscale('log')
			ax.plot(times, norm_factor, '--')
			plt.show()
	
			#FIXME: You should be very very careful on your choice of P! How do you set P? Probably you will need an average of some sort
	
			white_batch = white_batch/np.sqrt(norm_factor)
			
			#white_batch /= np.sqrt(np.var(white_batch[self.trim:-self.trim]))
			print(np.var(white_batch))
			
			white_batch = TimeSeries(white_batch, times = times)

			self.white_strain_segments.append(white_batch)
		return 





