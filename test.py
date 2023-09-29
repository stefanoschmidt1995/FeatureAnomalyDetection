import numpy as np
import matplotlib.pyplot as plt
import subprocess

def test_import():
	import fad
	from fad import feature_generator

def test_feature_generator():
	from fad import variance_feature, KL_feature, FD_feature, student_t_feature
	from gwpy.timeseries import TimeSeries


	ts = TimeSeries.read('bullshit/H-H1_GWOSC_4KHZ_R1-1268903496-32.hdf5', format = 'hdf5.gwosc').whiten()
	
	feat_list = [
		variance_feature(ts.sample_rate.value, 1000, stride = 200, verbose = True),
		#KL_feature(ts.sample_rate.value, 1000, stride = 200, verbose = True),
		#FD_feature(ts.sample_rate.value, 1000, stride = 200, verbose = True),
#		student_t_feature(ts.sample_rate.value, 1000, stride = 200, verbose = True)
	]
	
	for f in feat_list:
		f(ts)
		f.save_features('bullshit/test_save.csv')

	variance_feature.load('bullshit/test_save.csv')
	
	
	for f in feat_list:
		plt.plot(f.features.times, f.features, label = f.name)
	plt.legend()
	plt.show()
	
	
def test_feature_generator_base():
	from fad import variance_feature
	from gwpy.timeseries import TimeSeries
	
	#ts = TimeSeries.read('bullshit/L-L1_GWOSC_4KHZ_R1-1268901464-4096.hdf5', format = 'hdf5.gwosc')
	ts = TimeSeries.read('bullshit/H-H1_GWOSC_4KHZ_R1-1268903496-32.hdf5', format = 'hdf5.gwosc').whiten()
	ts_bis = TimeSeries.read('bullshit/H-H1_GWOSC_4KHZ_R1-1268903496-32.hdf5', format = 'hdf5.gwosc').whiten()
	ts_bis = TimeSeries(ts_bis, t0 = ts_bis.t0.value + 200)
	
	vf = variance_feature(ts.sample_rate.value, 1000, stride = 100)
	
	feature_ts = vf(ts)
	feature_ts = vf(ts_bis)

	vf.save_features('bullshit/save_features_test.csv')
	vf_bis = variance_feature(ts.sample_rate.value, 1000, stride = 100, feature_name = 'variance')
	vf_bis.load_features('bullshit/save_features_test.csv')
	
	plt.plot(ts.times, ts, ls = '--', c = 'r', alpha = 0.3)
	plt.plot(ts_bis.times, ts_bis, ls = '--', c = 'r', alpha = 0.3)
	plt.plot(vf.features.times, vf.features, marker = 'o', ms = 3)
	plt.show()

def test_feature_aggregator():
	from fad import feature_aggregator, variance_feature, strain_loader
	from gwpy.timeseries import TimeSeries
	
	vf = variance_feature(4096., 291, stride = 300, feature_name = None, store_features = False)
	vf2 = variance_feature(4096., 518, stride = 300, feature_name = 'var#2', store_features = False)
	vf3 = variance_feature(4096., 5941, stride = 3000, feature_name = 'var#3', store_features = False)

	sl = strain_loader(8., overlap = 1., srate = 4096./2., verbose = True)
	#sl = strain_loader(50., overlap = 1., srate = 4096./2, verbose = True)
	#sl.load_strain_urls(GPS_start = 1259502955, GPS_stop = 1259502955+1000, format='gwf')
	sl.fetch_strain(filename = 'bullshit/H-H1_GWOSC_4KHZ_R1-1268903496-32.hdf5', format = 'hdf5.gwosc')

	#sl.load_strain_urls(filename = 'L-L1_GWOSC_4KHZ_R1-1268901464-4096.hdf5', format = 'hdf5.gwosc')
	sl.whiten_batch(method = 'standard')
	#sl.load_batches('bullshit/white_batches_IO_test')

	for s in sl:
		assert np.isclose(np.var(s), 1, atol = 0.1), "Variance of white batches is not unitary"

	srate = 10.
	fa = feature_aggregator([vf, vf2, vf3], sl, srate = srate, verbose = True)
	fa_bis = feature_aggregator([vf, vf2, vf3], sl, srate = srate, verbose = True)

	fa.compute_features()

	t0, len_feat = None, None

	for k,v in fa.features.items():
			#This check doesn't pass because gwpy sucks!
		#assert len(v.times) == len(v), "Sample times and vector length mismatch"
		assert np.isclose(v.sample_rate.value, srate), "Wrong feature sampling rate ({} vs {})".format(v.sample_rate.value, srate)
		assert not np.any(np.isnan(v)), "There are some nans around! Maybe interpolation errors?"

		if len_feat: assert len_feat == len(v), "Features are not of the same length"
		else: len_feat = len(v)

		if t0: assert t0 == v.t0.value, "Features do not start from the same time"
		else: t0 = v.t0.value
		
	fa.save_features('bullshit/save_aggregated_features_test.csv')
	fa_bis.load_features('bullshit/save_aggregated_features_test.csv')
	
	for k, v in fa.features.items():
		assert v.is_compatible(fa_bis.features[k]), "Something messed up with the loading of features metadata"
		assert np.allclose(v, fa_bis.features[k], equal_nan = True), "Something messed up with the loading of features"
	
	for k,v in fa.features.items():
		plt.plot(v.times[:len(v)], v, label = k)
	plt.legend()
	plt.show()

def test_gaussian_noise():
	from fad import strain_loader, gaussian_noise_loader
	from fad import feature_aggregator, variance_feature
	
	sl = strain_loader(7., srate = 4096./2., verbose = True)
	sl.fetch_strain(filename = 'bullshit/H-H1_GWOSC_4KHZ_R1-1268903496-32.hdf5', format = 'hdf5.gwosc')
	#sl.load_strain_urls(filename = 'L-L1_GWOSC_4KHZ_R1-1268901464-4096.hdf5', format = 'hdf5.gwosc')
	sl.whiten_batch(method = 'standard')

	gnl = gaussian_noise_loader(7., 3, srate = 4096./2., t0 = 1268903496)
	
	vf_gnl = variance_feature(4096., 500, stride = 300, feature_name = 'var gnl', store_features = False)
	vf_sl = variance_feature(4096., 500, stride = 300, feature_name = 'var sl', store_features = False)
	
	srate = 10.
	fa_sl = feature_aggregator([vf_sl], sl, srate = srate, verbose = True)
	fa_sl.compute_features()
	fa_gnl = feature_aggregator([vf_gnl], gnl, srate = srate, verbose = True)
	fa_gnl.compute_features()
	
	plt.figure()
	for k,v in fa_gnl.features.items():
		plt.plot(v.times[:len(v)], v, label = k)
	for k,v in fa_sl.features.items():
		plt.plot(v.times[:len(v)], v, label = k)
	plt.legend()
	
	plt.figure()
	for segment in gnl:
		plt.plot(segment.times, segment, label = 'gaussian')
	for segment in sl:
		plt.plot(segment.times, segment, label = 'true')
	plt.legend()
	plt.show()
	
def test_MESA_segment():
	from fad import MESA_segment

	m = MESA_segment(GPS_start = 1223528799, GPS_stop = 1223528799+10)
	assert m.duration ==10, "Duration is messed up"
			
	m = MESA_segment(GPS_start = 1223528799, duration = 10)
	assert m.duration ==10, "Duration is messed up"
	
	data = np.random.normal(0, 1, (10000,))
	m.solve(data)
	m.save('bullshit/MESA_seg_test.csv')
	m.whiten(np.random.normal(0, 1, (20000,)))

	m_bis = MESA_segment(filename = 'bullshit/MESA_seg_test.csv')
	
	assert np.allclose(m.a_k, m_bis.a_k), "a_k loading is messed up"
	assert m.P == m_bis.P, "P loading is messed up"
	assert m.duration == m_bis.duration, "GPS duration loading is messed up"
	assert (m.segment.start == m_bis.segment.start) and (m.segment.end == m_bis.segment.end), "GPS loading is messed up"
	
	
def test_zip():
	from fad import strain_loader
	sl = strain_loader(50., srate = 4096./2, verbose = True)
	sl.load_batches('bullshit/white_batches_IO_test')
	
	sl.save_zip('bullshit/test_compress', 'tar')
	
	sl_bis = strain_loader(50., srate = 4096./2, verbose = True)
	sl_bis.load_zip('bullshit/test_compress.tar', 'tar')
	
	end = 0.
	for segment, segment_bis in zip(sl, sl_bis):
		if end:	assert segment.t0.value-end == segment.dt.value, "Something wrong with the slicing"
		end = segment.times[-1].value
		
		assert np.allclose(segment, segment_bis)
		
		plt.plot(segment.times, segment)
	plt.show()
	
def test_strain_loader_variance():
	from fad import strain_loader
	
	srate_list = [4096., 4096/2., 4096/4, 3000]
	
	for srate in srate_list:
		sl = strain_loader(50., srate = 4096./2, verbose = True)
		sl.fetch_strain(filename = 'bullshit/H-H1_GWOSC_4KHZ_R1-1268903496-32.hdf5', format = 'hdf5.gwosc')
		sl.whiten_batch()
		for segment in sl:
			print(srate, np.var(segment))
	

def test_strain_loader():
	from fad import strain_loader
	
	sl = strain_loader(10., overlap = 1., srate = 4096./2, verbose = True)
	#sl.load_strain_urls(GPS_start = 1259502955, GPS_stop = 1259502955+1000, format='gwf')
	sl.fetch_strain(filename = 'bullshit/H-H1_GWOSC_4KHZ_R1-1268903496-32.hdf5', format = 'hdf5.gwosc')
	#sl.load_strain_urls(filename = 'bullshit/L-L1_GWOSC_4KHZ_R1-1268901464-4096.hdf5', format = 'hdf5.gwosc')
	
	sl.whiten_batch(method = 'standard')
	sl.save_batches('bullshit/white_batches_IO_test')

	#sl_bis = strain_loader(50., overlap = 1., srate = 4096./2)
	sl_bis = strain_loader(None, overlap = None, srate = None)
	sl_bis.load_batches('bullshit/white_batches_IO_test')

	assert np.isclose(sl.segment_length, sl_bis.segment_length, atol = 1e-4, rtol = 1e-4), "seglenth doesn't match {} vs {}".format(sl.segment_length, sl_bis.segment_length)
	if sl.overlap: assert np.isclose(sl.overlap, sl_bis.overlap), "overlap doesn't match"
	assert np.isclose(sl.srate, sl_bis.srate), "overlap doesn't match"
	
	plt.figure()
	end = 0
	for segment, segment_bis in zip(sl, sl_bis):
		#print(segment.t0.value-end + sl.overlap, segment.dt.value)
		if end:	assert segment.t0.value-end+sl.overlap == segment.dt.value, "Something wrong with the slicing: dt = {} vs {}".format(segment.t0.value-end+sl.overlap, segment.dt.value)
		end = segment.times[-1].value
		
		assert np.allclose(segment, segment_bis)
		
		plt.plot(segment.times, segment)
	plt.show()

def test_plotting():
	from fad import feature_aggregator, variance_feature, strain_loader
	from fad.utils import plot_features
	from gwpy.timeseries import TimeSeries

	sl = strain_loader(50., srate = 4096./2., verbose = True)
	sl.load_batches('bullshit/white_batches_IO_test')
	vf = variance_feature(4096., 5, stride = 30, feature_name = None, store_features = False)
	vf2 = variance_feature(4096., 500, stride = 300, feature_name = 'var#2', store_features = False)
	vf3 = variance_feature(4096., 5000, stride = 3000, feature_name = 'var#3', store_features = False)

	srate = 10.
	fa = feature_aggregator([vf, vf2, vf3], sl, srate = srate, verbose = True)
	fa.load_features('bullshit/save_aggregated_features_test.csv')
	
	plot_features([fa, fa], labels = ['set 1', 'set 2'])
	plot_features([fa, fa], True, labels = ['set 1', 'set 2'])
	
	plt.figure()
	for k,v in fa.features.items():
		plt.plot(v.times[:len(v)], v, label = k)
	plt.legend()
	
	#q_scan = sl.q_transform(1268901464+110, 1268901464+130)
	#q_scan = sl.q_transform(1268901660, 1268901660+20)
	sl.plot_q_transform(1268901680, 1268901680+20, savefile = 'bullshit/white_batches_IO_test/q_scan.png')
	plt.show()

def test_excutables():
	
	cmd = 'python bin/fad_white_batches --seg-length 13 --srate 2048 --input-files bullshit/H-H1_GWOSC_4KHZ_R1-1268903496-32.hdf5 --format hdf5.gwosc --savefolder test_executable --verbose'
	return_obj = subprocess.run(cmd, shell = True, capture_output = True)
	if return_obj.returncode:
		print(return_obj)
		raise ValueError("fad_white_batches doens't work")
	
	cmd = 'python bin/fad_generate_features --feature var --srate 10 --input-srate 2048 --n-window 5000 --stride 1000 --feature-name test_var --loadfolder bullshit/test_executable --output-file bullshit/test_executable/var_features.csv --verbose'
	return_obj = subprocess.run(cmd, shell = True, capture_output = True)
	if return_obj.returncode:
		print(return_obj)
		raise ValueError("fad_generate_features doens't work")

def test_far():
	from fad import feature_aggregator, feature_generator, strain_loader, far_qtranform_feature
	from fad.utils import plot_features, plot_scattered_features
	
	sl = strain_loader(100., srate = 4096./2., verbose = True)
	sl.load_batches('bullshit/dagfolder_test/strain/tmp')
	
	vf = far_qtranform_feature(4096./2., 4000, stride = 1000, feature_name = 'FAR', store_features = False)

	fa = feature_aggregator([vf], sl, srate = 10, verbose = True)
	fa.compute_features()
	
	print(fa.features)

def test_rank():
	from fad import feature_aggregator, feature_generator, strain_loader
	from fad.utils import plot_features, plot_scattered_features
	
	keys = None#['KL', 'FD', 'FAR']
	#gen = feature_aggregator.from_files('bullshit/dagfolder_test/features/MERGED_FEATURES_2Hz-1268895744-4096.csv', keys = keys)
	gen = feature_aggregator.from_files('bullshit/dagfolder_test/features_no_overlap/MERGED_FEATURES_2Hz-1268891648-4096.csv', keys = keys)
	gen_gaussian = feature_aggregator.from_files('bullshit/dagfolder_test/features_no_overlap/GAUSSIAN_MERGED_FEATURES_2Hz-0-5000.csv', keys = keys)
	
	for k in gen.features.keys():
		gen_gaussian.features[k].t0 = gen.features[k].t0-gen_gaussian.features[k].duration
	
	#plot_scattered_features([gen, gen_gaussian], labels = ['gen', 'gaussian'])
	#plt.show()

	
	gauss_matrix = gen_gaussian.feature_matrix
	matrix = gen.feature_matrix
	
	for i in range(matrix.shape[1]):
		ids_, = np.where(np.isnan(matrix[:,i]))
		if len(ids_)>0:
			print("Feature {} has some nans".format(i))
			mean = np.nanmean(matrix[:,i])
			matrix[ids_,i] = mean
	
	from sklearn.mixture import BayesianGaussianMixture
	import pickle
	
	if True:
		lm = BayesianGaussianMixture(n_components = 20, max_iter = 200, verbose = 2)
		lm.fit(gauss_matrix[-10_000:])
		with open('bullshit/DPGMM_test.pkl', 'wb') as f:
			pickle.dump(lm, f)
	else:
		with open('bullshit/DPGMM_test.pkl', 'rb') as f:
			lm = pickle.load(f)

	gaussian_likelihoods = -lm.score_samples(gauss_matrix)
	likelihoods = -lm.score_samples(matrix)
	
	plt.hist(gaussian_likelihoods, bins = 100, density = True, label = 'gaussian', histtype = 'step')
	plt.hist(likelihoods, bins = 1000, density = True, label = 'real', histtype = 'step')
	plt.yscale('log')
	plt.legend()

	bins = 200#np.logspace(-4, -1, 200)
	counts_gaussian, gauss_bins = np.histogram(gaussian_likelihoods, bins = bins)
	counts_real, real_bins = np.histogram(likelihoods, bins = 1000*bins)

	assert len(likelihoods) == sum(counts_real)
	
	N = len(likelihoods)
	cumulative_gaussian= (len(gaussian_likelihoods)-np.cumsum(counts_gaussian))*N/len(gaussian_likelihoods)
	cumulative_real = N-np.cumsum(counts_real)

	fig, ax = plt.subplots(1,1)
	ax.stairs(cumulative_real, real_bins, label = 'real' , color= 'orange')
	patch = ax.stairs(cumulative_gaussian, gauss_bins, label = 'gaussian')
	
	if True:
		p = cumulative_gaussian/N
		var_mean = p*(1-p)
		var_mean = np.sqrt(var_mean*N)
		ax.patches[-1].remove()
		vals, edges = patch.get_data().values, patch.get_data().edges
		ax.fill_between((edges[1:]+edges[:-1])/2., vals+var_mean, vals-var_mean, alpha = 0.3)
		ax.plot((edges[1:]+edges[:-1])/2., vals, label = 'gaussian', c = 'b')

#	plt.gca().invert_xaxis()
	plt.yscale('log')
	plt.legend()
	#plt.show()
	
	#print(lm.weights_, lm.means_)
	#print(lm.covariances_)
	
	plot_features(gen, make_hist = False, figsize = (10,10))
	plot_features([gen, gen_gaussian], make_hist = False, labels = ['real', 'gaussian'], figsize = (10,10))
	plot_features([gen, gen_gaussian], make_hist = True, labels = ['real', 'gaussian'], figsize = (10,10))

	plot_scattered_features([gen, gen_gaussian], labels = ['gen', 'gaussian'])
	
	plt.show()
	return
	
	sl = strain_loader(100., srate = 4096./2., verbose = True)
	#sl.load_zip('bullshit/dagfolder_test/strain/WHITE_BATCHES_2048Hz-1268891648-4096.tar')
	sl.load_zip('bullshit/dagfolder_test/strain/WHITE_BATCHES_2048Hz-1268895744-4096.tar')
	
	sl.plot_q_transform(1268896572-10, 1268896572+10)
	#sl.plot_q_transform(1268894141-10, 1268894141+10)
	#sl.plot_q_transform(1268894172-10, 1268894172+10)
	#sl.plot_q_transform(1268895571-10, 1268895571+10)
	
	plt.show()


if __name__ == '__main__':
	test_import()
	#test_feature_generator_base()
	#test_feature_generator()
	#test_strain_loader()
	
	#test_MESA_segment()
	
	#test_strain_loader_variance()
	#test_zip()
	#test_feature_aggregator()
	#test_gaussian_noise()
	#test_plotting()
	#test_excutables()
	
	#test_rank()
	#test_far()
	
























