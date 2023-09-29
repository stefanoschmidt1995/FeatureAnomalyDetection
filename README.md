Feature Anomaly Detection for GW data
=====================================

TODO
----
- How to make sure that you compute features always on the same amount of datapoints? This affects a lot the performances...
- Test other types of features
- Add DQ features (like Andrew says) & test
- Maybe autoencoder is not a bad idea after all?
- Add some plots in ranks
- Add extra layer to merge all the times and compare bg & fg
- BIG BIG PROBLEM: how to test this?? I.e. what should we do with all of this??
- Think of injection mechanism... (far away)

RANDOM OBSERVATIONS
-------------------

- Q scan has its energy clipped, remember this when you do things
- Can maybe do **PCA** over the image and try to reconstruct the image? If the model is trained with gaussian noise, it should be reasonably easy to do this and would pick nicely every yellow point in the spectrogram
- We need something for the phase, checking phase consistency between different parts: maybe a NN designed to look for CBCs?
- You REALLY should look into sigma cut a bit more


PROBLEMS
--------

- Whitening is done in batch:
	* You have edge effects for the feature computation: you want long batches
	* You want to track PSD changes: you want short batches
  We can try to fix this by doing a smooth change of the AR coefficients computed in different batches. Then the whitening convolution will have a time dependent convulution kernel, which may fix the edge effect problem and effectively track the PSD changes... -> GOOD IDEA FOR A PROJECT!!!

- Feature distribution in real and gaussian noise is very very different: how do you detect anomalies? You cannot compare them to the gaussian behavior...
	* You can still use the gaussian likelihood to rank the anomalies from the most to the least likely to happen in gaussian noise...
	* BUT far doesn't have much meaning

- Deciding which feature to implement is non-trivial
	* We want to look at the spectrograms
	* Delta-sigma cut: what is exactly sigma?
	* We have plenty of methods that look at amplitude excess, but none that looks at phase correlation between different times: maybe some non-parametric signal decomposition? EMD, ICA or others (this one? https://indico.ego-gw.it/event/464/contributions/4157/)
	
