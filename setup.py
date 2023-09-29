import setuptools
import sys

required_packages =['scipy>=1.9.3', 'numpy', 'matplotlib', 'gwpy', 'memspectrum'],

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="gw-fad",
	version="0.0.0",
	author="Stefano Schmidt",
	author_email="stefanoschmidt1995@gmail.com",
	description="Feature Anomaly Detection for GW data",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="",
	packages=setuptools.find_packages(),
	license = 'GNU GENERAL PUBLIC LICENSE v3',
	classifiers=[
		"Programming Language :: Python :: 3",
		"Operating System :: OS Independent",
	],
	scripts = ["bin/fad_white_batches", "bin/fad_generate_features", "bin/fad_rank", "bin/fad_generate_dag", "bin/fad_plot_spectrograms"],
	python_requires='>=3.8',
	install_requires=required_packages,
)

