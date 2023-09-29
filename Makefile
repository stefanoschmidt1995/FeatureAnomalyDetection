DIR=$(shell pwd)
MBANK_INSTALLED=$(shell python -c 'import fad' | cat)
INSTALL_DIR:=$(shell python -c "import numpy; print(numpy.__file__.replace('numpy/__init__.py', 'fad'))")
INSTALL_BIN_DIR:=$(shell which fad_generate_features | sed 's/fad_generate_features//g')

to_nikhef:
	scp -r fad/ bin/ test.py setup.py Nikhef:/data/gravwav/sschmidt/FeatureAnomalyDetection

install:
	python setup.py sdist && cd .. && pip install FeatureAnomalyDetection/dist/gw-fad*.tar.gz

install_cp:
	@if [ -d $(INSTALL_DIR) ]; then \
		cp bin/* $(INSTALL_BIN_DIR) ; \
		cp fad/*.py $(INSTALL_DIR); \
		echo Copied executables in $(INSTALL_BIN_DIR); \
		echo Copied modules in $(INSTALL_DIR) ;\
	else \
		echo Installation directory \"$(INSTALL_DIR)\" does not exist: unable to install the files.; \
		echo Have you ever installed the package normally?; \
	fi
clean:
	rm -rf dist/ fad/__pycache__ gw_fad.egg-info/
