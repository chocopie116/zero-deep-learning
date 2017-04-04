DOCKER=$(shell which docker)

up:
	$(DOCKER) run -i -t continuumio/anaconda3 /bin/bash

up/notebook:
	$(DOCKER) run -i -t -p 8888:8888 continuumio/anaconda3 \
		/bin/bash -c "/opt/conda/bin/conda install jupyter -y --quiet &&\
		mkdir /opt/notebooks && /opt/conda/bin/jupyter notebook \
		--notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser"

