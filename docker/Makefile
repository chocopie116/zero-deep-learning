DOCKER=$(shell which docker)
DOCKER_COMPOSE=$(shell which docker-compose)

docker-compose/up:
	$(DOCKER_COMPOSE) up

docker-compose/build:
	$(DOCKER_COMPOSE) build

docker-compose/clean:
	$(DOCKER_COMPOSE) stop
	$(DOCKER_COMPOSE) rm

#https://www.continuum.io/blog/developer-blog/anaconda-and-docker-better-together-reproducible-data-science
up:
	$(DOCKER) run -i -t continuumio/anaconda3 /bin/bash

#https://www.continuum.io/blog/developer-blog/anaconda-and-docker-better-together-reproducible-data-science
up/notebook:
	$(DOCKER) run -i -t -p 8888:8888 continuumio/anaconda3 \
		/bin/bash -c "/opt/conda/bin/conda install jupyter -y --quiet &&\
		mkdir /opt/notebooks && /opt/conda/bin/jupyter notebook \
		--notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser"

