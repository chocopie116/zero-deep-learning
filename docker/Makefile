DOCKER_COMPOSE=$(shell which docker-compose)
env=../.env
config=


up:
	$(DOCKER_COMPOSE) -f docker-compose.yml up

build:
	$(DOCKER_COMPOSE) -f docker-compose.yml build

clean:
	$(DOCKER_COMPOSE) -f docker-compose.yml stop
	$(DOCKER_COMPOSE) -f docker-compose.yml rm

flush_storage:
	rm -rf storage/mysql/**
	rm -rf storage/redis/**

