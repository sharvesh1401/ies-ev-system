.PHONY: up down build test verify logs clean install

up:
	docker-compose up -d

down:
	docker-compose down

build:
	docker-compose build

logs:
	docker-compose logs -f

test:
	docker-compose run --rm backend pytest

verify:
	chmod +x scripts/verify.sh
	./scripts/verify.sh

clean:
	docker-compose down -v
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf backend/.pytest_cache
	rm -rf frontend/node_modules
	rm -rf frontend/dist

install:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install
