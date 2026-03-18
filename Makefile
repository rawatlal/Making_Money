run:
	streamlit run app/Home.py

docker-build:
	docker compose build

docker-run:
	docker compose up

test:
	pytest tests/ -v

download-data:
	python -m src.data.downloader

backtest:
	python -m src.backtest.engine
