from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apscheduler.schedulers.background import BackgroundScheduler

from agents.market_data_agent.fetch_crypto import fetch_crypto
from agents.market_data_agent.fetch_macro import fetch_macro
from agents.market_data_agent.fetch_stocks import fetch_stocks


def run_stock_ingestion():
    print("Running stock ingestion...")
    fetch_stocks()


def run_crypto_ingestion():
    print("Running crypto ingestion...")
    fetch_crypto()


def run_macro_ingestion():
    print("Running macro ingestion...")
    fetch_macro()


def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_stock_ingestion, "interval", seconds=30)
    scheduler.add_job(run_crypto_ingestion, "interval", seconds=20)
    scheduler.add_job(run_macro_ingestion, "interval", seconds=40)
    scheduler.start()

    try:
        while True:
            time.sleep(60)
    finally:
        scheduler.shutdown()


if __name__ == "__main__":
    start_scheduler()
