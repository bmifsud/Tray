import logging
from datetime import datetime, timedelta, timezone
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.services.ingestion import ingest_ticks

logger = logging.getLogger(__name__)

def daily_tick_ingestion():
    """
    Task to run daily tick ingestion at market close.
    """
    symbol = "NAS100"
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=1)

    logger.info(f"Running scheduled daily tick ingestion for {symbol} ({start_date} to {end_date})")

    result = ingest_ticks(symbol, start_date, end_date)
    logger.info(f"Scheduled ingestion result: {result}")

def start_scheduler():
    """
    Initializes and starts the background scheduler for daily ingestion.
    Runs every weekday at 16:00 (4:00 PM EST market close).
    Note: Timezones should be adjusted based on broker server time in production.
    """
    scheduler = BackgroundScheduler()

    # Run at 16:00 daily (example for market close). Adjust as needed.
    scheduler.add_job(
        daily_tick_ingestion,
        CronTrigger(hour=16, minute=0, day_of_week='mon-fri')
    )

    scheduler.start()
    logger.info("Background scheduler started. Daily tick ingestion configured for 16:00.")
