"""Trading Intelligence Monitor background worker.

Run as a separate long-lived process: python worker.py
For scheduled cloud jobs, pass --duration-seconds to exit cleanly after a bounded run.
The Streamlit web process only reads/writes shared PostgreSQL data and never starts
scanner or Telegram polling threads itself.
"""
from __future__ import annotations

import argparse
import signal
import threading

import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TIM scanner and Telegram background worker.")
    parser.add_argument(
        "--local-sqlite", action="store_true",
        help="allow local SQLite for development when web and worker share the same machine and project folder",
    )
    parser.add_argument(
        "--duration-seconds", type=int, default=0,
        help="stop after this many seconds (0 means run until the process is stopped)",
    )
    args = parser.parse_args()

    if not app.database_url() and not args.local_sqlite:
        raise SystemExit(
            "DATABASE_URL is required for a separately deployed worker. "
            "For local development on the same machine as Streamlit, use: python worker.py --local-sqlite"
        )

    app.RT = app.Runtime()
    stop_event = app.RT.stop_event

    def request_shutdown(signum, _frame):
        app.log.info("Received signal %s; stopping TIM worker...", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)

    app.init_db()
    # Initialize shared settings before the first scan. For a new PostgreSQL deployment,
    # start the web app once first so its existing settings.json can seed app_settings.
    app.load_settings()

    if not app.acquire_worker_lock():
        raise SystemExit("Another TIM background worker already holds the database lock.")

    app.RT.status["started_at"] = app.utc_now()
    app.RT.status["worker_state"] = "starting"
    app.publish_monitor_status()

    telegram_thread = threading.Thread(
        target=app.telegram_poll_loop,
        args=(stop_event,),
        name="tim-telegram-poller",
        daemon=True,
    )
    telegram_thread.start()

    app.log.info("TIM worker started. Data provider: Yahoo Finance via yfinance.")
    stop_timer = None
    try:
        if args.duration_seconds > 0:
            stop_timer = threading.Timer(args.duration_seconds, stop_event.set)
            stop_timer.daemon = True
            stop_timer.start()
        app.monitor_loop(stop_event)
    finally:
        stop_event.set()
        if stop_timer is not None:
            stop_timer.cancel()
        telegram_thread.join(timeout=35)
        app.RT.status["worker_state"] = "stopped"
        try:
            app.publish_monitor_status()
        except Exception:
            app.log.exception("Could not publish final worker status")
        app.release_worker_lock()
        app.close_database_pool()
        app.log.info("TIM worker stopped.")


if __name__ == "__main__":
    main()
