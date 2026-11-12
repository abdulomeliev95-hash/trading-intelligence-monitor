"""One-time migration of local data/settings.json into PostgreSQL app_settings.

Run from the project folder after setting DATABASE_URL in .env:
    python migrate_settings_to_db.py
Use --force only when you intentionally want to overwrite settings already stored in DB.
"""
from __future__ import annotations

import argparse
import json

import app


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy TIM settings.json into shared PostgreSQL storage.")
    parser.add_argument("--force", action="store_true", help="overwrite the existing app_settings row")
    args = parser.parse_args()

    app.RT = app.Runtime()
    if not app.database_url():
        print("ERROR: DATABASE_URL is not set. Add it to your local .env or environment first.")
        return 2
    if not app.SETTINGS_PATH.exists():
        print(f"ERROR: {app.SETTINGS_PATH} was not found; nothing to migrate.")
        return 2

    try:
        with app.SETTINGS_PATH.open("r", encoding="utf-8") as handle:
            settings = json.load(handle)
        if not isinstance(settings, dict):
            raise ValueError("settings.json must contain a JSON object")

        app.init_db()
        existing = app.db_query("SELECT updated_at FROM app_settings WHERE id=1")
        if existing and not args.force:
            print("Shared settings already exist; no changes made.")
            print("Use --force only if you intentionally want local settings.json to replace them.")
            return 0

        app.save_settings(settings)
        print("Settings successfully copied into PostgreSQL app_settings.")
        print(f"Watchlist assets: {len(app.load_settings()['watchlist'])}")
        print("The configured neutral threshold and prediction horizons were preserved.")
        return 0
    finally:
        app.close_database_pool()


if __name__ == "__main__":
    raise SystemExit(main())
