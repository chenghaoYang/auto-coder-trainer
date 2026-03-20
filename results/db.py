"""SQLite-backed result database for experiment tracking."""

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA_PATH = Path(__file__).parent / "schema.sql"
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "results.db"


class ResultDB:
    """SQLite-backed experiment result database.

    Stores experiments, ablation results, and judge verdicts.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Connect to the database and initialize schema if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        schema = SCHEMA_PATH.read_text()
        self._conn.executescript(schema)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a sqlite3.Row to a dict, deserializing JSON fields."""
        d = dict(row)
        for key in ("metrics_json", "checks_json", "suggestions_json"):
            if key in d and d[key] is not None:
                d[key] = json.loads(d[key])
        return d

    def insert_experiment(self, experiment: dict[str, Any]) -> str:
        """Insert a new experiment record. Returns experiment ID."""
        assert self._conn is not None, "Database not connected"
        metrics = experiment.get("metrics_json")
        if isinstance(metrics, dict):
            metrics = json.dumps(metrics)
        self._conn.execute(
            """INSERT INTO experiments
               (id, recipe_id, config_hash, status, trainer_type, backend, model_base,
                metrics_json, checkpoint_path, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment["id"],
                experiment["recipe_id"],
                experiment["config_hash"],
                experiment["status"],
                experiment["trainer_type"],
                experiment["backend"],
                experiment["model_base"],
                metrics,
                experiment.get("checkpoint_path"),
                experiment.get("error"),
            ),
        )
        self._conn.commit()
        return experiment["id"]

    def insert_ablation(self, ablation: dict[str, Any]) -> int:
        """Insert an ablation result. Returns ablation ID."""
        assert self._conn is not None, "Database not connected"
        value = ablation.get("value")
        if not isinstance(value, str):
            value = json.dumps(value)
        metrics = ablation.get("metrics_json")
        if isinstance(metrics, dict):
            metrics = json.dumps(metrics)
        cur = self._conn.execute(
            """INSERT INTO ablations (experiment_id, variable, value, metrics_json)
               VALUES (?, ?, ?, ?)""",
            (
                ablation["experiment_id"],
                ablation["variable"],
                value,
                metrics,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def insert_verdict(self, verdict: dict[str, Any]) -> int:
        """Insert a judge verdict. Returns verdict ID."""
        assert self._conn is not None, "Database not connected"
        checks = verdict.get("checks_json")
        if isinstance(checks, dict):
            checks = json.dumps(checks)
        suggestions = verdict.get("suggestions_json")
        if isinstance(suggestions, list):
            suggestions = json.dumps(suggestions)
        cur = self._conn.execute(
            """INSERT INTO verdicts
               (experiment_id, verdict, reasoning, checks_json, suggestions_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                verdict["experiment_id"],
                verdict["verdict"],
                verdict.get("reasoning"),
                checks,
                suggestions,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """Get an experiment by ID."""
        assert self._conn is not None, "Database not connected"
        cur = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def find_by_recipe(self, recipe_id: str) -> list[dict[str, Any]]:
        """Find all experiments for a given recipe."""
        assert self._conn is not None, "Database not connected"
        cur = self._conn.execute(
            "SELECT * FROM experiments WHERE recipe_id = ? ORDER BY timestamp DESC",
            (recipe_id,),
        )
        return [self._row_to_dict(row) for row in cur.fetchall()]

    def find_by_config_hash(self, config_hash: str) -> list[dict[str, Any]]:
        """Find experiments with matching config hash (dedup check)."""
        assert self._conn is not None, "Database not connected"
        cur = self._conn.execute(
            "SELECT * FROM experiments WHERE config_hash = ? ORDER BY timestamp DESC",
            (config_hash,),
        )
        return [self._row_to_dict(row) for row in cur.fetchall()]

    def get_best_recipe(self, metric: str = "resolve_rate") -> dict[str, Any] | None:
        """Get the best-performing experiment by a given metric.

        Extracts the metric value from the metrics_json blob using
        json_extract and returns the experiment with the highest value.
        """
        assert self._conn is not None, "Database not connected"
        cur = self._conn.execute(
            """SELECT *
               FROM experiments
               WHERE status = 'success'
                 AND metrics_json IS NOT NULL
                 AND json_extract(metrics_json, ?) IS NOT NULL
               ORDER BY json_extract(metrics_json, ?) DESC
               LIMIT 1""",
            (f"$.{metric}", f"$.{metric}"),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)
