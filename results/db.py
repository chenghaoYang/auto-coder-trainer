"""SQLite-backed result database for experiment tracking."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "results.db"
_JSON_FIELDS = {
    "metrics_json",
    "train_metrics_json",
    "checks_json",
    "suggestions_json",
    "recipe_json",
    "budget_json",
    "details_json",
    "metadata_json",
    "payload_json",
}


def _resolve_default_db_path() -> Path:
    env_override = os.environ.get("ACT_RESULTS_DB")
    return Path(env_override) if env_override else _DEFAULT_DB_PATH


class ResultDB:
    """SQLite-backed experiment result database.

    Stores experiments, eval runs, ablations, verdicts, tasks, and artifacts.
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else _resolve_default_db_path()
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

    def _ensure_conn(self) -> sqlite3.Connection:
        assert self._conn is not None, "Database not connected"
        return self._conn

    @staticmethod
    def _serialize_json(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value)

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a sqlite3.Row to a dict, deserializing known JSON fields."""
        payload = dict(row)
        for key in _JSON_FIELDS:
            if key in payload and payload[key] is not None:
                try:
                    payload[key] = json.loads(payload[key])
                except (TypeError, json.JSONDecodeError):
                    pass
        return payload

    def _fetch_related_rows(self, table: str, experiment_id: str) -> list[dict[str, Any]]:
        """Fetch rows from a child table keyed by experiment_id."""
        conn = self._ensure_conn()
        cur = conn.execute(
            f"SELECT * FROM {table} WHERE experiment_id = ? ORDER BY timestamp, id",
            (experiment_id,),
        )
        return [self._row_to_dict(row) for row in cur.fetchall()]

    def insert_experiment(self, experiment: dict[str, Any]) -> str:
        """Insert or update an experiment record. Returns experiment ID."""
        conn = self._ensure_conn()
        conn.execute(
            """INSERT INTO experiments
               (id, recipe_id, config_hash, timestamp, status, trainer_type, backend,
                model_base, metrics_json, train_metrics_json, recipe_json, budget_json,
                checkpoint_path, error)
               VALUES (
                   ?,
                   ?,
                   ?,
                   COALESCE((SELECT timestamp FROM experiments WHERE id = ?), datetime('now')),
                   ?,
                   ?,
                   ?,
                   ?,
                   ?,
                   ?,
                   ?,
                   ?,
                   ?,
                   ?
               )
               ON CONFLICT(id) DO UPDATE SET
                   recipe_id = excluded.recipe_id,
                   config_hash = excluded.config_hash,
                   status = excluded.status,
                   trainer_type = excluded.trainer_type,
                   backend = excluded.backend,
                   model_base = excluded.model_base,
                   metrics_json = excluded.metrics_json,
                   train_metrics_json = excluded.train_metrics_json,
                   recipe_json = excluded.recipe_json,
                   budget_json = excluded.budget_json,
                   checkpoint_path = excluded.checkpoint_path,
                   error = excluded.error""",
            (
                experiment["id"],
                experiment["recipe_id"],
                experiment["config_hash"],
                experiment["id"],
                experiment["status"],
                experiment["trainer_type"],
                experiment["backend"],
                experiment["model_base"],
                self._serialize_json(experiment.get("metrics_json")),
                self._serialize_json(experiment.get("train_metrics_json")),
                self._serialize_json(experiment.get("recipe_json")),
                self._serialize_json(experiment.get("budget_json")),
                experiment.get("checkpoint_path"),
                experiment.get("error"),
            ),
        )
        conn.commit()
        return experiment["id"]

    def list_experiments(
        self,
        *,
        recipe_id: str | None = None,
        limit: int | None = 50,
    ) -> list[dict[str, Any]]:
        """List experiments, newest first."""
        conn = self._ensure_conn()
        query = "SELECT * FROM experiments"
        params: list[Any] = []
        if recipe_id:
            query += " WHERE recipe_id = ?"
            params.append(recipe_id)
        query += " ORDER BY timestamp DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        cur = conn.execute(query, tuple(params))
        return [self._row_to_dict(row) for row in cur.fetchall()]

    def insert_eval_run(self, eval_run: dict[str, Any]) -> int:
        """Insert or update a benchmark evaluation run."""
        conn = self._ensure_conn()
        cur = conn.execute(
            """INSERT INTO eval_runs
               (experiment_id, benchmark, seed, metrics_json, details_json)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(experiment_id, benchmark, seed) DO UPDATE SET
                   metrics_json = excluded.metrics_json,
                   details_json = excluded.details_json""",
            (
                eval_run["experiment_id"],
                eval_run["benchmark"],
                eval_run.get("seed", 42),
                self._serialize_json(eval_run.get("metrics_json")),
                self._serialize_json(eval_run.get("details_json")),
            ),
        )
        conn.commit()
        return cur.lastrowid

    def insert_eval_runs(self, eval_runs: list[dict[str, Any]]) -> None:
        """Insert many evaluation runs."""
        for eval_run in eval_runs:
            self.insert_eval_run(eval_run)

    def get_eval_runs_for_experiment(self, experiment_id: str) -> list[dict[str, Any]]:
        """Return all eval runs linked to an experiment."""
        return self._fetch_related_rows("eval_runs", experiment_id)

    def get_eval_runs_for_recipe(self, recipe_id: str) -> list[dict[str, Any]]:
        """Return eval runs across every experiment in a recipe."""
        conn = self._ensure_conn()
        cur = conn.execute(
            """SELECT eval_runs.*
               FROM eval_runs
               INNER JOIN experiments ON experiments.id = eval_runs.experiment_id
               WHERE experiments.recipe_id = ?
               ORDER BY eval_runs.timestamp, eval_runs.id""",
            (recipe_id,),
        )
        return [self._row_to_dict(row) for row in cur.fetchall()]

    def insert_artifact(self, artifact: dict[str, Any]) -> int:
        """Insert an artifact record."""
        conn = self._ensure_conn()
        cur = conn.execute(
            """INSERT INTO artifacts (recipe_id, experiment_id, kind, path, metadata_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                artifact["recipe_id"],
                artifact.get("experiment_id"),
                artifact["kind"],
                artifact["path"],
                self._serialize_json(artifact.get("metadata_json")),
            ),
        )
        conn.commit()
        return cur.lastrowid

    def get_artifacts_for_experiment(self, experiment_id: str) -> list[dict[str, Any]]:
        """Return all artifacts linked to an experiment."""
        return self._fetch_related_rows("artifacts", experiment_id)

    def get_artifacts_for_recipe(self, recipe_id: str) -> list[dict[str, Any]]:
        """Return all artifacts linked to a recipe."""
        conn = self._ensure_conn()
        cur = conn.execute(
            "SELECT * FROM artifacts WHERE recipe_id = ? ORDER BY timestamp, id",
            (recipe_id,),
        )
        return [self._row_to_dict(row) for row in cur.fetchall()]

    def upsert_task(self, task: dict[str, Any]) -> str:
        """Insert or update a task row."""
        conn = self._ensure_conn()
        conn.execute(
            """INSERT INTO tasks (id, recipe_id, experiment_id, kind, title, status, priority, payload_json, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   recipe_id = excluded.recipe_id,
                   experiment_id = excluded.experiment_id,
                   kind = excluded.kind,
                   title = excluded.title,
                   status = excluded.status,
                   priority = excluded.priority,
                   payload_json = excluded.payload_json,
                   notes = excluded.notes,
                   updated_at = datetime('now')""",
            (
                task["id"],
                task["recipe_id"],
                task.get("experiment_id"),
                task["kind"],
                task["title"],
                task["status"],
                task.get("priority", "medium"),
                self._serialize_json(task.get("payload_json")),
                task.get("notes"),
            ),
        )
        conn.commit()
        return task["id"]

    def get_tasks(
        self,
        *,
        recipe_id: str | None = None,
        experiment_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query tracked tasks."""
        conn = self._ensure_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if recipe_id:
            clauses.append("recipe_id = ?")
            params.append(recipe_id)
        if experiment_id:
            clauses.append("(experiment_id = ? OR experiment_id IS NULL)")
            params.append(experiment_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        query = "SELECT * FROM tasks"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += (
            " ORDER BY CASE priority "
            "WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, "
            "updated_at DESC, timestamp DESC, id DESC"
        )
        cur = conn.execute(query, tuple(params))
        return [self._row_to_dict(row) for row in cur.fetchall()]

    def get_open_tasks(self, recipe_id: str | None = None) -> list[dict[str, Any]]:
        """Return pending / blocked / in-progress tasks."""
        tasks = self.get_tasks(recipe_id=recipe_id)
        return [
            task
            for task in tasks
            if task.get("status") in {"pending", "blocked", "in_progress"}
        ]

    def insert_ablation(self, ablation: dict[str, Any]) -> int:
        """Insert an ablation result. Returns ablation ID."""
        conn = self._ensure_conn()
        value = ablation.get("value")
        if not isinstance(value, str):
            value = json.dumps(value)
        metrics = ablation.get("metrics_json")
        cur = conn.execute(
            """INSERT INTO ablations (experiment_id, variable, value, metrics_json)
               VALUES (?, ?, ?, ?)""",
            (
                ablation["experiment_id"],
                ablation["variable"],
                value,
                self._serialize_json(metrics),
            ),
        )
        conn.commit()
        return cur.lastrowid

    def insert_verdict(self, verdict: dict[str, Any]) -> int:
        """Insert a judge verdict. Returns verdict ID."""
        conn = self._ensure_conn()
        cur = conn.execute(
            """INSERT INTO verdicts
               (experiment_id, verdict, reasoning, checks_json, suggestions_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                verdict["experiment_id"],
                verdict["verdict"],
                verdict.get("reasoning"),
                self._serialize_json(verdict.get("checks_json")),
                self._serialize_json(verdict.get("suggestions_json")),
            ),
        )
        conn.commit()
        return cur.lastrowid

    def get_ablations_for_experiment(self, experiment_id: str) -> list[dict[str, Any]]:
        """Return all ablation records linked to an experiment."""
        return self._fetch_related_rows("ablations", experiment_id)

    def get_verdicts_for_experiment(self, experiment_id: str) -> list[dict[str, Any]]:
        """Return all verdict records linked to an experiment."""
        return self._fetch_related_rows("verdicts", experiment_id)

    def get_latest_verdict(self, experiment_id: str) -> dict[str, Any] | None:
        """Return the newest verdict linked to an experiment."""
        conn = self._ensure_conn()
        cur = conn.execute(
            """SELECT *
               FROM verdicts
               WHERE experiment_id = ?
               ORDER BY timestamp DESC, id DESC
               LIMIT 1""",
            (experiment_id,),
        )
        row = cur.fetchone()
        return self._row_to_dict(row) if row is not None else None

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """Get an experiment by ID."""
        conn = self._ensure_conn()
        cur = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        row = cur.fetchone()
        return self._row_to_dict(row) if row is not None else None

    def get_experiment_bundle(self, experiment_id: str) -> dict[str, Any]:
        """Return an experiment together with its related records."""
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            return {
                "experiment": None,
                "eval_runs": [],
                "ablations": [],
                "verdicts": [],
                "tasks": [],
                "artifacts": [],
            }
        recipe_id = experiment.get("recipe_id", "")
        return {
            "experiment": experiment,
            "eval_runs": self.get_eval_runs_for_experiment(experiment_id),
            "ablations": self.get_ablations_for_experiment(experiment_id),
            "verdicts": self.get_verdicts_for_experiment(experiment_id),
            "tasks": self.get_tasks(recipe_id=recipe_id, experiment_id=experiment_id),
            "artifacts": self.get_artifacts_for_experiment(experiment_id),
        }

    def find_by_recipe(self, recipe_id: str) -> list[dict[str, Any]]:
        """Find all experiments for a given recipe."""
        return self.list_experiments(recipe_id=recipe_id, limit=None)

    def find_by_recipe_with_details(self, recipe_id: str) -> list[dict[str, Any]]:
        """Find all experiments for a recipe and attach detailed records."""
        return [self.get_experiment_bundle(experiment["id"]) for experiment in self.find_by_recipe(recipe_id)]

    def find_by_config_hash(self, config_hash: str) -> list[dict[str, Any]]:
        """Find experiments with matching config hash (dedup check)."""
        conn = self._ensure_conn()
        cur = conn.execute(
            "SELECT * FROM experiments WHERE config_hash = ? ORDER BY timestamp DESC, id DESC",
            (config_hash,),
        )
        return [self._row_to_dict(row) for row in cur.fetchall()]

    def get_best_recipe(self, metric: str = "resolve_rate") -> dict[str, Any] | None:
        """Get the best-performing experiment by a given metric."""
        experiments = [
            experiment
            for experiment in self.list_experiments(limit=None)
            if experiment.get("status") == "success"
        ]
        ranked: list[tuple[float, dict[str, Any]]] = []
        for experiment in experiments:
            metrics = experiment.get("metrics_json", {})
            if isinstance(metrics, dict) and isinstance(metrics.get(metric), (int, float)):
                ranked.append((float(metrics[metric]), experiment))
        if not ranked:
            return None
        return max(ranked, key=lambda item: item[0])[1]
