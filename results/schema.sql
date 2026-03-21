-- Result DB schema for auto-coder-trainer experiments

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    recipe_id TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    status TEXT NOT NULL CHECK(status IN ('planned', 'prepared', 'running', 'success', 'failed', 'timeout', 'blocked')),
    trainer_type TEXT NOT NULL,
    backend TEXT NOT NULL,
    model_base TEXT NOT NULL,
    metrics_json TEXT,  -- summary metrics (prefer eval metrics when available)
    train_metrics_json TEXT,  -- raw training metrics
    recipe_json TEXT,  -- normalized recipe payload for recovery
    budget_json TEXT,  -- declared budget for this run
    checkpoint_path TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS ablations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    variable TEXT NOT NULL,
    value TEXT NOT NULL,  -- JSON-encoded value
    metrics_json TEXT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS verdicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    verdict TEXT NOT NULL CHECK(verdict IN ('accept', 'reject', 'needs_ablation', 'needs_rerun')),
    reasoning TEXT,
    checks_json TEXT,  -- JSON blob of check results
    suggestions_json TEXT,  -- JSON array of suggestions
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    benchmark TEXT NOT NULL,
    seed INTEGER NOT NULL,
    metrics_json TEXT,
    details_json TEXT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(experiment_id, benchmark, seed)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recipe_id TEXT NOT NULL,
    experiment_id TEXT REFERENCES experiments(id),
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    metadata_json TEXT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    recipe_id TEXT NOT NULL,
    experiment_id TEXT,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'in_progress', 'completed', 'blocked', 'cancelled')),
    priority TEXT NOT NULL DEFAULT 'medium' CHECK(priority IN ('low', 'medium', 'high')),
    payload_json TEXT,
    notes TEXT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_experiments_recipe ON experiments(recipe_id);
CREATE INDEX IF NOT EXISTS idx_experiments_hash ON experiments(config_hash);
CREATE INDEX IF NOT EXISTS idx_ablations_experiment ON ablations(experiment_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_experiment ON verdicts(experiment_id);
CREATE INDEX IF NOT EXISTS idx_eval_runs_experiment ON eval_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_recipe ON artifacts(recipe_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_experiment ON artifacts(experiment_id);
CREATE INDEX IF NOT EXISTS idx_tasks_recipe ON tasks(recipe_id);
CREATE INDEX IF NOT EXISTS idx_tasks_experiment ON tasks(experiment_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
