-- Result DB schema for auto-coder-trainer experiments

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    recipe_id TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    status TEXT NOT NULL CHECK(status IN ('running', 'success', 'failed', 'timeout')),
    trainer_type TEXT NOT NULL,
    backend TEXT NOT NULL,
    model_base TEXT NOT NULL,
    metrics_json TEXT,  -- JSON blob of metrics
    checkpoint_path TEXT,
    error TEXT,
    UNIQUE(config_hash, recipe_id)
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

CREATE INDEX IF NOT EXISTS idx_experiments_recipe ON experiments(recipe_id);
CREATE INDEX IF NOT EXISTS idx_experiments_hash ON experiments(config_hash);
CREATE INDEX IF NOT EXISTS idx_ablations_experiment ON ablations(experiment_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_experiment ON verdicts(experiment_id);
