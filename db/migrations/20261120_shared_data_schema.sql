-- Run once against an existing database created before the shared data schema.
-- Example:
--   docker compose exec -T db psql -U str -d trajectories < db/migrations/20261120_shared_data_schema.sql
BEGIN;

CREATE SCHEMA IF NOT EXISTS data;

ALTER TABLE tsmini.dataset SET SCHEMA data;
ALTER TABLE tsmini.trajectory SET SCHEMA data;
ALTER TABLE tsmini.ground_truth_set SET SCHEMA data;
ALTER TABLE tsmini.ground_truth SET SCHEMA data;

COMMIT;
