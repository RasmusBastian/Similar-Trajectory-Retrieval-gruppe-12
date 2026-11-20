// Web server for the demo UI.
// Serves the page in public/ and a small JSON API that reads from the project's PostgreSQL database.
// Start: npm start  ->  http://localhost:3000

import express from "express";
import pg from "pg";

// The models and ground-truth measures the UI can show.
// "implemented: false" = shown in the UI as "coming soon" but cannot be picked yet.
export const MODELS = [
  { id: "tsmini", name: "TSMini", implemented: true, table: "tsmini.embedding" },
  { id: "trajcl", name: "TrajCL", implemented: true, table: "trajcl.embedding" },
  { id: "convtraj", name: "ConvTraj", implemented: false, table: null },
];
export const MEASURES = [
  { id: "frechet", name: "Fréchet" },
  { id: "hausdorff", name: "Hausdorff" },
  { id: "dtw", name: "DTW" },
];
export const K = { min: 1, max: 50, default: 10 };

// Connection to the Docker database from docker-compose.yml (override with PGHOST, PGPORT, ... like the Python scripts).
const db = new pg.Pool({
  host: process.env.PGHOST ?? "localhost",
  port: Number(process.env.PGPORT ?? 5433),
  user: process.env.PGUSER ?? "str",
  password: process.env.PGPASSWORD ?? "str",
  database: process.env.PGDATABASE ?? "trajectories",
});

const app = express();
app.use(express.static("public")); // index.html, style.css, app.js

// GET /api/options: everything the page needs to fill its drop-downs.
// For each dataset: which ground-truth measures and which models have data.
app.get("/api/options", async (req, res) => {
  const datasets = await db.query(`
    SELECT d.name, d.n AS "trajectoryCount",
           ARRAY(SELECT g.measure FROM tsmini.ground_truth_set g
                 WHERE g.dataset = d.name ORDER BY g.measure) AS measures
    FROM tsmini.dataset d
    ORDER BY d.name`);

  for (const dataset of datasets.rows) {
    dataset.models = [];
    for (const model of MODELS.filter((m) => m.implemented)) {
      // to_regclass returns NULL if the table does not exist (e.g. TrajCL data never imported)
      const { rows } = await db.query(
        `SELECT to_regclass($1) IS NOT NULL AS exists`, [model.table]);
      if (!rows[0].exists) continue;
      const found = await db.query(
        `SELECT 1 FROM ${model.table} WHERE dataset = $1 LIMIT 1`, [dataset.name]);
      if (found.rowCount > 0) dataset.models.push(model.id);
    }
  }

  res.json({
    models: MODELS.map(({ id, name, implemented }) => ({ id, displayName: name, implemented })),
    measures: MEASURES.map(({ id, name }) => ({ id, displayName: name })),
    k: { min: K.min, max: K.max, default: K.default },
    datasets: datasets.rows,
  });
});

// If a database query fails, send the error to the page as { error: "..." } instead of crashing.
app.use((err, req, res, next) => {
  console.error(err);
  const message = err.code === "ECONNREFUSED"
    ? "Cannot reach the database. Is it running (docker compose up -d)?"
    : err.message;
  res.status(500).json({ error: message });
});

const port = Number(process.env.PORT ?? 3000);
app.listen(port, () => console.log(`Demo UI on http://localhost:${port}`));
