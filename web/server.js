// Web server for the demo UI.
// Serves the page in public/ and a small JSON API that reads from the project's PostgreSQL database.
// Start: npm start  ->  http://localhost:3000

import { fileURLToPath } from "node:url";
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
export const db = new pg.Pool({
  host: process.env.PGHOST ?? "localhost",
  port: Number(process.env.PGPORT ?? 5433),
  user: process.env.PGUSER ?? "str",
  password: process.env.PGPASSWORD ?? "str",
  database: process.env.PGDATABASE ?? "trajectories",
});

export const app = express();
// index.html, style.css, app.js. The path is relative to this file, so it works from any folder.
app.use(express.static(fileURLToPath(new URL("public", import.meta.url))));

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

// GET /api/datasets/beijing_20200/queries?measure=frechet
// The trajectory ids that can be used as query: those that have ground truth for the measure.
// We use the test split, i.e. trajectories the models were not trained on. If a dataset has no
// test split, we fall back to all trajectories with ground truth.
app.get("/api/datasets/:dataset/queries", async (req, res) => {
  const { dataset } = req.params;
  const { measure } = req.query;
  if (!MEASURES.some((m) => m.id === measure)) {
    return res.status(400).json({ error: `Unknown measure '${measure}'.` });
  }

  const sql = `
    SELECT idx FROM tsmini.ground_truth
    WHERE dataset = $1 AND measure = $2 AND ($3::text IS NULL OR split = $3)
    ORDER BY idx`;
  let result = await db.query(sql, [dataset, measure, "test"]);
  if (result.rowCount === 0) result = await db.query(sql, [dataset, measure, null]);

  res.json({ dataset, measure, ids: result.rows.map((row) => row.idx) });
});

// ---------- Search ----------

/**
 * Checks the search parameters from the URL. Returns an error message, or null if they are valid.
 * Exported so the tests can call it directly.
 */
export function validateSearch({ dataset, model, measure, query, k }) {
  if (!dataset) return "dataset is required.";
  const modelInfo = MODELS.find((m) => m.id === model);
  if (!modelInfo) return `Unknown model '${model}'.`;
  if (!modelInfo.implemented) return `${modelInfo.name} is not available yet.`;
  if (!MEASURES.some((m) => m.id === measure)) return `Unknown measure '${measure}'.`;
  if (!Number.isInteger(query)) return "query must be a trajectory id (a whole number).";
  if (!Number.isInteger(k) || k < K.min || k > K.max) return `k must be between ${K.min} and ${K.max}.`;
  return null;
}

/**
 * How many of the true top-k the model also found (order does not matter). recall = hits / k.
 * Example: model [5, 1, 9], truth [1, 2, 5] -> hits 2, k 3, recall 0.67
 */
export function overlap(modelIds, truthIds) {
  const truth = new Set(truthIds);
  const hits = new Set(modelIds.filter((id) => truth.has(id))).size;
  return { hits, k: truth.size, recall: truth.size === 0 ? 0 : hits / truth.size };
}

/**
 * PostGIS gives a line as GeoJSON with [longitude, latitude] points.
 * The map (Leaflet) wants [latitude, longitude], so we swap them.
 */
export function toLatLon(geoJson) {
  return JSON.parse(geoJson).coordinates.map(([lon, lat]) => [lat, lon]);
}

/** Turns database rows (idx, distance, geojson) into a ranked list starting at rank 1. */
function toRanked(rows) {
  return rows.map((row, i) => ({
    rank: i + 1,
    idx: row.idx,
    distance: row.distance,
    path: toLatLon(row.geojson),
  }));
}

// GET /api/search?dataset=beijing_20200&model=tsmini&measure=frechet&query=42953&k=10
// Returns the query trajectory, the model's top-k, the true top-k and how much they overlap.
// Both lists are searched among the same candidates: the other trajectories in the query's split,
// because ground truth only exists between trajectories in the same split.
app.get("/api/search", async (req, res) => {
  const params = {
    dataset: req.query.dataset,
    model: req.query.model,
    measure: req.query.measure,
    query: Number(req.query.query),
    k: Number(req.query.k ?? K.default),
  };
  const error = validateSearch(params);
  if (error) return res.status(400).json({ error });
  const { dataset, model, measure, query, k } = params;
  const table = MODELS.find((m) => m.id === model).table;

  // 1. The query trajectory and which split (train/eval/test) it is in.
  const queryResult = await db.query(`
    SELECT g.split, ST_AsGeoJSON(t.geom_wgs84) AS geojson
    FROM tsmini.ground_truth g
    JOIN tsmini.trajectory t ON t.dataset = g.dataset AND t.idx = g.idx
    WHERE g.dataset = $1 AND g.measure = $2 AND g.idx = $3
    LIMIT 1`, [dataset, measure, query]);
  if (queryResult.rowCount === 0) {
    return res.status(404).json({ error: `Trajectory ${query} has no ${measure} ground truth in ${dataset}.` });
  }
  const { split, geojson } = queryResult.rows[0];

  // 2. Which trained model (checkpoint) to use. Prefer one trained on the chosen measure,
  //    e.g. beijing_20200_trajsimi_TSMini_frechet_best.pt for frechet.
  const checkpointResult = await db.query(`
    SELECT model FROM ${table}
    WHERE dataset = $1 AND idx = $2
    ORDER BY model ILIKE '%' || $3 || '%' DESC, model
    LIMIT 1`, [dataset, query, measure]);
  if (checkpointResult.rowCount === 0) {
    const name = MODELS.find((m) => m.id === model).name;
    return res.status(404).json({ error: `${name} has no embedding for trajectory ${query} in ${dataset}.` });
  }
  const checkpoint = checkpointResult.rows[0].model;

  // 3. The model's top-k: the candidates whose embedding is closest to the query's (L1 distance =
  //    sum of absolute differences of the numbers in the two embeddings).
  const modelResult = await db.query(`
    WITH q AS (
      SELECT embedding FROM ${table}
      WHERE dataset = $1 AND model = $2 AND idx = $3
    ), candidate AS (
      SELECT idx FROM tsmini.ground_truth
      WHERE dataset = $1 AND measure = $4 AND split = $5 AND idx <> $3
    ), ranked AS (
      SELECT e.idx,
             (SELECT sum(abs(a - b)) FROM unnest(e.embedding, q.embedding) AS u(a, b))::float8 AS distance
      FROM ${table} e
      JOIN candidate c ON c.idx = e.idx
      CROSS JOIN q
      WHERE e.dataset = $1 AND e.model = $2
      ORDER BY distance, e.idx
      LIMIT $6
    )
    SELECT r.idx, r.distance, ST_AsGeoJSON(t.geom_wgs84) AS geojson
    FROM ranked r JOIN tsmini.trajectory t ON t.dataset = $1 AND t.idx = r.idx
    ORDER BY r.distance, r.idx`, [dataset, checkpoint, query, measure, split, k]);

  // 4. The true top-k. ground_truth.distances[j + 1] is the true distance in metres from the
  //    query to the trajectory in row j of the same split (see db/init/03_tsmini.sql).
  const truthResult = await db.query(`
    WITH q AS (
      SELECT distances FROM tsmini.ground_truth
      WHERE dataset = $1 AND measure = $2 AND split = $3 AND idx = $4
    )
    SELECT g.idx, u.distance::float8 AS distance, ST_AsGeoJSON(t.geom_wgs84) AS geojson
    FROM q
    CROSS JOIN LATERAL unnest(q.distances) WITH ORDINALITY AS u(distance, position)
    JOIN tsmini.ground_truth g
      ON g.dataset = $1 AND g.measure = $2 AND g.split = $3 AND g.row = u.position - 1
    JOIN tsmini.trajectory t ON t.dataset = $1 AND t.idx = g.idx
    WHERE g.idx <> $4
    ORDER BY u.distance, g.idx
    LIMIT $5`, [dataset, measure, split, query, k]);

  // 5. Put it together.
  const modelTopK = toRanked(modelResult.rows);
  const groundTruthTopK = toRanked(truthResult.rows);
  res.json({
    dataset, model, measure, checkpoint,
    query: { idx: query, split, path: toLatLon(geojson) },
    modelTopK,
    groundTruthTopK,
    overlap: overlap(modelTopK.map((r) => r.idx), groundTruthTopK.map((r) => r.idx)),
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

// Only start listening when this file is run directly (npm start / node server.js),
// not when the tests import it. The tests start their own server on a free port.
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT ?? 3000);
  app.listen(port, () => console.log(`Demo UI on http://localhost:${port}`));
}
