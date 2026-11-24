// Tests for the API in server.js, run against the demo database.
// Run from web/:   $env:PGPORT = "5435"; npm test
// (the database must be running: docker start str-demo-db)

import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { app, db } from "../server.js";

let server;
let baseUrl;

before(async () => {
  // Fail early with a clear message if the database is not reachable.
  try {
    await db.query("SELECT 1");
  } catch (error) {
    throw new Error(`Cannot reach the database on port ${db.options.port} (${error.code}). ` +
      `Start it with "docker start str-demo-db" and set $env:PGPORT = "5435".`);
  }
  // Port 0 = let the operating system pick a free port.
  server = app.listen(0);
  await new Promise((resolve) => server.once("listening", resolve));
  baseUrl = `http://localhost:${server.address().port}`;
});

after(async () => {
  server?.close();
  await db.end();
});

/** Calls the API and returns the HTTP status and the JSON body. */
async function get(path) {
  const response = await fetch(baseUrl + path);
  return { status: response.status, body: await response.json() };
}

// ---------- GET /api/options ----------

test("options lists the three models, with ConvTraj not implemented yet", async () => {
  const { status, body } = await get("/api/options");

  assert.equal(status, 200);
  assert.deepEqual(body.models.map((m) => m.id), ["tsmini", "trajcl", "convtraj"]);
  assert.equal(body.models.find((m) => m.id === "convtraj").implemented, false);
});

test("options has k limits 1-50 with default 10", async () => {
  const { body } = await get("/api/options");

  assert.deepEqual(body.k, { min: 1, max: 50, default: 10 });
});

test("options says which measures and models have data in beijing_20200", async () => {
  const { body } = await get("/api/options");
  const beijing = body.datasets.find((d) => d.name === "beijing_20200");

  assert.ok(beijing, "beijing_20200 should be in the database");
  assert.ok(beijing.measures.includes("frechet"));
  assert.ok(beijing.models.includes("tsmini"));
  assert.ok(beijing.models.includes("trajcl"));
});

// ---------- GET /api/datasets/:dataset/queries ----------

test("queries returns sorted trajectory ids with Fréchet ground truth", async () => {
  const { status, body } = await get("/api/datasets/beijing_20200/queries?measure=frechet");

  assert.equal(status, 200);
  assert.ok(body.ids.length > 0);
  assert.ok(body.ids.every(Number.isInteger));
  assert.deepEqual(body.ids, [...body.ids].sort((a, b) => a - b));
});

test("queries only uses the test split", async () => {
  const { body } = await get("/api/datasets/beijing_20200/queries?measure=frechet");
  const { rows } = await db.query(
    `SELECT count(*)::int AS n FROM tsmini.ground_truth
     WHERE dataset = 'beijing_20200' AND measure = 'frechet' AND split = 'test'`);

  assert.equal(body.ids.length, rows[0].n);
});

test("queries is empty for a measure without ground truth", async () => {
  const { status, body } = await get("/api/datasets/beijing_20200/queries?measure=dtw");

  assert.equal(status, 200);
  assert.deepEqual(body.ids, []);
});

test("queries rejects an unknown measure with an error message", async () => {
  const { status, body } = await get("/api/datasets/beijing_20200/queries?measure=euclid");

  assert.equal(status, 400);
  assert.match(body.error, /Unknown measure/);
});
