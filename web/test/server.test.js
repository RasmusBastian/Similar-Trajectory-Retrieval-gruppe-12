// Tests for the API in server.js, run against the demo database.
// Run from web/:   $env:PGPORT = "5435"; npm test
// (the database must be running: docker start str-demo-db)

import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { app, db, validateSearch, overlap, toLatLon } from "../server.js";

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

// ---------- Helper functions (no database needed) ----------

test("validateSearch accepts a valid search", () => {
  assert.equal(validateSearch({ dataset: "beijing_20200", model: "tsmini", measure: "frechet", query: 5, k: 10 }), null);
});

test("validateSearch explains what is wrong", () => {
  const valid = { dataset: "beijing_20200", model: "tsmini", measure: "frechet", query: 5, k: 10 };

  assert.match(validateSearch({ ...valid, dataset: "" }), /dataset/);
  assert.match(validateSearch({ ...valid, model: "unknown" }), /Unknown model/);
  assert.match(validateSearch({ ...valid, model: "convtraj" }), /not available yet/);
  assert.match(validateSearch({ ...valid, measure: "euclid" }), /Unknown measure/);
  assert.match(validateSearch({ ...valid, query: NaN }), /query/);
  assert.match(validateSearch({ ...valid, k: 0 }), /k must be/);
  assert.match(validateSearch({ ...valid, k: 51 }), /k must be/);
});

test("overlap counts shared ids regardless of order", () => {
  assert.deepEqual(overlap([5, 1, 9, 7], [1, 2, 5, 7]), { hits: 3, k: 4, recall: 0.75 });
  assert.deepEqual(overlap([1, 2], [3, 4]), { hits: 0, k: 2, recall: 0 });
  assert.deepEqual(overlap([1, 2], []), { hits: 0, k: 0, recall: 0 }); // no division by zero
});

test("toLatLon swaps GeoJSON's [lon, lat] to [lat, lon]", () => {
  const geoJson = '{"type":"LineString","coordinates":[[116.4,39.9],[116.41,39.91]]}';

  assert.deepEqual(toLatLon(geoJson), [[39.9, 116.4], [39.91, 116.41]]);
});

// ---------- GET /api/search ----------

/** The first id that can be used as query, so the tests do not depend on specific ids. */
async function firstQueryId() {
  const { body } = await get("/api/datasets/beijing_20200/queries?measure=frechet");
  return body.ids[0];
}

for (const model of ["tsmini", "trajcl"]) {
  test(`search with ${model} returns k results from the model and from the ground truth`, async () => {
    const query = await firstQueryId();
    const { status, body } = await get(
      `/api/search?dataset=beijing_20200&model=${model}&measure=frechet&query=${query}&k=5`);

    assert.equal(status, 200);
    assert.equal(body.query.idx, query);
    assert.equal(body.query.split, "test");
    assert.equal(body.modelTopK.length, 5);
    assert.equal(body.groundTruthTopK.length, 5);
    assert.match(body.checkpoint, /frechet/i);
  });
}

test("search results are ranked 1..k, closest first, and never include the query itself", async () => {
  const query = await firstQueryId();
  const { body } = await get(`/api/search?dataset=beijing_20200&model=tsmini&measure=frechet&query=${query}&k=10`);

  for (const list of [body.modelTopK, body.groundTruthTopK]) {
    assert.deepEqual(list.map((r) => r.rank), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const distances = list.map((r) => r.distance);
    assert.deepEqual(distances, [...distances].sort((a, b) => a - b));
    assert.ok(list.every((r) => r.idx !== query));
  }
});

test("search overlap matches the two lists", async () => {
  const query = await firstQueryId();
  const { body } = await get(`/api/search?dataset=beijing_20200&model=tsmini&measure=frechet&query=${query}&k=10`);
  const truthIds = new Set(body.groundTruthTopK.map((r) => r.idx));
  const shared = body.modelTopK.filter((r) => truthIds.has(r.idx)).length;

  assert.equal(body.overlap.hits, shared);
  assert.equal(body.overlap.k, 10);
});

test("search paths are [lat, lon] points inside Beijing", async () => {
  const query = await firstQueryId();
  const { body } = await get(`/api/search?dataset=beijing_20200&model=tsmini&measure=frechet&query=${query}&k=3`);

  for (const [lat, lon] of body.query.path) {
    assert.ok(lat > 39 && lat < 41, `latitude ${lat} should be around 40 (Beijing)`);
    assert.ok(lon > 115 && lon < 118, `longitude ${lon} should be around 116 (Beijing)`);
  }
});

test("search gives 400 for invalid parameters and 404 for a query without ground truth", async () => {
  const bad = await get("/api/search?dataset=beijing_20200&model=convtraj&measure=frechet&query=1");
  assert.equal(bad.status, 400);
  assert.match(bad.body.error, /not available yet/);

  // DTW is a known measure (so not 400), but no DTW ground truth has been computed (so 404)
  const missing = await get("/api/search?dataset=beijing_20200&model=tsmini&measure=dtw&query=0");
  assert.equal(missing.status, 404);
  assert.match(missing.body.error, /no dtw ground truth/);
});
