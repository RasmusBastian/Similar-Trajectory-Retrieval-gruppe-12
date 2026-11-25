"""Compare stored Beijing embeddings with exact discrete Frechet search."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Search Beijing with a stored model and compare exact Frechet ranks.")
    parser.add_argument("--model", choices=("tsmini", "trajcl"), required=True)
    parser.add_argument("--idx", type=int, required=True, help="Beijing trajectory index (0–49933)")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest matches (default: 10)")
    parser.add_argument("--rerank", type=int, default=50, help="Embedding candidates for exact reranking (default: 50)")
    args = parser.parse_args()
    if not 0 <= args.idx < 49934:
        parser.error("--idx must be between 0 and 49933")
    if args.k < 1:
        parser.error("--k must be at least 1")
    if args.rerank < args.k:
        parser.error("--rerank must be at least --k")

    schema = args.model
    sql = f"""
    WITH q AS MATERIALIZED (
        SELECT idx, geom, n_points, split
        FROM data.trajectory WHERE dataset = 'beijing_20200' AND idx = {args.idx}
    ), embedding_query AS MATERIALIZED (
        SELECT model, embedding FROM {schema}.embedding
        WHERE dataset = 'beijing_20200' AND idx = {args.idx}
        ORDER BY model LIMIT 1
    ), embedding_stats AS (
        SELECT count(*) AS n, array_length(max(embedding), 1) AS dimensions
        FROM {schema}.embedding WHERE dataset = 'beijing_20200'
    ), embedding_distances AS MATERIALIZED (
        SELECT e.idx,
               (SELECT sum(abs(a - b)) FROM unnest(e.embedding, q.embedding) AS u(a, b))::double precision AS l1
        FROM {schema}.embedding e CROSS JOIN embedding_query q
        WHERE e.dataset = 'beijing_20200' AND e.model = q.model AND e.idx <> {args.idx}
    ), embedding_ranked AS MATERIALIZED (
        SELECT idx, l1, row_number() OVER (ORDER BY l1, idx) AS rank
        FROM embedding_distances
    ), exact_distances AS MATERIALIZED (
        SELECT t.idx, ST_FrechetDistance(q.geom, t.geom) AS frechet
        FROM q CROSS JOIN data.trajectory t
        WHERE t.dataset = 'beijing_20200' AND t.idx <> {args.idx}
    ), exact_ranked AS MATERIALIZED (
        SELECT idx, frechet, row_number() OVER (ORDER BY frechet, idx) AS rank
        FROM exact_distances
    ), top_embedding AS (
        SELECT e.rank, e.idx, e.l1, x.frechet, x.rank AS exact_rank
        FROM embedding_ranked e JOIN exact_ranked x USING (idx)
        WHERE e.rank <= {args.k}
    ), top_candidates AS (
        SELECT e.idx, e.l1, x.frechet
        FROM embedding_ranked e JOIN exact_distances x USING (idx)
        WHERE e.rank <= {args.rerank}
    ), reranked AS (
        SELECT idx, row_number() OVER (ORDER BY frechet, idx) AS rank
        FROM top_candidates
        ORDER BY frechet, idx
        LIMIT {args.k}
    ), metrics AS (
        SELECT (SELECT count(*) FROM top_embedding WHERE exact_rank <= {args.k}) AS embedding_hits,
               (SELECT count(*) FROM reranked r JOIN exact_ranked x USING (idx) WHERE x.rank <= {args.k}) AS rerank_hits
    ), output AS (
    SELECT 'info' AS section, NULL::text AS rank, NULL::text AS idx, NULL::text AS l1,
           NULL::text AS frechet, 'loaded database embeddings (' || s.n || ', ' || s.dimensions || ')' AS detail
    FROM embedding_stats s
    UNION ALL
    SELECT 'query', q.idx::text, q.idx::text, '', '', q.n_points::text || ' points, ' || q.split
    FROM q
    UNION ALL
    SELECT 'match', te.rank::text, te.idx::text, to_char(te.l1, 'FM999999990.000'),
           to_char(te.frechet, 'FM999999990.0'),
           CASE WHEN te.exact_rank <= {args.k} THEN 'yes' ELSE 'no' END
    FROM top_embedding te
    UNION ALL
    SELECT 'summary', m.embedding_hits::text, m.rerank_hits::text, '', '', ''
    FROM metrics m
    )
    SELECT section, rank, idx, l1, frechet, detail
    FROM output
    ORDER BY CASE section WHEN 'info' THEN 0 WHEN 'query' THEN 1 WHEN 'match' THEN 2 ELSE 3 END,
             CASE WHEN section = 'match' THEN rank::int END;
    """
    command = [
        "docker", "compose", "exec", "-T", "db", "psql",
        "-U", "str", "-d", "trajectories", "-A", "-t", "-F", "\t", "-c", sql,
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8")
    except FileNotFoundError:
        print("Docker was not found. Run this from PowerShell with Docker Desktop running.", file=sys.stderr)
        return 127
    if result.returncode:
        print(result.stderr, file=sys.stderr, end="")
        return result.returncode

    rows = [line.split("\t") for line in result.stdout.splitlines()]
    info = next((r for r in rows if r[0] == "info"), None)
    query = next((r for r in rows if r[0] == "query"), None)
    matches = [r for r in rows if r[0] == "match"]
    summary = next((r for r in rows if r[0] == "summary"), None)
    print(f"\n{info[5] if info else 'database embeddings'}")
    print(f"\nquery #{query[2]}: {query[5]}")
    print(f"\n{'rank':>4} {'idx':>7} {'emb L1':>9} {'exact Frechet (m)':>18}  in exact top-{args.k}?")
    for row in matches:
        print(f"{int(row[1]):>4} {int(row[2]):>7} {float(row[3]):>9.3f} {float(row[4]):>18.1f}  {row[5]}")
    if summary:
        print(f"\nembedding search found {summary[1]}/{args.k} of the true top-{args.k}")
        print(f"embedding top-{args.rerank} + exact rerank found {summary[2]}/{args.k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
