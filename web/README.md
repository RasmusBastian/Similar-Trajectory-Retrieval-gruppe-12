# Demo UI

Web page for the exam demo: compares a model's top-k similar trajectories with the ground truth on a map.

Requires Node.js 22+, Docker Desktop and the dump `beijing_20200_<date>.dump` from the team's Google Drive.

## Database (once)

Starts a separate database on port 5435 and loads the dump into it. Replace `C:\Users\you\Downloads`
with the folder that holds the dump, and wait ~10 seconds between the two commands:

```bash
docker run -d --name str-demo-db -p 5435:5432 -e POSTGRES_USER=str -e POSTGRES_PASSWORD=str -e POSTGRES_DB=trajectories -v str-demo-data:/var/lib/postgresql/data -v "C:\Users\you\Downloads:/dump:ro" postgis/postgis:17-3.5

docker exec str-demo-db sh -c "pg_restore --list /dump/beijing_20200_20261117.dump | grep -v -e 'EXTENSION - postgis' -e 'spatial_ref_sys' > /tmp/restore.list && pg_restore -U str -d trajectories --no-owner -L /tmp/restore.list /dump/beijing_20200_20261117.dump"
```

After a restart of the PC: `docker start str-demo-db`

## Run

```powershell
cd web
npm install
$env:PGPORT = "5435"     # macOS/Linux/Git Bash: export PGPORT=5435
npm start
```

Open <http://localhost:3000>. Run the tests with `npm test` (needs the database).
