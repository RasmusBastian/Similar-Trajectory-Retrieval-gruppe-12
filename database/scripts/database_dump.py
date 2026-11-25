"""Create or restore a portable custom-format dump of the project PostgreSQL database."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def docker_command(*args: str) -> list[str]:
    return ["docker", "compose", "exec", "-T", "db", *args]


def stream_to_process(path: Path, command: list[str], *, discard_output: bool = False) -> int:
    proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL if discard_output else None)
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                proc.stdin.write(chunk)
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    return proc.wait()


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    command = docker_command("pg_dump", "--format=custom", "--compress=6", "--no-owner", "--no-acl",
                             "-U", "str", "-d", "trajectories")
    print("Writing a compressed PostgreSQL dump to", path)
    try:
        with partial.open("wb") as target:
            result = subprocess.run(command, cwd=ROOT, stdout=target, check=False)
        if result.returncode:
            partial.unlink(missing_ok=True)
            return result.returncode
        os.replace(partial, path)

        verify = docker_command("pg_restore", "--list")
        print("Verifying the archive with pg_restore ...")
        if stream_to_process(path, verify, discard_output=True):
            path.unlink(missing_ok=True)
            raise RuntimeError("pg_restore could not read the generated archive")

        digest = checksum(path)
        sidecar = path.with_name(path.name + ".sha256")
        sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
        print(f"Dump ready: {path} ({path.stat().st_size / 1_000_000:.1f} MB)")
        print(f"SHA-256: {digest}")
        print(f"Checksum file: {sidecar}")
        return 0
    except FileNotFoundError as exc:
        partial.unlink(missing_ok=True)
        print(f"Could not run {exc.filename}; make sure Docker Desktop is running.", file=sys.stderr)
        return 127
    finally:
        partial.unlink(missing_ok=True)


def restore(path: Path) -> int:
    if not path.is_file():
        print(f"Dump not found: {path}", file=sys.stderr)
        return 2
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.exists():
        expected = sidecar.read_text(encoding="ascii").split()[0]
        actual = checksum(path)
        if actual.lower() != expected.lower():
            print("SHA-256 mismatch; refusing to restore this dump.", file=sys.stderr)
            return 2

    command = docker_command("pg_restore", "--exit-on-error", "--clean", "--if-exists",
                             "--no-owner", "--no-acl", "-U", "str", "-d", "trajectories")
    print("Restoring this dump will replace the contents of the local trajectories database.")
    return stream_to_process(path, command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    dump_parser = sub.add_parser("dump", help="write and verify a compressed database dump")
    dump_parser.add_argument("path", type=Path)
    restore_parser = sub.add_parser("restore", help="replace the local database from a verified dump")
    restore_parser.add_argument("path", type=Path)
    args = parser.parse_args()
    return dump(args.path.resolve()) if args.action == "dump" else restore(args.path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
