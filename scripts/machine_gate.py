from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

_VALID_MODES = {"shared", "isolated"}
_DONE_STATES = {"done", "failed", "stale"}


def _default_db_path() -> Path:
    raw = os.environ.get("MACHINE_GATE_DB")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".cache" / "agent-gate" / "machine-gate.sqlite3"


def _now() -> float:
    return time.time()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class MachineGateError(RuntimeError):
    pass


@dataclass
class MachineGateLease:
    db_path: Path
    job_id: int
    mode: str
    label: str
    _released: bool = False
    _stop_event: threading.Event | None = None
    _heartbeat_thread: threading.Thread | None = None

    def release(self, *, exit_code: int = 0, state: str | None = None) -> None:
        if self._released:
            return
        self._released = True
        if self._stop_event is not None:
            self._stop_event.set()
        final_state = state or ("done" if exit_code == 0 else "failed")
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE jobs
                   SET state = ?, exit_code = ?, finished_at = ?, heartbeat_at = ?
                 WHERE id = ?
                """,
                (final_state, int(exit_code), _now(), _now(), self.job_id),
            )
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=0.5)

    def __enter__(self) -> "MachineGateLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release(exit_code=1 if exc_type is not None else 0)


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT NOT NULL CHECK (mode IN ('shared', 'isolated')),
            state TEXT NOT NULL,
            label TEXT NOT NULL,
            pid INTEGER NOT NULL,
            command TEXT,
            created_at REAL NOT NULL,
            started_at REAL,
            finished_at REAL,
            heartbeat_at REAL NOT NULL,
            exit_code INTEGER
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS jobs_state_id_idx ON jobs(state, id)")
    return conn


def _cleanup_stale(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT id, pid, state FROM jobs WHERE state IN ('waiting', 'running')"
    ).fetchall()
    t = _now()
    for job_id, pid, state in rows:
        if not _pid_alive(int(pid)):
            conn.execute(
                """
                UPDATE jobs
                   SET state = 'stale', finished_at = ?, heartbeat_at = ?
                 WHERE id = ? AND state = ?
                """,
                (t, t, int(job_id), str(state)),
            )


def _insert_waiting_job(
    conn: sqlite3.Connection,
    *,
    mode: str,
    label: str,
    command: str | None,
) -> int:
    t = _now()
    cur = conn.execute(
        """
        INSERT INTO jobs(mode, state, label, pid, command, created_at, heartbeat_at)
        VALUES (?, 'waiting', ?, ?, ?, ?, ?)
        """,
        (mode, label, os.getpid(), command, t, t),
    )
    return int(cur.lastrowid)


def _is_eligible(conn: sqlite3.Connection, *, job_id: int, mode: str) -> bool:
    running_isolated = conn.execute(
        "SELECT 1 FROM jobs WHERE state = 'running' AND mode = 'isolated' LIMIT 1"
    ).fetchone()
    if mode == "shared":
        if running_isolated:
            return False
        earlier_waiting_isolated = conn.execute(
            """
            SELECT 1 FROM jobs
             WHERE state = 'waiting' AND mode = 'isolated' AND id < ?
             LIMIT 1
            """,
            (job_id,),
        ).fetchone()
        return earlier_waiting_isolated is None

    running_any = conn.execute(
        "SELECT 1 FROM jobs WHERE state = 'running' LIMIT 1"
    ).fetchone()
    if running_any:
        return False
    earlier_waiting = conn.execute(
        "SELECT 1 FROM jobs WHERE state = 'waiting' AND id < ? LIMIT 1",
        (job_id,),
    ).fetchone()
    return earlier_waiting is None


def _start_heartbeat(db_path: Path, job_id: int, stop_event: threading.Event) -> threading.Thread:
    def beat() -> None:
        while not stop_event.wait(2.0):
            try:
                with _connect(db_path) as conn:
                    conn.execute(
                        "UPDATE jobs SET heartbeat_at = ? WHERE id = ? AND state IN ('waiting', 'running')",
                        (_now(), job_id),
                    )
            except Exception:
                # Heartbeat failures should not kill the protected command.
                pass

    thread = threading.Thread(target=beat, name=f"machine-gate-heartbeat-{job_id}", daemon=True)
    thread.start()
    return thread


def acquire(
    mode: str,
    *,
    label: str | None = None,
    command: list[str] | None = None,
    db_path: Path | None = None,
    quiet: bool = False,
) -> MachineGateLease:
    if os.environ.get("MACHINE_GATE_DISABLE") in {"1", "true", "yes", "on"}:
        return MachineGateLease(db_path or _default_db_path(), -1, mode, label or "disabled", _released=True)

    mode = mode.strip().lower()
    if mode not in _VALID_MODES:
        raise MachineGateError(f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}")

    db_path = db_path or _default_db_path()
    label = label or " ".join(command or []) or f"pid {os.getpid()}"
    command_text = " ".join(shlex.quote(part) for part in command) if command else None

    with _connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            _cleanup_stale(conn)
            job_id = _insert_waiting_job(conn, mode=mode, label=label, command=command_text)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    stop_event = threading.Event()
    heartbeat_thread = _start_heartbeat(db_path, job_id, stop_event)

    waited_notice = False
    start_wait = time.monotonic()
    while True:
        with _connect(db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _cleanup_stale(conn)
                if _is_eligible(conn, job_id=job_id, mode=mode):
                    t = _now()
                    conn.execute(
                        """
                        UPDATE jobs
                           SET state = 'running', started_at = ?, heartbeat_at = ?
                         WHERE id = ? AND state = 'waiting'
                        """,
                        (t, t, job_id),
                    )
                    conn.execute("COMMIT")
                    if not quiet:
                        waited = time.monotonic() - start_wait
                        print(
                            f"[machine-gate] acquired {mode} job #{job_id} after {waited:.1f}s: {label}",
                            file=sys.stderr,
                            flush=True,
                        )
                    return MachineGateLease(
                        db_path=db_path,
                        job_id=job_id,
                        mode=mode,
                        label=label,
                        _stop_event=stop_event,
                        _heartbeat_thread=heartbeat_thread,
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                stop_event.set()
                raise

        if not quiet and not waited_notice and time.monotonic() - start_wait >= 1.0:
            print(
                f"[machine-gate] waiting for {mode} job #{job_id}: {label}",
                file=sys.stderr,
                flush=True,
            )
            waited_notice = True
        time.sleep(0.15)


@contextlib.contextmanager
def machine_gate(
    mode: str | None,
    *,
    label: str | None = None,
    command: list[str] | None = None,
    quiet: bool = False,
) -> Iterator[MachineGateLease | None]:
    if mode is None or str(mode).lower() in {"", "off", "none", "0", "false", "no"}:
        yield None
        return
    lease = acquire(str(mode), label=label, command=command, quiet=quiet)
    try:
        yield lease
    except BaseException:
        lease.release(exit_code=1)
        raise
    else:
        lease.release(exit_code=0)


def _status(db_path: Path) -> int:
    with _connect(db_path) as conn:
        _cleanup_stale(conn)
        rows = conn.execute(
            """
            SELECT id, mode, state, pid, label, command, created_at, started_at, heartbeat_at
              FROM jobs
             WHERE state NOT IN ('done', 'failed', 'stale')
             ORDER BY id
            """
        ).fetchall()
    if not rows:
        print(f"machine-gate: idle ({db_path})")
        return 0
    print(f"machine-gate: active queue ({db_path})")
    for job_id, mode, state, pid, label, command, created_at, started_at, heartbeat_at in rows:
        age = _now() - float(created_at)
        started = "-" if started_at is None else f"{_now() - float(started_at):.1f}s ago"
        cmd = f" command={command}" if command else ""
        print(
            f"  #{job_id:<4} {state:<7} {mode:<8} pid={pid:<7} age={age:6.1f}s started={started} label={label}{cmd}"
        )
    return 0


def _cleanup(db_path: Path) -> int:
    with _connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            _cleanup_stale(conn)
            conn.execute(
                "DELETE FROM jobs WHERE state IN ('done', 'failed', 'stale') AND finished_at < ?",
                (_now() - 24 * 3600,),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return _status(db_path)


def _run_command(args: argparse.Namespace) -> int:
    if not args.command:
        raise MachineGateError("missing command after --")
    label = args.label or " ".join(shlex.quote(part) for part in args.command)
    lease = acquire(args.mode, label=label, command=args.command, quiet=args.quiet, db_path=args.db)
    proc: subprocess.Popen[bytes] | None = None

    def forward(signum: int, _frame) -> None:
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signum)
            except ProcessLookupError:
                pass

    old_int = signal.signal(signal.SIGINT, forward)
    old_term = signal.signal(signal.SIGTERM, forward)
    try:
        proc = subprocess.Popen(args.command)
        exit_code = proc.wait()
        lease.release(exit_code=exit_code)
        return int(exit_code)
    except BaseException:
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        lease.release(exit_code=1)
        raise
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fair local shared/isolated command gate")
    parser.add_argument("--db", type=Path, default=_default_db_path(), help="SQLite queue DB path")
    parser.add_argument("--quiet", action="store_true", help="Do not print wait/acquire messages")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run a command under the gate")
    run_p.add_argument("mode", choices=sorted(_VALID_MODES))
    run_p.add_argument("--label", default=None)
    run_p.add_argument("command", nargs=argparse.REMAINDER)

    status_p = sub.add_parser("status", help="show active queue")
    cleanup_p = sub.add_parser("cleanup", help="mark dead jobs stale and delete old completed rows")

    # Convenience: allow `machine_gate.py shared -- cargo test`.
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in _VALID_MODES:
        argv = ["run", argv[0], *argv[1:]]
    if "--" in argv:
        sep = argv.index("--")
        argv = [*argv[:sep], *argv[sep + 1:]]

    args = parser.parse_args(argv)
    if args.cmd == "run":
        return _run_command(args)
    if args.cmd == "status":
        return _status(args.db)
    if args.cmd == "cleanup":
        return _cleanup(args.db)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
