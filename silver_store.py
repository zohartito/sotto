"""Private, crash-safe storage for autonomous *silver* speech evidence.

This store deliberately has no dependency on :mod:`learning`: human consented
gold and machine pseudo-labels have different retention, clear, and evidence
rules.  Rows contain opaque ids and hashes except for an accepted normalized
reference.  In particular, teacher hypotheses and candidate hypotheses never
enter SQLite, logs, or retry records.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from comparator_spool import ComparatorSpool
from storage_lock import ensure_private_directory

SCHEMA_VERSION = 5
_REVOKE_OPERATIONS=frozenset({"revoke","delete","correct","retry","correct_as_is","no_speech","review"})


def _valid_revision(value: Any) -> bool:
    return value is None or (isinstance(value,int) and not isinstance(value,bool) and value >= 0)


def _valid_token(value: Any) -> bool:
    return isinstance(value,str) and len(value) == 32 and all(char in "0123456789abcdef" for char in value)


def _revoke_intent_key(value: dict[str, Any]) -> tuple[Any,...]:
    return (value["operation"],value["history_id"],value["history_revision"],value["epoch"],value["phase"])


def _revoke_intent_order(value: dict[str, Any]) -> tuple[Any,...]:
    return (value["operation"],value["history_id"],-1 if value["history_revision"] is None else value["history_revision"],
            value["epoch"],value["phase"],tuple(value["cohorts"]))


def _valid_revoke_intent(value: Any) -> bool:
    if not isinstance(value,dict) or set(value) != {"schema","kind","operation","history_id","history_revision","epoch","cohorts","phase","token"}:
        return False
    cohorts=value.get("cohorts")
    if not isinstance(cohorts,list) or not all(isinstance(item,str) and item for item in cohorts): return False
    return (value.get("schema") == 1 and value.get("kind") == "revoke" and value.get("operation") in _REVOKE_OPERATIONS and
            isinstance(value.get("history_id"),str) and bool(value["history_id"]) and _valid_revision(value.get("history_revision")) and
            isinstance(value.get("epoch"),int) and not isinstance(value["epoch"],bool) and value["epoch"] >= 0 and
            cohorts == sorted(set(cohorts)) and
            value.get("phase") == "silver_fenced" and _valid_token(value.get("token")))


def _valid_revoke_batch(value: Any) -> bool:
    if not isinstance(value,dict) or set(value) != {"schema","kind","intents","cohorts","phase","token"}:
        return False
    intents=value.get("intents"); cohorts=value.get("cohorts")
    if value.get("schema") != 2 or value.get("kind") != "revoke" or value.get("phase") != "silver_fenced" or not _valid_token(value.get("token")):
        return False
    if not isinstance(intents,list) or not intents or not all(_valid_revoke_intent(item) for item in intents): return False
    if intents != sorted(intents,key=_revoke_intent_order) or len({_revoke_intent_key(item) for item in intents}) != len(intents): return False
    if not isinstance(cohorts,list) or not all(isinstance(item,str) and item for item in cohorts): return False
    return cohorts == sorted(set(cohorts)) and cohorts == sorted({cohort for item in intents for cohort in item["cohorts"]})


def _valid_clear_marker(value: Any, *, epoch: int | None = None) -> bool:
    """Strict clear fence.  Never infer authority from a partial marker."""
    if not isinstance(value,dict) or set(value) != {"schema","kind","epoch","phase","token"}:
        return False
    marker_epoch=value.get("epoch")
    return (value.get("schema") == 1 and value.get("kind") == "clear" and
            value.get("phase") == "clear_fenced" and isinstance(marker_epoch,int) and
            not isinstance(marker_epoch,bool) and marker_epoch >= 0 and
            (epoch is None or marker_epoch == epoch) and _valid_token(value.get("token")))


def _valid_current_scrub(value: Any, *, epoch: int) -> bool:
    if _valid_clear_marker(value,epoch=epoch): return True
    if _valid_revoke_batch(value):
        return all(item["epoch"] == epoch for item in value["intents"])
    return _valid_revoke_intent(value) and value["epoch"] == epoch


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_only_status(base_dir: Path | str) -> dict[str, Any]:
    """Inspect an existing Silver ledger without constructing or migrating it.

    This operational path intentionally never calls :class:`SilverStore`:
    constructor-side directory creation, chmod, WAL recovery, or schema
    migration would make a supposedly read-only status command an authority
    mutation.  Malformed/unknown state is reported as blocked rather than
    guessed at.
    """
    base=Path(base_dir)
    path=base / "adaptive-learning" / "silver" / "silver.sqlite3"
    try:
        mode=os.lstat(path).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            return {"state":"blocked","reason":"unsafe_store_path"}
        db=sqlite3.connect(f"file:{path}?mode=ro",uri=True)
    except (OSError,sqlite3.Error):
        return {"state":"blocked","reason":"store_unavailable"}
    try:
        db.execute("PRAGMA query_only=ON")
        schema=db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        epoch=db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()
        if schema is None or schema[0] != str(SCHEMA_VERSION) or epoch is None:
            return {"state":"blocked","reason":"schema_unknown"}
        pending=db.execute("SELECT value FROM meta WHERE key='scrub_pending'").fetchone()
        worker=db.execute("SELECT value FROM meta WHERE key='worker_status'").fetchone()
        try:
            marker=json.loads(pending[0]) if pending is not None else None
            worker_value=json.loads(worker[0]) if worker is not None else None
        except (TypeError,ValueError,json.JSONDecodeError):
            return {"state":"blocked","reason":"metadata_malformed"}
        if pending is not None and not _valid_current_scrub(marker,epoch=int(epoch[0])):
            return {"state":"blocked","reason":"scrub_malformed"}
        jobs=dict(db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status").fetchall())
        labels=dict(db.execute("SELECT state,COUNT(*) FROM labels GROUP BY state").fetchall())
        return {"state":"ok","schema":SCHEMA_VERSION,"clear_epoch":int(epoch[0]),"jobs":jobs,"labels":labels,
                "worker":worker_value,"accepted":int(labels.get("accepted",0)),"abstained":int(labels.get("abstained",0)),
                "scrub_pending":pending is not None}
    except (sqlite3.Error,ValueError,TypeError):
        return {"state":"blocked","reason":"store_malformed"}
    finally:
        db.close()


class SilverStore:
    """A fail-closed SQLite outbox and immutable silver-label ledger.

    Mutating APIs reload through ``BEGIN IMMEDIATE`` and use expected epoch / a
    lease token as compare-and-swap predicates.  This lets a worker safely do
    expensive inference outside of all storage locks.
    """
    max_attempts = 4
    lease_seconds = 60 * 60

    @staticmethod
    def _runtime_evidence_fields(value: dict[str,Any]) -> tuple[int,int,float,int,int,int]:
        required={"success","fallback","latency","coverage_ok","hallucination_ok","identity_valid"}
        if not isinstance(value,dict) or set(value)!=required or any(not isinstance(value[key],bool) for key in required-{"latency"}): raise ValueError("invalid runtime evidence")
        if isinstance(value["latency"],bool) or not isinstance(value["latency"],(int,float)) or not math.isfinite(float(value["latency"])) or float(value["latency"])<0: raise ValueError("invalid runtime evidence")
        return (int(value['success']),int(value['fallback']),float(value['latency']),int(value['coverage_ok']),int(value['hallucination_ok']),int(value['identity_valid']))

    def _migration_checkpoint(self, phase: str) -> None:
        """Test-only failpoint seam; production leaves it unset."""
        hook=getattr(self,"_migration_failpoint",None)
        if hook is not None: hook(phase)

    def __init__(self, base_dir: Path | str, *, clock=time.time) -> None:
        self.base_dir = Path(base_dir)
        self.root = self.base_dir / "adaptive-learning" / "silver"
        self.path = self.root / "silver.sqlite3"
        self.clock = clock
        ensure_private_directory(self.base_dir)
        ensure_private_directory(self.base_dir / "adaptive-learning")
        ensure_private_directory(self.root)
        self._check_target(self.path, allow_missing=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA secure_delete=ON")
            self._migrate(db)
        self._tighten()

    def _check_target(self, path: Path, *, allow_missing: bool = False) -> None:
        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError:
            if allow_missing:
                return
            raise
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise RuntimeError("refusing unsafe silver SQLite target")

    def _tighten(self) -> None:
        ensure_private_directory(self.root)
        for item in (self.path, self.path.with_name(self.path.name + "-wal"), self.path.with_name(self.path.name + "-shm")):
            try:
                self._check_target(item)
                os.chmod(item, 0o600)
            except FileNotFoundError:
                continue

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._check_target(self.path, allow_missing=True)
        db = sqlite3.connect(self.path, isolation_level=None, timeout=15)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA secure_delete=ON")
            yield db
        finally:
            db.close()
            self._tighten()

    def _migrate(self, db: sqlite3.Connection) -> None:
        # Schema changes are one transaction.  In particular, never leave a
        # live authority DB between rename/create/copy/drop/ALTER phases.
        db.execute("BEGIN IMMEDIATE")
        db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        current = db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        migrate_comparator_v4=False
        # Version 1 keyed teacher jobs without the validated receipt.  Keeping
        # any terminal row from that schema would allow a later teacher build
        # to reuse it, so the only safe migration is to invalidate its private
        # evidence before recreating the ledger.
        if current is not None and current[0] == "1":
            for table in ("candidate_attempts", "prospective_members", "prospective_horizons",
                          "cohort_members", "cohorts", "attempts", "labels", "jobs",
                          "runtime_observations", "calibration_progress", "calibration_runs",
                          "calibration_items", "calibrations", "calibration_coverage"):
                db.execute(f"DROP TABLE IF EXISTS {table}")
            db.execute("UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION),))
        elif current is not None and current[0] == "2":
            try: db.execute("ALTER TABLE candidate_attempts ADD COLUMN not_before REAL NOT NULL DEFAULT 0")
            except sqlite3.OperationalError: pass
            db.execute("UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION),))
        elif current is not None and current[0] == "3":
            required={"jobs","labels","attempts","epochs","history_tombstones","alpha_ledger","cohort_looks","policy_attempts","cohorts","cohort_members","calibrations","calibration_items","calibration_runs","calibration_progress","calibration_coverage","calibration_universes","calibration_expected_v2","calibration_items_v2","prospective_horizons","prospective_members","candidate_attempts","runtime_observations","runtime_pairs","route_assignments","shadow_audits"}
            present={str(row[0]) for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not required <= present:
                raise RuntimeError("incomplete v3 schema; refusing migration")
        elif current is not None and current[0] == "4":
            columns={str(row[1]) for row in db.execute("PRAGMA table_info(comparator_intents)")}
            required={"deployment_revision","capture_id","session_hash","cohort_id","epoch","route_generation","candidate_arm","audio_sha256","spool_name","candidate_success","candidate_fallback","candidate_latency","candidate_coverage_ok","candidate_hallucination_ok","candidate_identity_valid","status","attempts","lease_owner","lease_epoch","lease_until","attempt_id","created_ts","updated_ts"}
            if columns == required:
                db.execute("ALTER TABLE comparator_intents RENAME TO comparator_intents_v4")
                migrate_comparator_v4=True
                self._migration_checkpoint("comparator_renamed")
            elif columns:
                # A present-but-different table is unknown data: refuse.
                raise RuntimeError("incomplete v4 comparator schema; refusing migration")
            # An absent table is a legitimate pre-comparator v4 store; there is
            # nothing to carry and the current-shape table is created below.
        elif current is not None and current[0] != str(SCHEMA_VERSION):
            raise RuntimeError("unknown silver schema; refusing to operate")
        self._migration_checkpoint("preflight")
        db.execute("INSERT INTO meta(key,value) VALUES ('schema_version',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(SCHEMA_VERSION),))
        db.execute("INSERT OR IGNORE INTO meta(key,value) VALUES ('clear_epoch','0')")
        db.execute("INSERT OR IGNORE INTO meta(key,value) VALUES ('quiescing','0')")
        db.execute("""CREATE TABLE IF NOT EXISTS jobs (
            job_key TEXT PRIMARY KEY, history_id TEXT NOT NULL, history_revision INTEGER NOT NULL,
            audio_sha256 TEXT NOT NULL, teacher_family_hash TEXT NOT NULL,
            consensus_policy_hash TEXT NOT NULL, clear_epoch INTEGER NOT NULL,
            status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, not_before REAL NOT NULL DEFAULT 0,
            lease_owner TEXT, lease_epoch INTEGER NOT NULL DEFAULT 0, lease_until REAL,
            receipt_hash TEXT NOT NULL, provenance_hash TEXT NOT NULL, created_ts REAL NOT NULL,
            updated_ts REAL NOT NULL, CHECK(status IN ('pending','leased','terminal','discarded')),
            UNIQUE(history_id,history_revision,audio_sha256,teacher_family_hash,consensus_policy_hash,receipt_hash)
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS labels (
            job_key TEXT PRIMARY KEY REFERENCES jobs(job_key) ON DELETE CASCADE,
            state TEXT NOT NULL, reference TEXT, reference_digest TEXT NOT NULL,
            vote_digest TEXT NOT NULL, policy_hash TEXT NOT NULL, receipt_hash TEXT NOT NULL,
            evidence_revision INTEGER NOT NULL, created_ts REAL NOT NULL, revoked_ts REAL,
            CHECK(state IN ('accepted','abstained','superseded','revoked'))
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS attempts (
            attempt_id TEXT PRIMARY KEY, job_key TEXT NOT NULL REFERENCES jobs(job_key) ON DELETE CASCADE,
            owner TEXT NOT NULL, lease_epoch INTEGER NOT NULL, started_ts REAL NOT NULL,
            finished_ts REAL, outcome TEXT, code TEXT, UNIQUE(job_key,owner,lease_epoch)
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS epochs (
            clear_epoch INTEGER PRIMARY KEY, reason TEXT NOT NULL, ts REAL NOT NULL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS history_tombstones (
            history_id TEXT NOT NULL, history_revision INTEGER NOT NULL,
            clear_epoch INTEGER NOT NULL, reason TEXT NOT NULL, ts REAL NOT NULL,
            PRIMARY KEY(history_id,history_revision,clear_epoch)
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS alpha_ledger (
            generation INTEGER PRIMARY KEY, alpha REAL NOT NULL, created_ts REAL NOT NULL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS cohort_looks (
            cohort_id TEXT PRIMARY KEY REFERENCES cohorts(cohort_id) ON DELETE CASCADE,
            generation INTEGER NOT NULL UNIQUE, alpha REAL NOT NULL, identity_hash TEXT NOT NULL,
            created_ts REAL NOT NULL
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS policy_attempts (
            kind TEXT NOT NULL, policy_hash TEXT NOT NULL, split_hash TEXT NOT NULL,
            status TEXT NOT NULL, created_ts REAL NOT NULL, PRIMARY KEY(kind,policy_hash,split_hash)
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS cohorts (
            cohort_id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
            member_hash TEXT NOT NULL, comparison_hash TEXT NOT NULL, metadata TEXT NOT NULL,
            clear_epoch INTEGER NOT NULL, created_ts REAL NOT NULL
        )""")
        # A successful evaluation is replayable exactly until its first
        # successful authorization.  Once a rollout has been authorized then
        # invalidated/rolled back, its terminal cohort is consumed forever;
        # a later shadow state must not resurrect it on worker restart.
        db.execute("""CREATE TABLE IF NOT EXISTS cohort_authorization (
            cohort_id TEXT PRIMARY KEY REFERENCES cohorts(cohort_id) ON DELETE CASCADE,
            disposition TEXT NOT NULL CHECK(disposition IN ('authorizing','authorized','consumed')),
            expected_revision INTEGER, expected_state_digest TEXT, token TEXT,
            updated_ts REAL NOT NULL
        )""")
        # SQLite cannot widen a CHECK constraint in place.  Preserve the
        # content-free rows while upgrading pre-intent installations.
        schema_row=db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='cohort_authorization'").fetchone()
        if schema_row is not None and "authorizing" not in str(schema_row[0]):
            db.execute("ALTER TABLE cohort_authorization RENAME TO cohort_authorization_old")
            db.execute("""CREATE TABLE cohort_authorization (
                cohort_id TEXT PRIMARY KEY REFERENCES cohorts(cohort_id) ON DELETE CASCADE,
                disposition TEXT NOT NULL CHECK(disposition IN ('authorizing','authorized','consumed')),
                expected_revision INTEGER, expected_state_digest TEXT, token TEXT,
                updated_ts REAL NOT NULL
            )""")
            db.execute("INSERT INTO cohort_authorization(cohort_id,disposition,updated_ts) SELECT cohort_id,disposition,updated_ts FROM cohort_authorization_old")
            db.execute("DROP TABLE cohort_authorization_old")
        for column, definition in (("expected_revision","INTEGER"),("expected_state_digest","TEXT"),("token","TEXT")):
            try: db.execute(f"ALTER TABLE cohort_authorization ADD COLUMN {column} {definition}")
            except sqlite3.OperationalError: pass
        self._migration_checkpoint("authority_table")
        db.execute("""CREATE TABLE IF NOT EXISTS cohort_members (
            cohort_id TEXT NOT NULL REFERENCES cohorts(cohort_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL, history_id TEXT NOT NULL, history_revision INTEGER NOT NULL,
            label_state TEXT NOT NULL, audio_sha256 TEXT NOT NULL, reference_digest TEXT NOT NULL,
            cluster_hash TEXT NOT NULL,
            PRIMARY KEY(cohort_id,ordinal), UNIQUE(cohort_id,history_id,history_revision)
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS calibrations (
            calibration_id TEXT PRIMARY KEY, manifest_hash TEXT NOT NULL, source_hash TEXT NOT NULL,
            policy_hash TEXT NOT NULL, split_hash TEXT NOT NULL, receipt_hash TEXT NOT NULL,
            clear_epoch INTEGER NOT NULL, state TEXT NOT NULL, accepted INTEGER NOT NULL,
            reference_errors INTEGER NOT NULL, clusters INTEGER NOT NULL, created_ts REAL NOT NULL,
            UNIQUE(policy_hash,split_hash)
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS calibration_items (
            calibration_id TEXT NOT NULL REFERENCES calibrations(calibration_id) ON DELETE CASCADE,
            item_hash TEXT NOT NULL, cluster_hash TEXT NOT NULL, outcome TEXT NOT NULL, reference_match INTEGER NOT NULL,
            PRIMARY KEY(calibration_id,item_hash)
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS calibration_runs (
            manifest_hash TEXT PRIMARY KEY, policy_hash TEXT NOT NULL, split_hash TEXT NOT NULL,
            source_hash TEXT NOT NULL, receipt_hash TEXT NOT NULL, clear_epoch INTEGER NOT NULL,
            state TEXT NOT NULL, created_ts REAL NOT NULL, updated_ts REAL NOT NULL,
            UNIQUE(policy_hash,split_hash)
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS calibration_progress (
            manifest_hash TEXT NOT NULL REFERENCES calibration_runs(manifest_hash) ON DELETE CASCADE,
            item_hash TEXT NOT NULL, cluster_hash TEXT NOT NULL, outcome TEXT NOT NULL,
            reference_match INTEGER NOT NULL, PRIMARY KEY(manifest_hash,item_hash)
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS calibration_coverage (
            manifest_hash TEXT PRIMARY KEY, total INTEGER NOT NULL, terminal INTEGER NOT NULL,
            accepted INTEGER NOT NULL, created_ts REAL NOT NULL
        )""")
        # V2 calibration is a public, one-shot source universe.  It is kept
        # outside personal clear epochs and stores only opaque item/cluster
        # identities plus terminal codes, never references or teacher text.
        db.execute("""CREATE TABLE IF NOT EXISTS calibration_universes (
            holdout_key TEXT PRIMARY KEY, manifest_hash TEXT NOT NULL, source_hash TEXT NOT NULL,
            protocol_hash TEXT NOT NULL, policy_hash TEXT NOT NULL, receipt_hash TEXT NOT NULL,
            state TEXT NOT NULL, created_ts REAL NOT NULL, updated_ts REAL NOT NULL
        )""")
        try: db.execute("ALTER TABLE calibration_universes ADD COLUMN result TEXT")
        except sqlite3.OperationalError: pass
        db.execute("""CREATE TABLE IF NOT EXISTS calibration_expected_v2 (
            holdout_key TEXT NOT NULL REFERENCES calibration_universes(holdout_key) ON DELETE RESTRICT,
            ordinal INTEGER NOT NULL, item_hash TEXT NOT NULL, cluster_hash TEXT NOT NULL,
            PRIMARY KEY(holdout_key,ordinal), UNIQUE(holdout_key,item_hash)
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS calibration_items_v2 (
            holdout_key TEXT NOT NULL REFERENCES calibration_universes(holdout_key) ON DELETE RESTRICT,
            item_hash TEXT NOT NULL, state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            lease_owner TEXT, lease_token TEXT, lease_until REAL,
            outcome TEXT, reference_match INTEGER, code TEXT, updated_ts REAL NOT NULL,
            PRIMARY KEY(holdout_key,item_hash), CHECK(state IN ('pending','leased','terminal')),
            CHECK(reference_match IS NULL OR reference_match IN (0,1))
        )""")
        # A prospective horizon is opened *before* its first teacher job.  It
        # therefore cannot be reconstructed from an appealing historical set
        # of shadow labels after a candidate has been chosen.
        db.execute("""CREATE TABLE IF NOT EXISTS prospective_horizons (
            horizon_id TEXT PRIMARY KEY, status TEXT NOT NULL, start_ts REAL NOT NULL,
            close_ts REAL NOT NULL, champion TEXT NOT NULL, candidates TEXT NOT NULL,
            receipt_hash TEXT NOT NULL, identity_hash TEXT NOT NULL, clear_epoch INTEGER NOT NULL,
            created_ts REAL NOT NULL, frozen_cohort_id TEXT
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS prospective_members (
            horizon_id TEXT NOT NULL REFERENCES prospective_horizons(horizon_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL, history_id TEXT NOT NULL, history_revision INTEGER NOT NULL,
            audio_sha256 TEXT NOT NULL, captured_ts REAL NOT NULL, cluster_hash TEXT NOT NULL,
            PRIMARY KEY(horizon_id,ordinal), UNIQUE(horizon_id,history_id,history_revision)
        )""")
        # Statistics only: candidate/champion hypotheses and reference text
        # never enter this table.
        db.execute("""CREATE TABLE IF NOT EXISTS candidate_attempts (
            cohort_id TEXT NOT NULL REFERENCES cohorts(cohort_id) ON DELETE CASCADE,
            arm_id TEXT NOT NULL, history_id TEXT NOT NULL, history_revision INTEGER NOT NULL,
            audio_sha256 TEXT NOT NULL, reference_digest TEXT NOT NULL, cluster_hash TEXT NOT NULL,
            status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            lease_owner TEXT, lease_epoch INTEGER NOT NULL DEFAULT 0, lease_until REAL, attempt_id TEXT,
            stats TEXT, code TEXT, not_before REAL NOT NULL DEFAULT 0, updated_ts REAL NOT NULL,
            PRIMARY KEY(cohort_id,arm_id,history_id,history_revision),
            CHECK(status IN ('pending','leased','complete','failed','discarded'))
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS runtime_observations (
            deployment_revision INTEGER NOT NULL, session_hash TEXT NOT NULL,
            arm_id TEXT NOT NULL, ts REAL NOT NULL, day INTEGER NOT NULL,
            success INTEGER NOT NULL, fallback INTEGER NOT NULL, latency REAL NOT NULL,
            coverage_ok INTEGER NOT NULL, hallucination_ok INTEGER NOT NULL,
            identity_valid INTEGER NOT NULL, PRIMARY KEY(deployment_revision,session_hash,ts,arm_id)
        )""")
        # One capture is a single paired canary datum.  Keeping both arms in
        # one immutable row prevents a crash/replay from turning hundreds of
        # candidate calls and one control call into a superficially complete
        # comparator.  It is deliberately content-free.
        db.execute("""CREATE TABLE IF NOT EXISTS runtime_pairs (
            deployment_revision INTEGER NOT NULL, capture_id TEXT NOT NULL,
            session_hash TEXT NOT NULL, cohort_id TEXT NOT NULL, epoch INTEGER NOT NULL,
            candidate_arm TEXT NOT NULL, candidate_success INTEGER NOT NULL,
            candidate_fallback INTEGER NOT NULL, candidate_latency REAL NOT NULL,
            candidate_coverage_ok INTEGER NOT NULL, candidate_hallucination_ok INTEGER NOT NULL,
            candidate_identity_valid INTEGER NOT NULL,
            incumbent_success INTEGER NOT NULL, incumbent_fallback INTEGER NOT NULL,
            incumbent_latency REAL NOT NULL, incumbent_coverage_ok INTEGER NOT NULL,
            incumbent_hallucination_ok INTEGER NOT NULL, incumbent_identity_valid INTEGER NOT NULL,
            ts REAL NOT NULL, day INTEGER NOT NULL, state TEXT NOT NULL DEFAULT 'complete',
            PRIMARY KEY(deployment_revision,capture_id)
        )""")
        try: db.execute("ALTER TABLE runtime_pairs ADD COLUMN state TEXT NOT NULL DEFAULT 'complete'")
        except sqlite3.OperationalError: pass
        db.execute("""CREATE TABLE IF NOT EXISTS comparator_intents (
            deployment_revision INTEGER NOT NULL, capture_id TEXT NOT NULL,
            session_hash TEXT NOT NULL, cohort_id TEXT NOT NULL, epoch INTEGER NOT NULL,
            route_generation INTEGER NOT NULL, candidate_arm TEXT NOT NULL,
            audio_sha256 TEXT NOT NULL, spool_name TEXT NOT NULL,
            candidate_success INTEGER NOT NULL, candidate_fallback INTEGER NOT NULL, candidate_latency REAL NOT NULL,
            candidate_coverage_ok INTEGER NOT NULL, candidate_hallucination_ok INTEGER NOT NULL, candidate_identity_valid INTEGER NOT NULL,
            status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            lease_owner TEXT, lease_epoch INTEGER NOT NULL DEFAULT 0, lease_until REAL, attempt_id TEXT,
            created_ts REAL NOT NULL, updated_ts REAL NOT NULL,
            PRIMARY KEY(deployment_revision,capture_id), CHECK(status IN ('prepared','pending','leased','complete','discarded'))
        )""")
        if migrate_comparator_v4:
            self._migration_checkpoint("comparator_created")
            db.execute("INSERT INTO comparator_intents SELECT * FROM comparator_intents_v4")
            self._migration_checkpoint("comparator_copied")
            db.execute("DROP TABLE comparator_intents_v4")
            self._migration_checkpoint("comparator_dropped")
        db.execute("""CREATE TABLE IF NOT EXISTS route_assignments (
            route_generation INTEGER NOT NULL, session_hash TEXT NOT NULL,
            selected INTEGER NOT NULL, created_ts REAL NOT NULL,
            PRIMARY KEY(route_generation,session_hash)
        )""")
        db.execute("""CREATE TABLE IF NOT EXISTS shadow_audits (
            job_key TEXT PRIMARY KEY REFERENCES jobs(job_key) ON DELETE CASCADE,
            word_distance INTEGER NOT NULL, reference_words INTEGER NOT NULL,
            character_distance INTEGER NOT NULL, reference_characters INTEGER NOT NULL,
            preboundary INTEGER NOT NULL, created_ts REAL NOT NULL
        )""")
        self._migration_checkpoint("finalize")
        db.commit()

    @staticmethod
    def job_key(history_id: str, history_revision: int, audio_sha256: str,
                teacher_family_hash: str, consensus_policy_hash: str, receipt_hash: str = "") -> str:
        return _digest([history_id, history_revision, audio_sha256, teacher_family_hash, consensus_policy_hash, receipt_hash])

    def epoch(self) -> int:
        with self._connect() as db:
            return int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])

    def active_session_id(self, *, inactivity_seconds: float = 30 * 60) -> str:
        """Return a durable content-free dictation session, rotating on idle.

        This lives beside the evidence ledger so a process restart inside the
        gap retains the same session for sticky routing and clustered gates.
        """
        now=self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT value FROM meta WHERE key='dictation_session'").fetchone()
            value={}
            try: value=json.loads(row[0]) if row else {}
            except (TypeError, ValueError, json.JSONDecodeError): value={}
            if not isinstance(value,dict) or now-float(value.get("last_ts",0)) > inactivity_seconds or not isinstance(value.get("id"),str):
                value={"id":uuid.uuid4().hex,"last_ts":now}
            else: value["last_ts"]=now
            db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('dictation_session',?)", (json.dumps(value,sort_keys=True),))
            db.commit(); return value["id"]

    def route_assignment(self, *, route_generation: int, session_id: str, percent: int) -> bool:
        """Persist a rollout arm once per active session/generation."""
        if not 0 <= int(percent) <= 100: raise ValueError("invalid rollout percent")
        session_hash=hashlib.sha256(session_id.encode()).hexdigest()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT selected FROM route_assignments WHERE route_generation=? AND session_hash=?",(int(route_generation),session_hash)).fetchone()
            if row is not None:
                db.commit(); return bool(row[0])
            selected=int(hashlib.sha256(session_id.encode()).hexdigest()[:8],16)%100 < int(percent)
            db.execute("INSERT INTO route_assignments(route_generation,session_hash,selected,created_ts) VALUES(?,?,?,?)",(int(route_generation),session_hash,int(selected),self.clock()))
            db.commit(); return selected

    def has_route_assignment(self, *, route_generation: int, session_id: str) -> bool:
        with self._connect() as db:
            row=db.execute("SELECT selected FROM route_assignments WHERE route_generation=? AND session_hash=?",(int(route_generation),hashlib.sha256(session_id.encode()).hexdigest())).fetchone()
            return bool(row and row[0])

    def has_route_assignment_hash(self, *, route_generation: int, session_hash: str) -> bool:
        with self._connect() as db:
            row=db.execute("SELECT selected FROM route_assignments WHERE route_generation=? AND session_hash=?",(int(route_generation),str(session_hash))).fetchone()
            return bool(row and row[0])

    def enqueue(self, *, history_id: str, history_revision: int, audio_sha256: str,
                teacher_family_hash: str, consensus_policy_hash: str, receipt_hash: str,
                provenance_hash: str = "", expected_epoch: int | None = None) -> str | None:
        key = self.job_key(history_id, history_revision, audio_sha256, teacher_family_hash, consensus_policy_hash, receipt_hash)
        now = self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT value FROM meta WHERE key='quiescing'").fetchone()[0] != "0":
                db.rollback(); raise RuntimeError("silver store is clearing")
            epoch = int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            # A scanner can have read an old marker before a Clear commits its
            # epoch fence.  Never reinterpret that retained history row as a
            # new-epoch job while cross-store deletion is still pending.
            if db.execute("SELECT 1 FROM meta WHERE key='scrub_pending'").fetchone() is not None:
                db.rollback(); return None
            if expected_epoch is not None and epoch != int(expected_epoch):
                db.rollback(); return None
            if db.execute("SELECT 1 FROM history_tombstones WHERE history_id=? AND history_revision=? AND clear_epoch=?",(history_id,int(history_revision),epoch)).fetchone() is not None:
                db.rollback(); return None
            db.execute("""INSERT OR IGNORE INTO jobs(job_key,history_id,history_revision,audio_sha256,teacher_family_hash,
                consensus_policy_hash,clear_epoch,status,receipt_hash,provenance_hash,created_ts,updated_ts)
                VALUES(?,?,?,?,?,?,?,'pending',?,?,?,?)""",
                (key, history_id, int(history_revision), audio_sha256, teacher_family_hash, consensus_policy_hash,
                 epoch, receipt_hash, provenance_hash, now, now))
            db.commit()
        return key

    def recover(self) -> int:
        """Release expired leases once, preserving bounded attempts and epoch."""
        now = self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT job_key FROM jobs WHERE status='leased' AND lease_until<?", (now,)).fetchall()
            for (key,) in rows:
                db.execute("UPDATE jobs SET status='pending',lease_owner=NULL,lease_until=NULL,updated_ts=? WHERE job_key=?", (now, key))
            # Retry exhaustion is terminal abstention-like operational state;
            # it must not spin forever or reappear after a service restart.
            db.execute("UPDATE jobs SET status='discarded',updated_ts=? WHERE status='pending' AND attempts>=?", (now, self.max_attempts))
            db.execute("UPDATE prospective_horizons SET status='invalidated' WHERE status IN ('collecting','closed','frozen') AND EXISTS (SELECT 1 FROM prospective_members m JOIN jobs j ON j.history_id=m.history_id AND j.history_revision=m.history_revision AND j.audio_sha256=m.audio_sha256 WHERE m.horizon_id=prospective_horizons.horizon_id AND j.status='discarded')")
            db.execute("UPDATE candidate_attempts SET status=CASE WHEN attempts>=? THEN 'failed' ELSE 'pending' END,lease_owner=NULL,lease_until=NULL,attempt_id=NULL,updated_ts=? WHERE status='leased' AND lease_until<?", (self.max_attempts,now,now))
            comparator_rows=db.execute("SELECT deployment_revision,capture_id,attempts FROM comparator_intents WHERE status='leased' AND lease_until<?",(now,)).fetchall()
            for revision,capture,attempts in comparator_rows:
                status='discarded' if int(attempts) >= self.max_attempts else 'pending'
                db.execute("UPDATE comparator_intents SET status=?,lease_owner=NULL,lease_until=NULL,attempt_id=NULL,updated_ts=? WHERE deployment_revision=? AND capture_id=? AND status='leased' AND lease_until<?",(status,now,revision,capture,now))
            # A prepared row was never foreground-published.  Recovery makes
            # that crash disposition terminal so workers cannot infer delivery.
            expired_prepared=db.execute("""SELECT deployment_revision,capture_id FROM comparator_intents
                WHERE status='prepared' AND created_ts<? AND NOT EXISTS(
                    SELECT 1 FROM meta WHERE key=('comparator_adoption:' || comparator_intents.deployment_revision || ':' || comparator_intents.capture_id))""",(now-self.lease_seconds,)).fetchall()
            db.execute("""UPDATE comparator_intents SET status='discarded',updated_ts=?
                WHERE status='prepared' AND created_ts<? AND NOT EXISTS(
                    SELECT 1 FROM meta WHERE key=('comparator_adoption:' || comparator_intents.deployment_revision || ':' || comparator_intents.capture_id))""",(now,now-self.lease_seconds))
            for revision,capture in expired_prepared:
                db.execute("DELETE FROM meta WHERE key=?",(self._comparator_adoption_key(int(revision),str(capture)),))
            db.execute("UPDATE comparator_intents SET status='discarded',lease_owner=NULL,lease_until=NULL,attempt_id=NULL,updated_ts=? WHERE status='pending' AND attempts>=?",(now,self.max_attempts))
            db.commit()
            return len(rows)+len(comparator_rows)

    def claim(self, owner: str) -> dict[str, Any] | None:
        now = self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT value FROM meta WHERE key='quiescing'").fetchone()[0] != "0":
                db.rollback(); return None
            if db.execute("SELECT 1 FROM meta WHERE key='scrub_pending'").fetchone() is not None:
                db.rollback(); return None
            row = db.execute("""SELECT * FROM jobs WHERE status='pending' AND attempts<? AND not_before<=?
                              ORDER BY created_ts,job_key LIMIT 1""", (self.max_attempts, now)).fetchone()
            if row is None:
                db.commit(); return None
            names = [x[0] for x in db.execute("SELECT * FROM jobs LIMIT 0").description]
            job = dict(zip(names, row)); lease_epoch = int(job["lease_epoch"]) + 1
            attempt_id = uuid.uuid4().hex
            updated = db.execute("""UPDATE jobs SET status='leased',attempts=attempts+1,lease_owner=?,lease_epoch=?,lease_until=?,updated_ts=?
                          WHERE job_key=? AND status='pending'""", (owner, lease_epoch, now + self.lease_seconds, now, job["job_key"]))
            if updated.rowcount != 1:
                db.rollback(); return None
            db.execute("INSERT INTO attempts(attempt_id,job_key,owner,lease_epoch,started_ts) VALUES(?,?,?,?,?)",
                       (attempt_id, job["job_key"], owner, lease_epoch, now))
            db.commit()
            job.update({"lease_owner": owner, "lease_epoch": lease_epoch, "attempt_id": attempt_id,
                        "clear_epoch": int(job["clear_epoch"]), "attempts": int(job["attempts"]) + 1})
            return job

    @staticmethod
    def _shadow_statistics(value: dict[str, Any] | None, captured_ts: float | None) -> tuple[int,int,int,int,float] | None:
        """Validate the content-free, non-authority old-capture aggregate.

        Text is deliberately not accepted here.  The caller has already
        compared the immutable history hypothesis with consensus in memory;
        this boundary accepts only the four sufficient statistics and capture
        time required to decide whether that comparison predates a frozen
        prospective horizon.
        """
        if value is None:
            return None
        if not isinstance(value,dict) or set(value) != {"word_distance","reference_words","character_distance","reference_characters"}:
            raise ValueError("invalid shadow sufficient statistics")
        numbers=[]
        for key in ("word_distance","reference_words","character_distance","reference_characters"):
            item=value[key]
            if isinstance(item,bool) or not isinstance(item,int) or item < 0:
                raise ValueError("invalid shadow sufficient statistics")
            numbers.append(item)
        if numbers[1] <= 0 or numbers[3] <= 0 or isinstance(captured_ts,bool) or not isinstance(captured_ts,(int,float)) or not math.isfinite(float(captured_ts)):
            raise ValueError("invalid shadow sufficient statistics")
        return (*numbers,float(captured_ts))

    @staticmethod
    def _scrub_unneeded_references(db: sqlite3.Connection, epoch: int, now: float) -> int:
        """Drop plaintext labels once no unfinished exact consumer exists.

        The immutable digest in ``labels`` and the frozen member ledger are
        sufficient after this point.  A reference remains readable only while
        it belongs to an active prospective horizon or an unfinished frozen
        candidate attempt in the same clear epoch.
        """
        rows = db.execute("""SELECT l.job_key FROM labels l
            JOIN jobs j USING(job_key)
            WHERE l.state='accepted' AND l.reference IS NOT NULL AND j.clear_epoch=?
              AND NOT EXISTS(
                SELECT 1 FROM prospective_members pm
                JOIN prospective_horizons ph ON ph.horizon_id=pm.horizon_id
                WHERE pm.history_id=j.history_id AND pm.history_revision=j.history_revision
                  AND pm.audio_sha256=j.audio_sha256 AND ph.clear_epoch=j.clear_epoch
                  AND ph.status IN ('collecting','closed','frozen'))
              AND NOT EXISTS(
                SELECT 1 FROM cohort_members cm
                JOIN cohorts c ON c.cohort_id=cm.cohort_id
                JOIN candidate_attempts a ON a.cohort_id=cm.cohort_id
                  AND a.history_id=cm.history_id AND a.history_revision=cm.history_revision
                  AND a.audio_sha256=cm.audio_sha256
                WHERE cm.history_id=j.history_id AND cm.history_revision=j.history_revision
                  AND cm.audio_sha256=j.audio_sha256 AND c.clear_epoch=j.clear_epoch
                  AND c.status IN ('frozen','evaluating') AND a.status IN ('pending','leased'))""", (epoch,)).fetchall()
        for (job_key,) in rows:
            db.execute("UPDATE labels SET reference=NULL WHERE job_key=? AND reference IS NOT NULL", (job_key,))
        return len(rows)

    def finish(self, job: dict[str, Any], *, outcome: str, reference: str | None = None,
               vote_digest: str = "", code: str | None = None,
               shadow_audit: dict[str, Any] | None = None, captured_ts: float | None = None) -> bool:
        """CAS-finalize a lease. ``reference`` is accepted only for silver evidence."""
        if outcome not in {"accepted", "abstained", "retry", "discarded"}:
            raise ValueError("invalid silver outcome")
        if outcome == "accepted" and not isinstance(reference, str):
            raise ValueError("accepted silver requires canonical reference")
        if shadow_audit is not None and outcome != "accepted":
            raise ValueError("shadow audit requires accepted silver")
        shadow=self._shadow_statistics(shadow_audit,captured_ts)
        now = self.clock(); key = str(job["job_key"]); owner = str(job["lease_owner"]); lease = int(job["lease_epoch"])
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            epoch = int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            found = db.execute("""SELECT clear_epoch,status,lease_until FROM jobs WHERE job_key=? AND lease_owner=? AND lease_epoch=?
                AND EXISTS(SELECT 1 FROM attempts WHERE attempt_id=? AND job_key=jobs.job_key AND owner=? AND lease_epoch=? AND finished_ts IS NULL)""", (key, owner, lease,job.get("attempt_id"),owner,lease)).fetchone()
            if (found is None or found[0] != epoch or int(job["clear_epoch"]) != epoch or found[1] != "leased" or
                    found[2] is None or float(found[2]) < now or db.execute("SELECT 1 FROM meta WHERE key='scrub_pending'").fetchone() is not None):
                db.rollback(); return False
            db.execute("UPDATE attempts SET finished_ts=?,outcome=?,code=? WHERE attempt_id=?", (now, outcome, code, job["attempt_id"]))
            if outcome in {"accepted", "abstained"}:
                db.execute("UPDATE jobs SET status='terminal',lease_owner=NULL,lease_until=NULL,updated_ts=? WHERE job_key=?", (now, key))
                db.execute("""INSERT INTO labels(job_key,state,reference,reference_digest,vote_digest,policy_hash,receipt_hash,evidence_revision,created_ts)
                    VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(job_key) DO NOTHING""",
                    (key, outcome, reference if outcome == "accepted" else None, _digest(reference or ""), vote_digest,
                     job["consensus_policy_hash"], job["receipt_hash"], int(job["history_revision"]), now))
                # Old-capture diagnostics are non-authority data.  They may
                # exist only for a capture strictly older than the earliest
                # prospective boundary in this clear epoch, and are inserted
                # in the same transaction as the accepted teacher label.
                if outcome == "accepted" and shadow is not None:
                    start=db.execute("SELECT MIN(start_ts) FROM prospective_horizons WHERE clear_epoch=?",(epoch,)).fetchone()[0]
                    if start is not None and shadow[4] < float(start):
                        values=shadow[:4]
                        db.execute("INSERT INTO shadow_audits VALUES(?,?,?,?,?,?,?)",(key,*values,1,now))
            elif outcome == "retry":
                delay = min(3600.0, 15.0 * (2 ** max(0, int(job["attempts"]) - 1)))
                attempts=max(0,int(job["attempts"])-(1 if code in {"scheduler_busy","live_preempted"} else 0))
                if attempts >= self.max_attempts:
                    db.execute("UPDATE jobs SET status='discarded',attempts=?,lease_owner=NULL,lease_until=NULL,updated_ts=? WHERE job_key=?", (attempts,now,key))
                    db.execute("UPDATE prospective_horizons SET status='invalidated' WHERE status IN ('collecting','closed','frozen') AND EXISTS (SELECT 1 FROM prospective_members WHERE horizon_id=prospective_horizons.horizon_id AND history_id=? AND history_revision=? AND audio_sha256=?)", (job["history_id"],int(job["history_revision"]),job["audio_sha256"]))
                else:
                    db.execute("UPDATE jobs SET status='pending',attempts=?,lease_owner=NULL,lease_until=NULL,not_before=?,updated_ts=? WHERE job_key=?", (attempts,now + delay, now, key))
            else:
                db.execute("UPDATE jobs SET status='discarded',lease_owner=NULL,lease_until=NULL,updated_ts=? WHERE job_key=?", (now, key))
                db.execute("UPDATE prospective_horizons SET status='invalidated' WHERE status IN ('collecting','closed','frozen') AND EXISTS (SELECT 1 FROM prospective_members WHERE horizon_id=prospective_horizons.horizon_id AND history_id=? AND history_revision=? AND audio_sha256=?)", (job["history_id"],int(job["history_revision"]),job["audio_sha256"]))
            self._scrub_unneeded_references(db, epoch, now)
            db.commit(); return True

    def renew(self, job: dict[str, Any]) -> bool:
        """Extend an owned lease without accepting a stale clear epoch."""
        now = self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute("""UPDATE jobs SET lease_until=?,updated_ts=? WHERE job_key=? AND status='leased'
                                  AND lease_owner=? AND lease_epoch=? AND lease_until>=? AND clear_epoch=(SELECT value FROM meta WHERE key='clear_epoch')
                                  AND EXISTS(SELECT 1 FROM attempts WHERE attempt_id=? AND job_key=jobs.job_key AND owner=? AND lease_epoch=? AND finished_ts IS NULL)
                                  AND NOT EXISTS(SELECT 1 FROM meta WHERE key='scrub_pending')""",
                                 (now + self.lease_seconds, now, job["job_key"], job["lease_owner"], job["lease_epoch"],now,job.get("attempt_id"),job["lease_owner"],job["lease_epoch"]))
            if changed.rowcount != 1: db.rollback(); return False
            epoch=int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            self._scrub_unneeded_references(db, epoch, now)
            db.commit(); return True

    def _revoke_history_mutation(self, db: sqlite3.Connection, history_id: str, *,
                                 history_revision: int | None, reason: str,
                                 operation: str | None = None) -> tuple[set[str], int, dict[str, Any] | None]:
        """Apply the one-transaction history fence and optionally stage its intent."""
        if history_revision is not None and (isinstance(history_revision,bool) or not isinstance(history_revision,int) or history_revision < 0):
            raise ValueError("invalid history revision")
        now=self.clock()
        dependencies={str(row[0]) for row in db.execute("SELECT DISTINCT cohort_id FROM cohort_members WHERE history_id=?",(history_id,))}
        epoch=int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
        # Tombstone before deleting matching jobs.  A worker that already
        # observed a marker but has not enqueued yet will fail its enqueue
        # CAS even if this mutation is waiting on the history advisory
        # lock for the actual history edit.
        revisions={int(row[0]) for row in db.execute("SELECT DISTINCT history_revision FROM jobs WHERE history_id=?",(history_id,))}
        if history_revision is not None: revisions.add(history_revision)
        for revision in revisions:
            db.execute("INSERT OR IGNORE INTO history_tombstones(history_id,history_revision,clear_epoch,reason,ts) VALUES(?,?,?,?,?)",(history_id,revision,epoch,reason,now))
        # Comparator delivery adoption is a separate content-free outbox.
        # A human mutation wins before its History revision changes: discard
        # every exact prepared link for this history in the same SQLite
        # transaction, so a racing acknowledgement cannot promote stale
        # candidate evidence after the correction/revoke commits.
        adoption_rows=db.execute("SELECT key,value FROM meta WHERE key LIKE 'comparator_adoption:%'").fetchall()
        for key,raw in adoption_rows:
            try: adoption=json.loads(raw)
            except (TypeError,ValueError,json.JSONDecodeError) as exc:
                raise RuntimeError("comparator adoption outbox malformed") from exc
            if not self._valid_comparator_adoption(adoption):
                raise RuntimeError("comparator adoption outbox malformed")
            if adoption["history_id"] != history_id:
                continue
            # ``commit_retry`` first fences the old History revision, then
            # commits exactly its successor.  Its prewritten delivery link is
            # for that successor and must survive this one internal fence;
            # every correction/delete/revoke still discards all links.
            if (operation == "retry" and history_revision is not None
                    and adoption["history_revision"] == history_revision + 1):
                continue
            revision=int(adoption["deployment_revision"]); capture=str(adoption["capture_id"])
            db.execute("UPDATE comparator_intents SET status='discarded',lease_owner=NULL,lease_until=NULL,attempt_id=NULL,updated_ts=? WHERE deployment_revision=? AND capture_id=? AND status IN ('prepared','pending','leased')",(now,revision,capture))
            db.execute("DELETE FROM meta WHERE key=?",(key,))
        keys = [row[0] for row in db.execute("SELECT job_key FROM jobs WHERE history_id=? AND status NOT IN ('discarded')", (history_id,))]
        for key in keys:
            db.execute("UPDATE jobs SET status='discarded',updated_ts=? WHERE job_key=?", (now, key))
            db.execute("UPDATE labels SET state='revoked',reference=NULL,reference_digest=?,revoked_ts=? WHERE job_key=? AND state IN ('accepted','abstained')", (_digest(["revoked", key, now]), now, key))
            # These aggregates are private diagnostics for the same retained
            # capture, so a targeted revoke must remove them with its label.
            db.execute("DELETE FROM shadow_audits WHERE job_key=?",(key,))
        horizons=[row[0] for row in db.execute("SELECT DISTINCT horizon_id FROM prospective_members WHERE history_id=?", (history_id,))]
        for horizon_id in horizons:
            db.execute("DELETE FROM prospective_horizons WHERE horizon_id=?", (horizon_id,))
        cohorts=[str(row[0]) for row in db.execute("SELECT cohort_id FROM cohort_members WHERE history_id=?", (history_id,))]
        for cohort_id in cohorts:
            db.execute("DELETE FROM cohorts WHERE cohort_id=?", (cohort_id,))
        prior=db.execute("SELECT value FROM meta WHERE key='scrub_pending'").fetchone()
        try: pending=json.loads(prior[0]) if prior else None
        except (TypeError,ValueError,json.JSONDecodeError) as exc:
            raise RuntimeError("scrub marker malformed") from exc
        if prior is not None and not _valid_current_scrub(pending,epoch=epoch):
            raise RuntimeError("scrub marker invalid")
        if _valid_clear_marker(pending,epoch=epoch):
            return dependencies,len(keys),pending
        if operation is not None:
            semantic={"schema":1,"kind":"revoke","operation":operation,"history_id":history_id,
                      "history_revision":history_revision,"epoch":epoch,"cohorts":sorted(set(cohorts)),"phase":"silver_fenced"}
            if _valid_revoke_batch(pending):
                existing=[dict(item) for item in pending["intents"]]
                if any(_revoke_intent_key(item) == _revoke_intent_key(semantic) for item in existing):
                    return dependencies,len(keys),pending
                intents=sorted(existing+[{**semantic,"token":uuid.uuid4().hex}],key=_revoke_intent_order)
            elif _valid_revoke_intent(pending):
                existing=dict(pending)
                if _revoke_intent_key(existing) == _revoke_intent_key(semantic):
                    return dependencies,len(keys),existing
                intents=sorted([existing,{**semantic,"token":uuid.uuid4().hex}],key=_revoke_intent_order)
            else:
                # Schema-1 cohort-only markers cannot identify their target,
                # but retaining their cohort union preserves their cleanup
                # scope while new exact intents begin accumulating.
                legacy_cohorts=(pending or {}).get("cohorts",[]) if isinstance(pending,dict) else []
                inherited={value for value in legacy_cohorts if isinstance(value,str) and value}
                if inherited:
                    semantic={**semantic,"cohorts":sorted(set(semantic["cohorts"]) | inherited)}
                intents=[{**semantic,"token":uuid.uuid4().hex}]
            batch={"schema":2,"kind":"revoke","intents":intents,
                   "cohorts":sorted({cohort for item in intents for cohort in item["cohorts"]}),
                   "phase":"silver_fenced","token":uuid.uuid4().hex}
            db.execute("INSERT INTO meta(key,value) VALUES('scrub_pending',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (json.dumps(batch,sort_keys=True),))
            return dependencies,len(keys),batch
        if cohorts:
            # Legacy callers retain their cohort-union marker behavior until
            # they are migrated to exact staged intents.
            if _valid_revoke_batch(pending):
                return dependencies,len(keys),pending
            all_cohorts=set(str(value) for value in (pending or {}).get("cohorts",[]) if isinstance(value,str)) | set(cohorts)
            intent={"kind":"revoke","cohorts":sorted(all_cohorts),"token":uuid.uuid4().hex}
            db.execute("INSERT INTO meta(key,value) VALUES('scrub_pending',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (json.dumps(intent,sort_keys=True),))
            return dependencies,len(keys),intent
        return dependencies,len(keys),pending if isinstance(pending,dict) else None

    def revoke_history(self, history_id: str, *, history_revision: int | None = None,
                       reason: str = "history_changed", return_dependencies: bool = False) -> int | set[str]:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            dependencies,count,_intent=self._revoke_history_mutation(db,history_id,history_revision=history_revision,reason=reason)
            db.commit(); return dependencies if return_dependencies else count

    def revoke_history_with_intent(self, history_id: str, *, history_revision: int | None = None,
                                   reason: str = "history_changed", operation: str = "revoke") -> tuple[set[str],dict[str,Any]]:
        if operation not in _REVOKE_OPERATIONS:
            raise ValueError("invalid revoke operation")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            dependencies,_count,intent=self._revoke_history_mutation(db,history_id,history_revision=history_revision,reason=reason,operation=operation)
            if not isinstance(intent,dict):
                db.rollback(); raise RuntimeError("revoke intent missing")
            db.commit(); return dependencies,dict(intent)

    def revoke_history_dependencies(self, history_id: str, *, history_revision: int | None = None,
                                    reason: str = "history_changed") -> set[str]:
        """Return cohort dependencies before privacy scrub removes their IDs."""
        result=self.revoke_history(history_id,history_revision=history_revision,reason=reason,return_dependencies=True)
        return set(result) if isinstance(result,set) else set()

    def clear(self, *, reason: str = "user_clear") -> int:
        """Epoch fence and secure delete; never unlinks a live WAL/SHM file."""
        now = self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE meta SET value='1' WHERE key='quiescing'")
            old = int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            epoch = old + 1
            db.execute("UPDATE meta SET value=? WHERE key='clear_epoch'", (str(epoch),))
            count = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            db.execute("UPDATE labels SET state='revoked',reference=NULL,reference_digest=?,revoked_ts=? WHERE state IN ('accepted','abstained')", (_digest(["cleared", epoch, now]), now))
            db.execute("DELETE FROM attempts"); db.execute("DELETE FROM jobs")
            db.execute("DELETE FROM history_tombstones")
            db.execute("DELETE FROM prospective_horizons")
            db.execute("DELETE FROM cohorts")
            db.execute("DELETE FROM runtime_observations")
            db.execute("DELETE FROM runtime_pairs")
            # Comparator audio is scrubbed by the exact clear outbox before
            # acknowledgement; removing rows here prevents any leased/pending
            # comparator from resurrecting after the new epoch commits.
            db.execute("DELETE FROM comparator_intents")
            db.execute("DELETE FROM meta WHERE key LIKE 'comparator_adoption:%'")
            db.execute("DELETE FROM route_assignments")
            db.execute("DELETE FROM shadow_audits")
            db.execute("DELETE FROM meta WHERE key='dictation_session'")
            db.execute("INSERT INTO meta(key,value) VALUES('scrub_pending',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (json.dumps({"schema":1,"kind":"clear","epoch":epoch,"phase":"clear_fenced","token":uuid.uuid4().hex},sort_keys=True),))
            db.execute("INSERT INTO epochs(clear_epoch,reason,ts) VALUES(?,?,?)", (epoch, reason, now))
            db.execute("UPDATE meta SET value='0' WHERE key='quiescing'")
            db.commit()
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            # VACUUM runs only after the transaction and does not remove active WAL files.
            db.execute("VACUUM")
            return int(count)

    def scrub_pending(self) -> dict[str, Any] | None:
        with self._connect() as db:
            row=db.execute("SELECT value FROM meta WHERE key='scrub_pending'").fetchone()
            epoch=int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            try: value=json.loads(row[0]) if row else None
            except (TypeError,ValueError,json.JSONDecodeError): return {"kind":"unknown","blocked":True}
            if value is None: return None
            return value if _valid_current_scrub(value,epoch=epoch) else {"kind":"unknown","blocked":True}

    def complete_scrub(self, expected: dict[str, Any] | None = None) -> bool:
        if expected is None:
            return False
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            epoch=int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            if not _valid_current_scrub(expected,epoch=epoch): db.rollback(); return False
            row=db.execute("SELECT value FROM meta WHERE key='scrub_pending'").fetchone()
            try: current=json.loads(row[0]) if row else None
            except (TypeError,ValueError,json.JSONDecodeError): current=None
            if current != expected: db.rollback(); return False
            db.execute("DELETE FROM meta WHERE key='scrub_pending'"); db.commit(); return True

    def alpha(self, generation: int) -> float:
        if generation < 1: raise ValueError("generation must be positive")
        value = .05 / (generation * (generation + 1))
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT alpha FROM alpha_ledger WHERE generation=?", (generation,)).fetchone()
            if old is not None and float(old[0]) != value:
                db.rollback(); raise RuntimeError("alpha ledger corruption")
            db.execute("INSERT OR IGNORE INTO alpha_ledger(generation,alpha,created_ts) VALUES(?,?,?)", (generation, value, self.clock()))
            db.commit()
        return value

    def record_policy_attempt(self, kind: str, policy_hash: str, split_hash: str) -> bool:
        """Returns False when a calibration policy/split was already consumed."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute("INSERT INTO policy_attempts(kind,policy_hash,split_hash,status,created_ts) VALUES(?,?,?,?,?)",
                           (kind, policy_hash, split_hash, "started", self.clock()))
            except sqlite3.IntegrityError:
                db.rollback(); return False
            db.commit(); return True

    @staticmethod
    def manifest_hash(manifest: dict[str, Any]) -> str:
        """Content address model semantics, not a convenient arm alias.

        ``stable_id`` names a routing arm and must be unique in a family, but
        is deliberately excluded here: aliases for the exact same frozen model
        cannot manufacture a distinct comparison.
        """
        public = {key: manifest.get(key) for key in ("backend", "repo", "revision",
                 "package_versions", "decode_settings", "language", "glossary_hash",
                 "glossary_identity", "evaluator_id", "evaluator_hash", "snapshot_digest")}
        return _digest(public)

    def ensure_prospective_horizon(self, *, start_ts: float, champion: dict[str, Any],
                                   candidates: list[dict[str, Any]], receipt_hash: str) -> dict[str, Any]:
        """Open one immutable post-boundary horizon before teacher processing.

        The first call pins the complete arm family; subsequent calls can only
        recover that exact object.  A receipt or arm change fails closed.
        """
        arms = [dict(champion), *[dict(item) for item in candidates]]
        if not arms or any(not isinstance(item.get("stable_id"), str) for item in arms):
            raise ValueError("prospective horizon lacks frozen arms")
        if len({str(item["stable_id"]) for item in arms}) != len(arms):
            raise ValueError("prospective horizon arm ids are not unique")
        identities = [self.manifest_hash(item) for item in arms]
        if len(set(identities)) != len(identities):
            raise RuntimeError("prospective horizon arms are not semantically distinct")
        identity_hash = _digest({"champion": identities[0], "candidates": sorted(identities[1:]), "receipt": receipt_hash})
        now = self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            epoch = int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            # A frozen/evaluated horizon is immutable historical evidence,
            # not an active collection boundary.  Allow the next prospective
            # holdout to open after authorization without mutating the old
            # cohort's retained lineage.
            row = db.execute("SELECT horizon_id,status,start_ts,close_ts,champion,candidates,receipt_hash,identity_hash,clear_epoch,frozen_cohort_id FROM prospective_horizons WHERE status IN ('collecting','closed') ORDER BY created_ts DESC LIMIT 1").fetchone()
            if row is not None:
                value = dict(zip(("horizon_id","status","start_ts","close_ts","champion","candidates","receipt_hash","identity_hash","clear_epoch","cohort_id"), row))
                if value["clear_epoch"] != epoch or value["receipt_hash"] != receipt_hash or value["identity_hash"] != identity_hash:
                    db.rollback(); raise RuntimeError("existing prospective horizon identity changed")
                value["champion"] = json.loads(value["champion"]); value["candidates"] = json.loads(value["candidates"])
                db.commit(); return value
            horizon_id = _digest([epoch, float(start_ts), identity_hash])
            db.execute("INSERT INTO prospective_horizons VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (horizon_id, "collecting", float(start_ts), float(start_ts) + 30 * 86400,
                        json.dumps(champion, sort_keys=True), json.dumps(candidates, sort_keys=True),
                        receipt_hash, identity_hash, epoch, now, None))
            db.commit()
        return {"horizon_id": horizon_id, "status": "collecting", "start_ts": float(start_ts),
                "close_ts": float(start_ts) + 30 * 86400, "champion": champion,
                "candidates": candidates, "receipt_hash": receipt_hash, "identity_hash": identity_hash,
                "clear_epoch": epoch, "cohort_id": None}

    def admit_prospective_member(self, job: dict[str, Any], *, captured_ts: float,
                                 session_id: str | None = None) -> bool:
        """Admit eligible post-boundary capture metadata before teacher output."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            epoch = int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            row = db.execute("SELECT horizon_id,status,start_ts,close_ts,clear_epoch,receipt_hash FROM prospective_horizons WHERE status='collecting' ORDER BY created_ts DESC LIMIT 1").fetchone()
            if row is None or row[4] != epoch or int(job.get("clear_epoch", -1)) != epoch:
                db.commit(); return False
            horizon_id, _status, start, close, _, receipt = row
            # A receipt/family/policy change cannot mix teacher evidence into
            # the already frozen prospective protocol.  Close it fail-closed.
            if (job.get("receipt_hash") != receipt or not isinstance(job.get("teacher_family_hash"),str)
                    or not isinstance(job.get("consensus_policy_hash"),str)):
                db.execute("UPDATE prospective_horizons SET status='invalidated' WHERE horizon_id=?", (horizon_id,)); db.commit(); return False
            from teacher_backends import family_hash
            from teacher_consensus import policy_hash
            if job["teacher_family_hash"] != family_hash() or job["consensus_policy_hash"] != policy_hash():
                db.execute("UPDATE prospective_horizons SET status='invalidated' WHERE horizon_id=?", (horizon_id,)); db.commit(); return False
            if float(captured_ts) < float(start):
                db.commit(); return False
            count = int(db.execute("SELECT COUNT(*) FROM prospective_members WHERE horizon_id=?", (horizon_id,)).fetchone()[0])
            if float(captured_ts) > float(close):
                db.execute("UPDATE prospective_horizons SET status='closed' WHERE horizon_id=?", (horizon_id,)); db.commit(); return False
            cluster = _digest([session_id or "", int(float(captured_ts) // 86400)])
            day=int(float(captured_ts)//86400)
            if count >= 500:
                # Fixed-size, outcome-blind temporal reservoir.  Do not close
                # merely because a high-volume user filled the cap before the
                # seven-day gate: reserve one slot for each newly observed day
                # by replacing the oldest member from an overrepresented day.
                known={int(row[0]) for row in db.execute("SELECT DISTINCT CAST(captured_ts / 86400 AS INTEGER) FROM prospective_members WHERE horizon_id=?", (horizon_id,))}
                if day not in known:
                    victim=db.execute("""SELECT ordinal FROM prospective_members WHERE horizon_id=?
                        AND CAST(captured_ts / 86400 AS INTEGER) IN
                          (SELECT CAST(captured_ts / 86400 AS INTEGER) FROM prospective_members WHERE horizon_id=?
                           GROUP BY CAST(captured_ts / 86400 AS INTEGER) HAVING COUNT(*)>1)
                        ORDER BY captured_ts,ordinal LIMIT 1""", (horizon_id,horizon_id)).fetchone()
                    if victim is not None:
                        db.execute("UPDATE prospective_members SET history_id=?,history_revision=?,audio_sha256=?,captured_ts=?,cluster_hash=? WHERE horizon_id=? AND ordinal=?",
                                   (job["history_id"],int(job["history_revision"]),job["audio_sha256"],float(captured_ts),cluster,horizon_id,victim[0]))
                    else:
                        db.commit(); return False
                else:
                    db.commit(); return False
            else:
                db.execute("INSERT OR IGNORE INTO prospective_members VALUES(?,?,?,?,?,?,?)",
                           (horizon_id, count, job["history_id"], int(job["history_revision"]), job["audio_sha256"], float(captured_ts), cluster))
            count = int(db.execute("SELECT COUNT(*) FROM prospective_members WHERE horizon_id=?", (horizon_id,)).fetchone()[0])
            days=int(db.execute("SELECT COUNT(DISTINCT CAST(captured_ts / 86400 AS INTEGER)) FROM prospective_members WHERE horizon_id=?", (horizon_id,)).fetchone()[0])
            if count >= 500 and days >= 7: db.execute("UPDATE prospective_horizons SET status='closed' WHERE horizon_id=?", (horizon_id,))
            db.commit(); return True

    def prospective_status(self) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT horizon_id,status,start_ts,close_ts,champion,candidates,receipt_hash,identity_hash,clear_epoch,frozen_cohort_id FROM prospective_horizons ORDER BY created_ts DESC LIMIT 1").fetchone()
            if row is None: return None
            value = dict(zip(("horizon_id","status","start_ts","close_ts","champion","candidates","receipt_hash","identity_hash","clear_epoch","cohort_id"), row))
            value["champion"] = json.loads(value["champion"]); value["candidates"] = json.loads(value["candidates"])
            value["members"] = int(db.execute("SELECT COUNT(*) FROM prospective_members WHERE horizon_id=?", (value["horizon_id"],)).fetchone()[0])
            return value

    def allocate_cohort_look(self, cohort_id: str, identity_hash: str) -> tuple[int, float]:
        """Allocate the fixed sequential alpha look exactly once in SQLite."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT generation,alpha,identity_hash FROM cohort_looks WHERE cohort_id=?",(cohort_id,)).fetchone()
            if row is not None:
                if row[2] != identity_hash: db.rollback(); raise RuntimeError("cohort evaluation identity changed")
                db.commit(); return int(row[0]),float(row[1])
            cohort=db.execute("SELECT clear_epoch,status FROM cohorts WHERE cohort_id=?",(cohort_id,)).fetchone()
            epoch=int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            if cohort is None or int(cohort[0]) != epoch or cohort[1] not in {"frozen","evaluating"}:
                db.rollback(); raise RuntimeError("cohort is not available for one-look allocation")
            # The alpha ledger is deliberately public/persistent across
            # personal clears.  Cohort rows cascade on clear, so allocating
            # from them would incorrectly reuse alpha=0.025 after every
            # consent epoch.
            generation=int(db.execute("SELECT COALESCE(MAX(generation),0)+1 FROM alpha_ledger").fetchone()[0])
            alpha=.05/(generation*(generation+1))
            db.execute("INSERT INTO alpha_ledger(generation,alpha,created_ts) VALUES(?,?,?) ON CONFLICT(generation) DO NOTHING",(generation,alpha,self.clock()))
            db.execute("INSERT INTO cohort_looks(cohort_id,generation,alpha,identity_hash,created_ts) VALUES(?,?,?,?,?)",(cohort_id,generation,alpha,identity_hash,self.clock()))
            db.commit(); return generation,alpha

    def horizon_for_cohort(self, cohort_id: str) -> dict[str, Any] | None:
        """Return the immutable horizon belonging to a cohort, including terminal ones."""
        with self._connect() as db:
            row = db.execute("SELECT horizon_id,status,start_ts,close_ts,champion,candidates,receipt_hash,identity_hash,clear_epoch,frozen_cohort_id FROM prospective_horizons WHERE frozen_cohort_id=?", (cohort_id,)).fetchone()
            if row is None:
                return None
            value = dict(zip(("horizon_id","status","start_ts","close_ts","champion","candidates","receipt_hash","identity_hash","clear_epoch","cohort_id"), row))
            value["champion"] = json.loads(value["champion"]); value["candidates"] = json.loads(value["candidates"])
            value["members"] = int(db.execute("SELECT COUNT(*) FROM prospective_members WHERE horizon_id=?", (value["horizon_id"],)).fetchone()[0])
            return value

    def is_prospective_member(self, history_id: str, history_revision: int) -> bool:
        with self._connect() as db:
            return db.execute("SELECT 1 FROM prospective_members m JOIN prospective_horizons h USING(horizon_id) WHERE m.history_id=? AND m.history_revision=? AND h.clear_epoch=? AND h.status IN ('collecting','closed','frozen')", (history_id,int(history_revision),self.epoch())).fetchone() is not None

    def should_hold_marker(self, history_id: str, history_revision: int) -> bool:
        """Retention fence includes unprocessed durable teacher work."""
        with self._connect() as db:
            epoch=int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            active=db.execute("SELECT 1 FROM jobs WHERE history_id=? AND history_revision=? AND clear_epoch=? AND status IN ('pending','leased')",(history_id,int(history_revision),epoch)).fetchone()
            if active: return True
        return self.is_prospective_member(history_id,history_revision)

    def close_and_freeze_prospective(self, *, now: float | None = None) -> dict[str, Any] | None:
        """Atomically freeze exactly the labels and audio admitted to a horizon.

        This deliberately does *not* call ``freeze_personal_horizon``: that
        older public helper cannot safely bridge a read/commit race with a
        human correction.  The compare-and-swap below checks the horizon,
        epoch, admitted audio, policy/receipt lineage and exact label digest in
        one transaction.
        """
        now = self.clock() if now is None else now
        from teacher_consensus import policy_hash
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            epoch = int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            row = db.execute("SELECT horizon_id,status,start_ts,close_ts,champion,candidates,receipt_hash,identity_hash,clear_epoch,frozen_cohort_id FROM prospective_horizons WHERE status IN ('collecting','closed','frozen') ORDER BY created_ts DESC LIMIT 1").fetchone()
            if row is None or int(row[8]) != epoch:
                db.commit(); return None
            state = dict(zip(("horizon_id","status","start_ts","close_ts","champion","candidates","receipt_hash","identity_hash","clear_epoch","cohort_id"), row))
            state["champion"] = json.loads(state["champion"]); state["candidates"] = json.loads(state["candidates"])
            if state["cohort_id"]:
                # Candidate seeding used to happen after this transaction.
                # If a process died in that gap, a resumed frozen horizon must
                # converge to the exact immutable arm ledger rather than be
                # evaluated as a permanently missing-arm cohort.
                cohort_id=str(state["cohort_id"])
                db.commit()
                self.seed_candidate_attempts(cohort_id,[state["champion"],*state["candidates"]])
                return self.cohort_status(cohort_id)
            if state["status"] == "collecting":
                if now < float(state["close_ts"]):
                    db.commit(); return None
                if db.execute("UPDATE prospective_horizons SET status='closed' WHERE horizon_id=? AND status='collecting' AND clear_epoch=?", (state["horizon_id"], epoch)).rowcount != 1:
                    db.rollback(); return None
            if state["status"] not in {"collecting", "closed"}:
                db.commit(); return None
            rows = db.execute("SELECT history_id,history_revision,audio_sha256,captured_ts,cluster_hash FROM prospective_members WHERE horizon_id=? ORDER BY ordinal", (state["horizon_id"],)).fetchall()
            member_days=len({int(float(row[3] or 0)//86400) for row in rows})
            if len(rows) < 500 or member_days < 7:
                # An expired boundary below the frozen minimum personal
                # evidence (at least 500 raw captures over at least seven
                # distinct days) must never freeze into an evaluable cohort.
                # Retire it atomically so a later real capture opens a fresh
                # boundary rather than being silently kept behind an
                # undersized closed row forever.
                db.execute("UPDATE prospective_horizons SET status='invalidated' WHERE horizon_id=? AND clear_epoch=? AND status='closed'",(state["horizon_id"],epoch))
                db.commit(); return None
            members=[]; accepted=words=0
            for ordinal, (ident, revision, audio, ts, cluster) in enumerate(rows):
                label = db.execute("""SELECT l.state,l.reference,l.reference_digest,j.audio_sha256
                    FROM labels l JOIN jobs j USING(job_key)
                    WHERE j.history_id=? AND j.history_revision=? AND j.audio_sha256=?
                      AND j.clear_epoch=? AND j.status='terminal' AND l.evidence_revision=?
                      AND l.receipt_hash=? AND l.policy_hash=?
                    ORDER BY l.created_ts DESC LIMIT 1""",
                    (ident, revision, audio, epoch, revision, state["receipt_hash"], policy_hash())).fetchone()
                if label is None or label[0] not in {"accepted", "abstained"} or label[3] != audio:
                    db.commit(); return None
                if label[0] == "accepted":
                    accepted += 1; words += len((label[1] or "").split())
                members.append((ordinal, ident, revision, label[0], audio, label[2], cluster))
            member_hash = _digest(members)
            candidate_hashes = [self.manifest_hash(item) for item in state["candidates"]]
            evaluator = str(state["champion"].get("evaluator_hash", ""))
            comparison = _digest({"candidate_hashes": sorted(candidate_hashes), "evaluator_hash": evaluator,
                                  "acceptance_hash": state["identity_hash"], "member_hash": member_hash})
            existing = db.execute("SELECT cohort_id,metadata FROM cohorts WHERE comparison_hash=?", (comparison,)).fetchone()
            if existing:
                cohort_id=existing[0]
            else:
                cohort_id=hashlib.sha256((comparison + str(epoch)).encode()).hexdigest()
                metadata={"raw":len(members),"accepted":accepted,"abstained":len(members)-accepted,"words":words,
                          "sessions":len({item[6] for item in members}),"days":len({int(float(item[3] or 0)//86400) for item in rows}),
                          "acceptance_hash":state["identity_hash"],"evaluator_hash":evaluator,
                          "candidate_hashes":sorted(candidate_hashes),"teacher_receipt_hash":state["receipt_hash"],
                          "teacher_policy_hash":policy_hash(),
                          "all_terminal":True}
                db.execute("INSERT INTO cohorts(cohort_id,kind,status,member_hash,comparison_hash,metadata,clear_epoch,created_ts) VALUES(?, 'personal_horizon','frozen',?,?,?,?,?)", (cohort_id,member_hash,comparison,json.dumps(metadata,sort_keys=True),epoch,self.clock()))
                db.executemany("INSERT INTO cohort_members(cohort_id,ordinal,history_id,history_revision,label_state,audio_sha256,reference_digest,cluster_hash) VALUES(?,?,?,?,?,?,?,?)", [(cohort_id,*item) for item in members])
            if db.execute("UPDATE prospective_horizons SET status='frozen',frozen_cohort_id=? WHERE horizon_id=? AND clear_epoch=? AND status IN ('collecting','closed')", (cohort_id,state["horizon_id"],epoch)).rowcount != 1:
                db.rollback(); return None
            db.commit()
        self.seed_candidate_attempts(cohort_id, [state["champion"], *state["candidates"]])
        return self.cohort_status(cohort_id)

    def _freeze_personal_horizon_unsafe_legacy(self, captures: list[dict[str, Any]], *, candidate_hashes: list[str],
                                evaluator_hash: str, acceptance_hash: str) -> dict[str, Any]:
        """Freeze chronological raw membership before any candidate inference.

        ``captures`` contains only local history metadata (id/revision/time,
        session id).  Every member must already have one terminal accepted or
        abstained label under the current epoch; abstentions stay in the fixed
        denominator and accepted references remain private in ``labels``.
        """
        ordered = sorted(captures, key=lambda x: (float(x.get("ts", 0)), str(x.get("history_id", ""))))
        if not ordered:
            raise RuntimeError("no eligible raw captures")
        start = float(ordered[0].get("ts", 0)); selected: list[dict[str, Any]] = []
        for row in ordered:
            if len(selected) >= 500 or float(row.get("ts", 0)) - start > 30 * 86400:
                break
            selected.append(row)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            epoch = int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            members: list[tuple[Any, ...]] = []; words = 0; accepted = 0
            for ordinal, row in enumerate(selected):
                ident, revision = row.get("history_id"), row.get("history_revision")
                if not isinstance(ident, str) or not isinstance(revision, int):
                    db.rollback(); raise ValueError("horizon member lacks identity")
                label = db.execute("""SELECT l.state,l.reference,l.reference_digest FROM labels l JOIN jobs j USING(job_key)
                                      WHERE j.history_id=? AND j.history_revision=? AND j.clear_epoch=? AND j.status='terminal'
                                      ORDER BY l.created_ts DESC LIMIT 1""", (ident, revision, epoch)).fetchone()
                if label is None or label[0] not in {"accepted", "abstained"}:
                    db.rollback(); raise RuntimeError("horizon has unresolved member")
                if label[0] == "accepted":
                    accepted += 1; words += len((label[1] or "").split())
                members.append((ordinal, ident, revision, label[0], row.get("audio_sha256", ""), label[2], str(row.get("cluster_hash", ""))))
            days = len({int(float(row.get("ts", 0)) // 86400) for row in selected})
            sessions = len({str(row.get("cluster_hash", row.get("session_id", ""))) for row in selected if row.get("cluster_hash") or row.get("session_id")})
            metadata = {"raw": len(selected), "accepted": accepted, "abstained": len(selected) - accepted,
                        "words": words, "sessions": sessions, "days": days, "acceptance_hash": acceptance_hash,
                        "evaluator_hash": evaluator_hash, "candidate_hashes": sorted(candidate_hashes), "all_terminal": True}
            # The horizon is a durable one-look object; its id includes every
            # frozen source/gate identity but no transcript-bearing text.
            member_hash = _digest(members)
            comparison = _digest({"candidate_hashes": sorted(candidate_hashes), "evaluator_hash": evaluator_hash,
                                  "acceptance_hash": acceptance_hash, "member_hash": member_hash})
            existing = db.execute("SELECT cohort_id,metadata FROM cohorts WHERE comparison_hash=?", (comparison,)).fetchone()
            if existing:
                db.commit(); return {"cohort_id": existing[0], **json.loads(existing[1]), "existing": True}
            cohort_id = hashlib.sha256((comparison + str(epoch)).encode()).hexdigest()
            db.execute("INSERT INTO cohorts(cohort_id,kind,status,member_hash,comparison_hash,metadata,clear_epoch,created_ts) VALUES(?, 'personal_horizon','frozen',?,?,?,?,?)",
                       (cohort_id, member_hash, comparison, json.dumps(metadata, sort_keys=True), epoch, self.clock()))
            db.executemany("INSERT INTO cohort_members(cohort_id,ordinal,history_id,history_revision,label_state,audio_sha256,reference_digest,cluster_hash) VALUES(?,?,?,?,?,?,?,?,?)",
                           [(cohort_id, *item) for item in members])
            db.commit()
            return {"cohort_id": cohort_id, **metadata, "existing": False}

    def seed_candidate_attempts(self, cohort_id: str, arms: list[dict[str, Any]]) -> int:
        """Create idempotent, per-arm work only for the frozen accepted IDs."""
        if not arms or len({str(item.get("stable_id")) for item in arms}) != len(arms):
            raise ValueError("candidate arm family malformed")
        now = self.clock(); created = 0
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cohort = db.execute("SELECT clear_epoch,status FROM cohorts WHERE cohort_id=?", (cohort_id,)).fetchone()
            if cohort is None or cohort[0] != self.epoch() or cohort[1] not in {"frozen", "evaluating"}:
                db.rollback(); raise RuntimeError("candidate cohort is invalid")
            accepted = db.execute("""SELECT history_id,history_revision,audio_sha256,reference_digest,cluster_hash
                FROM cohort_members WHERE cohort_id=? AND label_state='accepted'""", (cohort_id,)).fetchall()
            for arm in arms:
                arm_id = str(arm["stable_id"])
                for ident, revision, digest, reference_digest, cluster_hash in accepted:
                    result=db.execute("INSERT OR IGNORE INTO candidate_attempts(cohort_id,arm_id,history_id,history_revision,audio_sha256,reference_digest,cluster_hash,status,updated_ts) VALUES(?,?,?,?,?,?,?,'pending',?)", (cohort_id,arm_id,ident,revision,digest,reference_digest,cluster_hash,now))
                    created += result.rowcount
            db.execute("UPDATE cohorts SET status='evaluating' WHERE cohort_id=? AND status='frozen'", (cohort_id,))
            db.commit()
        return created

    def claim_candidate_attempt(self, owner: str) -> dict[str, Any] | None:
        now=self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM meta WHERE key='scrub_pending'").fetchone() is not None:
                db.rollback(); return None
            row=db.execute("""SELECT a.cohort_id,a.arm_id,a.history_id,a.history_revision,a.audio_sha256,a.reference_digest,a.cluster_hash,a.attempts,a.lease_epoch,c.clear_epoch
                FROM candidate_attempts a JOIN cohorts c USING(cohort_id)
                WHERE a.status='pending' AND a.attempts<? AND a.not_before<=? AND c.status='evaluating'
                ORDER BY a.cohort_id,a.history_id,a.arm_id LIMIT 1""", (self.max_attempts,now)).fetchone()
            if row is None: db.commit(); return None
            value=dict(zip(("cohort_id","arm_id","history_id","history_revision","audio_sha256","reference_digest","cluster_hash","attempts","lease_epoch","clear_epoch"),row))
            if value["clear_epoch"] != self.epoch(): db.rollback(); return None
            lease=int(value["lease_epoch"])+1; attempt_id=uuid.uuid4().hex
            changed=db.execute("UPDATE candidate_attempts SET status='leased',attempts=attempts+1,lease_owner=?,lease_epoch=?,lease_until=?,attempt_id=?,updated_ts=? WHERE cohort_id=? AND arm_id=? AND history_id=? AND history_revision=? AND status='pending'", (owner,lease,now+self.lease_seconds,attempt_id,now,value["cohort_id"],value["arm_id"],value["history_id"],value["history_revision"]))
            if changed.rowcount != 1: db.rollback(); return None
            value.update({"lease_owner":owner,"lease_epoch":lease,"attempt_id":attempt_id,
                          "attempts":int(value["attempts"])+1}); db.commit(); return value

    def candidate_reference(self, attempt: dict[str, Any]) -> str | None:
        now=self.clock()
        with self._connect() as db:
            row=db.execute("""SELECT l.reference FROM candidate_attempts a
                JOIN cohort_members m ON m.cohort_id=a.cohort_id AND m.history_id=a.history_id AND m.history_revision=a.history_revision
                JOIN jobs j ON j.history_id=m.history_id AND j.history_revision=m.history_revision AND j.audio_sha256=m.audio_sha256
                JOIN labels l ON l.job_key=j.job_key
                WHERE a.cohort_id=? AND a.arm_id=? AND a.history_id=? AND a.history_revision=?
                  AND a.audio_sha256=m.audio_sha256 AND a.reference_digest=m.reference_digest
                  AND a.status='leased' AND a.lease_owner=? AND a.lease_epoch=? AND a.attempt_id=? AND a.lease_until>=?
                  AND j.clear_epoch=? AND j.status='terminal' AND l.state='accepted'
                  AND l.reference_digest=m.reference_digest
                  AND (SELECT value FROM meta WHERE key='clear_epoch')=?
                  AND NOT EXISTS(SELECT 1 FROM meta WHERE key='scrub_pending')""", (attempt["cohort_id"],attempt["arm_id"],attempt["history_id"],attempt["history_revision"],attempt.get("lease_owner"),attempt.get("lease_epoch"),attempt.get("attempt_id"),now,attempt["clear_epoch"],attempt["clear_epoch"])).fetchone()
            return row[0] if row and isinstance(row[0],str) else None

    def renew_candidate_attempt(self, attempt: dict[str, Any]) -> bool:
        """Extend only the exact current candidate lease; expiry is final."""
        now=self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed=db.execute("""UPDATE candidate_attempts SET lease_until=?,updated_ts=?
                WHERE cohort_id=? AND arm_id=? AND history_id=? AND history_revision=?
                  AND status='leased' AND lease_owner=? AND lease_epoch=? AND attempt_id=?
                  AND lease_until>=? AND (SELECT clear_epoch FROM cohorts WHERE cohort_id=?)=?
                  AND NOT EXISTS(SELECT 1 FROM meta WHERE key='scrub_pending')""",
                (now+self.lease_seconds,now,attempt["cohort_id"],attempt["arm_id"],attempt["history_id"],attempt["history_revision"],
                 attempt["lease_owner"],attempt["lease_epoch"],attempt["attempt_id"],now,attempt["cohort_id"],attempt["clear_epoch"]))
            if changed.rowcount != 1: db.rollback(); return False
            db.commit(); return True

    def finish_candidate_attempt(self, attempt: dict[str, Any], *, stats: dict[str, Any] | None = None,
                                 code: str | None = None, retry: bool = False) -> bool:
        """Lease/epoch CAS.  ``stats`` must be sufficient numeric metrics only."""
        allowed={"word_distance","reference_words","character_distance","reference_characters","hallucination","latency","cluster_hash"}
        if stats is not None:
            if not isinstance(stats,dict) or set(stats) != allowed: raise ValueError("candidate stats are not content-free")
            for key in ("word_distance","reference_words","character_distance","reference_characters"):
                value=stats.get(key)
                if isinstance(value,bool) or not isinstance(value,int) or value < 0 or (key in {"reference_words","reference_characters"} and value <= 0):
                    raise ValueError("candidate stats invalid")
            if not isinstance(stats.get("hallucination"),bool): raise ValueError("candidate stats invalid")
            latency=stats.get("latency")
            if isinstance(latency,bool) or not isinstance(latency,(int,float)) or not math.isfinite(float(latency)) or float(latency) < 0:
                raise ValueError("candidate stats invalid")
            if not isinstance(stats.get("cluster_hash"),str) or len(stats["cluster_hash"]) != 64 or any(char not in "0123456789abcdef" for char in stats["cluster_hash"]):
                raise ValueError("candidate stats invalid")
        now=self.clock()
        attempts=max(0,int(attempt.get("attempts",0))-(1 if retry and code in {"scheduler_busy","live_preempted"} else 0))
        status="pending" if retry and attempts < self.max_attempts else ("complete" if stats is not None else "failed")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            delay=min(3600.0,15.0*(2 ** max(0,int(attempt.get("attempts",0))))) if retry else 0.0
            expected_cluster=db.execute("SELECT cluster_hash FROM candidate_attempts WHERE cohort_id=? AND arm_id=? AND history_id=? AND history_revision=?",(attempt["cohort_id"],attempt["arm_id"],attempt["history_id"],attempt["history_revision"])).fetchone()
            if stats is not None and (expected_cluster is None or stats["cluster_hash"] != expected_cluster[0]):
                db.rollback(); return False
            changed=db.execute("""UPDATE candidate_attempts SET status=?,attempts=?,stats=?,code=?,not_before=?,lease_owner=NULL,lease_until=NULL,attempt_id=NULL,updated_ts=?
                WHERE cohort_id=? AND arm_id=? AND history_id=? AND history_revision=? AND status='leased'
                AND lease_owner=? AND lease_epoch=? AND attempt_id=? AND lease_until>=? AND (SELECT clear_epoch FROM cohorts WHERE cohort_id=?)=?
                AND NOT EXISTS(SELECT 1 FROM meta WHERE key='scrub_pending')""",
                (status,attempts,json.dumps(stats,sort_keys=True) if stats is not None else None,code,now+delay,now,attempt["cohort_id"],attempt["arm_id"],attempt["history_id"],attempt["history_revision"],attempt["lease_owner"],attempt["lease_epoch"],attempt["attempt_id"],now,attempt["cohort_id"],attempt["clear_epoch"]))
            if changed.rowcount != 1: db.rollback(); return False
            db.commit(); return True

    def candidate_attempt_rows(self, cohort_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows=db.execute("SELECT arm_id,history_id,history_revision,audio_sha256,reference_digest,cluster_hash,status,stats,attempts,code FROM candidate_attempts WHERE cohort_id=? ORDER BY history_id,history_revision,arm_id",(cohort_id,)).fetchall()
            result=[]
            for a,h,r,audio,reference,cluster,s,st,attempts,code in rows:
                try: parsed=json.loads(st) if st else None
                except (TypeError,ValueError,json.JSONDecodeError): parsed=None; code="corrupt_stats"
                result.append({"arm_id":a,"history_id":h,"history_revision":r,"audio_sha256":audio,"reference_digest":reference,"cluster_hash":cluster,"status":s,"stats":parsed,"attempts":int(attempts),"code":code})
            return result

    def frozen_accepted_members(self, cohort_id: str) -> list[dict[str, Any]]:
        """Exact immutable accepted identities used by every evaluation arm."""
        with self._connect() as db:
            rows=db.execute("SELECT history_id,history_revision,audio_sha256,reference_digest,cluster_hash FROM cohort_members WHERE cohort_id=? AND label_state='accepted' ORDER BY ordinal", (cohort_id,)).fetchall()
            return [dict(zip(("history_id","history_revision","audio_sha256","reference_digest","cluster_hash"), row)) for row in rows]

    def releasable_cohort_members(self, cohort_id: str) -> list[tuple[str, int]]:
        """Audio retention fence: release only after all arms/decision terminal."""
        with self._connect() as db:
            cohort=db.execute("SELECT status FROM cohorts WHERE cohort_id=?",(cohort_id,)).fetchone()
            if cohort is None or cohort[0] not in {"evaluated_passed","evaluated_failed","invalidated"}: return []
            pending=db.execute("SELECT COUNT(*) FROM candidate_attempts WHERE cohort_id=? AND status IN ('pending','leased')",(cohort_id,)).fetchone()[0]
            if pending: return []
            return [(row[0],int(row[1])) for row in db.execute("SELECT history_id,history_revision FROM cohort_members WHERE cohort_id=?",(cohort_id,))]

    def record_runtime_observation(self, *, deployment_revision: int, session_id: str, arm_id: str,
                                   success: bool, fallback: bool, latency: float, coverage_ok: bool,
                                   hallucination_ok: bool, identity_valid: bool) -> None:
        session_hash=_digest(session_id); now=self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR REPLACE INTO runtime_observations VALUES(?,?,?,?,?,?,?,?,?,?,?)", (int(deployment_revision),session_hash,arm_id,now,int(now//86400),int(success),int(fallback),float(latency),int(coverage_ok),int(hallucination_ok),int(identity_valid)))
            db.commit()

    def record_runtime_pair(self, *, deployment_revision: int, capture_id: str, session_id: str,
                            cohort_id: str, epoch: int, route_generation: int, candidate_arm: str,
                            candidate: dict[str, Any], incumbent: dict[str, Any]) -> str | bool:
        """Atomically commit one exact candidate/incumbent capture pair.

        The epoch/cohort/scrub predicates live in this same SQLite transaction
        as the insert.  A clear which wins before the insert therefore cannot
        leave a post-clear observation behind.
        """
        if (not isinstance(capture_id,str) or not capture_id or not isinstance(candidate_arm,str) or not candidate_arm or
                set(candidate) != {"success","fallback","latency","coverage_ok","hallucination_ok","identity_valid"} or
                set(incumbent) != set(candidate)):
            raise ValueError("invalid content-free runtime pair")
        def fields(value: dict[str, Any]) -> tuple[int,int,float,int,int,int]:
            latency=float(value["latency"])
            if not math.isfinite(latency) or latency < 0: raise ValueError("invalid runtime latency")
            return (int(bool(value["success"])),int(bool(value["fallback"])),latency,int(bool(value["coverage_ok"])),
                    int(bool(value["hallucination_ok"])),int(bool(value["identity_valid"])))
        candidate_fields=fields(candidate); incumbent_fields=fields(incumbent); now=self.clock(); session_hash=hashlib.sha256(session_id.encode()).hexdigest()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current=int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            cohort=db.execute("SELECT status,clear_epoch FROM cohorts WHERE cohort_id=?",(cohort_id,)).fetchone()
            pending=db.execute("SELECT 1 FROM meta WHERE key='scrub_pending'").fetchone()
            assigned=db.execute("SELECT selected FROM route_assignments WHERE route_generation=? AND session_hash=?",(int(route_generation),session_hash)).fetchone()
            if current != int(epoch) or pending is not None or assigned is None or not bool(assigned[0]) or cohort is None or cohort[0] != "evaluated_passed" or int(cohort[1]) != current:
                db.rollback(); return False
            row=(int(deployment_revision),capture_id,session_hash,cohort_id,current,candidate_arm,*candidate_fields,*incumbent_fields,now,int(now//86400))
            try:
                db.execute("INSERT INTO runtime_pairs(deployment_revision,capture_id,session_hash,cohort_id,epoch,candidate_arm,candidate_success,candidate_fallback,candidate_latency,candidate_coverage_ok,candidate_hallucination_ok,candidate_identity_valid,incumbent_success,incumbent_fallback,incumbent_latency,incumbent_coverage_ok,incumbent_hallucination_ok,incumbent_identity_valid,ts,day,state) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(*row,"complete"))
            except sqlite3.IntegrityError:
                prior=db.execute("SELECT * FROM runtime_pairs WHERE deployment_revision=? AND capture_id=?",(int(deployment_revision),capture_id)).fetchone()
                # ``ts``/day are write-time bookkeeping, not evidence.  A
                # lost acknowledgement can replay the exact pair later while
                # retaining the original capture timestamp.
                if prior is None or tuple(prior[:18]) != tuple(row[:18]): db.rollback(); return False
                db.commit(); return "replayed"
            db.commit(); return "inserted"

    def begin_runtime_pair(self, *, deployment_revision: int, capture_id: str, session_id: str,
                           cohort_id: str, epoch: int, route_generation: int, candidate_arm: str,
                           candidate: dict[str, Any]) -> str | bool:
        """Durably record the candidate half before any comparator work."""
        required={"success","fallback","latency","coverage_ok","hallucination_ok","identity_valid"}
        if set(candidate) != required or not isinstance(capture_id,str) or not capture_id or not isinstance(candidate_arm,str) or not candidate_arm:
            raise ValueError("invalid content-free runtime candidate")
        if any(not isinstance(candidate[key],bool) for key in ("success","fallback","coverage_ok","hallucination_ok","identity_valid")):
            raise ValueError("runtime flags must be bool")
        if isinstance(candidate["latency"],bool): raise ValueError("invalid runtime latency")
        latency=float(candidate["latency"])
        if not math.isfinite(latency) or latency < 0: raise ValueError("invalid runtime latency")
        session_hash=hashlib.sha256(session_id.encode()).hexdigest(); now=self.clock()
        fields=(int(bool(candidate["success"])),int(bool(candidate["fallback"])),latency,int(bool(candidate["coverage_ok"])),int(bool(candidate["hallucination_ok"])),int(bool(candidate["identity_valid"])))
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current=int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0]); cohort=db.execute("SELECT status,clear_epoch FROM cohorts WHERE cohort_id=?",(cohort_id,)).fetchone()
            assigned=db.execute("SELECT selected FROM route_assignments WHERE route_generation=? AND session_hash=?",(route_generation,session_hash)).fetchone()
            if current != epoch or db.execute("SELECT 1 FROM meta WHERE key='scrub_pending'").fetchone() or not cohort or cohort[0] != "evaluated_passed" or cohort[1] != epoch or not assigned or not assigned[0]:
                db.rollback(); return False
            row=(deployment_revision,capture_id,session_hash,cohort_id,epoch,candidate_arm,*fields,0,1,0.0,0,0,0,now,int(now//86400),"candidate_recorded")
            try:
                db.execute("INSERT INTO runtime_pairs(deployment_revision,capture_id,session_hash,cohort_id,epoch,candidate_arm,candidate_success,candidate_fallback,candidate_latency,candidate_coverage_ok,candidate_hallucination_ok,candidate_identity_valid,incumbent_success,incumbent_fallback,incumbent_latency,incumbent_coverage_ok,incumbent_hallucination_ok,incumbent_identity_valid,ts,day,state) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",row)
            except sqlite3.IntegrityError:
                old=db.execute("SELECT deployment_revision,capture_id,session_hash,cohort_id,epoch,candidate_arm,candidate_success,candidate_fallback,candidate_latency,candidate_coverage_ok,candidate_hallucination_ok,candidate_identity_valid,state FROM runtime_pairs WHERE deployment_revision=? AND capture_id=?",(deployment_revision,capture_id)).fetchone()
                if old != (*row[:12],"candidate_recorded"): db.rollback(); return False
                db.commit(); return "replayed"
            db.commit(); return "inserted"

    def complete_runtime_pair(self, *, deployment_revision: int, capture_id: str, session_id: str,
                              cohort_id: str, epoch: int, route_generation: int, candidate_arm: str,
                              candidate: dict[str, Any], incumbent: dict[str, Any]) -> str | bool:
        required={"success","fallback","latency","coverage_ok","hallucination_ok","identity_valid"}
        if set(candidate) != required or set(incumbent) != required: raise ValueError("invalid content-free runtime pair")
        def fields(value):
            if any(not isinstance(value[key],bool) for key in ("success","fallback","coverage_ok","hallucination_ok","identity_valid")):
                raise ValueError("runtime flags must be bool")
            latency=float(value["latency"])
            if isinstance(value["latency"],bool) or not math.isfinite(latency) or latency < 0: raise ValueError("invalid runtime latency")
            return (int(value["success"]),int(value["fallback"]),latency,int(value["coverage_ok"]),int(value["hallucination_ok"]),int(value["identity_valid"]))
        candidate_fields=fields(candidate); incumbent_fields=fields(incumbent); session_hash=hashlib.sha256(session_id.encode()).hexdigest()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current=int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0]); cohort=db.execute("SELECT status,clear_epoch FROM cohorts WHERE cohort_id=?",(cohort_id,)).fetchone(); assigned=db.execute("SELECT selected FROM route_assignments WHERE route_generation=? AND session_hash=?",(route_generation,session_hash)).fetchone()
            if current != epoch or db.execute("SELECT 1 FROM meta WHERE key='scrub_pending'").fetchone() or not cohort or cohort[0] != "evaluated_passed" or cohort[1] != epoch or not assigned or not assigned[0]: db.rollback(); return False
            row=db.execute("SELECT session_hash,cohort_id,epoch,candidate_arm,candidate_success,candidate_fallback,candidate_latency,candidate_coverage_ok,candidate_hallucination_ok,candidate_identity_valid,state,incumbent_success,incumbent_fallback,incumbent_latency,incumbent_coverage_ok,incumbent_hallucination_ok,incumbent_identity_valid FROM runtime_pairs WHERE deployment_revision=? AND capture_id=?",(deployment_revision,capture_id)).fetchone()
            if row is None or tuple(row[:10]) != (session_hash,cohort_id,epoch,candidate_arm,*candidate_fields): db.rollback(); return False
            if row[10] == "complete":
                ok=tuple(row[11:]) == incumbent_fields; db.commit(); return "replayed" if ok else False
            changed=db.execute("UPDATE runtime_pairs SET incumbent_success=?,incumbent_fallback=?,incumbent_latency=?,incumbent_coverage_ok=?,incumbent_hallucination_ok=?,incumbent_identity_valid=?,state='complete' WHERE deployment_revision=? AND capture_id=? AND state='candidate_recorded'",(*incumbent_fields,deployment_revision,capture_id))
            if changed.rowcount != 1: db.rollback(); return False
            db.commit(); return "inserted"

    def enqueue_comparator_intent(self, *, deployment_revision: int, capture_id: str, session_id: str,
                                  cohort_id: str, epoch: int, route_generation: int, candidate_arm: str,
                                  audio_sha256: str, spool_name: str, candidate: dict[str,Any]) -> str | bool:
        """Persist the exact audio comparator outbox before foreground return."""
        if not all(isinstance(x,str) and x for x in (capture_id,session_id,cohort_id,candidate_arm,audio_sha256,spool_name)) or len(audio_sha256)!=64:
            raise ValueError("invalid comparator intent")
        required={"success","fallback","latency","coverage_ok","hallucination_ok","identity_valid"}
        if not isinstance(candidate,dict) or set(candidate)!=required or any(not isinstance(candidate[k],bool) for k in required-{"latency"}) or isinstance(candidate["latency"],bool) or not math.isfinite(float(candidate["latency"])) or float(candidate["latency"])<0: raise ValueError("invalid comparator candidate")
        fields=self._runtime_evidence_fields(candidate)
        session_hash=hashlib.sha256(session_id.encode()).hexdigest(); now=self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE"); current=int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            if not self._valid_comparator_spool_file(capture_id=capture_id,spool_name=spool_name,audio_sha256=audio_sha256):
                db.rollback(); return False
            cohort=db.execute("SELECT status,clear_epoch FROM cohorts WHERE cohort_id=?",(cohort_id,)).fetchone(); assigned=db.execute("SELECT selected FROM route_assignments WHERE route_generation=? AND session_hash=?",(route_generation,session_hash)).fetchone()
            if current != epoch or db.execute("SELECT 1 FROM meta WHERE key='scrub_pending'").fetchone() or not cohort or cohort[0] != 'evaluated_passed' or cohort[1] != epoch or not assigned or not assigned[0]: db.rollback(); return False
            row=(deployment_revision,capture_id,session_hash,cohort_id,epoch,route_generation,candidate_arm,audio_sha256,spool_name,*fields)
            try: db.execute("INSERT INTO comparator_intents(deployment_revision,capture_id,session_hash,cohort_id,epoch,route_generation,candidate_arm,audio_sha256,spool_name,candidate_success,candidate_fallback,candidate_latency,candidate_coverage_ok,candidate_hallucination_ok,candidate_identity_valid,status,created_ts,updated_ts) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'prepared',?,?)",(*row,now,now))
            except sqlite3.IntegrityError:
                old=db.execute("SELECT deployment_revision,capture_id,session_hash,cohort_id,epoch,route_generation,candidate_arm,audio_sha256,spool_name,candidate_success,candidate_fallback,candidate_latency,candidate_coverage_ok,candidate_hallucination_ok,candidate_identity_valid,status FROM comparator_intents WHERE deployment_revision=? AND capture_id=?",(deployment_revision,capture_id)).fetchone()
                # A discarded row records that its foreground publication did
                # not occur (for example, an acknowledgement was lost and the
                # caller returned fallback).  It must never be replayed as
                # candidate delivery authority, even when every identity field
                # matches.  Pending/leased belongs to an earlier delivered
                # call; complete is an immutable already-paired delivery.
                if old is None or tuple(old[:15]) != row or old[15] not in {'pending','leased','complete'}:
                    db.rollback(); return False
                db.commit(); return "replayed"
            db.commit(); return "inserted"

    @staticmethod
    def _comparator_adoption_key(deployment_revision: int, capture_id: str) -> str:
        if (isinstance(deployment_revision,bool) or not isinstance(deployment_revision,int)
                or deployment_revision < 0 or not isinstance(capture_id,str) or not capture_id):
            raise ValueError("invalid comparator adoption identity")
        return f"comparator_adoption:{deployment_revision}:{capture_id}"

    @staticmethod
    def _valid_comparator_adoption(value: Any) -> bool:
        return (isinstance(value,dict) and set(value) == {"schema","deployment_revision","capture_id",
                "history_id","history_revision","audio_sha256","spool_name"}
                and value.get("schema") == 1
                and not isinstance(value.get("deployment_revision"),bool)
                and isinstance(value.get("deployment_revision"),int) and value["deployment_revision"] >= 0
                and isinstance(value.get("capture_id"),str) and bool(value["capture_id"])
                and isinstance(value.get("history_id"),str) and bool(value["history_id"])
                and not isinstance(value.get("history_revision"),bool)
                and isinstance(value.get("history_revision"),int) and value["history_revision"] >= 0
                and isinstance(value.get("audio_sha256"),str)
                and re.fullmatch(r"[0-9a-f]{64}",value["audio_sha256"]) is not None
                and isinstance(value.get("spool_name"),str) and Path(value["spool_name"]).name == value["spool_name"]
                and value["spool_name"] == f"{value['capture_id']}.wav")

    def adopt_comparator_publication(self, *, deployment_revision: int, capture_id: str,
                                     spool_name: str, audio_sha256: str, candidate: dict[str,Any],
                                     history_id: str, history_revision: int) -> bool:
        """Durably link an unclaimable publication to its pending History write.

        The link is content-free SQLite outbox state rather than History
        metadata, so a restart can distinguish a committed primary/retry row
        from a prepared candidate that was never delivered.
        """
        if (not self._valid_comparator_spool_identity(deployment_revision,capture_id,spool_name,audio_sha256)
                or not isinstance(history_id,str) or not history_id
                or isinstance(history_revision,bool) or not isinstance(history_revision,int) or history_revision < 0
                or not isinstance(candidate,dict)):
            raise ValueError("invalid comparator publication adoption")
        fields=self._runtime_evidence_fields(candidate)
        value={"schema":1,"deployment_revision":deployment_revision,"capture_id":capture_id,
               "history_id":history_id,"history_revision":history_revision,
               "audio_sha256":audio_sha256,"spool_name":spool_name}
        key=self._comparator_adoption_key(deployment_revision,capture_id)
        now=self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT spool_name,audio_sha256,candidate_success,candidate_fallback,candidate_latency,candidate_coverage_ok,candidate_hallucination_ok,candidate_identity_valid,status FROM comparator_intents WHERE deployment_revision=? AND capture_id=?",(deployment_revision,capture_id)).fetchone()
            if row is None or tuple(row[:8]) != (spool_name,audio_sha256,*fields) or row[8] != "prepared":
                db.rollback(); return False
            old=db.execute("SELECT value FROM meta WHERE key=?",(key,)).fetchone()
            if old is not None:
                try: prior=json.loads(old[0])
                except (TypeError,ValueError,json.JSONDecodeError): db.rollback(); return False
                if prior != value:
                    db.rollback(); return False
                db.commit(); return True
            db.execute("INSERT INTO meta(key,value) VALUES(?,?)",(key,json.dumps(value,sort_keys=True,separators=(",",":"))))
            db.execute("UPDATE comparator_intents SET updated_ts=? WHERE deployment_revision=? AND capture_id=? AND status='prepared'",(now,deployment_revision,capture_id))
            db.commit(); return True

    def comparator_publication_adoptions(self) -> list[dict[str,Any]]:
        """Read strict content-free pending-delivery links for restart recovery."""
        with self._connect() as db:
            rows=db.execute("SELECT key,value FROM meta WHERE key LIKE 'comparator_adoption:%' ORDER BY key").fetchall()
        result=[]
        for key,raw in rows:
            try: value=json.loads(raw)
            except (TypeError,ValueError,json.JSONDecodeError) as exc: raise RuntimeError("comparator adoption outbox malformed") from exc
            if (not self._valid_comparator_adoption(value)
                    or key != self._comparator_adoption_key(value["deployment_revision"],value["capture_id"])):
                raise RuntimeError("comparator adoption outbox malformed")
            result.append(value)
        return result

    def acknowledge_adopted_comparator_publication(self, *, deployment_revision: int, capture_id: str,
                                                    spool_name: str, audio_sha256: str,
                                                    candidate: dict[str,Any]) -> bool | None:
        """Promote an exactly adopted prepared publication, deleting its outbox link.

        ``None`` means no adoption link exists and preserves the direct-runtime
        acknowledgement seam; ``False`` is a malformed/stale link.
        """
        if (not self._valid_comparator_spool_identity(deployment_revision,capture_id,spool_name,audio_sha256)
                or not isinstance(candidate,dict)):
            raise ValueError("invalid comparator acknowledgement")
        fields=self._runtime_evidence_fields(candidate); key=self._comparator_adoption_key(deployment_revision,capture_id); now=self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            raw=db.execute("SELECT value FROM meta WHERE key=?",(key,)).fetchone()
            if raw is None:
                db.rollback(); return None
            if db.execute("SELECT 1 FROM meta WHERE key='scrub_pending'").fetchone() is not None:
                db.rollback(); return False
            try: adoption=json.loads(raw[0])
            except (TypeError,ValueError,json.JSONDecodeError): db.rollback(); return False
            if (not self._valid_comparator_adoption(adoption)
                    or adoption["spool_name"] != spool_name or adoption["audio_sha256"] != audio_sha256):
                db.rollback(); return False
            row=db.execute("SELECT spool_name,audio_sha256,candidate_success,candidate_fallback,candidate_latency,candidate_coverage_ok,candidate_hallucination_ok,candidate_identity_valid,status,created_ts FROM comparator_intents WHERE deployment_revision=? AND capture_id=?",(deployment_revision,capture_id)).fetchone()
            if row is None or tuple(row[:8]) != (spool_name,audio_sha256,*fields) or row[8] != "prepared":
                db.rollback(); return False
            changed=db.execute("UPDATE comparator_intents SET status='pending',updated_ts=? WHERE deployment_revision=? AND capture_id=? AND status='prepared'",(now,deployment_revision,capture_id))
            if changed.rowcount != 1:
                db.rollback(); return False
            db.execute("DELETE FROM meta WHERE key=?",(key,)); db.commit(); return True

    def comparator_candidate(self, deployment_revision: int, capture_id: str) -> dict[str,Any] | None:
        """Return strictly validated content-free candidate evidence for recovery."""
        if isinstance(deployment_revision,bool) or not isinstance(deployment_revision,int) or deployment_revision < 0 or not isinstance(capture_id,str) or not capture_id:
            return None
        with self._connect() as db:
            row=db.execute("SELECT candidate_success,candidate_fallback,candidate_latency,candidate_coverage_ok,candidate_hallucination_ok,candidate_identity_valid FROM comparator_intents WHERE deployment_revision=? AND capture_id=?",(deployment_revision,capture_id)).fetchone()
        if row is None:
            return None
        try:
            candidate={"success":bool(row[0]),"fallback":bool(row[1]),"latency":float(row[2]),
                       "coverage_ok":bool(row[3]),"hallucination_ok":bool(row[4]),"identity_valid":bool(row[5])}
            if tuple(row) != self._runtime_evidence_fields(candidate):
                return None
            return candidate
        except (TypeError,ValueError):
            return None

    def acknowledge_comparator_intent(self, *, deployment_revision: int, capture_id: str,
                                      spool_name: str, audio_sha256: str,
                                      candidate: dict[str,Any]) -> bool:
        """Make an exact prepared publication eligible for worker claim."""
        if (not self._valid_comparator_spool_identity(deployment_revision,capture_id,spool_name,audio_sha256)
                or not isinstance(candidate,dict)):
            raise ValueError("invalid comparator acknowledgement")
        fields=self._runtime_evidence_fields(candidate); now=self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT spool_name,audio_sha256,candidate_success,candidate_fallback,candidate_latency,candidate_coverage_ok,candidate_hallucination_ok,candidate_identity_valid,status,created_ts FROM comparator_intents WHERE deployment_revision=? AND capture_id=?",(deployment_revision,capture_id)).fetchone()
            if row is None or tuple(row[:8]) != (spool_name,audio_sha256,*fields):
                db.rollback(); return False
            if row[8] in {'pending','leased','complete'}:
                db.commit(); return True
            if row[8] != 'prepared' or float(row[9]) < now-self.lease_seconds:
                db.rollback(); return False
            changed=db.execute("UPDATE comparator_intents SET status='pending',updated_ts=? WHERE deployment_revision=? AND capture_id=? AND status='prepared'",(now,deployment_revision,capture_id))
            if changed.rowcount != 1:
                db.rollback(); return False
            db.execute("DELETE FROM meta WHERE key=?",(self._comparator_adoption_key(deployment_revision,capture_id),))
            db.commit(); return True

    def claim_comparator_intent(self, owner: str) -> dict[str,Any] | None:
        now=self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM meta WHERE key='scrub_pending'").fetchone(): db.rollback(); return None
            db.execute("UPDATE comparator_intents SET status='discarded',updated_ts=? WHERE status IN ('pending','prepared') AND attempts>=?",(now,self.max_attempts))
            row=db.execute("SELECT deployment_revision,capture_id,session_hash,cohort_id,epoch,route_generation,candidate_arm,audio_sha256,spool_name,candidate_success,candidate_fallback,candidate_latency,candidate_coverage_ok,candidate_hallucination_ok,candidate_identity_valid,attempts,lease_epoch FROM comparator_intents WHERE status='pending' AND attempts<? ORDER BY created_ts LIMIT 1",(self.max_attempts,)).fetchone()
            if row is None: db.commit(); return None
            keys=("deployment_revision","capture_id","session_hash","cohort_id","epoch","route_generation","candidate_arm","audio_sha256","spool_name","candidate_success","candidate_fallback","candidate_latency","candidate_coverage_ok","candidate_hallucination_ok","candidate_identity_valid","attempts","lease_epoch")
            value=dict(zip(keys,row))
            try:
                candidate={"success":value.pop("candidate_success"),"fallback":value.pop("candidate_fallback"),"latency":value.pop("candidate_latency"),"coverage_ok":value.pop("candidate_coverage_ok"),"hallucination_ok":value.pop("candidate_hallucination_ok"),"identity_valid":value.pop("candidate_identity_valid")}
                if any(item not in (0,1) for key,item in candidate.items() if key != "latency") or not math.isfinite(float(candidate["latency"])) or float(candidate["latency"]) < 0: raise ValueError
                value["candidate"]={"success":bool(candidate["success"]),"fallback":bool(candidate["fallback"]),"latency":float(candidate["latency"]),"coverage_ok":bool(candidate["coverage_ok"]),"hallucination_ok":bool(candidate["hallucination_ok"]),"identity_valid":bool(candidate["identity_valid"])}
            except (TypeError,ValueError):
                db.execute("UPDATE comparator_intents SET status='discarded',updated_ts=? WHERE deployment_revision=? AND capture_id=? AND status='pending'",(now,value["deployment_revision"],value["capture_id"])); db.commit(); return None
            lease=int(value['lease_epoch'])+1; token=uuid.uuid4().hex
            changed=db.execute("UPDATE comparator_intents SET status='leased',attempts=attempts+1,lease_owner=?,lease_epoch=?,attempt_id=?,lease_until=?,updated_ts=? WHERE deployment_revision=? AND capture_id=? AND status='pending'",(owner,lease,token,now+self.lease_seconds,now,value['deployment_revision'],value['capture_id']))
            if changed.rowcount!=1: db.rollback(); return None
            value.update({'lease_owner':owner,'lease_epoch':lease,'attempt_id':token,'lease_until':now+self.lease_seconds}); db.commit(); return value

    def finish_comparator_intent(self, intent: dict[str,Any], *, complete: bool, code: str | None=None) -> bool:
        now=self.clock(); status='complete' if complete else 'discarded'
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed=db.execute("UPDATE comparator_intents SET status=?,lease_owner=NULL,lease_until=NULL,attempt_id=NULL,updated_ts=? WHERE deployment_revision=? AND capture_id=? AND status='leased' AND lease_owner=? AND lease_epoch=? AND attempt_id=? AND lease_until>=? AND NOT EXISTS(SELECT 1 FROM meta WHERE key='scrub_pending')",(status,now,intent['deployment_revision'],intent['capture_id'],intent['lease_owner'],intent['lease_epoch'],intent['attempt_id'],now))
            if changed.rowcount!=1: db.rollback(); return False
            db.commit(); return True

    def renew_comparator_intent(self, intent: dict[str,Any]) -> bool:
        now=self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed=db.execute("UPDATE comparator_intents SET lease_until=?,updated_ts=? WHERE deployment_revision=? AND capture_id=? AND status='leased' AND lease_owner=? AND lease_epoch=? AND attempt_id=? AND lease_until>=? AND NOT EXISTS(SELECT 1 FROM meta WHERE key='scrub_pending')",(now+self.lease_seconds,now,intent['deployment_revision'],intent['capture_id'],intent['lease_owner'],intent['lease_epoch'],intent['attempt_id'],now))
            if changed.rowcount!=1: db.rollback(); return False
            db.commit(); return True

    def release_comparator_intent(self, intent: dict[str,Any], *, retry: bool=True, refund: bool=False) -> bool:
        """Release an interrupted comparator; only scheduler/preemption refunds.

        Backend/runtime failures consume a bounded attempt.  A foreground
        scheduler denial or shutdown before inference is not a failure and
        restores the prior attempt count.
        """
        now=self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            status='discarded' if not retry else 'pending'
            attempts_sql="attempts=CASE WHEN attempts>0 THEN attempts-1 ELSE 0 END," if refund else ""
            # The status decision for genuine failures uses the claimed count;
            # refunding first preserves the intent exactly as it was before a
            # busy/preempted scheduler lease.
            if retry and not refund:
                row=db.execute("SELECT attempts FROM comparator_intents WHERE deployment_revision=? AND capture_id=? AND status='leased' AND lease_owner=? AND lease_epoch=? AND attempt_id=? AND lease_until>=?",(intent['deployment_revision'],intent['capture_id'],intent['lease_owner'],intent['lease_epoch'],intent['attempt_id'],now)).fetchone()
                if row is None: db.rollback(); return False
                status='discarded' if int(row[0]) >= self.max_attempts else 'pending'
            changed=db.execute(f"UPDATE comparator_intents SET status=?,{attempts_sql}lease_owner=NULL,lease_until=NULL,attempt_id=NULL,updated_ts=? WHERE deployment_revision=? AND capture_id=? AND status='leased' AND lease_owner=? AND lease_epoch=? AND attempt_id=? AND lease_until>=?",(status,now,intent['deployment_revision'],intent['capture_id'],intent['lease_owner'],intent['lease_epoch'],intent['attempt_id'],now))
            if changed.rowcount!=1: db.rollback(); return False
            db.commit(); return True

    def comparator_intent_terminal(self, intent: dict[str,Any]) -> bool:
        """Content-free exact status read for owned spool cleanup."""
        try:
            with self._connect() as db:
                row=db.execute("SELECT status FROM comparator_intents WHERE deployment_revision=? AND capture_id=?",(int(intent['deployment_revision']),str(intent['capture_id']))).fetchone()
            return row is not None and row[0] in {'complete','discarded'}
        except (KeyError,TypeError,ValueError):
            return False

    @staticmethod
    def _comparator_spool_cleanup_state(db: sqlite3.Connection, *, deployment_revision: int,
                                        capture_id: str, spool_name: str,
                                        audio_sha256: str) -> str:
        row=db.execute("SELECT spool_name,audio_sha256,status FROM comparator_intents WHERE deployment_revision=? AND capture_id=?",(deployment_revision,capture_id)).fetchone()
        if row is None:
            return "unlink"
        if row[0] != spool_name or row[1] != audio_sha256:
            return "blocked"
        if row[2] in {"pending","leased"}:
            return "retain"
        if row[2] in {"complete","discarded"}:
            return "unlink"
        return "blocked"

    @staticmethod
    def _valid_comparator_spool_identity(deployment_revision: int, capture_id: str,
                                         spool_name: str, audio_sha256: str) -> bool:
        return (not isinstance(deployment_revision,bool) and isinstance(deployment_revision,int)
                and deployment_revision >= 0 and isinstance(capture_id,str) and bool(capture_id)
                and isinstance(spool_name,str) and Path(spool_name).name == spool_name
                and spool_name.endswith(".wav") and spool_name != ".wav"
                and "/" not in spool_name and "\\" not in spool_name
                and isinstance(audio_sha256,str)
                and re.fullmatch(r"[0-9a-f]{64}",audio_sha256) is not None)

    def _valid_comparator_spool_file(self, *, capture_id: str, spool_name: str,
                                     audio_sha256: str) -> bool:
        """Validate the owned spool while the comparator DB write fence is held."""
        if spool_name != f"{capture_id}.wav":
            return False
        try:
            ComparatorSpool(self.base_dir).read(spool_name,audio_sha256)
            return True
        except (FileNotFoundError,OSError,RuntimeError,ValueError,EOFError):
            return False

    def comparator_spool_cleanup_state(self, *, deployment_revision: int, capture_id: str,
                                       spool_name: str, audio_sha256: str) -> str:
        """Return the only safe foreground cleanup disposition for one spool.

        This is deliberately a read-only transaction.  A foreground replay can
        have different timing evidence from an already durable outbox item; it
        must never unlink the first item's audio merely because its own enqueue
        was rejected.  ``retain`` covers a live exact owner, ``unlink`` only an
        absent or terminal exact owner, and ``blocked`` is fail-closed.
        """
        if not self._valid_comparator_spool_identity(deployment_revision,capture_id,spool_name,audio_sha256):
            return "blocked"
        with self._connect() as db:
            db.execute("BEGIN")
            try:
                result=self._comparator_spool_cleanup_state(db,deployment_revision=deployment_revision,
                    capture_id=capture_id,spool_name=spool_name,audio_sha256=audio_sha256)
            finally:
                db.rollback()
        return result

    @contextmanager
    def comparator_spool_cleanup_fence(self, *, deployment_revision: int, capture_id: str,
                                        spool_name: str, audio_sha256: str) -> Iterator[str]:
        """Hold the enqueue write fence until an exact spool unlink completes."""
        if not self._valid_comparator_spool_identity(deployment_revision,capture_id,spool_name,audio_sha256):
            yield "blocked"
            return
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield self._comparator_spool_cleanup_state(db,deployment_revision=deployment_revision,
                    capture_id=capture_id,spool_name=spool_name,audio_sha256=audio_sha256)
            finally:
                db.rollback()

    def terminal_comparator_spools(self) -> dict[str,str]:
        """Exact owned terminal spool names mapped to their audio digest."""
        with self._connect() as db:
            rows=db.execute("SELECT deployment_revision,capture_id,spool_name,audio_sha256 FROM comparator_intents WHERE status IN ('complete','discarded')").fetchall()
        result: dict[str,str]={}
        for deployment_revision,capture_id,spool_name,audio_sha256 in rows:
            if (not self._valid_comparator_spool_identity(deployment_revision,capture_id,spool_name,audio_sha256)
                    or spool_name != f"{capture_id}.wav"):
                raise RuntimeError("malformed terminal comparator intent")
            previous=result.get(spool_name)
            if previous is None:
                result[spool_name]=audio_sha256
            else:
                raise RuntimeError("ambiguous terminal comparator spool")
        return result

    def discard_comparator_intent(self, *, capture_id: str, deployment_revision: int,
                                  spool_name: str | None=None, audio_sha256: str | None=None,
                                  candidate: dict[str,Any] | None=None) -> bool:
        """Terminalize one exact outbox item before a stale foreground return.

        This deliberately accepts both pending and leased states: foreground
        human authority wins over a background comparator that has just
        claimed the same intent.  A completed pair is immutable and is not
        rewritten here.
        """
        if (isinstance(deployment_revision,bool) or not isinstance(deployment_revision,int) or deployment_revision < 0
                or not isinstance(capture_id,str) or not capture_id):
            raise ValueError("invalid comparator identity")
        exact=(spool_name is not None or audio_sha256 is not None or candidate is not None)
        if exact and (not isinstance(spool_name,str) or not isinstance(audio_sha256,str)
                      or not self._valid_comparator_spool_identity(deployment_revision,capture_id,spool_name,audio_sha256)
                      or not isinstance(candidate,dict)):
            raise ValueError("invalid exact comparator identity")
        candidate_fields=self._runtime_evidence_fields(candidate) if exact else None
        now=self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if exact:
                row=db.execute("SELECT spool_name,audio_sha256,candidate_success,candidate_fallback,candidate_latency,candidate_coverage_ok,candidate_hallucination_ok,candidate_identity_valid FROM comparator_intents WHERE deployment_revision=? AND capture_id=?",(deployment_revision,capture_id)).fetchone()
                if row is None or tuple(row) != (spool_name,audio_sha256,*candidate_fields):
                    db.rollback(); return False
            changed=db.execute("UPDATE comparator_intents SET status='discarded',lease_owner=NULL,lease_until=NULL,attempt_id=NULL,updated_ts=? WHERE deployment_revision=? AND capture_id=? AND status IN ('prepared','pending','leased')",(now,deployment_revision,capture_id))
            if changed.rowcount != 1:
                db.rollback(); return False
            db.execute("DELETE FROM meta WHERE key=?",(self._comparator_adoption_key(deployment_revision,capture_id),))
            db.commit(); return True

    def _leased_comparator_row(self, db: sqlite3.Connection, intent: dict[str,Any], now: float) -> dict[str,Any] | None:
        required={"deployment_revision","capture_id","lease_owner","lease_epoch","attempt_id"}
        if not isinstance(intent,dict) or not required <= set(intent): raise ValueError("invalid comparator lease")
        if isinstance(intent['deployment_revision'],bool) or not isinstance(intent['deployment_revision'],int) or intent['deployment_revision'] < 0 or not isinstance(intent['capture_id'],str) or not intent['capture_id']:
            raise ValueError("invalid comparator lease")
        row=db.execute("SELECT session_hash,cohort_id,epoch,route_generation,candidate_arm,audio_sha256,spool_name,candidate_success,candidate_fallback,candidate_latency,candidate_coverage_ok,candidate_hallucination_ok,candidate_identity_valid,lease_until FROM comparator_intents WHERE deployment_revision=? AND capture_id=? AND status='leased' AND lease_owner=? AND lease_epoch=? AND attempt_id=?",(intent['deployment_revision'],intent['capture_id'],intent['lease_owner'],intent['lease_epoch'],intent['attempt_id'])).fetchone()
        if row is None or row[-1] is None or float(row[-1]) < now: return None
        keys=("session_hash","cohort_id","epoch","route_generation","candidate_arm","audio_sha256","spool_name","candidate_success","candidate_fallback","candidate_latency","candidate_coverage_ok","candidate_hallucination_ok","candidate_identity_valid","lease_until")
        value=dict(zip(keys,row)); current=int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0]); cohort=db.execute("SELECT status,clear_epoch FROM cohorts WHERE cohort_id=?",(value['cohort_id'],)).fetchone(); assigned=db.execute("SELECT selected FROM route_assignments WHERE route_generation=? AND session_hash=?",(value['route_generation'],value['session_hash'])).fetchone()
        if current != value['epoch'] or db.execute("SELECT 1 FROM meta WHERE key='scrub_pending'").fetchone() or not cohort or cohort[0] != 'evaluated_passed' or cohort[1] != current or not assigned or not assigned[0]: return None
        value['candidate']={"success":bool(value.pop('candidate_success')),"fallback":bool(value.pop('candidate_fallback')),"latency":float(value.pop('candidate_latency')),"coverage_ok":bool(value.pop('candidate_coverage_ok')),"hallucination_ok":bool(value.pop('candidate_hallucination_ok')),"identity_valid":bool(value.pop('candidate_identity_valid'))}
        value['deployment_revision']=intent['deployment_revision']; value['capture_id']=intent['capture_id']
        return value

    def _insert_fresh_comparator_runtime_pair(self, db: sqlite3.Connection, row: dict[str,Any], incumbent_fields: tuple[int,int,float,int,int,int], now: float) -> bool:
        candidate_fields=self._runtime_evidence_fields(row['candidate'])
        values=(row['deployment_revision'],row['capture_id'],row['session_hash'],row['cohort_id'],row['epoch'],row['candidate_arm'],
                *candidate_fields,*incumbent_fields,now,int(now//86400),'complete')
        try:
            db.execute("INSERT INTO runtime_pairs(deployment_revision,capture_id,session_hash,cohort_id,epoch,candidate_arm,candidate_success,candidate_fallback,candidate_latency,candidate_coverage_ok,candidate_hallucination_ok,candidate_identity_valid,incumbent_success,incumbent_fallback,incumbent_latency,incumbent_coverage_ok,incumbent_hallucination_ok,incumbent_identity_valid,ts,day,state) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",values)
            return True
        except sqlite3.IntegrityError:
            return False

    def _exact_comparator_runtime_pair(self, db: sqlite3.Connection, row: dict[str,Any], incumbent_fields: tuple[int,int,float,int,int,int]) -> bool:
        existing=db.execute("SELECT session_hash,cohort_id,epoch,candidate_arm,candidate_success,candidate_fallback,candidate_latency,candidate_coverage_ok,candidate_hallucination_ok,candidate_identity_valid,incumbent_success,incumbent_fallback,incumbent_latency,incumbent_coverage_ok,incumbent_hallucination_ok,incumbent_identity_valid,state FROM runtime_pairs WHERE deployment_revision=? AND capture_id=?",(row['deployment_revision'],row['capture_id'])).fetchone()
        if existing is None: return False
        candidate_fields=self._runtime_evidence_fields(row['candidate'])
        return tuple(existing) == (row['session_hash'],row['cohort_id'],row['epoch'],row['candidate_arm'],*candidate_fields,*incumbent_fields,'complete')

    def finish_comparator_pair(self, intent: dict[str,Any], incumbent: dict[str,Any], _fail_after_pair: bool=False) -> str | bool:
        incumbent_fields=self._runtime_evidence_fields(incumbent); now=self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row=self._leased_comparator_row(db,intent,now)
                if row is None: db.rollback(); return False
                inserted=self._insert_fresh_comparator_runtime_pair(db,row,incumbent_fields,now)
                if not inserted and not self._exact_comparator_runtime_pair(db,row,incumbent_fields): db.rollback(); return False
                if _fail_after_pair: raise RuntimeError("injected comparator pair failure")
                changed=db.execute("UPDATE comparator_intents SET status='complete',lease_owner=NULL,lease_until=NULL,attempt_id=NULL,updated_ts=? WHERE deployment_revision=? AND capture_id=? AND status='leased' AND lease_owner=? AND lease_epoch=? AND attempt_id=? AND lease_until>=?",(now,row['deployment_revision'],row['capture_id'],intent['lease_owner'],intent['lease_epoch'],intent['attempt_id'],now))
                if changed.rowcount != 1: db.rollback(); return False
                db.commit(); return 'inserted' if inserted else 'replayed'
            except Exception:
                db.rollback(); raise

    def _runtime_pair_write(self, *, deployment_revision: int, capture_id: str, session_id: str, cohort_id: str,
                            epoch: int, route_generation: int, candidate_arm: str, candidate: dict[str, Any],
                            incumbent: dict[str, Any], state: str) -> str | bool:
        if state == "complete":
            return self.record_runtime_pair(deployment_revision=deployment_revision,capture_id=capture_id,session_id=session_id,
                cohort_id=cohort_id,epoch=epoch,route_generation=route_generation,candidate_arm=candidate_arm,candidate=candidate,incumbent=incumbent)
        # Candidate halves use the same authoritative insert predicates as a
        # complete pair, but retain an explicitly unresolved comparator.
        return self.record_runtime_pair(deployment_revision=deployment_revision,capture_id=capture_id,session_id=session_id,
            cohort_id=cohort_id,epoch=epoch,route_generation=route_generation,candidate_arm=candidate_arm,candidate=candidate,incumbent=incumbent)

    def record_preboundary_shadow_audit(self, *args: Any, **kwargs: Any) -> bool:
        """Retired unsafe split-write API.

        Shadow aggregates must share the lease CAS that terminalizes their
        accepted label; accepting an independently supplied job key would
        permit crash gaps and unbounded post-boundary classifications.
        """
        raise RuntimeError("shadow audit must be finalized with accepted silver")

    def shadow_audit_status(self) -> dict[str, Any]:
        with self._connect() as db:
            row=db.execute("SELECT COUNT(*),COALESCE(SUM(word_distance),0),COALESCE(SUM(reference_words),0),COALESCE(SUM(character_distance),0),COALESCE(SUM(reference_characters),0) FROM shadow_audits WHERE preboundary=1").fetchone()
        count,wd,rw,cd,rc=(int(value) for value in row)
        return {"count":count,"word_distance":wd,"reference_words":rw,"character_distance":cd,"reference_characters":rc,
                "wer":wd/rw if rw else None,"cer":cd/rc if rc else None}

    def runtime_metrics(self, deployment_revision: int) -> dict[str, Any]:
        with self._connect() as db:
            pairs=db.execute("SELECT session_hash,candidate_arm,candidate_success,candidate_fallback,candidate_latency,candidate_coverage_ok,candidate_hallucination_ok,candidate_identity_valid,incumbent_success,incumbent_fallback,incumbent_latency,incumbent_coverage_ok,incumbent_hallucination_ok,incumbent_identity_valid,ts,day,state FROM runtime_pairs WHERE deployment_revision=? ORDER BY ts",(int(deployment_revision),)).fetchall()
        if pairs:
            complete=[row for row in pairs if row[16] == "complete"]
            candidate_latency=[float(row[4]) for row in complete]; incumbent_latency=[float(row[10]) for row in complete]
            p95=lambda value: sorted(value)[max(0,(95*len(value)+99)//100-1)] if value else None
            candidate_failure=lambda row: not bool(row[2]) or bool(row[3])
            incumbent_valid=lambda row: bool(row[8]) and not bool(row[9]) and bool(row[11]) and bool(row[12]) and bool(row[13])
            consecutive=0
            for row in reversed(pairs):
                if not candidate_failure(row): break
                consecutive += 1
            return {"candidate_records":len(pairs),"captures":len(complete),"days":len({row[15] for row in complete}),
                    "fallback_error":sum(candidate_failure(row) for row in complete)/len(complete) if complete else 1.0,
                    "incumbent_error":sum(not incumbent_valid(row) for row in complete)/len(complete) if complete else None,
                    "p95_ratio":p95(candidate_latency)/max(p95(incumbent_latency) or 1e-9,1e-9) if complete else None,
                    "coverage_ok":bool(complete) and all(bool(row[5]) for row in complete),"hallucination_ok":bool(complete) and all(bool(row[6]) for row in complete),
                    "identity_invalid":any(not bool(row[7]) or (row[16] == "complete" and not bool(row[13])) for row in pairs),
                    "consecutive_failures":consecutive,"comparator_resolved":bool(complete) and len(complete) == len(pairs) and all(incumbent_valid(row) for row in complete)}
        with self._connect() as db:
            rows=db.execute("SELECT session_hash,arm_id,ts,day,success,fallback,latency,coverage_ok,hallucination_ok,identity_valid FROM runtime_observations WHERE deployment_revision=?",(int(deployment_revision),)).fetchall()
        if not rows: return {"candidate_records":0,"captures":0,"days":0,"fallback_error":1.0,"incumbent_error":None,"p95_ratio":None,"coverage_ok":False,"hallucination_ok":False,"identity_invalid":False,"consecutive_failures":0,"comparator_resolved":False}
        # Baseline is observed on a separate arm when routed sessions fall
        # back.  Until that comparator exists latency is unresolved, never a
        # fabricated absolute-seconds ratio.
        by_arm: dict[str,list[float]]={}
        for row in rows: by_arm.setdefault(str(row[1]),[]).append(float(row[6]))
        incumbent=by_arm.get("whisper-large-v3-turbo-en", [])
        candidate_arms={arm: values for arm,values in by_arm.items() if arm != "whisper-large-v3-turbo-en"}
        candidate=[value for values in candidate_arms.values() for value in values]
        p95=lambda value: sorted(value)[max(0,(95*len(value)+99)//100-1)] if value else None
        ratio=(p95(candidate)/max(p95(incumbent) or 1e-9,1e-9)) if candidate and incumbent else None
        candidate_rows=[row for row in rows if row[1] != "whisper-large-v3-turbo-en"]
        baseline_rows=[row for row in rows if row[1] == "whisper-large-v3-turbo-en"]
        # A failed/fallback/unsafe incumbent record documents an unresolved
        # comparator, never a successful paired control merely because it
        # shares the session hash.
        baseline_valid=[row for row in baseline_rows if bool(row[4]) and not bool(row[5]) and bool(row[7]) and bool(row[8]) and bool(row[9])]
        failure=lambda row: not bool(row[4]) or bool(row[5])
        consecutive=0
        for row in sorted(candidate_rows, key=lambda value: (value[2], value[0]))[::-1]:
            if not failure(row): break
            consecutive += 1
        return {"captures":len(candidate_rows),"days":len({x[3] for x in candidate_rows}),
                "fallback_error":sum(failure(x) for x in candidate_rows)/max(1,len(candidate_rows)),
                "incumbent_error":sum(failure(x) for x in baseline_rows)/len(baseline_rows) if baseline_rows else None,
                "p95_ratio":ratio,"coverage_ok":bool(candidate_rows) and all(bool(x[7]) for x in candidate_rows),
                "hallucination_ok":bool(candidate_rows) and all(bool(x[8]) for x in candidate_rows),
                "identity_invalid":any(not bool(x[9]) for x in rows),"consecutive_failures":consecutive,
                # A session can contain many captures.  Merely seeing one
                # incumbent row for its hash is not a comparator for hundreds
                # of candidate calls.  Until the durable per-capture token
                # migration is available, require exact cardinality per
                # session rather than the old subset test.
                "comparator_resolved":bool(candidate_rows and baseline_rows) and
                    (lambda left,right: left == right)(
                        {key:sum(row[0] == key for row in candidate_rows) for key in {row[0] for row in candidate_rows}},
                        {key:sum(row[0] == key for row in baseline_valid) for key in {row[0] for row in baseline_valid}})}

    def cohort_status(self, cohort_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT status,metadata,clear_epoch FROM cohorts WHERE cohort_id=?", (cohort_id,)).fetchone()
            if row is None: return None
            meta = json.loads(row[1]); meta.update({"cohort_id": cohort_id, "status": row[0], "clear_epoch": row[2]})
            return meta

    def terminal_cohort_ids(self) -> list[str]:
        """Current-epoch terminal cohorts needing restart reconciliation."""
        with self._connect() as db:
            return [str(row[0]) for row in db.execute("SELECT cohort_id FROM cohorts WHERE clear_epoch=? AND status IN ('evaluated_passed','evaluated_failed','invalidated') ORDER BY created_ts", (self.epoch(),))]

    def authorization_replayable(self, cohort_id: str) -> bool:
        """Whether a passed terminal is an unconsumed eval→authorize crash.

        This deliberately has no compatibility default: malformed/unknown
        authority is not replayable.
        """
        with self._connect() as db:
            row=db.execute("SELECT disposition FROM cohort_authorization WHERE cohort_id=?",(cohort_id,)).fetchone()
            return row is None

    def authorization_disposition(self, cohort_id: str) -> str | None:
        with self._connect() as db:
            row=db.execute("SELECT disposition FROM cohort_authorization WHERE cohort_id=?",(cohort_id,)).fetchone()
            return str(row[0]) if row is not None else None

    def authorization_intent(self, cohort_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row=db.execute("SELECT disposition,expected_revision,expected_state_digest,token FROM cohort_authorization WHERE cohort_id=?",(cohort_id,)).fetchone()
            if row is None: return None
            disposition,revision,digest,token=row
            if disposition not in {"authorizing","authorized","consumed"}: return None
            return {"disposition":disposition,"expected_revision":revision,"expected_state_digest":digest,"token":token}

    def claim_cohort_authorization(self, cohort_id: str, *, expected_revision: int, expected_state_digest: str) -> bool:
        """Durably reserve eval→deployment publication before JSON write.

        ``authorizing`` is a recoverable two-store intent: no published
        rollout can exist without at least this SQLite fence.
        """
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT clear_epoch,status FROM cohorts WHERE cohort_id=?",(cohort_id,)).fetchone()
            epoch=int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            prior=db.execute("SELECT disposition FROM cohort_authorization WHERE cohort_id=?",(cohort_id,)).fetchone()
            if (not isinstance(expected_revision,int) or isinstance(expected_revision,bool) or expected_revision < 0 or
                    not isinstance(expected_state_digest,str) or len(expected_state_digest) != 64 or
                    any(ch not in "0123456789abcdef" for ch in expected_state_digest)):
                db.rollback(); return False
            if row is None or int(row[0]) != epoch or row[1] != "evaluated_passed" or (prior is not None and prior[0] == "consumed"):
                db.rollback(); return False
            if prior is None:
                db.execute("INSERT INTO cohort_authorization(cohort_id,disposition,expected_revision,expected_state_digest,token,updated_ts) VALUES(?,?,?,?,?,?)",(cohort_id,"authorizing",expected_revision,expected_state_digest,uuid.uuid4().hex,self.clock()))
            db.commit(); return True

    def release_cohort_authorization_claim(self, cohort_id: str) -> None:
        """Undo only an unpublished authorization reservation."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM cohort_authorization WHERE cohort_id=? AND disposition='authorizing'",(cohort_id,))
            db.commit()

    def mark_cohort_authorized(self, cohort_id: str) -> bool:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT clear_epoch,status FROM cohorts WHERE cohort_id=?",(cohort_id,)).fetchone()
            epoch=int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            if row is None or int(row[0]) != epoch or row[1] != "evaluated_passed":
                db.rollback(); return False
            prior=db.execute("SELECT disposition FROM cohort_authorization WHERE cohort_id=?",(cohort_id,)).fetchone()
            if prior is None or prior[0] not in {"authorizing","authorized"}:
                db.rollback(); return False
            db.execute("UPDATE cohort_authorization SET disposition='authorized',updated_ts=? WHERE cohort_id=? AND disposition IN ('authorizing','authorized')",(self.clock(),cohort_id))
            db.commit(); return True

    def consume_cohort_authorization(self, cohort_id: str | None) -> None:
        if not isinstance(cohort_id,str) or not cohort_id: return
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM cohorts WHERE cohort_id=?",(cohort_id,)).fetchone() is not None:
                db.execute("INSERT INTO cohort_authorization(cohort_id,disposition,updated_ts) VALUES(?,?,?) ON CONFLICT(cohort_id) DO UPDATE SET disposition=excluded.disposition,updated_ts=excluded.updated_ts",(cohort_id,"consumed",self.clock()))
            db.commit()

    def cohort_depends_on_history(self, cohort_id: str, history_id: str) -> bool:
        with self._connect() as db:
            return db.execute("SELECT 1 FROM cohort_members WHERE cohort_id=? AND history_id=?", (cohort_id,history_id)).fetchone() is not None

    def fresh_holdout_status(self, *, excluding: str, after_ts: float = 0) -> dict[str, Any]:
        """Only a later independently frozen passed horizon can be a holdout."""
        with self._connect() as db:
            source=db.execute("SELECT member_hash,metadata,clear_epoch FROM cohorts WHERE cohort_id=?", (excluding,)).fetchone()
            if source is None or int(source[2]) != self.epoch(): return {}
            source_meta=json.loads(source[1])
            row=db.execute("""SELECT c.cohort_id,c.metadata,c.member_hash FROM cohorts c
                JOIN prospective_horizons h ON h.frozen_cohort_id=c.cohort_id
                WHERE c.cohort_id<>? AND c.status='evaluated_passed' AND c.clear_epoch=?
                  AND h.start_ts>? ORDER BY c.created_ts DESC LIMIT 1""", (excluding,self.epoch(),float(after_ts))).fetchone()
            if row is None: return {}
            meta=json.loads(row[1]); meta["cohort_id"]=row[0]
            if (row[2] == source[0] or meta.get("teacher_receipt_hash") != source_meta.get("teacher_receipt_hash") or
                    meta.get("evaluator_hash") != source_meta.get("evaluator_hash") or
                    meta.get("candidate_hashes") != source_meta.get("candidate_hashes") or
                    meta.get("winner") != source_meta.get("winner") or not bool(meta.get("coverage_frozen"))):
                return {}
            return meta

    def complete_cohort(self, cohort_id: str, outcome: dict[str, Any]) -> bool:
        """CAS-publish content-free evaluation metadata for deployment gating."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT clear_epoch,status,metadata FROM cohorts WHERE cohort_id=?",(cohort_id,)).fetchone()
            epoch=int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            if row is None or row[0]!=epoch: db.rollback();return False
            if row[1] in {'evaluated_passed','evaluated_failed'}:
                stored=json.loads(row[2]).get("evaluation_outcome")
                db.commit()
                return isinstance(stored,dict) and stored == outcome
            if row[1] not in {'frozen','evaluating'}: db.rollback();return False
            meta=json.loads(row[2]); coverage=int(meta.get("accepted",0))/max(1,int(meta.get("raw",0)))
            horizon_row=db.execute("SELECT receipt_hash FROM prospective_horizons WHERE frozen_cohort_id=?", (cohort_id,)).fetchone()
            horizon={"receipt_hash": horizon_row[0]} if horizon_row else {}
            meta.update({'paired_gates':bool(outcome.get('winner')),'identity_hash':outcome.get('identity_hash'),
                         'winner':outcome.get('winner'),
                         'teacher_receipt_hash':horizon.get('receipt_hash'),
                         'coverage_frozen': .40 <= coverage <= .90, 'coverage': coverage,
                         # This is content-free sufficient statistics only.
                         # It is the authority used to repair a crashed JSON
                         # mirror; never derive deployment state from JSON.
                         'evaluation_outcome':outcome})
            changed=db.execute("UPDATE cohorts SET status=?,metadata=? WHERE cohort_id=? AND status=?",('evaluated_passed' if outcome.get('winner') else 'evaluated_failed',json.dumps(meta,sort_keys=True),cohort_id,row[1]))
            if changed.rowcount!=1:db.rollback();return False
            db.execute("UPDATE prospective_horizons SET status='evaluated' WHERE frozen_cohort_id=? AND clear_epoch=?", (cohort_id,epoch))
            self._scrub_unneeded_references(db, epoch, self.clock())
            db.commit();return True

    def store_calibration(self, *, manifest_hash: str, source_hash: str, policy_hash: str, split_hash: str,
                          receipt_hash: str, results: list[dict[str, Any]], expected_total: int | None = None) -> dict[str, Any]:
        """Persist only content-free calibration outcomes after one consumed run."""
        now = self.clock(); calibration_id = _digest([manifest_hash, policy_hash, split_hash])
        accepted = sum(row.get("outcome") == "accepted" for row in results)
        errors = sum(not bool(row.get("reference_match")) for row in results if row.get("outcome") == "accepted")
        clusters = len({row.get("cluster_hash") for row in results if row.get("outcome") == "accepted"})
        total = len(results) if expected_total is None else int(expected_total)
        terminal = len(results)
        coverage = accepted / total if total else 0.0
        state = "passed" if terminal == total and .40 <= coverage <= .90 and accepted >= 149 and errors == 0 and clusters >= 149 else "failed"
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            epoch = int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            run=db.execute("SELECT source_hash,policy_hash,split_hash,receipt_hash,state FROM calibration_runs WHERE manifest_hash=?", (manifest_hash,)).fetchone()
            progress=[dict(zip(("item_hash","cluster_hash","outcome","reference_match"), row)) for row in db.execute("SELECT item_hash,cluster_hash,outcome,reference_match FROM calibration_progress WHERE manifest_hash=? ORDER BY item_hash", (manifest_hash,))]
            supplied=sorted(({"item_hash":str(item.get("item_hash")),"cluster_hash":str(item.get("cluster_hash")),"outcome":str(item.get("outcome")),"reference_match":int(bool(item.get("reference_match")))} for item in results), key=lambda item:item["item_hash"])
            if (run is None or tuple(run[:4]) != (source_hash,policy_hash,split_hash,receipt_hash) or run[4] != "running" or
                    supplied != progress or expected_total is None or int(expected_total) != len(progress)):
                db.rollback(); raise RuntimeError("calibration finalization is not the persisted run")
            exists = db.execute("SELECT calibration_id FROM calibrations WHERE policy_hash=? AND split_hash=?", (policy_hash, split_hash)).fetchone()
            if exists:
                db.rollback(); raise RuntimeError("calibration policy/split already consumed")
            db.execute("INSERT INTO calibrations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                       (calibration_id, manifest_hash, source_hash, policy_hash, split_hash, receipt_hash, epoch, state,
                        accepted, errors, clusters, now))
            db.executemany("INSERT INTO calibration_items(calibration_id,item_hash,cluster_hash,outcome,reference_match) VALUES(?,?,?,?,?)",
                           [(calibration_id, r["item_hash"], r["cluster_hash"], r["outcome"], int(bool(r.get("reference_match")))) for r in results])
            db.execute("INSERT INTO calibration_coverage VALUES(?,?,?,?,?)", (manifest_hash,total,terminal,accepted,now))
            db.commit()
            return {"calibration_id": calibration_id, "state": state, "accepted": accepted, "total": total, "terminal": terminal,
                    "coverage": coverage, "reference_errors": errors, "clusters": clusters}

    def begin_calibration(self, *, manifest_hash: str, source_hash: str, policy_hash: str, split_hash: str, receipt_hash: str) -> list[dict[str, str]]:
        """Resume only the exact frozen run; a completed split cannot retune."""
        now = self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE"); epoch = int(db.execute("SELECT value FROM meta WHERE key='clear_epoch'").fetchone()[0])
            row = db.execute("SELECT manifest_hash,state,clear_epoch,source_hash,receipt_hash FROM calibration_runs WHERE policy_hash=? AND split_hash=?", (policy_hash, split_hash)).fetchone()
            if row is None:
                db.execute("INSERT INTO calibration_runs VALUES(?,?,?,?,?,?, 'running',?,?)", (manifest_hash, policy_hash, split_hash, source_hash, receipt_hash, epoch, now, now))
            elif row[0] != manifest_hash or row[1] != "running" or row[3] != source_hash or row[4] != receipt_hash:
                db.rollback(); raise RuntimeError("calibration policy/split already consumed")
            rows = db.execute("SELECT item_hash,cluster_hash,outcome,reference_match FROM calibration_progress WHERE manifest_hash=?", (manifest_hash,)).fetchall()
            db.commit()
            return [dict(zip(("item_hash", "cluster_hash", "outcome", "reference_match"), row)) for row in rows]

    def record_calibration_item(self, manifest_hash: str, result: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR REPLACE INTO calibration_progress(manifest_hash,item_hash,cluster_hash,outcome,reference_match) VALUES(?,?,?,?,?)",
                       (manifest_hash, result["item_hash"], result["cluster_hash"], result["outcome"], int(bool(result["reference_match"]))))
            db.execute("UPDATE calibration_runs SET updated_ts=? WHERE manifest_hash=? AND state='running'", (self.clock(), manifest_hash))
            db.commit()

    def finish_calibration(self, manifest_hash: str, *, expected_total: int | None = None) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT source_hash,policy_hash,split_hash,receipt_hash FROM calibration_runs WHERE manifest_hash=? AND state='running'", (manifest_hash,)).fetchone()
            if row is None: raise RuntimeError("calibration run unavailable")
            results = [dict(zip(("item_hash", "cluster_hash", "outcome", "reference_match"), x)) for x in db.execute("SELECT item_hash,cluster_hash,outcome,reference_match FROM calibration_progress WHERE manifest_hash=?", (manifest_hash,))]
        result = self.store_calibration(manifest_hash=manifest_hash, source_hash=row[0], policy_hash=row[1], split_hash=row[2], receipt_hash=row[3], results=results, expected_total=expected_total)
        with self._connect() as db:
            db.execute("UPDATE calibration_runs SET state='completed',updated_ts=? WHERE manifest_hash=?", (self.clock(), manifest_hash))
        return result

    def calibration_status(self) -> dict[str, Any] | None:
        with self._connect() as db:
            v2=db.execute("SELECT holdout_key,manifest_hash,source_hash,protocol_hash,policy_hash,receipt_hash,state,result FROM calibration_universes ORDER BY created_ts DESC LIMIT 1").fetchone()
            if v2 is not None:
                from calibration_manifest_builder import SOURCE_HASH, _protocol_hash, LEDGER_HASH
                from teacher_consensus import policy_hash as current_policy_hash
                expected_count=db.execute("SELECT COUNT(*) FROM calibration_expected_v2 WHERE holdout_key=?",(v2[0],)).fetchone()[0]
                item_count=db.execute("SELECT COUNT(*) FROM calibration_items_v2 WHERE holdout_key=? AND state='terminal'",(v2[0],)).fetchone()[0]
                raw_item_count=db.execute("SELECT COUNT(*) FROM calibration_items_v2 WHERE holdout_key=?",(v2[0],)).fetchone()[0]
                clusters=db.execute("SELECT COUNT(DISTINCT cluster_hash) FROM calibration_expected_v2 WHERE holdout_key=?",(v2[0],)).fetchone()[0]
                frozen=[tuple(row) for row in db.execute("SELECT ordinal,item_hash,cluster_hash FROM calibration_expected_v2 WHERE holdout_key=? ORDER BY ordinal",(v2[0],))]
                ledger=hashlib.sha256(json.dumps(frozen,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
                if (v2[0] != SOURCE_HASH or v2[2] != SOURCE_HASH or v2[3] != _protocol_hash() or v2[4] != current_policy_hash() or
                        expected_count != 500 or item_count != 500 or raw_item_count != 500 or clusters != 339 or ledger != LEDGER_HASH or
                        [row[0] for row in frozen] != list(range(500))):
                    return None
                try:
                    result=json.loads(v2[7]) if v2[7] else {}
                except (TypeError, ValueError, json.JSONDecodeError):
                    return None
                if not isinstance(result,dict) or set(result) != {"state","accepted","clusters","reference_errors","total"}:
                    return None
                rows=db.execute("SELECT e.ordinal,e.item_hash,e.cluster_hash,i.state,i.outcome,i.reference_match FROM calibration_expected_v2 e LEFT JOIN calibration_items_v2 i ON i.holdout_key=e.holdout_key AND i.item_hash=e.item_hash WHERE e.holdout_key=? ORDER BY e.ordinal",(v2[0],)).fetchall()
                if (len(rows) != 500 or any(row[0] != index or row[3] != "terminal" or row[4] not in {"accepted","abstained"} or
                        (row[4] == "accepted" and row[5] not in {0,1}) or (row[4] == "abstained" and row[5] is not None)
                        for index,row in enumerate(rows))):
                    return None
                accepted=sum(row[4] == "accepted" for row in rows)
                accepted_clusters=len({row[2] for row in rows if row[4] == "accepted"})
                errors=sum(row[4] == "accepted" and row[5] != 1 for row in rows)
                passed=(5*accepted >= 2*500 and 10*accepted <= 9*500 and accepted_clusters >= 149 and errors == 0)
                expected_state="passed" if passed else "failed"
                if (v2[6] != expected_state or result.get("state") != expected_state or result.get("accepted") != accepted or
                        result.get("clusters") != accepted_clusters or result.get("reference_errors") != errors or result.get("total") != 500):
                    return None
                return {"calibration_id":v2[0],"manifest_hash":v2[1],"source_hash":v2[2],"protocol_hash":v2[3],"policy_hash":v2[4],"receipt_hash":v2[5],"state":v2[6],"accepted":result.get("accepted",0),"reference_errors":result.get("reference_errors",0),"clusters":result.get("clusters",0),"total":result.get("total",500),"terminal":result.get("total",0),"clear_epoch":None}
            # Legacy calibration rows lack immutable expected-item and public
            # universe identity.  They are retained only for migration audit,
            # never for deployment authority.
            return None
            row = db.execute("SELECT c.calibration_id,c.state,c.manifest_hash,c.source_hash,c.policy_hash,c.split_hash,c.receipt_hash,c.accepted,c.reference_errors,c.clusters,c.clear_epoch,COALESCE(v.total,0),COALESCE(v.terminal,0) FROM calibrations c LEFT JOIN calibration_coverage v ON v.manifest_hash=c.manifest_hash ORDER BY c.created_ts DESC LIMIT 1").fetchone()
            if row is None: return None
            return dict(zip(("calibration_id", "state", "manifest_hash", "source_hash", "policy_hash", "split_hash", "receipt_hash", "accepted", "reference_errors", "clusters", "clear_epoch", "total", "terminal"), row))

    def begin_calibration_v2(self, *, holdout_key: str, manifest_hash: str, source_hash: str,
                             protocol_hash: str, policy_hash: str, receipt_hash: str,
                             expected: list[dict[str, str]]) -> None:
        """Consume a canonical public source universe and freeze its exact items."""
        # This authority seam refuses caller-selected source/protocol/policy
        # aliases even when invoked directly (rather than through the worker).
        from calibration_manifest_builder import SOURCE_HASH, _protocol_hash, LEDGER_HASH
        from teacher_consensus import policy_hash as current_policy_hash
        if holdout_key != SOURCE_HASH or source_hash != SOURCE_HASH or protocol_hash != _protocol_hash() or policy_hash != current_policy_hash():
            raise RuntimeError("calibration authority identity is not current")
        ledger=hashlib.sha256(json.dumps([(index,str(item.get("item_hash","")),str(item.get("cluster_hash",""))) for index,item in enumerate(expected)],separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
        if ledger != LEDGER_HASH:
            raise RuntimeError("calibration expected ledger is not compiled")
        if len(expected) != 500 or len({(x.get("item_hash"),x.get("cluster_hash")) for x in expected}) != len(expected) or len({str(x.get("cluster_hash")) for x in expected}) != 339 or any(sum(str(y.get("cluster_hash")) == str(x.get("cluster_hash")) for y in expected) > 2 for x in expected):
            raise RuntimeError("invalid immutable calibration plan")
        now=self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT manifest_hash,source_hash,protocol_hash,policy_hash,receipt_hash FROM calibration_universes WHERE holdout_key=?",(holdout_key,)).fetchone()
            identity=(manifest_hash,source_hash,protocol_hash,policy_hash,receipt_hash)
            if row is None:
                db.execute("INSERT INTO calibration_universes(holdout_key,manifest_hash,source_hash,protocol_hash,policy_hash,receipt_hash,state,created_ts,updated_ts) VALUES(?,?,?,?,?,?, 'running',?,?)",(holdout_key,*identity,now,now))
                db.executemany("INSERT INTO calibration_expected_v2 VALUES(?,?,?,?)",[(holdout_key,i,str(item["item_hash"]),str(item["cluster_hash"])) for i,item in enumerate(expected)])
                db.executemany("INSERT INTO calibration_items_v2(holdout_key,item_hash,state,updated_ts) VALUES(?,?,'pending',?)",[(holdout_key,str(item["item_hash"]),now) for item in expected])
            elif tuple(row) != identity:
                db.rollback(); raise RuntimeError("public calibration universe already consumed")
            else:
                frozen=[{"item_hash":r[0],"cluster_hash":r[1]} for r in db.execute("SELECT item_hash,cluster_hash FROM calibration_expected_v2 WHERE holdout_key=? ORDER BY ordinal",(holdout_key,))]
                supplied=[{"item_hash":str(item["item_hash"]),"cluster_hash":str(item["cluster_hash"])} for item in expected]
                if frozen != supplied:
                    db.rollback(); raise RuntimeError("calibration resume plan changed")
            db.commit()

    def claim_calibration_v2(self, holdout_key: str, owner: str) -> dict[str, Any] | None:
        now=self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE calibration_items_v2 SET state='pending',lease_owner=NULL,lease_token=NULL,lease_until=NULL,updated_ts=? WHERE holdout_key=? AND state='leased' AND lease_until<?",(now,holdout_key,now))
            row=db.execute("SELECT i.item_hash,e.cluster_hash,i.attempts FROM calibration_items_v2 i JOIN calibration_expected_v2 e USING(holdout_key,item_hash) WHERE i.holdout_key=? AND i.state='pending' ORDER BY e.ordinal LIMIT 1",(holdout_key,)).fetchone()
            if row is None: db.commit(); return None
            token=uuid.uuid4().hex
            if db.execute("UPDATE calibration_items_v2 SET state='leased',attempts=attempts+1,lease_owner=?,lease_token=?,lease_until=?,updated_ts=? WHERE holdout_key=? AND item_hash=? AND state='pending'",(owner,token,now+self.lease_seconds,now,holdout_key,row[0])).rowcount != 1:
                db.rollback(); return None
            db.commit(); return {"holdout_key":holdout_key,"item_hash":row[0],"cluster_hash":row[1],"attempts":int(row[2])+1,"owner":owner,"token":token}

    def finish_calibration_v2(self, item: dict[str, Any], *, outcome: str, reference_match: bool | None,
                              code: str = "terminal") -> bool:
        allowed_codes={"exact_unanimous","teacher_configuration","teacher_failure","unicode_unsupported","blank_or_no_speech","mismatch","high_risk","terminal"}
        if outcome not in {"accepted","abstained"} or (outcome == "accepted" and not isinstance(reference_match,bool)) or (outcome == "abstained" and reference_match is not None):
            raise ValueError("invalid calibration terminal result")
        if code not in allowed_codes: raise ValueError("invalid content-free calibration code")
        now=self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed=db.execute("UPDATE calibration_items_v2 SET state='terminal',outcome=?,reference_match=?,code=?,lease_owner=NULL,lease_token=NULL,lease_until=NULL,updated_ts=? WHERE holdout_key=? AND item_hash=? AND state='leased' AND lease_owner=? AND lease_token=? AND lease_until>=?",(outcome,None if reference_match is None else int(reference_match),code,now,item["holdout_key"],item["item_hash"],item["owner"],item["token"],now))
            if changed.rowcount != 1:
                # A crash after terminal commit but before the caller received
                # its acknowledgement is idempotent only for byte-identical
                # terminal evidence; conflicting replay is an authority error.
                old=db.execute("SELECT state,outcome,reference_match,code FROM calibration_items_v2 WHERE holdout_key=? AND item_hash=?",(item["holdout_key"],item["item_hash"])).fetchone()
                if old and old[0] == "terminal" and old[1] == outcome and old[2] == (None if reference_match is None else int(reference_match)) and old[3] == code:
                    db.commit(); return True
                db.rollback(); return False
            db.commit(); return True

    def release_calibration_v2(self, item: dict[str, Any], *, code: str) -> bool:
        """Return an infrastructure-interrupted V2 lease without padding data."""
        if code not in {"RuntimeError","OSError","TimeoutError","TimeoutExpired","InfrastructureError"}:
            raise ValueError("invalid calibration infrastructure code")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed=db.execute("UPDATE calibration_items_v2 SET state='pending',lease_owner=NULL,lease_token=NULL,lease_until=NULL,code=?,updated_ts=? WHERE holdout_key=? AND item_hash=? AND state='leased' AND lease_owner=? AND lease_token=?",(code,self.clock(),item.get("holdout_key"),item.get("item_hash"),item.get("owner"),item.get("token")))
            if changed.rowcount != 1: db.rollback(); return False
            db.commit(); return True

    def finalize_calibration_v2(self, holdout_key: str) -> dict[str, Any] | None:
        """Derive the pass result transactionally from the immutable ledger."""
        # A completed row is only a durable mirror of the independently
        # recomputed authority read.  Never return its JSON blindly after a
        # later code/source/ledger change or corruption.
        with self._connect() as db:
            completed=db.execute("SELECT state FROM calibration_universes WHERE holdout_key=?",(holdout_key,)).fetchone()
        if completed and completed[0] in {"passed","failed"}:
            status=self.calibration_status()
            if not status or status.get("calibration_id") != holdout_key or status.get("state") != completed[0]:
                raise RuntimeError("completed calibration authority invalid")
            return {"state":status["state"],"accepted":status["accepted"],"clusters":status["clusters"],"reference_errors":status["reference_errors"],"total":status["total"]}
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            universe=db.execute("SELECT state,result,source_hash,protocol_hash,policy_hash FROM calibration_universes WHERE holdout_key=?",(holdout_key,)).fetchone()
            if universe is None: db.rollback(); raise RuntimeError("unknown calibration universe")
            if universe[0] in {"passed","failed"}:
                # Another finalizer may have completed between the optimistic
                # read above and this transaction.  Reuse only the same
                # independently recomputed authority result.
                db.commit()
                status=self.calibration_status()
                if not status or status.get("calibration_id") != holdout_key or status.get("state") != universe[0]:
                    raise RuntimeError("completed calibration authority invalid")
                return {"state":status["state"],"accepted":status["accepted"],"clusters":status["clusters"],"reference_errors":status["reference_errors"],"total":status["total"]}
            if universe[0] != "running":
                db.rollback(); raise RuntimeError("invalid calibration universe state")
            from calibration_manifest_builder import SOURCE_HASH, _protocol_hash, LEDGER_HASH
            from teacher_consensus import policy_hash as current_policy_hash
            frozen=[tuple(row) for row in db.execute("SELECT ordinal,item_hash,cluster_hash FROM calibration_expected_v2 WHERE holdout_key=? ORDER BY ordinal",(holdout_key,))]
            raw_item_count=db.execute("SELECT COUNT(*) FROM calibration_items_v2 WHERE holdout_key=?",(holdout_key,)).fetchone()[0]
            if (holdout_key != SOURCE_HASH or tuple(universe[2:]) != (SOURCE_HASH,_protocol_hash(),current_policy_hash()) or len(frozen) != 500 or
                    [row[0] for row in frozen] != list(range(500)) or len({row[2] for row in frozen}) != 339 or
                    hashlib.sha256(json.dumps(frozen,separators=(",",":"),ensure_ascii=False).encode()).hexdigest() != LEDGER_HASH or raw_item_count != 500):
                db.rollback(); raise RuntimeError("calibration authority identity invalid")
            rows=db.execute("SELECT e.ordinal,e.item_hash,e.cluster_hash,i.state,i.outcome,i.reference_match FROM calibration_expected_v2 e LEFT JOIN calibration_items_v2 i ON i.holdout_key=e.holdout_key AND i.item_hash=e.item_hash WHERE e.holdout_key=? ORDER BY e.ordinal",(holdout_key,)).fetchall()
            if (len(rows) != 500 or any(row[0] != index or row[3] != "terminal" or row[4] not in {"accepted","abstained"} or
                    (row[4] == "accepted" and row[5] not in {0,1}) or (row[4] == "abstained" and row[5] is not None)
                    for index,row in enumerate(rows))): db.commit(); return None
            accepted=sum(row[4] == "accepted" for row in rows); clusters=len({row[2] for row in rows if row[4] == "accepted"}); errors=sum(row[4] == "accepted" and row[5] != 1 for row in rows)
            passed=(5*accepted >= 2*500 and 10*accepted <= 9*500 and clusters >= 149 and errors == 0)
            result={"state":"passed" if passed else "failed","accepted":accepted,"clusters":clusters,"reference_errors":errors,"total":500}
            changed=db.execute("UPDATE calibration_universes SET state=?,result=?,updated_ts=? WHERE holdout_key=? AND state='running'",(result["state"],json.dumps(result,sort_keys=True),self.clock(),holdout_key))
            if changed.rowcount != 1:
                db.rollback(); raise RuntimeError("calibration finalization race")
            db.commit(); return result

    def status(self) -> dict[str, Any]:
        with self._connect() as db:
            counts = dict(db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status").fetchall())
            labels = dict(db.execute("SELECT state,COUNT(*) FROM labels GROUP BY state").fetchall())
            worker=db.execute("SELECT value FROM meta WHERE key='worker_status'").fetchone()
            try: worker_status=json.loads(worker[0]) if worker else None
            except (TypeError, ValueError, json.JSONDecodeError): worker_status=None
            return {"schema": SCHEMA_VERSION, "clear_epoch": self.epoch(), "jobs": counts, "labels": labels,
                    "worker":worker_status,
                    "accepted": int(labels.get("accepted", 0)), "abstained": int(labels.get("abstained", 0)),
                    "preboundary_shadow":self.shadow_audit_status()}

    def set_worker_status(self, state: str, reason: str | None = None) -> None:
        """Durable, content-free operational visibility for launchd workers."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('worker_status',?)", (json.dumps({"state":state,"reason":reason},sort_keys=True),))
            db.commit()
