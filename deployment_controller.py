"""Fail-closed deployment state machine for frozen silver evidence.

It intentionally does not select a model or start a canary.  Provisioning must
call these durable transitions after it has independently performed inference.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from history import _atomic_jsonl
from silver_experiment import SilverExperiment
from silver_store import SilverStore
from storage_lock import advisory_lock, ensure_private_directory, ensure_private_file


class DeploymentController:
    def __init__(self, base_dir: Path | str, *, silver: SilverStore | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.root = self.base_dir / "adaptive-learning" / "silver"
        self.path = self.root / "deployment.jsonl"
        self.manifest_path = self.root / "provisional_silver.json"
        self.silver = silver or SilverStore(self.base_dir)
        ensure_private_directory(self.root); ensure_private_file(self.path)
        if not self.path.exists() or not self.path.read_text("utf-8").strip():
            self._write({"schema": 1, "revision": 0, "tier": "shadow_only", "current": None, "lkg": None,
                         "canary": None, "calibration": None, "route_generation":0, "audit": []})

    def _read(self) -> dict[str, Any]:
        try:
            row = json.loads(self.path.read_text("utf-8").splitlines()[-1])
            if not isinstance(row, dict) or row.get("schema") != 1:
                raise ValueError
            return row
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("unknown deployment schema; refusing deployment") from exc

    def _write(self, row: dict[str, Any]) -> None:
        _atomic_jsonl(self.path, [row])

    @staticmethod
    def _state_digest(state: dict[str, Any]) -> str:
        """Content-free exact pre-publication controller identity."""
        return hashlib.sha256(json.dumps(state,sort_keys=True,separators=(",",":")).encode()).hexdigest()

    @staticmethod
    def _comparator_intent_authorized(state: dict[str,Any], intent: dict[str,Any]) -> bool:
        try:
            required={"deployment_revision","cohort_id","epoch","route_generation","candidate_arm"}
            if not isinstance(state,dict) or not isinstance(intent,dict) or not required <= set(intent): return False
            if state.get('tier') not in {'provisional_silver','canary','deployed_silver'}: return False
            current=state.get('current'); plan=state.get('canary')
            if not isinstance(current,dict) or not isinstance(plan,dict): return False
            for key in ('deployment_revision','epoch','route_generation'):
                if isinstance(intent[key],bool) or not isinstance(intent[key],int): return False
            if not all(isinstance(intent[key],str) and intent[key] for key in ('cohort_id','candidate_arm')): return False
            return (current.get('stable_id') == intent['candidate_arm'] and state.get('cohort_id') == intent['cohort_id'] and
                    state.get('route_generation') == intent['route_generation'] and plan.get('epoch') == intent['epoch'] and
                    int(plan.get('observation_revision',state.get('revision',-1))) == intent['deployment_revision'])
        except (TypeError,ValueError): return False

    def _mutate(self, fn):
        with advisory_lock(self.base_dir):
            state = self._read(); result = fn(state)
            # Idempotent metric delivery and rejected stale CAS attempts must
            # not churn the authority revision used by in-flight route tokens.
            if result in {False,"stale","replayed"}:
                return result
            state["revision"] = int(state["revision"]) + 1
            state["audit"] = state.get("audit", [])[-99:]
            self._write(state); return result

    def _winner(self, cohort_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Resolve candidate solely from SQLite's completed cohort record.

        The JSON experiment file is an intentionally non-authoritative mirror
        and may be absent/tampered after a crash without changing routing.
        """
        horizon=self.silver.horizon_for_cohort(cohort_id)
        if not horizon or horizon.get("cohort_id") != cohort_id: return None
        cohort=self.silver.cohort_status(cohort_id) or {}
        outcome=cohort.get("evaluation_outcome")
        if not isinstance(outcome,dict) or outcome.get("cohort_id") != cohort_id or outcome.get("identity_hash") != horizon.get("identity_hash"):
            return None
        winner=outcome.get("winner") if isinstance(outcome,dict) else None
        candidate=next((item for item in horizon["candidates"] if item.get("stable_id")==winner),None)
        return (candidate,outcome) if isinstance(candidate,dict) and cohort.get("winner") == winner else None

    def _signed_manifest(self, candidate: dict[str, Any], cohort_id: str, outcome: dict[str, Any]) -> dict[str, Any]:
        # Route authority covers every local semantic component that can alter
        # cohort admission, evaluation, receipt validation, or runtime
        # rollback.  A code update consequently fails old signed routes closed.
        code_hash=hashlib.sha256(b"".join(Path(__file__).with_name(name).read_bytes() for name in
            ("deployment_controller.py", "silver_experiment.py", "adaptive_runtime.py", "adaptive_worker.py",
             "silver_store.py", "evaluation.py", "audio_codec.py", "speech_backends.py", "teacher_consensus.py"))).hexdigest()
        horizon=self.silver.horizon_for_cohort(cohort_id) or {}
        body={"schema":1,"kind":"provisional_silver","candidate":candidate,"cohort_id":cohort_id,
              "identity_hash":outcome.get("identity_hash"),"winner":outcome.get("winner"),
              "receipt_hash":horizon.get("receipt_hash"),"epoch":self.silver.epoch(),"code_hash":code_hash}
        body["content_hash"]=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        body["signature"]=hashlib.sha256((body["content_hash"]+str(outcome.get("identity_hash",""))).encode()).hexdigest()
        return body

    def authorize_provisional(self, candidate: dict[str, Any] | str, cohort_id: str | None = None, *, receipt_hash: str | None = None) -> bool:
        """Publish only the winner already proven by immutable evidence.

        Compatibility arguments are intentionally ignored as authority.
        """
        if cohort_id is None: cohort_id=str(candidate)
        if self.silver.scrub_pending() is not None:
            return False
        resolved=self._winner(str(cohort_id))
        if resolved is None: return False
        frozen_candidate,outcome=resolved
        calibration = self.silver.calibration_status() or {}
        cohort = self.silver.cohort_status(str(cohort_id)) or {}
        # Calibration is public, immutable one-shot evidence.  Personal clear
        # epochs fence cohorts and routing, not a still-matching receipt/policy
        # calibration record.
        ready = (calibration.get("state") == "passed" and
                 cohort.get("status") == "evaluated_passed" and
                 int(cohort.get("raw", 0)) >= 500 and int(cohort.get("accepted", 0)) >= 200 and
                 int(cohort.get("words", 0)) >= 2000 and int(cohort.get("sessions", 0)) >= 20 and
                 int(cohort.get("days", 0)) >= 7 and bool(cohort.get("all_terminal")) and
                 bool(cohort.get("coverage_frozen")) and bool(cohort.get("paired_gates")) and
                 cohort.get("identity_hash") == outcome.get("identity_hash") and
                 calibration.get("receipt_hash") == cohort.get("teacher_receipt_hash") and
                 calibration.get("policy_hash") == cohort.get("teacher_policy_hash"))
        try:
            pre_state=self._read()
        except RuntimeError:
            return False
        pre_revision=int(pre_state.get("revision",-1))
        pre_digest=self._state_digest(pre_state)
        if not ready or not self.silver.claim_cohort_authorization(str(cohort_id),expected_revision=pre_revision,expected_state_digest=pre_digest):
            return False
        def change(state):
            if int(state.get("revision",-1)) != pre_revision or self._state_digest(state) != pre_digest:
                return False
            if not ready:
                return False
            if state.get("cohort_id") == str(cohort_id) and state.get("tier") in {"provisional_silver","canary","deployed_silver"}:
                return True
            # Later compatible cohorts are holdouts for the active rollout;
            # they must not restart the candidate at five percent.
            if state.get("tier") in {"provisional_silver","canary","deployed_silver"}:
                return False
            # Re-resolve while holding the deployment lock so a later clear or
            # revoke cannot publish an earlier caller's candidate.
            current=self._winner(str(cohort_id))
            if current is None or current[0] != frozen_candidate:
                return False
            manifest=self._signed_manifest(frozen_candidate,str(cohort_id),outcome)
            _atomic_jsonl(self.manifest_path,[manifest])
            state.update({"tier": "provisional_silver", "current": dict(frozen_candidate), "cohort_id": str(cohort_id), "canary": {"stage": "five_percent",
                          "captures": 0, "days": 0, "failures": 0, "sticky_percent": 5, "epoch": self.silver.epoch(),
                          "observation_revision": int(state["revision"]) + 1, "started_ts": self.silver.clock()}})
            state["route_generation"]=int(state.get("route_generation",0))+1
            state["audit"].append({"event": "provisional_authorized", "tier": "provisional_silver"})
            return True
        authorized=bool(self._mutate(change))
        # The evaluation remains replayable only across the tiny window after
        # SQLite evaluation and before this durable deployment transition.
        if authorized:
            self.silver.mark_cohort_authorized(str(cohort_id))
        else:
            self.silver.release_cohort_authorization_claim(str(cohort_id))
        return authorized

    def observe_canary(self, metrics: dict[str, Any] | None = None) -> str:
        """Advance from observations stored by the runtime, never caller data."""
        consumed: list[str] = []
        def change(state):
            plan = state.get("canary")
            if state.get("tier") not in {"provisional_silver", "canary", "deployed_silver"} or not isinstance(plan, dict): return "shadow_only"
            measured=self.silver.runtime_metrics(int(plan.get("observation_revision", state["revision"])))
            if int(measured.get("candidate_records",measured.get("captures",0))) == 0:
                return str(plan.get("stage","five_percent"))
            enough_error_volume=int(measured.get("captures",0)) >= 100
            bad = (int(plan.get("epoch", -1)) != self.silver.epoch() or bool(measured.get("identity_invalid")) or
                   int(measured.get("consecutive_failures",0)) >= 3 or (enough_error_volume and float(measured.get("fallback_error", 1)) > .02) or
                   (enough_error_volume and (not bool(measured.get("comparator_resolved")) or measured.get("incumbent_error") is None or float(measured.get("fallback_error", 1)) > float(measured.get("incumbent_error")) + .01)) or
                   (enough_error_volume and (measured.get("p95_ratio") is None or float(measured.get("p95_ratio", 99)) > 1.5)) or
                   (enough_error_volume and (not bool(measured.get("coverage_ok")) or not bool(measured.get("hallucination_ok")))))
            if bad:
                if isinstance(state.get("cohort_id"),str): consumed.append(state["cohort_id"])
                state.update({"tier": "shadow_only", "current": state.get("lkg"), "canary": None})
                state["audit"].append({"event": "automatic_rollback"}); return "rolled_back"
            plan["captures"] = int(measured.get("captures", plan["captures"])); plan["days"] = int(measured.get("days", plan["days"]))
            if plan["stage"] == "five_percent" and plan["captures"] >= 100 and plan["days"] >= 7:
                plan.update({"stage": "twenty_five_percent", "sticky_percent": 25}); state["tier"] = "canary"
                # Assignment generation is fixed for this authorized rollout:
                # sessions already observed at 5% retain their persisted arm.
                # Unseen sessions are assigned against the expanded percent.
                return "twenty_five_percent"
            if plan["stage"] == "twenty_five_percent" and plan["captures"] >= 500 and plan["days"] >= 14:
                holdout = self.silver.fresh_holdout_status(excluding=str(state.get("cohort_id","")), after_ts=float(plan.get("started_ts",0)))
                if (int(holdout.get("accepted", 0)) >= 200 and int(holdout.get("words", 0)) >= 2000 and
                        int(holdout.get("sessions", 0)) >= 20 and int(holdout.get("days", 0)) >= 7 and
                        bool(holdout.get("all_terminal")) and bool(holdout.get("paired_gates")) and
                        bool(holdout.get("coverage_frozen")) and holdout.get("winner") == (state.get("current") or {}).get("stable_id")):
                    # Keep the same durable monitor at full traffic.  Dropping
                    # it made post-full failures unobservable and impossible
                    # to roll back.
                    state["tier"] = "deployed_silver"; state["lkg"] = state.get("current")
                    plan.update({"stage":"full","sticky_percent":100})
                    # Keep the same sticky assignment generation at full
                    # traffic; only new sessions see the 100% threshold.
                    return "deployed_silver"
                # A fresh prospective cohort is still collecting; never turn
                # absence of a holdout into a permanent rollout failure.
                return "awaiting_holdout"
            return plan["stage"]
        result=str(self._mutate(change))
        if result == "rolled_back":
            for cohort_id in consumed: self.silver.consume_cohort_authorization(cohort_id)
        return result

    def reconcile_runtime(self) -> str:
        """Recover a crash after durable pair evidence but before rollout observation."""
        try: state=self._read()
        except RuntimeError: return "shadow_only"
        plan=state.get("canary")
        if state.get("tier") not in {"provisional_silver","canary","deployed_silver"} or not isinstance(plan,dict):
            return "shadow_only"
        metrics=self.silver.runtime_metrics(int(plan.get("observation_revision",state.get("revision",0))))
        if int(metrics.get("candidate_records",0)) == 0: return str(plan.get("stage","five_percent"))
        return self.observe_canary()

    def route(self, session_id: str) -> dict[str, Any] | None:
        """Deterministic, session-sticky candidate selection; invalid state is baseline."""
        try: state=self._read()
        except RuntimeError: return None
        candidate=state.get("current")
        if not isinstance(candidate,dict) or state.get("tier") == "shadow_only": return None
        invalid=False
        try:
            cohort_id=str(state.get("cohort_id","")); resolved=self._winner(cohort_id)
            if resolved is None: raise ValueError("authoritative winner missing")
            frozen,outcome=resolved
            cohort=self.silver.cohort_status(cohort_id) or {}
            calibration=self.silver.calibration_status() or {}
            signed=json.loads(self.manifest_path.read_text("utf-8").splitlines()[-1])
            expected=self._signed_manifest(frozen,cohort_id,outcome)
            if (signed.get("content_hash") != expected.get("content_hash") or signed.get("signature") != expected.get("signature")
                    or signed.get("candidate") != candidate or frozen != candidate or cohort.get("status") != "evaluated_passed"
                    or cohort.get("clear_epoch") != self.silver.epoch() or signed.get("epoch") != self.silver.epoch()
                    or signed.get("receipt_hash") != cohort.get("teacher_receipt_hash")
                    or calibration.get("state") != "passed" or calibration.get("receipt_hash") != cohort.get("teacher_receipt_hash")
                    or calibration.get("policy_hash") != cohort.get("teacher_policy_hash")):
                raise ValueError("deployment lineage changed")
        except (OSError, ValueError, json.JSONDecodeError):
            invalid=True
        if invalid:
            # Corrupt or stale persisted routing state is itself a deployment
            # event.  Atomically roll it back so later sessions cannot see a
            # half-valid candidate.
            self.invalidate("route_authority_invalid", expected_revision=int(state.get("revision",-1)), expected_candidate=candidate)
            return None
        plan=state.get("canary")
        percent=100 if state.get("tier")=="deployed_silver" else int(plan.get("sticky_percent",0)) if isinstance(plan,dict) else 0
        if not self.silver.route_assignment(route_generation=int(state.get("route_generation",0)),session_id=session_id,percent=percent):
            return None
        selected=dict(candidate)
        selected["_deployment_token"]={"deployment_revision":int(plan.get("observation_revision",state.get("revision",-1))) if isinstance(plan,dict) else int(state.get("revision",-1)),"route_generation":int(state.get("route_generation",0)),"candidate_hash":hashlib.sha256(json.dumps(candidate,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"tier":state.get("tier"),"cohort_id":state.get("cohort_id"),"epoch":self.silver.epoch(),"session_hash":hashlib.sha256(session_id.encode()).hexdigest()}
        return selected

    def validate_route_token(self, token: dict[str, Any] | None) -> bool:
        if not isinstance(token,dict): return False
        try: state=self._read()
        except RuntimeError: return False
        current=state.get("current")
        if not isinstance(current,dict) or state.get("tier") == "shadow_only": return False
        cohort=self.silver.cohort_status(str(token.get("cohort_id",""))) or {}
        plan=state.get("canary") if isinstance(state.get("canary"),dict) else {}
        return (token.get("cohort_id") == state.get("cohort_id") and int(token.get("deployment_revision",-1)) == int(plan.get("observation_revision",state.get("revision",-2))) and int(token.get("route_generation",-1)) == int(state.get("route_generation",-2)) and token.get("tier") == state.get("tier") and
                token.get("epoch") == self.silver.epoch() and cohort.get("status") == "evaluated_passed" and cohort.get("clear_epoch") == self.silver.epoch() and
                token.get("candidate_hash") == hashlib.sha256(json.dumps(current,sort_keys=True,separators=(",",":")).encode()).hexdigest() and
                self.silver.has_route_assignment_hash(route_generation=int(token.get("route_generation",-1)),session_hash=str(token.get("session_hash",""))))

    def _pair_route_authorized(self, state: dict[str, Any], token: dict[str, Any] | None,
                               session_id: str, candidate_arm: str) -> bool:
        """Bind pair evidence to this caller's selected session and candidate."""
        if not isinstance(token,dict) or not isinstance(session_id,str) or not session_id or not isinstance(candidate_arm,str) or not candidate_arm:
            return False
        current=state.get("current")
        if not isinstance(current,dict) or current.get("stable_id") != candidate_arm or state.get("tier") == "shadow_only":
            return False
        current_hash=hashlib.sha256(json.dumps(current,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        cohort=self.silver.cohort_status(str(token.get("cohort_id",""))) or {}
        plan=state.get("canary") if isinstance(state.get("canary"),dict) else {}
        return (token.get("cohort_id") == state.get("cohort_id") and int(token.get("deployment_revision",-1)) == int(plan.get("observation_revision",state.get("revision",-2))) and int(token.get("route_generation",-1)) == int(state.get("route_generation",-2)) and
                token.get("candidate_hash") == current_hash and token.get("tier") == state.get("tier") and
                token.get("session_hash") == hashlib.sha256(session_id.encode()).hexdigest() and
                token.get("epoch") == self.silver.epoch() and cohort.get("status") == "evaluated_passed" and
                cohort.get("clear_epoch") == self.silver.epoch() and not self.silver.scrub_pending() and
                self.silver.has_route_assignment(route_generation=int(token.get("route_generation",-1)),session_id=session_id))

    def record_runtime(self, *, session_id: str, arm_id: str, success: bool, fallback: bool,
                       latency: float, coverage_ok: bool, hallucination_ok: bool, identity_valid: bool,
                       route_token: dict[str, Any] | None = None,
                       defer_observe: bool = False) -> str:
        # Runtime observations are rollout authority, not a public telemetry
        # API.  An unbound caller could otherwise fabricate canary gates.
        if not isinstance(route_token,dict): return "stale"
        def change(state):
            plan=state.get("canary") if isinstance(state.get("canary"),dict) else {}
            if route_token is not None:
                current=state.get("current")
                current_hash=hashlib.sha256(json.dumps(current,sort_keys=True,separators=(",",":")).encode()).hexdigest() if isinstance(current,dict) else ""
                cohort=self.silver.cohort_status(str(route_token.get("cohort_id",""))) or {}
                if (route_token.get("cohort_id") != state.get("cohort_id") or int(route_token.get("deployment_revision",-1)) != int(plan.get("observation_revision",state.get("revision",-2))) or int(route_token.get("route_generation",-1)) != int(state.get("route_generation",-2)) or
                        route_token.get("candidate_hash") != current_hash or route_token.get("tier") != state.get("tier")):
                    return "stale"
                if (route_token.get("session_hash") != hashlib.sha256(session_id.encode()).hexdigest() or route_token.get("epoch") != self.silver.epoch() or cohort.get("status") != "evaluated_passed" or
                        cohort.get("clear_epoch") != self.silver.epoch() or self.silver.scrub_pending() or
                        not self.silver.has_route_assignment(route_generation=int(route_token.get("route_generation",-1)),session_id=session_id)): return "stale"
            self.silver.record_runtime_observation(deployment_revision=int(plan.get("observation_revision", state["revision"])),session_id=session_id,arm_id=arm_id,
                success=success,fallback=fallback,latency=latency,coverage_ok=coverage_ok,hallucination_ok=hallucination_ok,identity_valid=identity_valid)
            if fallback: state.get("canary",{}).update({"failures":int(state.get("canary",{}).get("failures",0))+1})
            elif isinstance(state.get("canary"),dict): state["canary"]["failures"]=0
            return state.get("tier","shadow_only")
        result=self._mutate(change)
        if result == "stale": return "stale"
        # A routed candidate and its incumbent comparator are one logical
        # canary observation.  Do not let the first half advance/rollback the
        # canary (and thereby invalidate its own token) before the second half
        # is durably recorded.
        if defer_observe: return str(result)
        return self.observe_canary()

    def record_runtime_pair(self, *, session_id: str, capture_id: str, candidate_arm: str,
                            candidate: dict[str, Any], incumbent: dict[str, Any],
                            route_token: dict[str, Any] | None) -> str:
        """Commit both live arms as one authority-bound capture.

        Individual-arm writes are retained only for legacy diagnostics.  New
        rollout evidence must use this pair API so an incomplete control can
        never advance the canary.
        """
        if not isinstance(route_token,dict): return "stale"
        def change(state):
            if not self._pair_route_authorized(state,route_token,session_id,candidate_arm):
                return "stale"
            plan=state.get("canary") if isinstance(state.get("canary"),dict) else {}
            revision=int(plan.get("observation_revision",state["revision"]))
            pair_write=self.silver.record_runtime_pair(deployment_revision=revision,capture_id=capture_id,session_id=session_id,
                                                    cohort_id=str(route_token["cohort_id"]),epoch=int(route_token["epoch"]),route_generation=int(route_token["route_generation"]),
                                                    candidate_arm=candidate_arm,candidate=candidate,incumbent=incumbent)
            if not pair_write:
                return "stale"
            if pair_write == "replayed":
                return "replayed"
            if not bool(candidate.get("success")) or bool(candidate.get("fallback")):
                plan["failures"]=int(plan.get("failures",0))+1
            else:
                plan["failures"]=0
            return state.get("tier","shadow_only")
        result=self._mutate(change)
        if result == "stale": return "stale"
        # A process may die after SQLite commits the pair but before this
        # controller observes it.  Exact replay is the recovery trigger, not
        # a no-op: the store keeps it idempotent while this call reconciles the
        # durable evidence into rollout state.
        return self.observe_canary()

    def begin_runtime_pair(self, *, session_id: str, capture_id: str, candidate_arm: str,
                           candidate: dict[str, Any], route_token: dict[str, Any] | None) -> str:
        """Persist a candidate half without mutating deployment JSON."""
        with advisory_lock(self.base_dir):
            state=self._read()
            if not self._pair_route_authorized(state,route_token,session_id,candidate_arm): return "stale"
            plan=state.get("canary") if isinstance(state.get("canary"),dict) else {}
            result=self.silver.begin_runtime_pair(deployment_revision=int(plan.get("observation_revision",state["revision"])),capture_id=capture_id,
                session_id=session_id,cohort_id=str(route_token["cohort_id"]),epoch=int(route_token["epoch"]),route_generation=int(route_token["route_generation"]),candidate_arm=candidate_arm,candidate=candidate)
        if result in {"inserted","replayed"} and (not candidate.get("success") or candidate.get("fallback")):
            self.observe_canary()
        return "stale" if not result else str(result)

    def enqueue_comparator_intent(self, *, session_id: str, capture_id: str, candidate_arm: str,
                                  candidate: dict[str, Any], audio_sha256: str, spool_name: str,
                                  route_token: dict[str, Any] | None) -> str:
        """Fence one background comparator against the exact routed canary."""
        with advisory_lock(self.base_dir):
            state=self._read()
            if not self._pair_route_authorized(state,route_token,session_id,candidate_arm): return "stale"
            plan=state.get("canary") if isinstance(state.get("canary"),dict) else {}
            result=self.silver.enqueue_comparator_intent(deployment_revision=int(route_token["deployment_revision"]),
                capture_id=capture_id,session_id=session_id,cohort_id=str(route_token["cohort_id"]),epoch=int(route_token["epoch"]),
                route_generation=int(route_token["route_generation"]),candidate_arm=candidate_arm,candidate=candidate,
                audio_sha256=audio_sha256,spool_name=spool_name)
        return "stale" if not result else str(result)

    def finish_comparator_pair(self, intent: dict[str,Any], incumbent: dict[str,Any]) -> str:
        with advisory_lock(self.base_dir):
            try:
                state=self._read()
            except (RuntimeError,ValueError,TypeError):
                return 'stale'
            if not self._comparator_intent_authorized(state,intent): return 'stale'
            result=self.silver.finish_comparator_pair(intent,incumbent)
        return 'stale' if not result else str(result)

    def complete_runtime_pair(self, *, session_id: str, capture_id: str, candidate_arm: str,
                              candidate: dict[str, Any], incumbent: dict[str, Any], route_token: dict[str, Any] | None) -> str:
        with advisory_lock(self.base_dir):
            state=self._read()
            if not self._pair_route_authorized(state,route_token,session_id,candidate_arm): return "stale"
            plan=state.get("canary") if isinstance(state.get("canary"),dict) else {}
            result=self.silver.complete_runtime_pair(deployment_revision=int(plan.get("observation_revision",state["revision"])),capture_id=capture_id,
                session_id=session_id,cohort_id=str(route_token["cohort_id"]),epoch=int(route_token["epoch"]),route_generation=int(route_token["route_generation"]),candidate_arm=candidate_arm,candidate=candidate,incumbent=incumbent)
        if result in {"inserted","replayed"}: return self.observe_canary() if result == "inserted" else (self.observe_canary() or "replayed")
        return "stale"

    def invalidate(self, reason: str, *, expected_revision: int | None = None,
                   expected_candidate: dict[str, Any] | None = None) -> bool:
        consumed: list[str] = []
        def change(state):
            if ((expected_revision is not None and int(state.get("revision",-1)) != expected_revision) or
                    (expected_candidate is not None and state.get("current") != expected_candidate)):
                return False
            state.update({"tier": "shadow_only", "current": state.get("lkg"), "canary": None})
            if isinstance(state.get("cohort_id"),str): consumed.append(state["cohort_id"])
            state["route_generation"]=int(state.get("route_generation",0))+1
            state["audit"].append({"event": "invalidate", "reason": reason})
            try: self.manifest_path.unlink()
            except FileNotFoundError: pass
            return True
        changed=bool(self._mutate(change))
        if changed:
            for cohort_id in consumed: self.silver.consume_cohort_authorization(cohort_id)
        return changed

    def invalidate_candidate(self, candidate: dict[str, Any], reason: str) -> bool:
        """CAS invalidation for a runtime receipt failure after routing."""
        return self.invalidate(reason, expected_candidate=candidate)

    def invalidate_comparator_publication(self, token: dict[str,Any], reason: str) -> bool:
        """Fail closed only if an exact publication still names this rollout."""
        if not isinstance(token,dict): return False
        try:
            state=self._read(); current=state.get("current"); plan=state.get("canary") or {}
            if (not isinstance(current,dict) or not isinstance(plan,dict)
                    or token.get("cohort_id") != state.get("cohort_id")
                    or token.get("epoch") != self.silver.epoch()
                    or token.get("route_generation") != state.get("route_generation")
                    or token.get("deployment_revision") != plan.get("observation_revision",state.get("revision"))
                    or token.get("candidate_arm") != current.get("stable_id")
                    or token.get("candidate_hash") != hashlib.sha256(json.dumps(current,sort_keys=True,separators=(",",":")).encode()).hexdigest()):
                return False
            return self.invalidate(reason,expected_revision=int(state["revision"]),expected_candidate=current)
        except (RuntimeError,TypeError,ValueError):
            return False

    def invalidate_for_history(self, history_id: str, reason: str = "history_dependency_revoked") -> bool:
        """Do not reset an unrelated active rollout for ordinary corrections."""
        try: state=self._read()
        except RuntimeError: return False
        cohort_id=state.get("cohort_id")
        if not isinstance(cohort_id,str) or not self.silver.cohort_depends_on_history(cohort_id,history_id):
            return False
        return self.invalidate(reason, expected_revision=int(state.get("revision",-1)), expected_candidate=state.get("current"))

    def invalidate_for_cohorts(self, cohorts: set[str], reason: str = "history_dependency_revoked") -> bool:
        try: state=self._read()
        except RuntimeError: return False
        if not cohorts or state.get("cohort_id") not in cohorts: return False
        return self.invalidate(reason, expected_revision=int(state.get("revision",-1)), expected_candidate=state.get("current"))

    def clear_personal_state(self) -> None:
        """Crash-recovery-safe privacy scrub, including malformed state files."""
        prior_cohort: str | None=None
        with advisory_lock(self.base_dir):
            revision=0
            try:
                parsed=self._read(); revision=int(parsed.get("revision",0))+1
                prior_cohort=parsed.get("cohort_id") if isinstance(parsed.get("cohort_id"),str) else None
            except RuntimeError: pass
            self._write({"schema":1,"revision":revision,"tier":"shadow_only","current":None,"lkg":None,
                         "canary":None,"calibration":None,"route_generation":revision,"audit":[]})
            try: self.manifest_path.unlink()
            except FileNotFoundError: pass
        if prior_cohort is not None:
            self.silver.consume_cohort_authorization(prior_cohort)

    def status(self) -> dict[str, Any]:
        with advisory_lock(self.base_dir):
            state = self._read()
            calibration = self.silver.calibration_status() or {}
            return {"tier": state["tier"], "revision": state["revision"], "canary": state.get("canary"),
                    "route_generation": int(state.get("route_generation",0)),
                    "calibration_valid": calibration.get("state") == "passed"}
