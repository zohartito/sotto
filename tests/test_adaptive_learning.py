from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

import sys

if sys.platform != "darwin":
    raise unittest.SkipTest("adaptive lane is macOS-only in v1 (fcntl)")

from adaptive_learning import AdaptiveLearning


def manifest(stable_id: str, *, dependencies=()) -> dict:
    from speech_backends import evaluator_hash
    return {
        "stable_id": stable_id, "backend": "local", "repo": "repo", "revision": "abc",
        "package_versions": {"x": "1"}, "decode_settings": {"beam": 1}, "language": "en",
        "glossary_hash": "g", "glossary_identity": "generation-g", "evaluator_id": "sotto-paired-v1",
        "evaluator_hash": evaluator_hash(), "development_dependencies": list(dependencies),
    }


class AdaptiveLearningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = AdaptiveLearning(Path(self.temp.name), clock=lambda: 10.0)

    def test_idle_baseline_refresh_and_mid_collection_refusal(self):
        """Same pinned identity with a refreshed receipt binding replaces a
        stale idle baseline; a different identity, or any refresh during
        active collection, still refuses."""
        stale = {**manifest("base"), "snapshot_path": "/legacy/Sotto/path"}
        fresh = {**manifest("base"), "snapshot_path": "/current/sotto/path", "snapshot_digest": "d" * 64}
        self.store.ensure_baseline(stale)
        self.assertEqual(self.store.champion_manifest()["snapshot_path"], "/legacy/Sotto/path")
        self.assertEqual(self.store.ensure_baseline(fresh), fresh)
        self.assertEqual(self.store.champion_manifest(), fresh)
        with self.assertRaisesRegex(RuntimeError, "baseline identity changed"):
            self.store.ensure_baseline({**fresh, "revision": "other"})
        self.frozen_open()
        self.assertEqual(self.store.ensure_baseline({**fresh, "snapshot_digest": "e" * 64}), fresh)
        self.assertEqual(self.store.champion_manifest(), fresh)

    def tearDown(self):
        self.temp.cleanup()

    def capture(self, ident: str, timestamp: int, *, adaptive=True):
        self.store.register_capture({"history_id": ident, "captured_ts": timestamp, "duration": 1,
                                     "audio_digest": ident, "adaptive": adaptive})

    def frozen_open(self, generation="g", candidates=None):
        ident = self.store.create_generation(generation_id=generation, champion=manifest("champ"), evaluator_hash=manifest("champ")["evaluator_hash"], freeze_ts=0)
        self.store.freeze_generation(ident, candidates or [manifest("chall")])
        self.store.open_pool(ident)
        return ident

    def resolve_ready_pool(self, generation="g", prefix=""):
        for value in range(33):
            ident = f"{prefix}{value:02d}"
            self.capture(ident, value + 1)
            self.store.resolve_review(ident, "no_speech" if value < 3 else "corrected",
                                      reference="" if value < 3 else "word " * 10)
        self.assertTrue(self.store.close_pool(generation))

    def test_pending_gold_is_content_free_in_status_and_retryable(self):
        self.capture("one", 11)
        self.store.resolve_review("one", "corrected", reference="private words",
                                  auto_enroll=lambda *_: (_ for _ in ()).throw(OSError("no")))
        self.assertEqual(self.store.status()["pending_gold"], 1)
        self.assertNotIn("private words", repr(self.store.status()))
        self.assertEqual(self.store.retry_pending_gold(lambda *_: None), 1)
        self.assertEqual(self.store.status()["pending_gold"], 0)

    def test_frozen_family_precedes_pool_and_cannot_mutate_after_open(self):
        generation = self.store.create_generation(generation_id="g", champion=manifest("champ"), evaluator_hash=manifest("champ")["evaluator_hash"], freeze_ts=0)
        self.capture("before", -1)
        self.assertEqual(self.store.review_list("g"), [])
        self.store.freeze_generation(generation, [manifest("a"), manifest("b")])
        with self.assertRaises(RuntimeError):
            self.store.freeze_generation(generation, [manifest("a")])
        self.store.open_pool(generation)
        with self.assertRaises(RuntimeError):
            self.store.freeze_candidate(generation, manifest("late"))
        self.capture("after", 1)
        self.assertEqual([item["history_id"] for item in self.store.review_list(generation)], ["after"])

    def test_atomic_publication_aborts_legacy_partials_and_allows_one_winner(self):
        # Staged legacy rows cannot be safely resumed because their candidate
        # family may have changed while the process was down.
        self.store.create_generation(generation_id="legacy", champion=manifest("champ"), evaluator_hash=manifest("champ")["evaluator_hash"])
        self.assertEqual(self.store.recover_partial_publications(), ["legacy"])
        self.assertEqual(self.store.status()["generations"]["legacy"]["status"], "publication_aborted")
        barrier = threading.Barrier(2); results = []
        def publish():
            peer = AdaptiveLearning(Path(self.temp.name), clock=lambda: 10.0)
            barrier.wait()
            try:
                results.append(("ok", peer.publish_generation(champion=manifest("champ"), candidates=[manifest("chall")], evaluator_hash=manifest("champ")["evaluator_hash"])))
            except RuntimeError as exc:
                results.append(("blocked", str(exc)))
        threads = [threading.Thread(target=publish), threading.Thread(target=publish)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertEqual(sum(kind == "ok" for kind, _ in results), 1)
        self.assertEqual(sum(kind == "blocked" for kind, _ in results), 1)
        self.assertEqual(len([g for g in self.store.status()["generations"].values() if g["status"] == "pool_open"]), 1)

    def test_consumed_comparison_cannot_repeat_without_new_frozen_input(self):
        self.capture("dev", 11)
        self.store.resolve_review("dev", "corrected", reference="first gold")
        challenger = manifest("chall", dependencies=["dev"])
        first = self.store.publish_generation(generation_id="first", champion=manifest("champ"), candidates=[challenger], evaluator_hash=manifest("champ")["evaluator_hash"])
        self.store._mutate(lambda state: state["generations"][first].update({"status": "no_promotion", "consumed": True,
                                                                                "decision": {"outcome": "no_promotion"}}))
        with self.assertRaisesRegex(RuntimeError, "exact comparison already consumed"):
            self.store.publish_generation(champion=manifest("champ"), candidates=[challenger], evaluator_hash=manifest("champ")["evaluator_hash"])
        self.store.resolve_review("dev", "corrected", reference="revised gold")
        second = self.store.publish_generation(champion=manifest("champ"), candidates=[challenger], evaluator_hash=manifest("champ")["evaluator_hash"])
        self.assertEqual(self.store.pool_status(second)["status"], "pool_open")

    def test_one_look_identity_ignores_local_snapshot_transport_paths(self):
        champion = {**manifest("champ"), "snapshot_path": "/cache/a", "runtime_note": "first"}
        challenger = {**manifest("chall"), "snapshot_path": "/cache/a", "runtime_note": "first"}
        first = self.store.publish_generation(champion=champion, candidates=[challenger], evaluator_hash=champion["evaluator_hash"])
        self.store._mutate(lambda state: state["generations"][first].update({"status": "no_promotion", "consumed": True,
                                                                                "decision": {"outcome": "no_promotion"}}))
        with self.assertRaisesRegex(RuntimeError, "exact comparison already consumed"):
            self.store.publish_generation(champion={**champion, "snapshot_path": "/cache/b", "runtime_note": "second"},
                                          candidates=[{**challenger, "snapshot_path": "/cache/b", "runtime_note": "second"}],
                                          evaluator_hash=champion["evaluator_hash"])
        changed = {**challenger, "revision": "new-revision", "snapshot_path": "/cache/c"}
        allowed = self.store.publish_generation(champion={**champion, "snapshot_path": "/cache/c"}, candidates=[changed], evaluator_hash=champion["evaluator_hash"])
        self.assertEqual(self.store.pool_status(allowed)["status"], "pool_open")

    def test_manual_rollback_never_redeploys_matching_prior_identity(self):
        first_fallback = {**manifest("fallback"), "snapshot_path": "/cache/one", "invalid": False}
        duplicate_fallback = {**manifest("fallback"), "snapshot_path": "/cache/two", "generation_id": "old", "invalid": False}
        other = {**manifest("other"), "invalid": False}
        self.store._mutate(lambda state: state["champions"].update({"baseline": manifest("base"), "current": {**manifest("current"), "invalid": False},
                                                                       "prior": [first_fallback, duplicate_fallback, other]}))
        self.store.rollback()
        self.assertEqual(self.store.champion_manifest()["stable_id"], "fallback")
        self.store.rollback()
        self.assertEqual(self.store.champion_manifest()["stable_id"], "other")
        with self.store._locked():
            priors = self.store._read()["champions"]["prior"]
        self.assertTrue(all(item.get("invalid") for item in priors[:2]))

    def test_automatic_rollback_never_redeploys_matching_prior_identity(self):
        first_fallback = {**manifest("fallback"), "snapshot_path": "/cache/one", "invalid": False}
        duplicate_fallback = {**manifest("fallback"), "snapshot_path": "/cache/two", "generation_id": "old", "invalid": False}
        other = {**manifest("other"), "invalid": False}
        self.store._mutate(lambda state: state["champions"].update({"baseline": manifest("base"), "current": {**manifest("current"), "invalid": False},
                                                                       "prior": [first_fallback, duplicate_fallback, other]}))
        self.store.rollback()
        self.assertEqual(self.store.champion_manifest()["stable_id"], "fallback")
        self.store.record_runtime_failure("fallback", "x")
        self.store.record_runtime_failure("fallback", "x")
        result = self.store.record_runtime_failure("fallback", "x")
        self.assertTrue(result["rolled_back"])
        self.assertEqual(self.store.champion_manifest()["stable_id"], "other")

    def test_pool_chronology_pending_and_all_quota_boundaries(self):
        generation = self.frozen_open()
        for value in reversed(range(33)):
            self.capture(f"{value:02d}", value + 1)
        selected = [row["history_id"] for row in self.store.review_list(generation)]
        self.assertEqual(selected, [f"{value:02d}" for value in range(33)])
        self.assertFalse(self.store.close_pool(generation))  # unresolved remains open
        self.assertEqual(self.store.pool_status(generation)["status"], "pool_open")
        for value in range(33):
            self.store.resolve_review(f"{value:02d}", "no_speech" if value < 3 else "corrected",
                                      reference="" if value < 3 else "word " * (10 if value != 3 else 9))
        self.assertFalse(self.store.close_pool(generation))  # 299 words
        self.assertEqual(self.store.pool_status(generation)["status"], "invalidated")

    def test_skip_limit_and_reference_revision_invalidate_closed_pool(self):
        generation = self.frozen_open()
        self.resolve_ready_pool(generation)
        self.store.resolve_review("03", "corrected", reference="revised words")
        self.assertEqual(self.store.pool_status(generation)["status"], "invalidated")
        self.assertIn("03", [item["history_id"] for item in self.store.development_references()])
        other = self.frozen_open("h")
        for value in range(33):
            self.capture(f"x{value:02d}", value + 20)
            outcome = "skip" if value < 4 else ("no_speech" if value < 7 else "corrected")
            self.store.resolve_review(f"x{value:02d}", outcome, reference="word " * 10,
                                      skip_reason="private" if outcome == "skip" else None)
        self.assertFalse(self.store.close_pool(other))

    def test_evaluator_hash_mismatch_aborts_before_callback(self):
        generation = self.frozen_open()
        self.resolve_ready_pool(generation)
        self.store._mutate(lambda state: state["generations"][generation]["candidates"]["chall"]["manifest"].update({"evaluator_hash": "bad"}))
        self.assertEqual(self.store.evaluate(generation, lambda *_: self.fail("callback"))["outcome"], "evaluator_hash_mismatch")

    def test_attempt_correction_race_cannot_complete_newer_or_stale_authority(self):
        generation=self.frozen_open(); self.resolve_ready_pool(generation)
        self.store._mutate(lambda state: state['generations'][generation].update({'status':'evaluating'}))
        def callback(_manifest,_capture):
            self.store.resolve_review('03','corrected',reference='replacement reference words')
            return ('attractive stale transcript',.1)
        with self.assertRaisesRegex(RuntimeError,'human_evaluation_authority_changed'):
            self.store._attempt(generation,None,'03','champion',callback,evaluator_owner='owner')
        with self.store._locked():
            state=self.store._read(); current=state['generations'][generation]
            attempt=current['champion_attempts']['03'][0]
            self.assertEqual(current['status'],'invalidated')
            self.assertEqual((attempt['status'],attempt['code']),('failed','authority_changed'))
            self.assertNotIn('text',attempt)
        with self.assertRaisesRegex(RuntimeError,'generation not evaluable'):
            self.store.evaluate(generation,lambda *_: self.fail('must not retry invalidated generation'))
        self.assertIsNone(self.store.champion_manifest())

    def test_shared_champion_attempts_and_exact_empty_exclusion(self):
        generation = self.frozen_open(candidates=[manifest("a"), manifest("b")])
        self.resolve_ready_pool(generation)
        calls = {"champ": 0, "a": 0, "b": 0}
        def callback(candidate, capture):
            calls[candidate["stable_id"]] += 1
            if candidate["stable_id"] == "champ":
                return ("unexpected" if int(capture["history_id"]) < 3 else "wrong", .1)
            return ("" if int(capture["history_id"]) < 3 else "word " * 10, .1)
        result = self.store.evaluate(generation, callback)
        self.assertEqual(calls, {"champ": 33, "a": 33, "b": 33})
        self.assertEqual(result["outcome"], "promoted")

    def test_skips_resolve_pool_but_are_excluded_from_evaluation_and_dependencies(self):
        generation = self.frozen_open()
        skipped = {"03": "private", "04": "corrupt", "05": "wrong_capture"}
        for value in range(33):
            ident = f"{value:02d}"; self.capture(ident, value + 1)
            if ident in skipped:
                self.store.resolve_review(ident, "skip", skip_reason=skipped[ident])
            elif value < 3:
                self.store.resolve_review(ident, "no_speech")
            else:
                self.store.resolve_review(ident, "corrected", reference="word " * 12)
        pool = self.store.pool_status(generation)
        self.assertEqual((pool["resolved"], pool["verified"], pool["no_speech"], pool["skips"], pool["words"]), (33, 30, 3, 3, 324))
        self.assertTrue(self.store.close_pool(generation))
        calls = {"champ": [], "chall": []}
        def callback(candidate, capture):
            calls[candidate["stable_id"]].append(capture["history_id"])
            if candidate["stable_id"] == "champ":
                return ("" if int(capture["history_id"]) < 3 else "wrong " * 12, .1)
            return ("" if int(capture["history_id"]) < 3 else "word " * 12, .1)
        self.store.evaluate(generation, callback)
        self.assertEqual(len(calls["champ"]), 30)
        self.assertEqual(len(calls["chall"]), 30)
        self.assertTrue(all(ident not in calls["champ"] and ident not in calls["chall"] for ident in skipped))
        champion = self.store.champion_manifest()
        self.assertIsNotNone(champion)
        with self.store._locked():
            state = self.store._read()
            scored = {ident for ident in state["generations"][generation]["pool"]
                      if state["captures"][ident].get("outcome") in {"corrected", "correct_as_is", "no_speech"}}
            dependencies = set(champion.get("promotion_dependency_digests", ()))
            expected = {self.store._sample_digest(state, ident) for ident in scored}
        self.assertEqual(dependencies, expected)

    def test_unresolved_challenger_is_fixed_family_p_one_and_champion_abort_consumes(self):
        generation = self.frozen_open()
        self.resolve_ready_pool(generation)
        self.store._mutate(lambda state: state["generations"][generation]["champion_attempts"].update({ident: [{"status": "completed", "text": "word " * 10, "latency": .1}] for ident in state["generations"][generation]["pool"]}))
        first = self.store.decide(generation)
        self.assertEqual(first["candidates"]["chall"]["p_value"], 1.0)
        self.assertEqual(self.store.decide(generation), first)  # published decision is idempotent
        with self.assertRaises(RuntimeError):
            self.store.decide(generation, expected_revision=-1)
        abort = self.frozen_open("abort")
        self.resolve_ready_pool("abort", "z")
        self.assertEqual(self.store.evaluate(abort, lambda *_: (_ for _ in ()).throw(TimeoutError()))["outcome"], "technical_abort_champion")
        self.assertEqual(self.store.pool_status(abort)["status"], "technical_abort_champion")

    def test_decision_cas_and_dependency_rollback(self):
        self.capture("unrelated", 1)
        self.capture("dependent", 2)
        def install(state):
            digest = self.store._sample_digest(state, "dependent")
            current = {**manifest("current"), "development_dependency_digests": [digest], "promotion_dependency_digests": [], "invalid": False}
            invalid_prior = {**manifest("invalid"), "invalid": True}
            old_prior = {**manifest("old"), "invalid": False}
            state["champions"].update({"baseline": manifest("base"), "current": current, "prior": [invalid_prior, old_prior]})
        self.store._mutate(install)
        self.assertTrue(self.store.revoke("unrelated"))
        self.assertEqual(self.store.status()["current_champion"], "current")
        self.assertTrue(self.store.revoke("dependent"))
        self.assertEqual(self.store.status()["current_champion"], "old")

    def test_promotion_cohort_dependency_revocation_rolls_back(self):
        self.capture("cohort", 1)
        def install(state):
            digest = self.store._sample_digest(state, "cohort")
            state["champions"].update({
                "baseline": manifest("base"),
                "current": {**manifest("promoted"), "promotion_dependency_digests": [digest], "invalid": False},
                "prior": [{**manifest("independent"), "invalid": False}],
            })
        self.store._mutate(install)
        self.assertTrue(self.store.revoke("cohort"))
        self.assertEqual(self.store.status()["current_champion"], "independent")

    def test_reclaimed_owner_marks_running_attempt_retryable(self):
        generation = self.frozen_open()
        self.resolve_ready_pool(generation)
        self.store._mutate(lambda state: state["generations"][generation]["champion_attempts"].update(
            {"00": [{"status": "running", "owner": "dead-owner"}]}))
        self.assertEqual(self.store.recover_interrupted_attempts(["dead-owner"]), 1)
        with self.store._locked():
            attempt = self.store._read()["generations"][generation]["champion_attempts"]["00"][0]
        self.assertEqual((attempt["status"], attempt["code"]), ("failed", "interrupted"))

    def test_pending_pool_first_review_is_not_reference_invalidation_but_revoke_is(self):
        generation = self.frozen_open()
        self.capture("pending", 1)
        # It is a selected, still-pending cohort member: first gold is valid.
        self.store.pre_mutate_reference("pending")
        self.assertEqual(self.store.pool_status(generation)["status"], "pool_open")
        self.store.resolve_review("pending", "corrected", reference="gold")
        self.store.pre_mutate_reference("pending")
        self.assertEqual(self.store.pool_status(generation)["status"], "invalidated")

    def test_revoke_consumes_every_pool_state_and_dependency_generation(self):
        for status in ("pool_open", "pool_closed", "evaluating"):
            with self.subTest(status=status):
                generation = self.frozen_open(status)
                self.capture("member-" + status, 1)
                self.store._mutate(lambda state, g=generation, status=status: state["generations"][g].update({"status": status}))
                self.assertTrue(self.store.revoke("member-" + status))
                pool = self.store.pool_status(generation)
                self.assertEqual(pool["status"], "invalidated")
                with self.store._locked():
                    row = self.store._read()["captures"]["member-" + status]
                self.assertTrue(row["prior_pool"]); self.assertIsNone(row["pool_generation"])
        self.capture("dev", 1)
        generation = self.store.create_generation(generation_id="dep", champion=manifest("champ"), evaluator_hash=manifest("champ")["evaluator_hash"])
        self.store.freeze_generation(generation, [manifest("candidate", dependencies=["dev"])])
        self.assertTrue(self.store.revoke("dev"))
        self.assertEqual(self.store.status()["generations"][generation]["status"], "invalidated")

    def test_review_list_excludes_compact_revoked_and_reregistered_tombstone(self):
        self.capture("tomb", 1)
        # Force this opaque ID into the ordinary salted bucket to exercise the
        # old KeyError path after revoke removes captured_ts.
        self.store.review_selected = lambda _ident: True  # type: ignore[method-assign]
        self.assertTrue(self.store.revoke("tomb"))
        self.assertEqual(self.store.review_list(), [])
        self.store.register_capture({"history_id": "tomb", "captured_ts": 2, "duration": 1,
                                     "audio_digest": "tomb", "adaptive": True, "language": "en"})
        self.assertEqual(self.store.review_list(), [])

    def test_reference_change_rolls_back_development_and_cohort_deployments(self):
        self.capture("dev", 1); self.capture("cohort", 2)
        def install(state):
            dev_digest = self.store._sample_digest(state, "dev")
            cohort_digest = self.store._sample_digest(state, "cohort")
            base = manifest("base")
            current = {**manifest("current"), "development_dependency_digests": [dev_digest], "invalid": False}
            prior = {**manifest("prior"), "promotion_dependency_digests": [cohort_digest], "invalid": False}
            state["champions"].update({"baseline": base, "current": current, "prior": [prior]})
        self.store._mutate(install)
        self.store.pre_mutate_reference("dev")
        self.assertEqual(self.store.status()["current_champion"], "prior")
        self.store.pre_mutate_reference("cohort")
        self.assertEqual(self.store.status()["current_champion"], "base")

    def test_dependency_invalidation_fans_out_to_semantic_duplicate_deployments(self):
        self.capture("dev", 1)
        def install(state):
            digest = self.store._sample_digest(state, "dev")
            duplicate_current = {**manifest("duplicate"), "snapshot_path": "/cache/current", "invalid": False}
            duplicate_dependent_prior = {**manifest("duplicate"), "snapshot_path": "/cache/prior", "development_dependency_digests": [digest], "invalid": False}
            state["champions"].update({"baseline": manifest("base"), "current": duplicate_current,
                                        "prior": [duplicate_dependent_prior, {**manifest("other"), "invalid": False}]})
        self.store._mutate(install)
        self.store.pre_mutate_reference("dev")
        self.assertEqual(self.store.status()["current_champion"], "other")
        with self.store._locked():
            current, duplicate = self.store._read()["champions"]["current"], self.store._read()["champions"]["prior"][0]
        self.assertEqual(current["stable_id"], "other"); self.assertTrue(duplicate["invalid"])

    def test_revoke_dependency_fans_out_to_semantic_duplicate_deployments(self):
        self.capture("cohort", 1)
        def install(state):
            digest = self.store._sample_digest(state, "cohort")
            duplicate_current = {**manifest("duplicate"), "snapshot_path": "/cache/current", "invalid": False}
            duplicate_dependent_prior = {**manifest("duplicate"), "snapshot_path": "/cache/prior", "promotion_dependency_digests": [digest], "invalid": False}
            state["champions"].update({"baseline": manifest("base"), "current": duplicate_current,
                                        "prior": [duplicate_dependent_prior, {**manifest("other"), "invalid": False}]})
        self.store._mutate(install)
        self.assertTrue(self.store.revoke("cohort"))
        self.assertEqual(self.store.status()["current_champion"], "other")

    def test_runtime_failure_streak_is_scoped_to_semantic_revision_identity(self):
        base = manifest("base")
        revision_a = {**manifest("same"), "revision": "a", "invalid": False}
        revision_b = {**manifest("same"), "revision": "b", "invalid": False}
        self.store._mutate(lambda state: state["champions"].update({"baseline": base, "current": revision_a, "prior": []}))
        self.store.record_runtime_failure("same", "x", expected_manifest=revision_a)
        self.store.record_runtime_failure("same", "x", expected_manifest=revision_a)
        self.assertTrue(self.store.record_runtime_failure("same", "x", expected_manifest=revision_a)["rolled_back"])
        self.store._mutate(lambda state: state["champions"].update({"current": revision_b}))
        self.assertEqual(self.store.record_runtime_failure("same", "stale", expected_manifest=revision_a)["streak"], 0)
        fresh = self.store.record_runtime_failure("same", "fresh", expected_manifest=revision_b)
        self.assertEqual((fresh["streak"], fresh["rolled_back"]), (1, False))


if __name__ == "__main__":
    unittest.main()
