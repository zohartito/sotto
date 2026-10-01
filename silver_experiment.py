"""Deterministic evidence calculation for prospective frozen silver cohorts.

Inference integrations feed only sufficient paired statistics here; no
candidate transcript or teacher alternative is persisted in this state file.
"""
from __future__ import annotations
import hashlib, json, math, os, uuid, random
from pathlib import Path
from typing import Any, Iterable
from evaluation import PairedPromotionRow, holm_fixed_family, paired_bootstrap_statistics
from history import _atomic_jsonl
from silver_store import SilverStore
from storage_lock import advisory_lock, ensure_private_directory, ensure_private_file

def _p95(values: list[float]) -> float:
    if not values:return math.inf
    return sorted(values)[max(0,math.ceil(.95*len(values))-1)]

class SilverExperiment:
    """One-look metrics and material gates; caller cannot supply gate booleans."""
    def __init__(self, base_dir:Path|str, *, silver:SilverStore|None=None)->None:
        self.base_dir=Path(base_dir); self.silver=silver or SilverStore(self.base_dir)
        self.path=self.base_dir/'adaptive-learning'/'silver'/'experiments.jsonl';ensure_private_directory(self.path.parent);ensure_private_file(self.path)
        if not self.path.exists() or not self.path.read_text().strip(): _atomic_jsonl(self.path,[{'schema':1,'runs':{}}])
    def _read(self):
        try:
            x=json.loads(self.path.read_text().splitlines()[-1])
            if x.get('schema')!=1 or not isinstance(x.get('runs'),dict):raise ValueError
            return x
        except Exception as e:raise RuntimeError('silver experiment state malformed') from e

    def clear_personal_state(self) -> None:
        """Scrub durable cohort identifiers/outcomes after personal deletion."""
        with advisory_lock(self.base_dir):
            _atomic_jsonl(self.path,[{"schema":1,"runs":{}}])

    def clear_cohorts(self, cohort_ids: set[str] | list[str]) -> None:
        """Selective revoke scrub; unrelated rollout mirrors remain live."""
        doomed={str(value) for value in cohort_ids}
        if not doomed: return
        with advisory_lock(self.base_dir):
            state=self._read()
            state["runs"]={key:value for key,value in state["runs"].items()
                           if not isinstance(value,dict) or str(value.get("cohort_id","")) not in doomed}
            _atomic_jsonl(self.path,[state])

    @staticmethod
    def _gates(stats: dict[str, Any]) -> bool:
        """Material gates are intentionally recomputed after every CI update."""
        return bool(stats.get("resolved") and
            stats.get('champion_wer', 0)-stats.get('candidate_wer', 0)>=.02 and
            (stats.get('champion_wer', 0)-stats.get('candidate_wer', 0))/max(stats.get('champion_wer', 0),1e-12)>=.10 and
            stats.get('upper_ci', math.inf)<0 and stats.get('candidate_cer', math.inf)-stats.get('champion_cer', 0)<=.005 and
            stats.get('win_or_tie_rate', 0)>=.65 and stats.get('candidate_hallucinations', math.inf)<=stats.get('champion_hallucinations', -math.inf) and
            stats.get('candidate_latency_p95', math.inf)<=1.5*max(stats.get('champion_latency_p95', math.inf),1e-9) and
            int(stats.get('bootstrap_clusters', 0)) >= 20)
    def evaluate(self, cohort_id:str, champion_id:str, candidates:dict[str,Iterable[PairedPromotionRow]], *, identity_hash:str, persist: bool = True,
                 generation: int | None = None, alpha: float | None = None)->dict[str,Any]:
        cohort=self.silver.cohort_status(cohort_id)
        if not cohort or cohort.get('clear_epoch')!=self.silver.epoch() or cohort.get('status') not in {'frozen','evaluating','evaluated_passed','evaluated_failed'}: raise RuntimeError('cohort invalid')
        if generation is None or alpha is None:
            # Non-authoritative diagnostic callers may calculate a look, but
            # persisted evaluation always supplies the SQLite allocation.
            generation=len(self._read()['runs'])+1; alpha=self.silver.alpha(generation)
        stats={}
        for ident, rows0 in candidates.items():
            rows=tuple(rows0)
            if not rows: stats[ident]={'p_value':1.0,'resolved':False};continue
            stats[ident]=paired_bootstrap_statistics(rows,cohort_id+ident,replicates=10_000);stats[ident]['resolved']=True
            stats[ident]['candidate_latency_p95']=_p95([r.candidate_latency for r in rows]);stats[ident]['champion_latency_p95']=_p95([r.champion_latency for r in rows])
            # The ordinary evaluator is sample-level.  It cannot by itself
            # satisfy the independent-cluster requirement used for promotion.
            stats[ident]['bootstrap_clusters']=0; stats[ident]['gates']=self._gates(stats[ident])
        holm=holm_fixed_family([(k,float(v.get('p_value',1))) for k,v in stats.items()],alpha=alpha)
        winners=[k for k,v in stats.items() if v.get('gates') and holm.get(k,False)]
        outcome={'cohort_id':cohort_id,'generation':generation,'alpha':alpha,'identity_hash':identity_hash,'champion':champion_id,'candidates':stats,'holm':holm,
                 'winner':sorted(winners)[0] if winners else None,'tier':'provisional_silver' if winners else 'shadow_only'}
        # This compatibility method accepts caller-provided sufficient stats
        # for diagnostics only.  It is never an authority path: only
        # ``evaluate_persisted`` may complete a durable cohort.
        return outcome

    def evaluate_persisted(self, cohort_id: str, *, identity_hash: str) -> dict[str, Any] | None:
        """Evaluate the frozen durable arm ledger, never accepting caller metrics.

        Each challenger is paired to the same accepted id/reference as the
        champion.  Missing/failed arms deliberately remain in the fixed Holm
        family with p=1 rather than disappearing from multiplicity control.
        """
        if self.silver.scrub_pending() is not None:
            return None
        key=hashlib.sha256((cohort_id+identity_hash).encode()).hexdigest()
        authoritative=self.silver.cohort_status(cohort_id)
        if authoritative and authoritative.get("status") in {"evaluated_passed","evaluated_failed"}:
            existing=authoritative.get("evaluation_outcome")
            if not isinstance(existing,dict) or existing.get("identity_hash") != identity_hash:
                return None
            with advisory_lock(self.base_dir):
                state=self._read()
                if not isinstance(state["runs"].get(key),dict):
                    state["runs"][key]=existing; _atomic_jsonl(self.path,[state])
            return existing
        horizon = self.silver.horizon_for_cohort(cohort_id)
        if not horizon or horizon.get("cohort_id") != cohort_id or horizon.get("identity_hash") != identity_hash:
            raise RuntimeError("prospective experiment identity is invalid")
        champion=str(horizon["champion"].get("stable_id"))
        candidates=[str(item.get("stable_id")) for item in horizon["candidates"]]
        raw=self.silver.candidate_attempt_rows(cohort_id)
        corrupt_terminal=any(row.get("status") == "complete" and row.get("code") == "corrupt_stats" for row in raw)
        expected_members=self.silver.frozen_accepted_members(cohort_id)
        expected={(m["history_id"],int(m["history_revision"]),m["audio_sha256"],m["reference_digest"],m["cluster_hash"]) for m in expected_members}
        if not expected:
            raise RuntimeError("frozen cohort has no accepted identities")
        by_arm: dict[str,dict[tuple[str,int,str,str,str],dict[str,Any]]] = {}
        duplicate=False
        for row in raw:
            identity=(str(row["history_id"]),int(row["history_revision"]),str(row["audio_sha256"]),str(row["reference_digest"]),str(row["cluster_hash"]))
            bucket=by_arm.setdefault(str(row["arm_id"]),{})
            if identity in bucket: duplicate=True
            bucket[identity]=row
        # Do not evaluate until every arm got one bounded terminal attempt.
        if any(row["status"] in {"pending","leased"} for row in raw): return None
        rows_by_candidate: dict[str,list[PairedPromotionRow]]={ident:[] for ident in candidates}
        clusters: dict[str, list[str]]={ident:[] for ident in candidates}
        for ident in candidates:
            unresolved=duplicate or set(by_arm) != {champion,*candidates}
            if set(by_arm.get(champion,{})) != expected or set(by_arm.get(ident,{})) != expected:
                unresolved=True
            for sample, candidate in by_arm.get(ident,{}).items():
                incumbent=by_arm.get(champion,{}).get(sample)
                # Retried candidate work is reliability evidence, not a
                # successful-only sample.  This conservative fixed protocol
                # refuses promotion when an arm needed a genuine retry; the
                # scheduler_busy path refunds its attempt in the store.
                if (not incumbent or incumbent["status"] != "complete" or candidate["status"] != "complete" or
                        int(incumbent.get("attempts",0)) != 1 or int(candidate.get("attempts",0)) != 1):
                    unresolved=True; continue
                left,right=incumbent.get("stats"),candidate.get("stats")
                if not isinstance(left,dict) or not isinstance(right,dict): unresolved=True; continue
                if left.get("cluster_hash") != sample[4] or right.get("cluster_hash") != sample[4] or left.get("cluster_hash") != right.get("cluster_hash"):
                    unresolved=True; continue
                try:
                    # Both arms must report the same immutable reference
                    # denominators and physically meaningful sufficient stats;
                    # otherwise a malformed candidate row could borrow the
                    # incumbent denominator and manufacture an improvement.
                    required={"word_distance","reference_words","character_distance","reference_characters","hallucination","latency","cluster_hash"}
                    def valid(stats):
                        return (set(stats) == required and
                                all(isinstance(stats[key],int) and not isinstance(stats[key],bool) and stats[key] >= 0
                                    for key in ("word_distance","reference_words","character_distance","reference_characters")) and
                                stats["reference_words"] > 0 and stats["reference_characters"] > 0 and
                                isinstance(stats["hallucination"],bool) and isinstance(stats["latency"],(int,float)) and
                                not isinstance(stats["latency"],bool) and math.isfinite(float(stats["latency"])) and float(stats["latency"]) >= 0 and
                                isinstance(stats["cluster_hash"],str))
                    if (not valid(left) or not valid(right) or
                            left["reference_words"] != right["reference_words"] or
                            left["reference_characters"] != right["reference_characters"]):
                        unresolved=True; continue
                    rows_by_candidate[ident].append(PairedPromotionRow(sample_id="%s:%s" % (sample[0],sample[1]),
                        champion_word_distance=int(left["word_distance"]),candidate_word_distance=int(right["word_distance"]),
                        reference_words=int(left["reference_words"]),champion_character_distance=int(left["character_distance"]),
                        candidate_character_distance=int(right["character_distance"]),reference_characters=int(left["reference_characters"]),
                        champion_hallucination=bool(left["hallucination"]),candidate_hallucination=bool(right["hallucination"]),
                        champion_latency=float(left["latency"]),candidate_latency=float(right["latency"])))
                    clusters[ident].append(sample[4])
                except (KeyError, TypeError, ValueError): unresolved=True; break
            if unresolved: rows_by_candidate[ident]=[]; clusters[ident]=[]
        # Existing calculation is deterministic; replace its sample-level
        # bootstrap p value with a cluster bootstrap when multiple clusters
        # exist, preserving all other sufficient statistics and gates.
        candidate_values: dict[str, Iterable[PairedPromotionRow]] = {}
        for ident, rows in rows_by_candidate.items():
            candidate_values[ident]=rows
        generation,alpha=self.silver.allocate_cohort_look(cohort_id,identity_hash)
        outcome=self.evaluate(cohort_id,champion,candidate_values,identity_hash=identity_hash,persist=False,
                             generation=generation,alpha=alpha)
        for ident, rows in rows_by_candidate.items():
            if rows:
                outcome["candidates"][ident].update(self._cluster_bootstrap(rows,clusters[ident],cohort_id+ident))
            outcome["candidates"][ident]["gates"]=self._gates(outcome["candidates"][ident])
        # Holm must use the actual clustered p values, including unresolved=1.
        outcome["holm"]=holm_fixed_family([(ident,float(value.get("p_value",1.0))) for ident,value in outcome["candidates"].items()],alpha=outcome["alpha"])
        winners=[ident for ident,value in outcome["candidates"].items() if value.get("gates") and outcome["holm"].get(ident,False)]
        outcome["winner"]=sorted(winners)[0] if winners and not corrupt_terminal else None; outcome["tier"]="provisional_silver" if outcome["winner"] else "shadow_only"
        if corrupt_terminal: outcome["code"]="corrupt_candidate_stats"
        # Finalization is deliberately store-first.  If clear/revoke wins the
        # cohort CAS, no JSON mirror is allowed to resurrect its outcome.
        completed=self.silver.complete_cohort(cohort_id,outcome)
        current=self.silver.cohort_status(cohort_id)
        if not completed and (not current or current.get("clear_epoch") != self.silver.epoch()):
            return None
        with advisory_lock(self.base_dir):
            # Clear/revoke may have run while the evaluator was computing or
            # waiting for this lock.  Never recreate an experiment mirror for
            # a cohort no longer present in the current personal epoch.
            current=self.silver.cohort_status(cohort_id)
            if not current or current.get("clear_epoch") != self.silver.epoch():
                return None
            state=self._read()
            final=state["runs"].get(key)
            if not isinstance(final,dict):
                state["runs"][key]=outcome; _atomic_jsonl(self.path,[state]); final=outcome
        return final

    @staticmethod
    def _cluster_bootstrap(rows: list[PairedPromotionRow], clusters: list[str], seed: str) -> dict[str, float]:
        """One-sided deterministic bootstrap resampling whole session/day clusters."""
        grouped: dict[str,list[PairedPromotionRow]]={}
        for row,cluster in zip(rows,clusters): grouped.setdefault(cluster or row.sample_id,[]).append(row)
        keys=sorted(grouped)
        if len(keys) < 20:
            return {"p_value":1.0,"upper_ci":math.inf,"bootstrap_clusters":len(keys)}
        rng=random.Random(int(hashlib.sha256(seed.encode()).hexdigest(),16)); observed=0.0
        words=sum(row.reference_words for row in rows); observed=sum(row.candidate_word_distance-row.champion_word_distance for row in rows)/words
        draws=[]
        for _ in range(10_000):
            sample=[item for _key in range(len(keys)) for item in grouped[keys[rng.randrange(len(keys))]]]
            denom=sum(row.reference_words for row in sample)
            draws.append(sum(row.candidate_word_distance-row.champion_word_distance for row in sample)/denom)
        draws.sort()
        # Delta is candidate minus champion error: under the no-improvement
        # null, non-negative resamples are evidence against promotion.
        p=(1+sum(draw >= 0 for draw in draws))/10001
        return {"p_value":p,"upper_ci":draws[9499],"bootstrap_clusters":len(keys)}
