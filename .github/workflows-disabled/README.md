# CI tombstone

`ci.yml.disabled` is intentionally non-executable while the locally sealed
campaign evidence lacks a reviewed immutable digest for `actions/setup-python@v7`.
Do not move it into `.github/workflows` or replace its tag with an unverified SHA.

To re-enable it, an owner must record immutable release provenance for every action,
pin each `uses:` value to that reviewed 40-character digest, move the workflow back,
and run `python3 tests/test_workflow_action_pinning.py`.
