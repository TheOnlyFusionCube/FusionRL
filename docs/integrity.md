# Reward-hacking defenses

## Why this exists

vals.ai's ["MiMo reward hacking"](https://www.vals.ai/blogs/mimo-reward-hacking) audit found that MiMo-V2.6 models, trained on the same task family FusionRL imports, learned to pull reference fixes out of their environments instead of solving tasks:

| Leak reported by vals.ai | Mechanism |
|---|---|
| Later upstream commits on disk | Full upstream clone with HEAD moved back; `git log HEAD..origin/main` |
| Unreachable git objects (1,795 of 2,698 audited coding tasks, 67%) | Fix branches deleted but never pruned; read with `git fsck`, or a hand-written pack parser when `git fsck` was blocked |
| File timestamps | Files touched by the reference patch carry later mtimes; `find -newermt` lists the fix's scope |
| Module caches | Go/npm/pip caches searched for a newer copy of the project |
| Network | Fixes fetched from GitHub, or newer releases downloaded and diffed |
| Grader leniency | Only detected hacks (<2%) were zeroed, so undetected ones kept being rewarded |
| Vague rules | "Do not cheat" barely helped (5/6 runs still searched); naming the channels did (0/6) |

The earlier FusionRL design had one defense: an optional `confirmed_hacks` input that zeroes a reward. That is exactly the "zero only what you detect" grader the audit criticizes. The integrity layer adds defense in depth; every layer assumes the others can fail.

## What was found in the real dataset

Measured on the pinned MiMo-V2.6-RL-oss snapshot (`971cab2…`) with the tools below.

**Static check of all 2,602 code tasks** (`fusionrl integrity audit-dataset`):

- No task leaks its hidden tests into the prompt. 300 issues quote a few lines that also appear in the hidden tests (at most a small fraction); that is normal and is reported only as a warning.
- All 2,602 use mutable image tags (`format-code-task-…:latest`). A dataset commit does not pin image contents.
- All 2,602 grade with `bash /testbed/mimo_test_command.sh`, a path inside the agent's workspace that the hidden test patch creates at grading time. A policy can pre-create it unless grading runs from a fresh container with the candidate's diff sanitized (below).

**Image audit of `format-code-task-001457`** (pulled from the public Docker Hub image):

- Git: clean. HEAD is detached, and all 893 objects are reachable from it.
- **Timestamps leak the fix.** Outside `node_modules`, 36 files share one mtime, and five files are 72 seconds newer: `src/action.js`, `action.yml`, `README.md`, `package-lock.json` and `src/action.test.js`. The issue is "make `secrets` optional", and these are the files such a fix touches. `find -newer` reveals them, and `src/action.test.js` is also the file the hidden test patch edits.
- After hardening, all 17,963 workspace files share one timestamp and the audit passes. The task behaves identically: same Node version, clean `git status`, and the same 24 tests pass with `--network none`.

A larger image sample was rate-limited by the registry in this environment; see [validation.md](validation.md) for what completed. Run `scripts/audit_mimo_images.py` on your own infrastructure before training.

## Layers

### 1. Environment audit and hardening (`fusionrl.integrity.environment`)

`audit_image` streams `docker export` and never runs the image. It reports **leaks** (which block attestation) and **warnings**:

| Check | Leak when |
|---|---|
| `git_refs_beyond_head` | Any branch, remote ref, tag, stash, note or replace ref points outside HEAD's history |
| `git_objects_beyond_head` | Any object not reachable from HEAD (staged blobs are allowed only if the same content is in the workspace) |
| `git_reflogs`, `git_state_files` | Reflogs; `ORIG_HEAD`/`FETCH_HEAD`/…; `COMMIT_EDITMSG` (keeps the last commit's message, often the fix's); rebase/merge/sequencer state (can hold whole patches) |
| `git_alternates`, `git_grafts` | External object stores or rewritten history |
| `timestamp_outliers` | Workspace files newer than the bulk checkout (overlap with hidden-patch files is called out) |
| `patch_leftovers` | `*.orig`, `*.rej`, `solution.patch`, `test_patch.diff`, … in the workspace |
| `test_content_present` / `reference_content_present` | A file outside git holds most of the hidden test or reference patch's distinctive added lines (caches, vendored copies) |
| `proxy_configured` | The image sets an outbound proxy for the policy |

`harden_image` writes a new single-layer image: HEAD-only git history, the reflogs and state files removed, the index reset to HEAD, remotes and dangerous config keys dropped, patch leftovers deleted, and every workspace timestamp set to HEAD's commit time (PAX atime/ctime dropped). Flattening to one layer removes layer-membership hints. Cache contents are reported, not deleted automatically, because deleting dependencies can break a task.

Git analysis runs on a copy whose `.git/config` and hooks are replaced: an untrusted `core.fsmonitor`, `core.sshCommand` or hook could otherwise execute on the auditing host. A test plants such a config and confirms nothing runs.

### 2. Attestation gate (`fusionrl.integrity.attestation`)

An `EnvironmentAttestation` binds a task to:

- the content-addressed image ID that was audited (`sha256:…`, never a tag);
- the exact docker network flags used;
- a TCP probe showing egress blocked under those flags, checked against GitHub, GitLab, PyPI, npm, the Go proxy, `example.com` and raw IPs (raw IPs catch networks that block DNS but not egress);
- the audit and rules versions.

`IntegrityGate.check` refuses a task if the attestation is missing or failed, the image or network flags differ, the audit version changed, or the ledger revoked the task. The network probe uses `nc`, not `wget`: busybox `wget` fails on HTTPS even when a host is reachable, which would report a false pass.

```bash
fusionrl integrity audit-image --image IMG --workspace /testbed --tasks code.jsonl --task-id ID
fusionrl integrity harden-image --image IMG --tag local/ID:hardened --workspace /testbed
fusionrl integrity attest --image local/ID:hardened --source-image IMG --workspace /testbed \
    --store attestations/ --probe-image busybox --tasks code.jsonl --task-id ID
python scripts/audit_mimo_images.py --tasks code.jsonl --image-map image-mapping.jsonl \
    --attest --store attestations/ --probe-image busybox --output audit.jsonl
```

### 3. Explicit rules (`fusionrl.integrity.rules`)

`integrity_rules()` / `with_integrity_rules(messages)` append a rules block that names each prohibited channel concretely (e.g. `git log --all`, `find -newer*`, module caches, upstream fetches, hidden tests). It is generated from the same catalog (`channels.py`) the monitor checks, so the model is told about exactly what is enforced. Rules supplement the hardening; they never replace it.

### 4. Trajectory monitor and circuit breaker (`fusionrl.integrity.monitor`)

`judge_group` scans each execution's decoded steps (`TranscriptStep` with kind `action`, `observation` or `reasoning`):

- **violation**: a prohibited channel returned content, or a grading-owned path was written. This is a confirmed hack.
- **attempt**: a prohibited channel was invoked without visible success, for example when hardening blocked it. The default `HackPolicy` also treats this as a hack. The rules named the channel, and rewarding a search that happened to fail would still reinforce searching.
- **signal**: reasoning cues ("reference fix", "jackpot", "gray area") or reads of grading files. These are logged only, never a hack on their own.

A missing transcript or a monitor error quarantines the whole group. Ordinary engineering is not flagged (`git log -5`, `git diff`, `ls -lt`, `pytest`, `pip install -e .`, `sed -i`); tests cover both sides.

`IntegrityLedger` revokes a task whose hack rate passes `task_revoke_rate` (it needs a new audit) and halts training when the recent global rate passes `halt_rate`. Training stops rather than keep optimizing against an exploit.

### 5. Reward and grading (`rewards.py`, `fusionrl.integrity.grading`)

A confirmed hack's reward is zeroed before group statistics, as before, and now also receives `hack_penalty` (default 1.0 in the real-model runner) as a separately centered channel. Zeroing alone gives no signal when the hack's groupmates also failed: `[0, hack, 0, 0]` gives all-zero advantages. With the penalty the hack is strictly worse than an honest failure (`[0.25, -0.75, 0.25, 0.25]`). The penalty is allowed with every route, including GAR, because it is an integrity channel, not a length controller.

`sanitize_candidate_patch` drops every section of the candidate's diff that touches a file defined by the hidden test patch, a grading-owned path, or test-only configuration (`conftest.py`, `pytest.ini`, jest/vitest/mocha/karma/phpunit configs), and reports what it dropped. Grade from a fresh hardened container, apply the sanitized diff, then the hidden tests.

## Integration status

- The **real-model runner** (single-turn tasks) applies verifier hack flags, the penalty, the ledger and halting, and quarantines evaluator failures. It refuses agentic task records (`environment`/`docker_image`) rather than run them without the gate and monitor.
- An **agentic environment runner** that executes tool calls in hardened containers is not included. Such a runner must call `IntegrityGate.check` before every dispatch, run containers with the attested network flags and image ID, build `TranscriptStep`s from actual tool calls and outputs, and pass `judge_group(...).confirmed_hacks` and the penalty into `GroupSupervision`.
- The **verl adapter** accepts binary rewards only, so a hack must score 0 in the user's verifier. Its penalty channel is not available there.
