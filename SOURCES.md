# Source provenance

Inspected October 19, 2026. All six user-requested sources were read directly. GitHub's connected tools were used to verify the public fork and its exact revision; code interfaces were then checked against a local public-source checkout. Source reports are referenced rather than embedded in this archive.

| ID | Requested source | Version actually used | Relevant locations |
|---|---|---|---|
| DS41 | [DeepSeek V4.1 technical report PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) | Exact downloaded PDF SHA-256 below; equivalent report [arXiv HTML v1](https://arxiv.org/html/2609.19969v1) | §5.1 task pipeline; §5.1.4 Eqs. 8–10 effort; §§5.2.1–5.2.4 asynchronous infrastructure and OPD |
| MIMO | [MiMo V2.6: Scaling Reinforcement Learning Towards Self-Improvement](https://arxiv.org/html/2610.11959v1) | `2610.11959v1` | §4.3 Eqs. 2–5; §5.1 policy algorithm; §5.4 routing; §§6.1–6.4 systems; §7 public release |
| DATA | [XiaomiMiMo/MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss) | `971cab2ab421fd88db29770db0d0e52014b63615` | Dataset card, five parquet schemas, image mapping |
| VERL | [XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl) | Branch inspected `mimo-oss`; commit `79090e58195e80a9cfe251c95ae7a78a2132b6e2` | README; domain recipes; PPO core; V1 trainer; rollout metadata; FSDP loss and gradient reduction |
| SEER | [Seer paper, supplied unversioned PDF](https://arxiv.org/pdf/2511.14617) | Resolved to [2511.14617v3](https://arxiv.org/pdf/2511.14617v3), April 3, 2026 | §§3.2–3.4 divided rollout, context learning and scheduling; Appendix A.1 |
| GYRE | [QwenGyre: An Elastic Reinforcement Learning Framework for Training xLong-Horizon Agents](https://arxiv.org/html/2609.33848v1) | `2609.33848v1` | §§4.1–4.2 dispatch and elastic resources; §§5.1–5.3 target identity; Appendix A lag accounting; Appendix B Eqs. 26–30 objective |

DeepSeek input PDF:

```text
sha256 ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d
bytes  1809802
```

The unversioned Seer link was not assumed to mean v1. The resolved v3 is retained in the provenance record so subsequent reproductions can use the same text.

## Public code pins

The repository's gitlinks pin the optional MiMo agent submodule to `c3df8469f250b15e7906487a6d52be4c1827a509` and Uni-Agent to `c63e0b01c375ebede95e01fe92bc367df24e5bf3`. The bootstrap uses the repository's own `.gitmodules` routing. A rename of a README link is not substituted for the observed gitlink.

`fusionrl/integrations/verl/source_contract.json` stores original and patched hashes plus exact parsed function signatures for ten relevant upstream files. The local patch is tied to those files. Future changes require an updated contract and validation run.

## Attribution and independence

The equations are implemented independently and are cited at their point of use in `docs/algorithm.md`. FusionRL's additional decisions—strict compatibility guards, bounded scalar reward calibration, fixed dispatch-order membership, deterministic path selection, invalid-group quarantine, stage barriers and the toy environment—are identified as engineering choices. Original research metrics are not presented as FusionRL measurements.

The dataset card and code repository declare Apache-2.0. The package does not change terms for source reports, upstream projects, model weights, repository snapshots inside environments, or container images. The small upstream patch retains original source context and belongs with that upstream license. See `NOTICE`.

