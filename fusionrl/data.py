"""Strict, non-executing adapters for the five MiMo RL dataset configurations.

The canonical task envelope deliberately separates model-visible messages from
private verifier and environment metadata. Loading a task never executes any
dataset-supplied command, imports a verifier, downloads a container, or exposes
test patches through ``TaskRecord.model_input``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


MIMO_DATASET = "XiaomiMiMo/MiMo-V2.6-RL-oss"
MIMO_REVISION = "971cab2ab421fd88db29770db0d0e52014b63615"
MIMO_LICENSE = "apache-2.0"
MIMO_FILES = {
    "code": "code.parquet",
    "cyber": "cyber.parquet",
    "general": "general/train.parquet",
    "webdev": "webdev.parquet",
    "music": "music.parquet",
}
MIMO_ROW_COUNTS = {"code": 2602, "cyber": 1000, "general": 989, "webdev": 2093, "music": 1000}
_DOMAINS = {"code": "code", "cyber": "cyber", "general": "general", "webdev": "visual", "music": "music"}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return copy.deepcopy(dict(value))


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _prompt(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("prompt must be a nonempty message list")
    result = []
    for index, entry in enumerate(value):
        message = _mapping(entry, f"prompt[{index}]")
        if message.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"prompt[{index}].role is invalid")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"prompt[{index}].content must be a string")
        result.append(message)
    return result


@dataclass(frozen=True)
class TaskRecord:
    """Canonical schema v1; the complete envelope belongs to the trusted runner."""

    task_id: str
    subset: str
    domain: str
    data_source: str
    prompt: list[dict[str, Any]]
    agent_name: str | None = None
    ability: str | None = None
    private_extra_info: dict[str, Any] = field(default_factory=dict)
    private_instance: dict[str, Any] = field(default_factory=dict)
    private_reward_model: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def model_input(self) -> list[dict[str, Any]]:
        """Return only source messages; a harness must construct its own public instructions."""
        return copy.deepcopy(self.prompt)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskRecord":
        raw = _mapping(value, "record")
        if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
            raise ValueError("Unsupported canonical schema_version")
        required = {"task_id", "subset", "domain", "data_source", "prompt"}
        if not required.issubset(raw):
            raise ValueError(f"Missing canonical fields: {sorted(required - raw.keys())}")
        for name in required - {"prompt"}:
            _text(raw[name], name)
        raw["prompt"] = _prompt(raw["prompt"])
        for name in ("private_extra_info", "private_instance", "private_reward_model", "provenance"):
            raw[name] = _mapping(raw.get(name, {}), name)
        for name in ("agent_name", "ability"):
            if raw.get(name) is not None:
                _text(raw[name], name)
        if raw["subset"] in _DOMAINS and raw["domain"] != _DOMAINS[raw["subset"]]:
            raise ValueError("Canonical subset and domain disagree")
        serialized = raw["private_extra_info"].get("instance_json")
        if serialized is not None:
            try:
                decoded = _mapping(json.loads(_text(serialized, "instance_json")), "instance_json")
            except json.JSONDecodeError as exc:
                raise ValueError("Canonical instance_json is invalid JSON") from exc
            if decoded != raw["private_instance"]:
                raise ValueError("Canonical private_instance disagrees with source instance_json")
        known = set(cls.__dataclass_fields__)
        if set(raw) - known:
            raise ValueError(f"Unknown canonical fields: {sorted(set(raw) - known)}")
        return cls(**raw)


def normalize_mimo_row(
    row: Mapping[str, Any],
    subset: str,
    *,
    revision: str = MIMO_REVISION,
) -> TaskRecord:
    """Normalize a real MiMo row without assuming identical subset schemas.

    Music has no required agent_name/instance_json and uses extra_info.src_id.
    Cyber has no required ability. General contains multiple environment types;
    retain its dataset_type and dispatch only through a registered adapter.
    """
    if subset not in MIMO_FILES:
        raise ValueError(f"Unknown MiMo subset {subset!r}")
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("revision must be an immutable 40-character lowercase commit SHA")
    source = _mapping(row, "row")
    prompt = _prompt(source.get("prompt"))
    data_source = _text(source.get("data_source"), "data_source")
    extra = _mapping(source.get("extra_info"), "extra_info")
    reward_model = _mapping(source.get("reward_model"), "reward_model")
    _text(reward_model.get("style"), "reward_model.style")
    if not isinstance(reward_model.get("ground_truth"), str):
        raise ValueError("reward_model.ground_truth must be a string (it may be empty)")

    instance: dict[str, Any] = {}
    if subset == "music":
        instance_id = _text(extra.get("src_id"), "extra_info.src_id")
        for name in ("lang", "tag", "length", "meter"):
            _text(extra.get(name), f"extra_info.{name}")
        for name in ("nvoice_want", "bpm"):
            number = extra.get(name)
            if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
                raise ValueError(f"extra_info.{name} must be a positive integer")
        agent_name = source.get("agent_name")
    else:
        instance_id = _text(extra.get("instance_id"), "extra_info.instance_id")
        dtype = _text(extra.get("dataset_type"), "extra_info.dataset_type")
        serialized = _text(extra.get("instance_json"), "extra_info.instance_json")
        try:
            instance = _mapping(json.loads(serialized), "decoded instance_json")
        except json.JSONDecodeError as exc:
            raise ValueError("extra_info.instance_json is invalid JSON") from exc
        if instance.get("instance_id") != instance_id:
            raise ValueError("instance_id disagrees between extra_info and instance_json")
        if instance.get("dataset_type") != dtype:
            raise ValueError("dataset_type disagrees between extra_info and instance_json")
        _text(instance.get("docker_image"), "instance_json.docker_image")
        _text(instance.get("problem_statement"), "instance_json.problem_statement")
        agent_name = _text(source.get("agent_name"), "agent_name")
    if agent_name is not None:
        _text(agent_name, "agent_name")
    ability = source.get("ability")
    if ability is not None:
        _text(ability, "ability")
    return TaskRecord(
        task_id=f"mimo:{subset}:{instance_id}",
        subset=subset,
        domain=_DOMAINS[subset],
        data_source=data_source,
        prompt=prompt,
        agent_name=agent_name,
        ability=ability,
        private_extra_info=extra,
        private_instance=instance,
        private_reward_model=reward_model,
        provenance={
            "dataset": MIMO_DATASET,
            "revision": revision,
            "source_file": MIMO_FILES[subset],
            "declared_license": MIMO_LICENSE,
            "upstream_instance_id": instance_id,
        },
    )


def save_jsonl(records: Iterable[TaskRecord], path: str | Path) -> int:
    """Write complete trusted task envelopes, never model prompt batches."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    return count


def load_jsonl(path: str | Path) -> Iterator[TaskRecord]:
    """Read canonical JSONL and fail with the offending line number."""
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield TaskRecord.from_dict(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid canonical record at line {line_number}: {exc}") from exc


def load_parquet(
    path: str | Path,
    subset: str,
    *,
    revision: str = MIMO_REVISION,
    batch_size: int = 256,
) -> Iterator[TaskRecord]:
    """Stream raw MiMo parquet rows; install the optional pyarrow dependency."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("MiMo parquet import requires pyarrow; install FusionRL's data extra") from exc
    source = pq.ParquetFile(path)
    index = 0
    for batch in source.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            try:
                yield normalize_mimo_row(row, subset, revision=revision)
            except ValueError as exc:
                raise ValueError(f"Invalid {subset} parquet row {index}: {exc}") from exc
            index += 1


def read_image_mapping(path: str | Path) -> dict[str, str]:
    """Read upstream dataset_image -> dockerhub_image aliases without execution."""
    result: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = _mapping(json.loads(line), "image mapping")
                original = _text(row.get("dataset_image"), "dataset_image")
                target = _text(row.get("dockerhub_image"), "dockerhub_image")
            except ValueError as exc:
                raise ValueError(f"Invalid image mapping line {line_number}: {exc}") from exc
            if original in result and result[original] != target:
                raise ValueError(f"Conflicting image mapping for {original!r}")
            result[original] = target
    return result


def resolve_image_alias(record: TaskRecord, mapping: Mapping[str, str]) -> str | None:
    """Return the published alias; production runners should additionally pin its digest."""
    if record.subset == "music":
        return None
    logical_image = _text(record.private_instance.get("docker_image"), "docker_image")
    if logical_image not in mapping:
        raise ValueError(f"No image mapping for {logical_image!r}; refusing to guess a registry")
    return _text(mapping[logical_image], "resolved docker image")
