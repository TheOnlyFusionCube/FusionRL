"""Synthetic schema fixtures; no dataset content is bundled or executed."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fusionrl.data import (
    MIMO_REVISION,
    MIMO_ROW_COUNTS,
    TaskRecord,
    load_jsonl,
    load_parquet,
    normalize_mimo_row,
    read_image_mapping,
    resolve_image_alias,
    save_jsonl,
)


def fixture(subset="code"):
    if subset == "music":
        return {
            "prompt": [{"role": "user", "content": "Compose a short tune in ABC notation."}],
            "data_source": "music", "ability": "music_generation",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"src_id": "synthetic-music-1", "lang": "en", "tag": "dance",
                           "length": "short", "nvoice_want": 2, "bpm": 120, "meter": "4/4"},
        }
    dtype = {"code": "opensource-code", "cyber": "arvo", "general": "terminal_bench", "webdev": "webdev"}[subset]
    instance = {"instance_id": "synthetic-task-1", "dataset_type": dtype,
                "docker_image": "synthetic-image:v1", "cwd": "/work",
                "problem_statement": "Public environment instructions.",
                "test_patch": "PRIVATE_VERIFIER_PATCH", "test_command": "DO_NOT_EXECUTE"}
    row = {
        "prompt": [{"role": "user", "content": "Solve this synthetic task."}],
        "data_source": dtype, "agent_name": "mimo_swe_agent",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"index": 0, "instance_id": instance["instance_id"],
                       "dataset_type": dtype, "instance_json": json.dumps(instance)},
    }
    if subset != "cyber":
        row["ability"] = "agent"
    return row


class DataTests(unittest.TestCase):
    def test_all_five_schemas_normalize_without_fabricated_fields(self):
        for subset in MIMO_ROW_COUNTS:
            with self.subTest(subset=subset):
                record = normalize_mimo_row(fixture(subset), subset)
                self.assertTrue(record.task_id.startswith(f"mimo:{subset}:"))
                self.assertEqual(record.provenance["revision"], MIMO_REVISION)
        music = normalize_mimo_row(fixture("music"), "music")
        self.assertIsNone(music.agent_name)
        self.assertEqual(music.private_instance, {})
        self.assertIsNone(normalize_mimo_row(fixture("cyber"), "cyber").ability)

    def test_model_input_excludes_private_test_material(self):
        row = fixture()
        record = normalize_mimo_row(row, "code")
        self.assertNotIn("PRIVATE_VERIFIER_PATCH", json.dumps(record.model_input()))
        self.assertNotIn("DO_NOT_EXECUTE", json.dumps(record.model_input()))
        self.assertEqual(record.private_instance["test_patch"], "PRIVATE_VERIFIER_PATCH")
        model_messages = record.model_input()
        model_messages[0]["content"] = "Changed outside record"
        self.assertEqual(record.prompt[0]["content"], "Solve this synthetic task.")
        row["extra_info"]["instance_id"] = "Changed source"
        self.assertEqual(record.private_extra_info["instance_id"], "synthetic-task-1")

    def test_metadata_identifier_disagreement_fails(self):
        row = fixture()
        row["extra_info"]["instance_id"] = "wrong-id"
        with self.assertRaisesRegex(ValueError, "instance_id disagrees"):
            normalize_mimo_row(row, "code")
        row = fixture()
        row["extra_info"]["instance_json"] = "not JSON"
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            normalize_mimo_row(row, "code")

    def test_canonical_jsonl_round_trip_preserves_metadata(self):
        records = [normalize_mimo_row(fixture(s), s) for s in MIMO_ROW_COUNTS]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.jsonl"
            self.assertEqual(save_jsonl(records, path), 5)
            restored = list(load_jsonl(path))
            self.assertEqual([r.to_dict() for r in restored], [r.to_dict() for r in records])

    def test_unknown_schema_and_mutable_revision_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_mimo_row(fixture(), "code", revision="main")
        raw = normalize_mimo_row(fixture(), "code").to_dict()
        raw["schema_version"] = 42
        with self.assertRaises(ValueError):
            TaskRecord.from_dict(raw)

    def test_canonical_private_metadata_must_match_source_envelope(self):
        raw = normalize_mimo_row(fixture(), "code").to_dict()
        raw["private_instance"]["docker_image"] = "different-image:v2"
        with self.assertRaisesRegex(ValueError, "private_instance disagrees"):
            TaskRecord.from_dict(raw)
        raw = normalize_mimo_row(fixture(), "code").to_dict()
        raw["schema_version"] = True
        with self.assertRaises(ValueError):
            TaskRecord.from_dict(raw)

    def test_image_aliases_follow_explicit_mapping(self):
        record = normalize_mimo_row(fixture(), "code")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mapping.jsonl"
            path.write_text(json.dumps({"dataset_image": "synthetic-image:v1",
                                        "dockerhub_image": "docker.io/example/tasks:one"}) + "\n")
            mapping = read_image_mapping(path)
            self.assertEqual(resolve_image_alias(record, mapping), "docker.io/example/tasks:one")
        with self.assertRaisesRegex(ValueError, "refusing to guess"):
            resolve_image_alias(record, {})
        self.assertIsNone(resolve_image_alias(normalize_mimo_row(fixture("music"), "music"), {}))

    def test_registered_domain_does_not_imply_registered_environment_type(self):
        row = fixture("general")
        instance = json.loads(row["extra_info"]["instance_json"])
        instance["dataset_type"] = "general_agent"
        instance["env_task_dir"] = "general/envs/synthetic"
        row["extra_info"]["dataset_type"] = "general_agent"
        row["extra_info"]["instance_json"] = json.dumps(instance)
        record = normalize_mimo_row(row, "general")
        self.assertEqual(record.private_instance["dataset_type"], "general_agent")
        self.assertEqual(record.private_instance["env_task_dir"], "general/envs/synthetic")


@unittest.skipUnless(importlib.util.find_spec("pyarrow") is not None, "optional pyarrow dependency is not installed")
class ParquetDataTests(unittest.TestCase):
    @staticmethod
    def rows(subset):
        rows = [fixture(subset), fixture(subset)]
        rows[1]["prompt"][0]["content"] = "Second synthetic task: données / 数据."
        if subset == "music":
            rows[1]["extra_info"]["src_id"] += "-second"
        else:
            extra = rows[1]["extra_info"]
            extra["instance_id"] += "-second"
            extra["index"] = 1
            instance = json.loads(extra["instance_json"])
            instance["instance_id"] = extra["instance_id"]
            extra["instance_json"] = json.dumps(instance)
        return rows

    @staticmethod
    def write_parquet(rows, path, subset):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(rows)
        # The published webdev/music schemas use large-string scalar columns.
        if subset in {"webdev", "music"}:
            for column in ("data_source", "ability", "agent_name"):
                if column in table.column_names:
                    index = table.schema.get_field_index(column)
                    table = table.set_column(index, column, table[column].cast(pa.large_string()))
        pq.write_table(table, path, row_group_size=1)

    def test_actual_parquet_streams_all_five_schemas_and_round_trips(self):
        with tempfile.TemporaryDirectory() as temp:
            for subset in MIMO_ROW_COUNTS:
                with self.subTest(subset=subset):
                    rows = self.rows(subset)
                    parquet_path = Path(temp) / f"{subset}.parquet"
                    jsonl_path = Path(temp) / f"{subset}.jsonl"
                    self.write_parquet(rows, parquet_path, subset)
                    stream = load_parquet(parquet_path, subset, batch_size=1)
                    self.assertIs(iter(stream), stream)
                    self.assertEqual(save_jsonl(stream, jsonl_path), 2)
                    restored = list(load_jsonl(jsonl_path))
                    expected = [normalize_mimo_row(row, subset).to_dict() for row in rows]
                    self.assertEqual([record.to_dict() for record in restored], expected)

    def test_prepare_mimo_cli_reports_consumed_iterator_count(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "code.parquet"
            destination = Path(temp) / "code.jsonl"
            self.write_parquet(self.rows("code"), source, "code")
            result = subprocess.run(
                [sys.executable, "-m", "fusionrl", "prepare-mimo", "--subset", "code",
                 "--input", str(source), "--output", str(destination)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["tasks"], 2)
            self.assertEqual(len(list(load_jsonl(destination))), 2)


if __name__ == "__main__":
    unittest.main()
