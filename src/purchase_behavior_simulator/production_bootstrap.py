from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .models import Item, ObservationBatch


BOOTSTRAP_STATE_SCHEMA = "simplayer.production-bootstrap-state.v1"
BOOTSTRAP_LINEAGE_SCHEMA = "simplayer.production-bootstrap-lineage.v1"
PRODUCTION_MANIFEST_SCHEMA = "simuser.production-import.v1"
FORBIDDEN_MANIFEST_FLAGS = (
    "contains_answer_key",
    "contains_oracle_probability",
    "contains_model_predictions",
)
REQUIRED_NODE_COLUMNS = frozenset({":ID", ":LABEL"})
REQUIRED_EDGE_COLUMNS = frozenset(
    {":ID", ":START_ID", ":END_ID", ":TYPE"}
)


@dataclass(frozen=True)
class ProductionArtifact:
    root: Path
    manifest: Mapping[str, Any]
    memory_rows: tuple[Mapping[str, Any], ...]
    catalog_items: Mapping[str, Mapping[str, Any]]
    node_count: int
    edge_count: int
    fingerprint: str

    @property
    def memory_path(self) -> Path:
        return self.root / "memory_import.jsonl"

    @property
    def nodes_path(self) -> Path:
        return self.root / "neptune" / "nodes.csv"

    @property
    def edges_path(self) -> Path:
        return self.root / "neptune" / "edges.csv"

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_dir": str(self.root),
            "artifact_fingerprint": self.fingerprint,
            "memory_batches": len(self.memory_rows),
            "catalog_items": len(self.catalog_items),
            "neptune_nodes": self.node_count,
            "neptune_edges": self.edge_count,
            "export_mode": self.export_mode,
            "since": self.manifest.get("since"),
            "as_of": self.manifest.get("as_of"),
            "aws_called": False,
        }

    @property
    def export_mode(self) -> str:
        return str(self.manifest.get("export_mode", "snapshot"))


def read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            rows.append(value)
    return tuple(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.as_posix()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _csv_row_count(path: Path, required: frozenset[str]) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = frozenset(reader.fieldnames or ())
        missing = sorted(required - columns)
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(missing)}"
            )
        return sum(1 for _ in reader)


def validate_production_artifact(root: Path) -> ProductionArtifact:
    root = root.resolve()
    required_paths = (
        root / "manifest.json",
        root / "memory_import.jsonl",
        root / "catalog_items.jsonl",
        root / "neptune" / "nodes.csv",
        root / "neptune" / "edges.csv",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "production artifact is incomplete: " + ", ".join(missing)
        )

    manifest = json.loads(required_paths[0].read_text(encoding="utf-8"))
    if manifest.get("schema") != PRODUCTION_MANIFEST_SCHEMA:
        raise ValueError(
            "unsupported production manifest schema: "
            f"{manifest.get('schema')!r}"
        )
    unsafe_flags = [
        name for name in FORBIDDEN_MANIFEST_FLAGS if manifest.get(name) is not False
    ]
    if unsafe_flags:
        raise ValueError(
            "production artifact must explicitly exclude evaluation data: "
            + ", ".join(unsafe_flags)
        )
    export_mode = str(manifest.get("export_mode", "snapshot"))
    if export_mode not in {"snapshot", "incremental"}:
        raise ValueError(f"unsupported production export_mode: {export_mode!r}")
    since = manifest.get("since")
    if export_mode == "snapshot" and since is not None:
        raise ValueError("snapshot artifact must not declare since")
    if export_mode == "incremental" and not since:
        raise ValueError("incremental artifact must declare since")
    identity_namespace_id = str(
        manifest.get("identity_namespace_id", "")
    )
    if len(identity_namespace_id) != 24 or any(
        value not in "0123456789abcdef"
        for value in identity_namespace_id
    ):
        raise ValueError(
            "production manifest must include a valid identity_namespace_id"
        )

    memory_rows = read_jsonl(required_paths[1])
    for index, row in enumerate(memory_rows, start=1):
        if row.get("operation") != "record_observations":
            raise ValueError(
                f"memory_import.jsonl row {index} must use record_observations"
            )
        observation = row.get("observation")
        if not isinstance(observation, Mapping):
            raise ValueError(
                f"memory_import.jsonl row {index} has no observation object"
            )
        batch = ObservationBatch.from_dict(observation)
        if batch.source != "historical_import":
            raise ValueError(
                f"memory_import.jsonl row {index} must use historical_import"
            )

    catalog_rows = read_jsonl(required_paths[2])
    catalog: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(catalog_rows, start=1):
        item = Item.from_dict(row)
        if item.item_id in catalog:
            raise ValueError(
                f"catalog_items.jsonl contains duplicate item {item.item_id!r}"
            )
        catalog[item.item_id] = row

    node_count = _csv_row_count(required_paths[3], REQUIRED_NODE_COLUMNS)
    edge_count = _csv_row_count(required_paths[4], REQUIRED_EDGE_COLUMNS)
    expected_counts = {
        "memory_batches": len(memory_rows),
        "neptune_nodes": node_count,
        "neptune_edges": edge_count,
    }
    for field, observed in expected_counts.items():
        expected = manifest.get(field)
        if expected is None or int(expected) != observed:
            raise ValueError(
                f"manifest {field}={expected!r} does not match artifact count "
                f"{observed}"
            )

    fingerprint = artifact_fingerprint(required_paths)
    return ProductionArtifact(
        root=root,
        manifest=manifest,
        memory_rows=memory_rows,
        catalog_items=catalog,
        node_count=node_count,
        edge_count=edge_count,
        fingerprint=fingerprint,
    )


def memory_row_id(row: Mapping[str, Any]) -> str:
    observation = row["observation"]
    canonical = json.dumps(
        observation,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_bootstrap_state(
    path: Path,
    *,
    artifact_fingerprint_value: str,
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema": BOOTSTRAP_STATE_SCHEMA,
            "artifact_fingerprint": artifact_fingerprint_value,
            "memory_completed": [],
            "neptune": {},
            "canary": {},
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema") != BOOTSTRAP_STATE_SCHEMA:
        raise ValueError(f"unsupported bootstrap state schema in {path}")
    if state.get("artifact_fingerprint") != artifact_fingerprint_value:
        raise ValueError(
            "bootstrap state belongs to a different production artifact"
        )
    state.setdefault("memory_completed", [])
    state.setdefault("neptune", {})
    state.setdefault("canary", {})
    return state


def write_bootstrap_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_bootstrap_lineage(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema": BOOTSTRAP_LINEAGE_SCHEMA,
            "latest_as_of": None,
            "artifacts": [],
        }
    lineage = json.loads(path.read_text(encoding="utf-8"))
    if lineage.get("schema") != BOOTSTRAP_LINEAGE_SCHEMA:
        raise ValueError(f"unsupported bootstrap lineage schema in {path}")
    lineage.setdefault("latest_as_of", None)
    lineage.setdefault("artifacts", [])
    return lineage


def validate_artifact_lineage(
    artifact: ProductionArtifact,
    lineage: Mapping[str, Any],
) -> None:
    completed = {
        str(value.get("fingerprint"))
        for value in lineage.get("artifacts", ())
        if isinstance(value, Mapping)
    }
    if artifact.fingerprint in completed:
        return
    latest_as_of = lineage.get("latest_as_of")
    if artifact.export_mode == "snapshot":
        if latest_as_of is not None:
            raise ValueError(
                "a snapshot cannot follow an initialized lineage; create an "
                "incremental artifact or use a separate lineage path"
            )
        return
    expected_dataset = lineage.get("dataset_name")
    if expected_dataset != artifact.manifest.get("dataset_name"):
        raise ValueError(
            "incremental artifact dataset_name does not match the lineage"
        )
    expected_namespace = lineage.get("identity_namespace_id")
    if expected_namespace != artifact.manifest.get("identity_namespace_id"):
        raise ValueError(
            "incremental artifact identity salt does not match the lineage"
        )
    since = artifact.manifest.get("since")
    if latest_as_of is None:
        raise ValueError(
            "incremental bootstrap requires a completed snapshot lineage"
        )
    if str(since) != str(latest_as_of):
        raise ValueError(
            "incremental artifact since must equal the previous as_of: "
            f"expected {latest_as_of!r}, got {since!r}"
        )


def record_artifact_lineage(
    path: Path,
    lineage: dict[str, Any],
    artifact: ProductionArtifact,
) -> None:
    entries = [
        value
        for value in lineage.get("artifacts", ())
        if isinstance(value, Mapping)
        and value.get("fingerprint") != artifact.fingerprint
    ]
    entries.append(
        {
            "fingerprint": artifact.fingerprint,
            "export_mode": artifact.export_mode,
            "since": artifact.manifest.get("since"),
            "as_of": artifact.manifest.get("as_of"),
        }
    )
    lineage["artifacts"] = entries
    lineage["latest_as_of"] = artifact.manifest.get("as_of")
    lineage["dataset_name"] = artifact.manifest.get("dataset_name")
    lineage["identity_namespace_id"] = artifact.manifest.get(
        "identity_namespace_id"
    )
    write_bootstrap_state(path, lineage)


def import_memory_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    invoke: Callable[[Mapping[str, Any], str, str], Mapping[str, Any]],
    state: dict[str, Any],
    state_path: Path,
) -> dict[str, int]:
    completed = set(str(value) for value in state.get("memory_completed", ()))
    imported = 0
    skipped = 0
    events = 0
    long_term_records = 0

    for row in rows:
        row_id = memory_row_id(row)
        if row_id in completed:
            skipped += 1
            continue
        observation = row["observation"]
        batch = ObservationBatch.from_dict(observation)
        receipt = invoke(
            row,
            batch.user_id,
            f"bootstrap-memory-{row_id[:24]}",
        )
        if receipt.get("error"):
            raise RuntimeError(
                f"Memory import failed for {batch.user_id}/{batch.session_id}: "
                f"{receipt['error']}"
            )
        if str(receipt.get("user_id")) != batch.user_id:
            raise RuntimeError("Memory receipt user_id does not match request")
        if str(receipt.get("session_id")) != batch.session_id:
            raise RuntimeError("Memory receipt session_id does not match request")
        event_count = int(receipt.get("event_count", -1))
        if event_count != len(batch.events):
            raise RuntimeError(
                f"Memory receipt event_count={event_count} does not match "
                f"{len(batch.events)}"
            )
        record_count = int(receipt.get("long_term_record_count", 0))
        if record_count < 1:
            raise RuntimeError("Memory import created no long-term records")

        completed.add(row_id)
        state["memory_completed"] = sorted(completed)
        write_bootstrap_state(state_path, state)
        imported += 1
        events += event_count
        long_term_records += record_count

    return {
        "imported_batches": imported,
        "skipped_batches": skipped,
        "imported_events": events,
        "long_term_records": long_term_records,
        "completed_batches": len(completed),
    }


def build_canary_request(artifact: ProductionArtifact) -> tuple[str, str, dict[str, Any]]:
    if not artifact.memory_rows:
        raise ValueError("cannot build a canary without Memory import rows")
    row = artifact.memory_rows[0]
    observation = ObservationBatch.from_dict(row["observation"])
    item_id = next(
        (
            value
            for value in observation.recommended_item_ids
            if value in artifact.catalog_items
        ),
        None,
    )
    if item_id is None:
        item_id = next(iter(artifact.catalog_items), None)
    if item_id is None:
        raise ValueError("cannot build a canary without catalog items")
    target_product = dict(artifact.catalog_items[item_id])
    imported_session_id = observation.session_id
    canary_session_id = f"bootstrap-canary-{artifact.fingerprint[:24]}"
    request = {
        "operation": "simulate",
        "request": {
            "request_id": f"bootstrap-canary-{artifact.fingerprint[:16]}",
            "memory_session_id": canary_session_id,
            "user": {
                "user_id": observation.user_id,
                "persona_summary": "Imported historical game-store behavior",
                "pickiness": 0.5,
                "price_sensitivity": 0.5,
                "category_preferences": {},
                "engagement": 0.5,
                "variety": 0.5,
            },
            "target_product": target_product,
            "exposure_scenario": {
                "surface": "store_home",
                "session_fatigue": 0.0,
            },
            "game_state": {
                "currency_balance": max(
                    float(target_product.get("price", 0.0)) * 2.0,
                    1.0,
                ),
                "progression_need": 0.5,
                "recent_failure_intensity": 0.0,
                "inventory_overlap": 0.0,
                "event_urgency": 0.0,
                "purchase_cooldown": 0.0,
                "current_goals": [],
                "owned_item_ids": [],
            },
            "interactions": [],
        },
    }
    return observation.user_id, imported_session_id, request


def validate_canary_result(
    result: Mapping[str, Any],
    *,
    require_memory: bool,
    require_neptune: bool,
) -> dict[str, Any]:
    if "scalar_purchase_probability" not in result:
        raise RuntimeError("bootstrap canary returned no scalar probability")
    if "trajectory_purchase_probability" not in result:
        raise RuntimeError("bootstrap canary returned no trajectory probability")
    components = result.get("components", {})
    if not isinstance(components, Mapping):
        raise RuntimeError("bootstrap canary returned invalid components")
    memory_records = int(components.get("episodic_memory_records", 0))
    memory_events = int(components.get("episodic_memory_events", 0))
    memory_transitions = int(
        components.get("observed_memory_transitions", 0)
    )
    graph_support = float(
        components.get("knowledge_graph_retrieval_support", 0.0)
    )
    if require_memory and memory_records < 1:
        raise RuntimeError(
            "bootstrap canary did not retrieve imported long-term Memory records"
        )
    if require_neptune and graph_support <= 0.0:
        raise RuntimeError(
            "bootstrap canary did not retrieve evidence from the loaded "
            "Neptune graph"
        )
    return {
        "episodic_memory_records": memory_records,
        "episodic_memory_events": memory_events,
        "observed_memory_transitions": memory_transitions,
        "knowledge_graph_retrieval_support": graph_support,
        "action_graph_id": result.get("action_graph_id"),
    }


def extract_loader_status(response: Mapping[str, Any]) -> str:
    payload = response.get("payload", response)
    if isinstance(payload, Mapping):
        overall = payload.get("overallStatus")
        if isinstance(overall, Mapping) and overall.get("status"):
            return str(overall["status"]).upper()
        if payload.get("status"):
            return str(payload["status"]).upper()
    if response.get("status"):
        return str(response["status"]).upper()
    return "UNKNOWN"


def extract_loader_id(response: Mapping[str, Any]) -> str:
    payload = response.get("payload", response)
    if isinstance(payload, Mapping):
        for key in ("loadId", "load_id"):
            if payload.get(key):
                return str(payload[key])
    for key in ("loadId", "load_id"):
        if response.get(key):
            return str(response[key])
    raise ValueError("Neptune loader response did not include a load ID")
