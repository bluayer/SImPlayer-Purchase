from __future__ import annotations

import csv
import hashlib
import hmac
import json
import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .action_rollout import DEFAULT_ACTION_GRAPH
from .evaluation import (
    HoldoutCase,
    HoldoutProtocol,
    graph_evidence,
    item_payload,
)
from .models import BehaviorEvent, ProductComponent


# Legacy two-step fields remain readable while observed_action_path is the
# canonical graph-native label.
EXPOSURE_ACTIONS = frozenset({"CLICK", "SKIP", "EXIT", "PURCHASE_NOW"})
DETAIL_ACTIONS = frozenset({"PURCHASE", "BACK", "EXIT"})
EVENT_ACTIONS = frozenset({"IMPRESSION", "CLICK", "PURCHASE", "BACK", "EXIT"})
CART_ACTIONS = frozenset(
    {"ADD_TO_CART", "CART", "CHECKOUT", "REMOVE", "CONTINUE"}
)
GAME_STATE_FIELDS = (
    "currency_balance",
    "progression_need",
    "recent_failure_intensity",
    "inventory_overlap",
    "event_urgency",
    "purchase_cooldown",
    "current_goals",
    "owned_item_ids",
)


class DatasetValidationError(ValueError):
    pass


def build_path_coverage_protocol(
    protocol: HoldoutProtocol,
    *,
    max_cases: int = 100,
) -> HoldoutProtocol:
    """Select a deterministic diagnostic suite that favors rare action paths.

    The returned protocol is for path coverage only. Its action frequencies
    must not be interpreted as a natural traffic distribution.
    """
    if max_cases < 1:
        raise ValueError("max_cases must be positive")
    if not protocol.cases:
        raise ValueError("cannot build path coverage from an empty protocol")

    action_counts = Counter(
        action
        for case in protocol.cases
        for action in case.observed_action_path
    )
    remaining = sorted(protocol.cases, key=lambda case: case.case_id)
    selected: list[HoldoutCase] = []
    covered_actions: set[str] = set()
    covered_paths: set[tuple[str, ...]] = set()

    while remaining and len(selected) < min(max_cases, len(protocol.cases)):
        def priority(case: HoldoutCase) -> tuple[float, int, str]:
            path = tuple(case.observed_action_path)
            new_actions = len(set(path).difference(covered_actions))
            new_path = int(path not in covered_paths)
            rarity = sum(
                1.0 / max(1, action_counts[action])
                for action in path
            )
            score = 100.0 * new_actions + 25.0 * new_path + rarity
            return (score, len(path), case.case_id)

        chosen = max(remaining, key=priority)
        remaining.remove(chosen)
        selected.append(
            replace(chosen, ratio_membership=("path_coverage",))
        )
        covered_actions.update(chosen.observed_action_path)
        covered_paths.add(tuple(chosen.observed_action_path))

    source_protocol = dict(
        protocol.natural_metrics.get("protocol", {})
    )
    coverage_metadata = {
        **source_protocol,
        "sampling": "rare-path diagnostic coverage",
        "frequency_metrics_valid": False,
        "source_run_id": protocol.run_id,
        "selected_cases": len(selected),
        "covered_actions": sorted(covered_actions),
        "covered_path_count": len(covered_paths),
    }
    return HoldoutProtocol(
        run_id=f"{protocol.run_id}-coverage-{len(selected)}",
        users=len({case.original_user_id for case in selected}),
        history_fraction=protocol.history_fraction,
        history_transition_limit=protocol.history_transition_limit,
        cases=tuple(selected),
        bootstrap_payloads=protocol.bootstrap_payloads,
        natural_metrics={"protocol": coverage_metadata},
    )


def _timestamp(value: Any) -> datetime:
    if value is None or str(value).strip() == "":
        raise DatasetValidationError("timestamp is required")
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _float(value: Any, default: float = 0.0) -> float:
    optional = _optional_float(value)
    return default if optional is None else optional


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "y", "clicked", "purchased"}:
        return True
    if normalized in {"", "0", "false", "no", "n", "none", "null"}:
        return False
    raise DatasetValidationError(f"cannot parse boolean value {value!r}")


def _json_or_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    if stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _string_list(value: Any, *, separator: str = "|") -> tuple[str, ...]:
    parsed = _json_or_value(value)
    if parsed is None:
        return ()
    if isinstance(parsed, str):
        return tuple(
            part.strip() for part in parsed.split(separator) if part.strip()
        )
    if isinstance(parsed, Sequence) and not isinstance(parsed, (bytes, bytearray)):
        return tuple(str(part) for part in parsed if str(part).strip())
    return (str(parsed),)


def _float_mapping(value: Any) -> dict[str, float]:
    parsed = _json_or_value(value)
    if not isinstance(parsed, Mapping):
        return {}
    return {str(key): float(item) for key, item in parsed.items()}


def _product_components(
    value: Any,
    *,
    separator: str = "|",
) -> tuple[ProductComponent, ...]:
    parsed = _json_or_value(value)
    if parsed is None:
        return ()
    values: Sequence[Any]
    if isinstance(parsed, str):
        values = tuple(
            part.strip() for part in parsed.split(separator) if part.strip()
        )
    elif isinstance(parsed, Mapping):
        values = (parsed,)
    elif isinstance(parsed, Sequence) and not isinstance(parsed, (bytes, bytearray)):
        values = parsed
    else:
        values = (parsed,)

    components: list[ProductComponent] = []
    for component in values:
        if isinstance(component, str) and ":" in component:
            item_id, quantity = component.rsplit(":", 1)
            component = {
                "item_id": item_id.strip(),
                "quantity": int(quantity),
            }
        try:
            components.append(ProductComponent.from_value(component))
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError(
                f"invalid product component {component!r}: {exc}"
            ) from exc
    return tuple(components)


def _read_rows(path: Path, format_name: str | None = None) -> list[dict[str, Any]]:
    format_name = (format_name or path.suffix.lstrip(".")).lower()
    if format_name == "csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if format_name in {"jsonl", "ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetValidationError(
                        f"{path}:{line_number}: invalid JSON"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise DatasetValidationError(
                        f"{path}:{line_number}: row must be an object"
                    )
                rows.append(dict(value))
        return rows
    if format_name == "json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or any(
            not isinstance(row, Mapping) for row in value
        ):
            raise DatasetValidationError(f"{path}: JSON must contain an object list")
        return [dict(row) for row in value]
    raise DatasetValidationError(
        f"unsupported tabular format {format_name!r} for {path}"
    )


def _mapped(
    row: Mapping[str, Any],
    columns: Mapping[str, Any],
    name: str,
    default: Any = None,
) -> Any:
    source = columns.get(name)
    if source is None:
        return default
    return row.get(str(source), default)


def _mapped_object(
    row: Mapping[str, Any],
    columns: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    mapping = columns.get(name, {})
    if not isinstance(mapping, Mapping):
        raise DatasetValidationError(f"{name} mapping must be an object")
    return {
        str(canonical): row.get(str(source))
        for canonical, source in mapping.items()
        if row.get(str(source)) not in (None, "")
    }


def _mapped_game_state(
    row: Mapping[str, Any],
    columns: Mapping[str, Any],
) -> dict[str, Any]:
    mapping = columns.get("game_state", {})
    if not isinstance(mapping, Mapping):
        raise DatasetValidationError("game_state mapping must be an object")
    normalized: dict[str, Any] = {}
    for canonical, source in mapping.items():
        name = str(canonical)
        value = row.get(str(source))
        if name in {"current_goals", "owned_item_ids"}:
            normalized[name] = list(_string_list(value))
        elif name == "currency_balance":
            normalized[name] = _optional_float(value)
        elif value in (None, ""):
            continue
        else:
            normalized[name] = float(value)
    return normalized


@dataclass(frozen=True)
class CanonicalUser:
    user_id: str
    persona_summary: str = ""
    pickiness: float = 0.5
    price_sensitivity: float = 0.5
    category_preferences: Mapping[str, float] = field(default_factory=dict)
    engagement: float = 0.5
    variety: float = 0.5
    budget_reference: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalItem:
    item_id: str
    product_type: str = "item"
    categories: tuple[str, ...] = ()
    price: float = 0.0
    discount_rate: float = 0.0
    components: tuple[ProductComponent, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "components": [
                component.to_dict() for component in self.components
            ],
        }

    def to_evaluation_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "product_type": self.product_type,
            "categories": self.categories,
            "price": self.price,
            "discount_rate": self.discount_rate,
            "components": [
                component.to_dict() for component in self.components
            ],
            **dict(self.attributes),
        }


@dataclass(frozen=True)
class CanonicalImpression:
    impression_id: str
    user_id: str
    item_id: str
    timestamp: datetime
    session_id: str
    surface: str = "store_home"
    clicked: bool = False
    purchased: bool = False
    click_timestamp: datetime | None = None
    purchase_timestamp: datetime | None = None
    session_fatigue: float = 0.0
    budget_reference: float | None = None
    context_features: Mapping[str, float] = field(default_factory=dict)
    game_state: Mapping[str, Any] = field(default_factory=dict)
    observed_initial_state: str | None = None
    observed_next_action: str | None = None
    observed_detail_action: str | None = None
    observed_action_path: tuple[str, ...] = ()
    oracle_probability: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def action_labels(self) -> tuple[str, str, str | None]:
        if self.observed_next_action:
            initial_state = (
                self.observed_initial_state
                or ("ITEM_DETAIL" if self.surface == "checkout" else "ITEM_EXPOSURE")
            ).upper()
            next_action = self.observed_next_action.upper()
            detail_action = (
                self.observed_detail_action.upper()
                if self.observed_detail_action
                else None
            )
            if initial_state == "ITEM_EXPOSURE":
                if next_action not in EXPOSURE_ACTIONS:
                    raise DatasetValidationError(
                        f"{self.impression_id}: invalid exposure action {next_action}"
                    )
                if next_action == "CLICK" and detail_action not in DETAIL_ACTIONS:
                    raise DatasetValidationError(
                        f"{self.impression_id}: CLICK requires a detail action"
                    )
            elif initial_state == "ITEM_DETAIL":
                if next_action not in DETAIL_ACTIONS:
                    raise DatasetValidationError(
                        f"{self.impression_id}: invalid detail action {next_action}"
                    )
                if detail_action is not None:
                    raise DatasetValidationError(
                        f"{self.impression_id}: detail-start case cannot have "
                        "observed_detail_action"
                    )
            else:
                raise DatasetValidationError(
                    f"{self.impression_id}: unsupported state {initial_state}"
                )
            return initial_state, next_action, detail_action

        if self.surface == "checkout":
            return (
                "ITEM_DETAIL",
                "PURCHASE" if self.purchased else "BACK",
                None,
            )
        if self.clicked:
            return (
                "ITEM_EXPOSURE",
                "CLICK",
                "PURCHASE" if self.purchased else "BACK",
            )
        return (
            "ITEM_EXPOSURE",
            "PURCHASE_NOW" if self.purchased else "SKIP",
            None,
        )

    def action_path(self) -> tuple[str, tuple[str, ...]]:
        initial_state = (
            self.observed_initial_state
            or ("ITEM_DETAIL" if self.surface == "checkout" else "ITEM_EXPOSURE")
        ).upper()
        if self.observed_action_path:
            actions = tuple(
                str(action).upper() for action in self.observed_action_path
            )
        else:
            legacy_initial, next_action, detail_action = self.action_labels()
            initial_state = legacy_initial
            if initial_state == "ITEM_EXPOSURE" and next_action == "CLICK":
                actions = (
                    "CLICK",
                    *(
                        (
                            "START_PURCHASE",
                            "CONFIRM_PURCHASE",
                            "PAYMENT_SUCCESS",
                        )
                        if detail_action == "PURCHASE"
                        else ("BACK", "SKIP")
                    ),
                )
            elif next_action in {"PURCHASE", "PURCHASE_NOW"}:
                actions = (
                    *(() if initial_state == "ITEM_DETAIL" else ("PURCHASE_NOW",)),
                    *(("START_PURCHASE",) if initial_state == "ITEM_DETAIL" else ()),
                    "CONFIRM_PURCHASE",
                    "PAYMENT_SUCCESS",
                )
            else:
                actions = (
                    next_action,
                    *(
                        ("SKIP",)
                        if initial_state == "ITEM_DETAIL"
                        and next_action == "BACK"
                        else ()
                    ),
                )
        state = initial_state
        for index, action in enumerate(actions):
            if DEFAULT_ACTION_GRAPH.is_terminal(state):
                raise DatasetValidationError(
                    f"{self.impression_id}: action after terminal state at index {index}"
                )
            try:
                state = DEFAULT_ACTION_GRAPH.transition(
                    state,
                    action,
                ).next_state
            except ValueError as exc:
                raise DatasetValidationError(
                    f"{self.impression_id}: invalid action path at "
                    f"{state}/{action}"
                ) from exc
        if not DEFAULT_ACTION_GRAPH.is_terminal(state):
            raise DatasetValidationError(
                f"{self.impression_id}: action path does not reach a terminal state"
            )
        return initial_state, actions

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat(),
            "click_timestamp": (
                self.click_timestamp.isoformat()
                if self.click_timestamp is not None
                else None
            ),
            "purchase_timestamp": (
                self.purchase_timestamp.isoformat()
                if self.purchase_timestamp is not None
                else None
            ),
        }


@dataclass(frozen=True)
class CanonicalDataset:
    dataset_name: str
    users: Mapping[str, CanonicalUser]
    items: Mapping[str, CanonicalItem]
    impressions: tuple[CanonicalImpression, ...]
    complete_exposure: bool = True
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> dict[str, Any]:
        if not self.impressions:
            raise DatasetValidationError("dataset contains no impressions")
        for item in self.items.values():
            if item.product_type not in {"item", "bundle"}:
                raise DatasetValidationError(
                    f"{item.item_id}: product_type must be item or bundle"
                )
            if item.price < 0.0:
                raise DatasetValidationError(
                    f"{item.item_id}: price cannot be negative"
                )
            if not 0.0 <= item.discount_rate <= 1.0:
                raise DatasetValidationError(
                    f"{item.item_id}: discount_rate must be between 0 and 1"
                )
            if item.product_type == "bundle" and not item.components:
                raise DatasetValidationError(
                    f"{item.item_id}: bundle requires at least one component"
                )
            for component in item.components:
                if component.item_id == item.item_id:
                    raise DatasetValidationError(
                        f"{item.item_id}: bundle cannot contain itself"
                    )
                if component.item_id not in self.items:
                    raise DatasetValidationError(
                        f"{item.item_id}: unknown bundle component "
                        f"{component.item_id}"
                    )
        seen: set[str] = set()
        action_counts: Counter[str] = Counter()
        game_state_coverage: Counter[str] = Counter()
        game_state_values: dict[str, set[str]] = {
            field: set() for field in GAME_STATE_FIELDS
        }
        for impression in self.impressions:
            if impression.impression_id in seen:
                raise DatasetValidationError(
                    f"duplicate impression_id {impression.impression_id}"
                )
            seen.add(impression.impression_id)
            if impression.user_id not in self.users:
                raise DatasetValidationError(
                    f"{impression.impression_id}: unknown user {impression.user_id}"
                )
            if impression.item_id not in self.items:
                raise DatasetValidationError(
                    f"{impression.impression_id}: unknown item {impression.item_id}"
                )
            if impression.timestamp.tzinfo is None:
                raise DatasetValidationError(
                    f"{impression.impression_id}: timestamp must be timezone-aware"
                )
            initial, actions = impression.action_path()
            state = initial
            for action in actions:
                action_counts[f"{state}:{action}"] += 1
                state = DEFAULT_ACTION_GRAPH.transition(
                    state,
                    action,
                ).next_state
            for field in GAME_STATE_FIELDS:
                if field in impression.game_state:
                    game_state_coverage[field] += 1
                    game_state_values[field].add(
                        json.dumps(
                            impression.game_state[field],
                            sort_keys=True,
                        )
                    )
        impression_count = len(self.impressions)
        return {
            "dataset_name": self.dataset_name,
            "users": len(self.users),
            "items": len(self.items),
            "impressions": len(self.impressions),
            "complete_exposure": self.complete_exposure,
            "purchases": sum(item.purchased for item in self.impressions),
            "clicks": sum(item.clicked for item in self.impressions),
            "action_counts": dict(sorted(action_counts.items())),
            "game_state": {
                "coverage": {
                    field: round(
                        game_state_coverage[field]
                        / max(1, impression_count),
                        8,
                    )
                    for field in GAME_STATE_FIELDS
                },
                "unique_values": {
                    field: len(game_state_values[field])
                    for field in GAME_STATE_FIELDS
                },
            },
        }

    def write(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            output_dir / "users.jsonl",
            (user.to_dict() for user in self.users.values()),
        )
        _write_jsonl(
            output_dir / "items.jsonl",
            (item.to_dict() for item in self.items.values()),
        )
        _write_jsonl(
            output_dir / "impressions.jsonl",
            (item.to_dict() for item in self.impressions),
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "simuser.canonical-dataset.v2",
                    "validation": self.validate(),
                    "source_metadata": self.source_metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


class DatasetAdapter(ABC):
    @abstractmethod
    def load(self) -> CanonicalDataset:
        raise NotImplementedError


class CanonicalJsonlDatasetAdapter(DatasetAdapter):
    def __init__(self, data_dir: Path, *, dataset_name: str | None = None) -> None:
        self.data_dir = data_dir
        self.dataset_name = dataset_name or data_dir.name

    def load(self) -> CanonicalDataset:
        manifest_path = self.data_dir / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        users = {
            str(row["user_id"]): _canonical_user(row)
            for row in _read_rows(self.data_dir / "users.jsonl", "jsonl")
        }
        items = {
            str(row["item_id"]): _canonical_item(row)
            for row in _read_rows(self.data_dir / "items.jsonl", "jsonl")
        }
        impression_rows = _read_rows(
            self.data_dir / "impressions.jsonl", "jsonl"
        )
        impressions = tuple(_canonical_impression(row) for row in impression_rows)
        dataset = CanonicalDataset(
            dataset_name=(
                self.dataset_name
                if self.dataset_name != self.data_dir.name
                else str(
                    manifest.get("validation", {}).get(
                        "dataset_name", self.dataset_name
                    )
                )
            ),
            users=users,
            items=items,
            impressions=impressions,
            complete_exposure=bool(
                manifest.get("validation", {}).get(
                    "complete_exposure", True
                )
            ),
            source_metadata={
                **dict(manifest.get("source_metadata", {})),
                "adapter": "canonical_jsonl",
                "synthetic": bool(
                    manifest.get("source_metadata", {}).get("synthetic", False)
                    or any(bool(row.get("synthetic")) for row in impression_rows)
                ),
            },
        )
        dataset.validate()
        return dataset


class MappedTabularDatasetAdapter(DatasetAdapter):
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(self.config, Mapping):
            raise DatasetValidationError("adapter config must be a JSON object")

    def load(self) -> CanonicalDataset:
        mode = str(self.config.get("input_mode", "impression_rows"))
        if mode not in {"impression_rows", "event_rows"}:
            raise DatasetValidationError(
                "input_mode must be impression_rows or event_rows"
            )
        files = self.config.get("files", {})
        columns = self.config.get("columns", {})
        formats = self.config.get("formats", {})
        if not isinstance(files, Mapping) or not isinstance(columns, Mapping):
            raise DatasetValidationError("files and columns must be objects")
        base = self.config_path.parent

        interaction_path = base / str(files.get("interactions", ""))
        if not interaction_path.is_file():
            raise DatasetValidationError(
                f"interaction file does not exist: {interaction_path}"
            )
        interaction_rows = _read_rows(
            interaction_path,
            str(formats.get("interactions", "")) or None,
        )

        users = self._users(base, files, formats, columns, interaction_rows)
        items = self._items(base, files, formats, columns, interaction_rows)
        if mode == "impression_rows":
            impressions = self._impression_rows(interaction_rows, columns)
        else:
            impressions = self._event_rows(interaction_rows, columns)

        options = self.config.get("options", {})
        require_impressions = bool(
            options.get("require_impressions", True)
            if isinstance(options, Mapping)
            else True
        )
        dataset = CanonicalDataset(
            dataset_name=str(
                self.config.get("dataset_name", self.config_path.stem)
            ),
            users=users,
            items=items,
            impressions=tuple(
                sorted(
                    impressions,
                    key=lambda value: (
                        value.user_id,
                        value.timestamp,
                        value.impression_id,
                    ),
                )
            ),
            complete_exposure=require_impressions,
            source_metadata={
                "adapter": "mapped_tabular",
                "config": str(self.config_path),
                "input_mode": mode,
                "synthetic": bool(self.config.get("synthetic", False)),
            },
        )
        dataset.validate()
        return dataset

    def _users(
        self,
        base: Path,
        files: Mapping[str, Any],
        formats: Mapping[str, Any],
        columns: Mapping[str, Any],
        interaction_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, CanonicalUser]:
        user_columns = columns.get("users", {})
        user_file = files.get("users")
        if user_file:
            rows = _read_rows(
                base / str(user_file),
                str(formats.get("users", "")) or None,
            )
            users = {}
            for row in rows:
                user_id = str(_mapped(row, user_columns, "user_id", "")).strip()
                if not user_id:
                    raise DatasetValidationError("user row is missing user_id")
                if user_id in users:
                    raise DatasetValidationError(f"duplicate user_id {user_id}")
                users[user_id] = CanonicalUser(
                    user_id=user_id,
                    persona_summary=str(
                        _mapped(row, user_columns, "persona_summary", "")
                    ),
                    pickiness=_float(
                        _mapped(row, user_columns, "pickiness"), 0.5
                    ),
                    price_sensitivity=_float(
                        _mapped(row, user_columns, "price_sensitivity"), 0.5
                    ),
                    category_preferences=_float_mapping(
                        _mapped(row, user_columns, "category_preferences")
                    ),
                    engagement=_float(
                        _mapped(row, user_columns, "engagement"), 0.5
                    ),
                    variety=_float(_mapped(row, user_columns, "variety"), 0.5),
                    budget_reference=_optional_float(
                        _mapped(row, user_columns, "budget_reference")
                    ),
                    attributes=_mapped_object(row, user_columns, "attributes"),
                )
            return users

        interaction_columns = columns.get("interactions", {})
        user_ids = {
            str(_mapped(row, interaction_columns, "user_id", "")).strip()
            for row in interaction_rows
        }
        user_ids.discard("")
        return {user_id: CanonicalUser(user_id=user_id) for user_id in user_ids}

    def _items(
        self,
        base: Path,
        files: Mapping[str, Any],
        formats: Mapping[str, Any],
        columns: Mapping[str, Any],
        interaction_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, CanonicalItem]:
        item_columns = columns.get("items", {})
        item_file = files.get("items")
        separator = str(self.config.get("options", {}).get("categories_separator", "|"))
        if item_file:
            rows = _read_rows(
                base / str(item_file),
                str(formats.get("items", "")) or None,
            )
            items = {}
            for row in rows:
                item_id = str(_mapped(row, item_columns, "item_id", "")).strip()
                if not item_id:
                    raise DatasetValidationError("item row is missing item_id")
                if item_id in items:
                    raise DatasetValidationError(f"duplicate item_id {item_id}")
                items[item_id] = CanonicalItem(
                    item_id=item_id,
                    product_type=str(
                        _mapped(row, item_columns, "product_type", "item")
                    ).lower(),
                    categories=_string_list(
                        _mapped(row, item_columns, "categories"),
                        separator=separator,
                    ),
                    price=_float(_mapped(row, item_columns, "price"), 0.0),
                    discount_rate=_float(
                        _mapped(row, item_columns, "discount_rate"), 0.0
                    ),
                    components=_product_components(
                        _mapped(row, item_columns, "components"),
                        separator=separator,
                    ),
                    attributes=_mapped_object(row, item_columns, "attributes"),
                )
            return items

        interaction_columns = columns.get("interactions", {})
        item_ids = {
            str(_mapped(row, interaction_columns, "item_id", "")).strip()
            for row in interaction_rows
        }
        item_ids.discard("")
        return {item_id: CanonicalItem(item_id=item_id) for item_id in item_ids}

    def _impression_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        columns: Mapping[str, Any],
    ) -> list[CanonicalImpression]:
        interaction_columns = columns.get("interactions", {})
        defaults = self.config.get("defaults", {})
        return [
            _impression_from_mapped_row(
                row,
                interaction_columns,
                defaults if isinstance(defaults, Mapping) else {},
            )
            for row in rows
        ]

    def _event_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        columns: Mapping[str, Any],
    ) -> list[CanonicalImpression]:
        interaction_columns = columns.get("interactions", {})
        defaults = self.config.get("defaults", {})
        defaults = defaults if isinstance(defaults, Mapping) else {}
        action_map = {
            str(key).upper(): str(value).upper()
            for key, value in self.config.get("action_map", {}).items()
        }
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            impression_id = str(
                _mapped(row, interaction_columns, "impression_id", "")
            ).strip()
            if not impression_id:
                raise DatasetValidationError(
                    "event_rows require an impression_id column"
                )
            raw_action = str(
                _mapped(row, interaction_columns, "action", "")
            ).upper()
            action = action_map.get(raw_action, raw_action)
            if action in CART_ACTIONS:
                raise DatasetValidationError(
                    f"{impression_id}: cart action {action} is unsupported"
                )
            if action not in EVENT_ACTIONS:
                raise DatasetValidationError(
                    f"{impression_id}: unsupported event action {raw_action!r}"
                )
            grouped[impression_id].append(
                {
                    "row": row,
                    "action": action,
                    "timestamp": _timestamp(
                        _mapped(row, interaction_columns, "timestamp")
                    ),
                }
            )

        impressions = []
        require_impressions = bool(
            self.config.get("options", {}).get("require_impressions", True)
        )
        for impression_id, events in grouped.items():
            events.sort(key=lambda value: value["timestamp"])
            impression_events = [
                value for value in events if value["action"] == "IMPRESSION"
            ]
            if require_impressions and not impression_events:
                raise DatasetValidationError(
                    f"{impression_id}: no IMPRESSION event; SKIP is not observable"
                )
            first = (impression_events or events)[0]
            row = first["row"]
            user_id = str(
                _mapped(row, interaction_columns, "user_id", "")
            ).strip()
            item_id = str(
                _mapped(row, interaction_columns, "item_id", "")
            ).strip()
            if any(
                str(_mapped(value["row"], interaction_columns, "user_id", "")).strip()
                != user_id
                or str(
                    _mapped(value["row"], interaction_columns, "item_id", "")
                ).strip()
                != item_id
                for value in events
            ):
                raise DatasetValidationError(
                    f"{impression_id}: events disagree on user_id or item_id"
                )
            clicked_at = next(
                (
                    value["timestamp"]
                    for value in events
                    if value["action"] == "CLICK"
                ),
                None,
            )
            purchased_at = next(
                (
                    value["timestamp"]
                    for value in events
                    if value["action"] == "PURCHASE"
                ),
                None,
            )
            actions = {value["action"] for value in events}
            explicit_next = None
            explicit_detail = None
            surface = str(
                _mapped(
                    row,
                    interaction_columns,
                    "surface",
                    defaults.get("surface", "store_home"),
                )
            )
            if "EXIT" in actions and "CLICK" not in actions:
                explicit_next = "EXIT"
            elif "CLICK" in actions and "PURCHASE" not in actions:
                explicit_detail = "EXIT" if "EXIT" in actions else "BACK"
            elif "BACK" in actions:
                explicit_detail = "BACK"
            impressions.append(
                CanonicalImpression(
                    impression_id=impression_id,
                    user_id=user_id,
                    item_id=item_id,
                    timestamp=first["timestamp"],
                    session_id=str(
                        _mapped(
                            row,
                            interaction_columns,
                            "session_id",
                            f"session-{user_id}-{impression_id}",
                        )
                    ),
                    surface=surface,
                    clicked=clicked_at is not None,
                    purchased=purchased_at is not None,
                    click_timestamp=clicked_at,
                    purchase_timestamp=purchased_at,
                    session_fatigue=_float(
                        _mapped(
                            row,
                            interaction_columns,
                            "session_fatigue",
                            defaults.get("session_fatigue", 0.0),
                        )
                    ),
                    budget_reference=_optional_float(
                        _mapped(
                            row,
                            interaction_columns,
                            "budget_reference",
                            defaults.get("budget_reference"),
                        )
                    ),
                    context_features={
                        key: float(value)
                        for key, value in _mapped_object(
                            row, interaction_columns, "context_features"
                        ).items()
                    },
                    game_state=_mapped_game_state(
                        row,
                        interaction_columns,
                    ),
                    observed_next_action=explicit_next,
                    observed_detail_action=explicit_detail,
                    oracle_probability=_optional_float(
                        _mapped(row, interaction_columns, "oracle_probability")
                    ),
                    attributes=_mapped_object(
                        row, interaction_columns, "attributes"
                    ),
                )
            )
        return impressions


def _canonical_user(row: Mapping[str, Any]) -> CanonicalUser:
    return CanonicalUser(
        user_id=str(row["user_id"]),
        persona_summary=str(row.get("persona_summary", "")),
        pickiness=_float(row.get("pickiness"), 0.5),
        price_sensitivity=_float(row.get("price_sensitivity"), 0.5),
        category_preferences=_float_mapping(
            row.get("category_preferences", {})
        ),
        engagement=_float(row.get("engagement"), 0.5),
        variety=_float(row.get("variety", row.get("novelty_affinity")), 0.5),
        budget_reference=_optional_float(
            row.get("budget_reference", row.get("spending_power"))
        ),
        attributes=dict(row.get("attributes", {})),
    )


def _canonical_item(row: Mapping[str, Any]) -> CanonicalItem:
    excluded = {
        "item_id",
        "product_id",
        "product_type",
        "categories",
        "price",
        "discount_rate",
        "components",
        "attributes",
        "latent_vector",
        "random_effect",
    }
    attributes = dict(row.get("attributes", {}))
    attributes.update(
        {
            str(key): value
            for key, value in row.items()
            if key not in excluded
        }
    )
    return CanonicalItem(
        item_id=str(row.get("item_id", row.get("product_id"))),
        product_type=str(row.get("product_type", "item")).lower(),
        categories=_string_list(row.get("categories", ())),
        price=_float(row.get("price"), 0.0),
        discount_rate=_float(row.get("discount_rate"), 0.0),
        components=_product_components(row.get("components", ())),
        attributes=attributes,
    )


def _canonical_impression(row: Mapping[str, Any]) -> CanonicalImpression:
    return CanonicalImpression(
        impression_id=str(row["impression_id"]),
        user_id=str(row["user_id"]),
        item_id=str(row["item_id"]),
        timestamp=_timestamp(row.get("timestamp")),
        session_id=str(
            row.get(
                "session_id",
                f"session-{row['user_id']}-{row['impression_id']}",
            )
        ),
        surface=str(row.get("surface", "store_home")),
        clicked=_bool(row.get("clicked", False)),
        purchased=_bool(row.get("purchased", False)),
        click_timestamp=(
            _timestamp(row["click_timestamp"])
            if row.get("click_timestamp")
            else None
        ),
        purchase_timestamp=(
            _timestamp(row["purchase_timestamp"])
            if row.get("purchase_timestamp")
            else None
        ),
        session_fatigue=_float(row.get("session_fatigue"), 0.0),
        budget_reference=_optional_float(row.get("budget_reference")),
        context_features={
            str(key): float(value)
            for key, value in row.get("context_features", {}).items()
        },
        game_state=dict(row.get("game_state", {})),
        observed_initial_state=(
            str(row["observed_initial_state"])
            if row.get("observed_initial_state")
            else None
        ),
        observed_next_action=(
            str(row["observed_next_action"])
            if row.get("observed_next_action")
            else None
        ),
        observed_detail_action=(
            str(row["observed_detail_action"])
            if row.get("observed_detail_action")
            else None
        ),
        observed_action_path=tuple(
            str(value)
            for value in row.get("observed_action_path", ())
        ),
        oracle_probability=_optional_float(
            row.get(
                "oracle_probability",
                row.get("ground_truth_purchase_probability"),
            )
        ),
        attributes=dict(row.get("attributes", {})),
    )


def _impression_from_mapped_row(
    row: Mapping[str, Any],
    columns: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> CanonicalImpression:
    impression_id = str(_mapped(row, columns, "impression_id", "")).strip()
    user_id = str(_mapped(row, columns, "user_id", "")).strip()
    item_id = str(_mapped(row, columns, "item_id", "")).strip()
    if not impression_id or not user_id or not item_id:
        raise DatasetValidationError(
            "impression rows require impression_id, user_id, and item_id"
        )
    timestamp = _timestamp(_mapped(row, columns, "timestamp"))
    return CanonicalImpression(
        impression_id=impression_id,
        user_id=user_id,
        item_id=item_id,
        timestamp=timestamp,
        session_id=str(
            _mapped(
                row,
                columns,
                "session_id",
                f"session-{user_id}-{impression_id}",
            )
        ),
        surface=str(
            _mapped(
                row, columns, "surface", defaults.get("surface", "store_home")
            )
        ),
        clicked=_bool(_mapped(row, columns, "clicked", False)),
        purchased=_bool(_mapped(row, columns, "purchased", False)),
        click_timestamp=(
            _timestamp(_mapped(row, columns, "click_timestamp"))
            if _mapped(row, columns, "click_timestamp") not in (None, "")
            else None
        ),
        purchase_timestamp=(
            _timestamp(_mapped(row, columns, "purchase_timestamp"))
            if _mapped(row, columns, "purchase_timestamp") not in (None, "")
            else None
        ),
        session_fatigue=_float(
            _mapped(
                row,
                columns,
                "session_fatigue",
                defaults.get("session_fatigue", 0.0),
            )
        ),
        budget_reference=_optional_float(
            _mapped(
                row,
                columns,
                "budget_reference",
                defaults.get("budget_reference"),
            )
        ),
        context_features={
            key: float(value)
            for key, value in _mapped_object(
                row, columns, "context_features"
            ).items()
        },
        game_state=_mapped_game_state(
            row,
            columns,
        ),
        observed_initial_state=(
            str(_mapped(row, columns, "observed_initial_state"))
            if _mapped(row, columns, "observed_initial_state") not in (None, "")
            else None
        ),
        observed_next_action=(
            str(_mapped(row, columns, "observed_next_action"))
            if _mapped(row, columns, "observed_next_action") not in (None, "")
            else None
        ),
        observed_detail_action=(
            str(_mapped(row, columns, "observed_detail_action"))
            if _mapped(row, columns, "observed_detail_action") not in (None, "")
            else None
        ),
        observed_action_path=tuple(
            str(value)
            for value in (
                _json_or_value(
                    _mapped(row, columns, "observed_action_path", ())
                )
                or ()
            )
        ),
        oracle_probability=_optional_float(
            _mapped(row, columns, "oracle_probability")
        ),
        attributes=_mapped_object(row, columns, "attributes"),
    )


@dataclass(frozen=True)
class ProtocolBuildConfig:
    selected_users: int | None = 20
    cases_per_user: int | None = 25
    history_fraction: float = 0.5
    history_limit: int = 16
    seed: int = 20260819
    excluded_user_ids: frozenset[str] = frozenset()
    allow_incomplete_exposure: bool = False
    require_game_state: bool = False


class EvaluationProtocolBuilder:
    def __init__(
        self,
        dataset: CanonicalDataset,
        config: ProtocolBuildConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = config or ProtocolBuildConfig()

    def build(self) -> HoldoutProtocol:
        validation = self.dataset.validate()
        config = self.config
        if not 0.0 < config.history_fraction < 1.0:
            raise DatasetValidationError(
                "history_fraction must be between zero and one"
            )
        if (
            not self.dataset.complete_exposure
            and not config.allow_incomplete_exposure
        ):
            raise DatasetValidationError(
                "dataset has no complete impression exposure log; pass "
                "allow_incomplete_exposure only for selection-biased diagnostics"
            )
        if config.require_game_state:
            state_validation = validation["game_state"]
            incomplete = [
                field
                for field, coverage in state_validation[
                    "coverage"
                ].items()
                if coverage < 1.0
            ]
            constant = [
                field
                for field, count in state_validation[
                    "unique_values"
                ].items()
                if field not in {"current_goals", "owned_item_ids"}
                and count < 2
            ]
            if incomplete or constant:
                raise DatasetValidationError(
                    "state-rich evaluation requires complete, varying "
                    f"GameStateSnapshot fields; incomplete={incomplete}, "
                    f"constant={constant}"
                )

        grouped: dict[str, list[CanonicalImpression]] = defaultdict(list)
        for impression in self.dataset.impressions:
            if impression.user_id not in config.excluded_user_ids:
                grouped[impression.user_id].append(impression)
        for rows in grouped.values():
            rows.sort(key=lambda row: (row.timestamp, row.impression_id))

        eligible = []
        for user_id, rows in grouped.items():
            cutoff = max(1, int(len(rows) * config.history_fraction))
            holdout_count = len(rows) - cutoff
            required = config.cases_per_user or 1
            if holdout_count >= required:
                eligible.append(user_id)
        selected_count = config.selected_users or len(eligible)
        if len(eligible) < selected_count:
            raise DatasetValidationError(
                f"only {len(eligible)} users have enough temporal holdout cases; "
                f"requested {selected_count}"
            )

        # Evaluation sampling must be reproducible; this is not security randomness.
        rng = random.Random(config.seed)  # nosec B311
        chosen = (
            sorted(eligible)
            if selected_count == len(eligible)
            else sorted(rng.sample(eligible, selected_count))
        )
        run_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "dataset_name": self.dataset.dataset_name,
                    "seed": config.seed,
                    "history_fraction": config.history_fraction,
                    "history_limit": config.history_limit,
                    "selected_users": config.selected_users,
                    "cases_per_user": config.cases_per_user,
                    "impression_ids": sorted(
                        impression.impression_id
                        for impression in self.dataset.impressions
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:12]
        run_id = (
            f"adapter-{self.dataset.dataset_name}-"
            f"{config.seed}-{run_fingerprint}"
        )
        cases: list[HoldoutCase] = []
        bootstrap_payloads: list[Mapping[str, Any]] = []
        for user_id in chosen:
            rows = grouped[user_id]
            cutoff = max(1, int(len(rows) * config.history_fraction))
            history = self._history_window(
                rows[:cutoff],
                config.history_limit,
            )
            holdout = rows[cutoff:]
            if config.cases_per_user is not None:
                local_rng = random.Random(  # nosec B311
                    f"{config.seed}:{self.dataset.dataset_name}:{user_id}"
                )
                holdout = local_rng.sample(holdout, config.cases_per_user)
                holdout.sort(key=lambda row: (row.timestamp, row.impression_id))

            actor_id = f"eval-{run_id}-{user_id}"
            memory_session_id = f"history-{run_id}-{user_id}"
            history_events = self._history_events(history)
            profile = self._profile_payload(
                self.dataset.users[user_id], history, actor_id
            )
            bootstrap_payloads.append(
                {
                    "operation": "initialize_memory",
                    "observation": {
                        "user_id": actor_id,
                        "session_id": memory_session_id,
                        "source": "historical_import",
                        "page_id": f"dataset-adapter:{self.dataset.dataset_name}",
                        "recommended_item_ids": [
                            impression.item_id for impression in history
                        ],
                        "events": [
                            {
                                "event_type": event.event_type,
                                "timestamp": event.timestamp.isoformat(),
                                "item_id": event.item_id,
                                "categories": list(event.categories),
                            }
                            for event in history_events
                        ],
                        "transitions": self._transition_payloads(
                            history,
                            self.dataset.users[user_id],
                        ),
                    },
                }
            )
            for impression in holdout:
                item = self.dataset.items[impression.item_id]
                initial_state, next_action, detail_action = (
                    impression.action_labels()
                )
                path_initial_state, observed_action_path = (
                    impression.action_path()
                )
                preceding = [
                    value
                    for value in rows
                    if (
                        value.timestamp,
                        value.impression_id,
                    )
                    < (
                        impression.timestamp,
                        impression.impression_id,
                    )
                ]
                recent_history = self._history_window(
                    preceding,
                    config.history_limit,
                )
                recent_events = self._history_events(recent_history)
                recent_transitions = self._transition_payloads(
                    recent_history,
                    self.dataset.users[user_id],
                )
                recent_purchase_count = sum(
                    value.purchased for value in recent_history
                )
                cases.append(
                    HoldoutCase(
                        case_id=f"{user_id}:{impression.impression_id}",
                        original_user_id=user_id,
                        actor_id=actor_id,
                        session_id=memory_session_id,
                        item_id=impression.item_id,
                        label=int(impression.purchased),
                        oracle_probability=impression.oracle_probability,
                        observed_initial_state=path_initial_state,
                        observed_next_action=next_action,
                        observed_detail_action=detail_action,
                        observed_action_path=observed_action_path,
                        ratio_membership=("natural",),
                        request={
                            "user": profile,
                            "item": item_payload(item.to_evaluation_dict()),
                            "context": {
                                "surface": impression.surface,
                                "session_fatigue": impression.session_fatigue,
                                "budget_reference": (
                                    impression.budget_reference
                                    if impression.budget_reference is not None
                                    else self.dataset.users[
                                        user_id
                                    ].budget_reference
                                ),
                                "timestamp": impression.timestamp.isoformat(),
                                "features": dict(
                                    impression.context_features
                                )
                                | {
                                    "recent_history_transition_count": float(
                                        len(recent_transitions)
                                    ),
                                    "recent_history_purchase_count": float(
                                        recent_purchase_count
                                    ),
                                },
                            },
                            "game_state": dict(
                                impression.game_state
                            ),
                            "interactions": [
                                {
                                    "event_type": event.event_type,
                                    "timestamp": event.timestamp.isoformat(),
                                    "item_id": event.item_id,
                                    "categories": list(event.categories),
                                }
                                for event in recent_events
                            ],
                            "kg_evidence": graph_evidence(
                                target_item=item.to_evaluation_dict(),
                                events=recent_events,
                                items={
                                    key: value.to_evaluation_dict()
                                    for key, value in self.dataset.items.items()
                                },
                                now=impression.timestamp,
                            ),
                            "memory_session_id": memory_session_id,
                            "request_id": (
                                f"adapter-{run_id}-{impression.impression_id}"
                            ),
                        },
                    )
                )

        cases = self._progressive_case_order(cases)

        return HoldoutProtocol(
            run_id=run_id,
            users=len(chosen),
            history_fraction=config.history_fraction,
            history_transition_limit=config.history_limit,
            cases=tuple(cases),
            bootstrap_payloads=tuple(bootstrap_payloads),
            natural_metrics={
                "protocol": {
                    "adapter_schema": "simuser.canonical-dataset.v2",
                    "dataset_name": self.dataset.dataset_name,
                    "sampling": "uniform-within-user temporal holdout",
                    "history_unit": "observed_transitions",
                    "history_transition_limit": config.history_limit,
                    "cases_per_user": config.cases_per_user,
                    "complete_exposure": self.dataset.complete_exposure,
                    "purchase_rate": round(
                        sum(case.label for case in cases)
                        / max(1, len(cases)),
                        8,
                    ),
                    "oracle_available": all(
                        case.oracle_probability is not None for case in cases
                    ),
                    "recent_history": {
                        "cases_with_prior_purchase": sum(
                            any(
                                str(event.get("event_type", "")).lower()
                                == "purchase"
                                for event in case.request.get(
                                    "interactions",
                                    (),
                                )
                            )
                            for case in cases
                        ),
                        "coverage_rate": round(
                            sum(
                                any(
                                    str(
                                        event.get("event_type", "")
                                    ).lower()
                                    == "purchase"
                                    for event in case.request.get(
                                        "interactions",
                                        (),
                                    )
                                )
                                for case in cases
                            )
                            / max(1, len(cases)),
                            8,
                        ),
                    },
                    "validation": validation,
                }
            },
        )

    @staticmethod
    def _progressive_case_order(
        cases: Sequence[HoldoutCase],
    ) -> list[HoldoutCase]:
        def round_robin(
            values: Sequence[HoldoutCase],
        ) -> list[HoldoutCase]:
            grouped: dict[str, list[HoldoutCase]] = defaultdict(list)
            for value in values:
                grouped[value.original_user_id].append(value)
            return [
                grouped[user_id][index]
                for index in range(
                    max(
                        (len(items) for items in grouped.values()),
                        default=0,
                    )
                )
                for user_id in sorted(grouped)
                if index < len(grouped[user_id])
            ]

        def diverse(values: Sequence[HoldoutCase]) -> list[HoldoutCase]:
            # Keep progressive prefixes distributed across users. Diversity is
            # measured as a protocol gate rather than reordering one user's
            # cases ahead of another.
            return round_robin(values)

        positives = diverse([case for case in cases if case.label == 1])
        negatives = diverse([case for case in cases if case.label == 0])
        total = len(cases)
        positive_total = len(positives)
        positive_index = 0
        negative_index = 0
        ordered: list[HoldoutCase] = []
        for index in range(total):
            target_positive_count = (
                (index + 1) * positive_total // max(1, total)
            )
            if (
                target_positive_count > positive_index
                and positive_index < len(positives)
            ):
                ordered.append(positives[positive_index])
                positive_index += 1
            elif negative_index < len(negatives):
                ordered.append(negatives[negative_index])
                negative_index += 1
            else:
                ordered.append(positives[positive_index])
                positive_index += 1
        return ordered

    @staticmethod
    def _history_window(
        impressions: Sequence[CanonicalImpression],
        transition_limit: int,
    ) -> list[CanonicalImpression]:
        if transition_limit < 1:
            raise DatasetValidationError(
                "history_limit must be a positive transition count"
            )
        selected: list[CanonicalImpression] = []
        transition_count = 0
        for impression in reversed(impressions):
            path_length = len(impression.action_path()[1])
            if selected and transition_count + path_length > transition_limit:
                break
            selected.append(impression)
            transition_count += path_length
            if transition_count >= transition_limit:
                break
        selected.reverse()
        return selected

    def _history_events(
        self, impressions: Sequence[CanonicalImpression]
    ) -> tuple[BehaviorEvent, ...]:
        events = []
        for impression in impressions:
            categories = self.dataset.items[impression.item_id].categories
            events.append(
                BehaviorEvent(
                    event_type="view",
                    timestamp=impression.timestamp,
                    item_id=impression.item_id,
                    categories=categories,
                )
            )
            _, actions = impression.action_path()
            for offset, action in enumerate(actions, start=1):
                event_type = {
                    "CLICK": "click",
                    "PAYMENT_SUCCESS": "purchase",
                    "EXIT": "exit",
                    "SKIP": "dismiss",
                }.get(action, action.lower())
                events.append(
                    BehaviorEvent(
                        event_type=event_type,
                        timestamp=impression.timestamp
                        + timedelta(seconds=offset * 2),
                        item_id=impression.item_id,
                        categories=categories,
                    )
                )
        return tuple(events)

    def _profile_payload(
        self,
        user: CanonicalUser,
        history: Sequence[CanonicalImpression],
        actor_id: str,
    ) -> dict[str, Any]:
        preferences = dict(user.category_preferences)
        if not preferences:
            weighted: Counter[str] = Counter()
            for impression in history:
                weight = 1.0 if impression.purchased else (
                    0.4 if impression.clicked else 0.1
                )
                for category in self.dataset.items[
                    impression.item_id
                ].categories:
                    weighted[category] += weight
            maximum = max(weighted.values(), default=0.0)
            if maximum > 0:
                preferences = {
                    category: round(value / maximum, 8)
                    for category, value in weighted.items()
                }
        persona = user.persona_summary
        if not persona:
            top = sorted(
                preferences.items(), key=lambda value: value[1], reverse=True
            )[:3]
            categories = ", ".join(category for category, _ in top)
            persona = (
                f"Observed-history profile with {len(history)} impressions"
                + (f"; strongest categories: {categories}" if categories else "")
            )
        return {
            "user_id": actor_id,
            "persona_summary": persona,
            "pickiness": user.pickiness,
            "price_sensitivity": user.price_sensitivity,
            "category_preferences": preferences,
            "engagement": user.engagement,
            "variety": user.variety,
        }

    def _transition_payloads(
        self,
        impressions: Sequence[CanonicalImpression],
        user: CanonicalUser,
    ) -> list[dict[str, Any]]:
        transitions: list[dict[str, Any]] = []
        for impression in impressions:
            item = self.dataset.items[impression.item_id]
            budget = (
                impression.budget_reference
                if impression.budget_reference is not None
                else user.budget_reference
            )
            ratio = item.price / budget if budget and budget > 0.0 else None
            initial_state, actions = impression.action_path()

            def append(
                state: str,
                action: str,
                next_state: str,
                outcome: str,
            ) -> None:
                transitions.append(
                    {
                        "state": state,
                        "action": action,
                        "next_state": next_state,
                        "timestamp": impression.timestamp.isoformat(),
                        "item_id": impression.item_id,
                        "categories": list(item.categories),
                        "surface": impression.surface,
                        "price_budget_ratio": ratio,
                        "session_fatigue": impression.session_fatigue,
                        "outcome": outcome,
                    }
                )

            state = initial_state
            for action in actions:
                transition = DEFAULT_ACTION_GRAPH.transition(state, action)
                next_state = transition.next_state
                append(
                    state,
                    action,
                    next_state,
                    (
                        "purchase"
                        if DEFAULT_ACTION_GRAPH.purchased(next_state)
                        else "no_purchase"
                        if DEFAULT_ACTION_GRAPH.is_terminal(next_state)
                        else "continued"
                    ),
                )
                state = next_state
        return transitions


# Backward-compatible alias for code written while the adapter was introduced.
DatasetProtocolBuilder = EvaluationProtocolBuilder


@dataclass(frozen=True)
class ProductionExportConfig:
    as_of: datetime
    identity_salt: str
    since: datetime | None = None
    history_limit_per_user: int | None = None
    allow_synthetic: bool = False


class ProductionExportBuilder:
    """Create label-free Memory and Neptune import artifacts.

    This builder never emits evaluation answer keys or oracle probabilities.
    It only exports observations at or before the configured as-of timestamp.
    """

    def __init__(
        self,
        dataset: CanonicalDataset,
        config: ProductionExportConfig,
    ) -> None:
        self.dataset = dataset
        self.config = config

    def write(self, output_dir: Path) -> dict[str, Any]:
        validation = self.dataset.validate()
        if not self.config.identity_salt:
            raise DatasetValidationError(
                "production export requires a non-empty identity salt"
            )
        if self.config.as_of.tzinfo is None:
            raise DatasetValidationError(
                "production as_of timestamp must be timezone-aware"
            )
        if self.config.since is not None:
            if self.config.since.tzinfo is None:
                raise DatasetValidationError(
                    "production since timestamp must be timezone-aware"
                )
            if self.config.since >= self.config.as_of:
                raise DatasetValidationError(
                    "production since timestamp must be earlier than as_of"
                )
        if (
            self.config.history_limit_per_user is not None
            and self.config.history_limit_per_user <= 0
        ):
            raise DatasetValidationError(
                "history_limit_per_user must be positive"
            )
        if (
            self.dataset.source_metadata.get("synthetic")
            and not self.config.allow_synthetic
        ):
            raise DatasetValidationError(
                "refusing to create a production bundle from synthetic data"
            )

        as_of = self.config.as_of.astimezone(timezone.utc)
        since = (
            self.config.since.astimezone(timezone.utc)
            if self.config.since is not None
            else None
        )
        grouped: dict[str, list[CanonicalImpression]] = defaultdict(list)
        for impression in self.dataset.impressions:
            if impression.timestamp <= as_of and (
                since is None or impression.timestamp > since
            ):
                grouped[impression.user_id].append(impression)
        for rows in grouped.values():
            rows.sort(key=lambda value: (value.timestamp, value.impression_id))
            if self.config.history_limit_per_user is not None:
                del rows[: -self.config.history_limit_per_user]
        exported_impressions = sum(len(rows) for rows in grouped.values())
        if exported_impressions == 0:
            raise DatasetValidationError(
                "production export contains no observations in the selected "
                "time range"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        memory_rows = list(self._memory_import_rows(grouped))
        _write_jsonl(output_dir / "memory_import.jsonl", memory_rows)
        _write_jsonl(
            output_dir / "catalog_items.jsonl",
            (item.to_dict() for item in self.dataset.items.values()),
        )
        neptune_dir = output_dir / "neptune"
        neptune_dir.mkdir(parents=True, exist_ok=True)
        nodes = self._neptune_nodes(grouped)
        edges = self._neptune_edges(grouped)
        self._write_neptune_nodes(neptune_dir / "nodes.csv", nodes)
        self._write_neptune_edges(neptune_dir / "edges.csv", edges)

        manifest = {
            "schema": "simuser.production-import.v1",
            "dataset_name": self.dataset.dataset_name,
            "identity_namespace_id": hmac.new(
                self.config.identity_salt.encode("utf-8"),
                b"simplayer:identity-namespace:v1",
                hashlib.sha256,
            ).hexdigest()[:24],
            "export_mode": "incremental" if since is not None else "snapshot",
            "since": since.isoformat() if since is not None else None,
            "as_of": as_of.isoformat(),
            "neptune_update_semantics": "append_only_by_stable_id",
            "identity_policy": "hmac-sha256",
            "contains_answer_key": False,
            "contains_oracle_probability": False,
            "contains_model_predictions": False,
            "source_validation": validation,
            "exported_users": len(grouped),
            "exported_impressions": exported_impressions,
            "exported_purchases": sum(
                impression.purchased
                for impressions in grouped.values()
                for impression in impressions
            ),
            "memory_batches": len(memory_rows),
            "neptune_nodes": len(nodes),
            "neptune_edges": len(edges),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    def _pseudonym(self, kind: str, value: str) -> str:
        digest = hmac.new(
            self.config.identity_salt.encode("utf-8"),
            f"{kind}:{value}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{kind}-{digest[:32]}"

    def _memory_import_rows(
        self,
        grouped: Mapping[str, Sequence[CanonicalImpression]],
    ) -> Iterable[Mapping[str, Any]]:
        transition_builder = EvaluationProtocolBuilder(self.dataset)
        for user_id, impressions in grouped.items():
            actor_id = self._pseudonym("user", user_id)
            sessions: dict[str, list[CanonicalImpression]] = defaultdict(list)
            for impression in impressions:
                sessions[impression.session_id].append(impression)
            for source_session_id, session_impressions in sorted(
                sessions.items()
            ):
                events = transition_builder._history_events(
                    session_impressions
                )
                transition_payloads = transition_builder._transition_payloads(
                    session_impressions,
                    self.dataset.users[user_id],
                )
                yield {
                    "operation": "record_observations",
                    "observation": {
                        "user_id": actor_id,
                        "session_id": self._pseudonym(
                            "session", source_session_id
                        ),
                        "source": "historical_import",
                        "page_id": (
                            f"dataset-adapter:{self.dataset.dataset_name}"
                        ),
                        "recommended_item_ids": [
                            impression.item_id
                            for impression in session_impressions
                        ],
                        "events": [
                            {
                                "event_type": event.event_type,
                                "timestamp": event.timestamp.isoformat(),
                                "item_id": event.item_id,
                                "categories": list(event.categories),
                            }
                            for event in events
                        ],
                        "transitions": transition_payloads,
                    },
                }

    def _neptune_nodes(
        self,
        grouped: Mapping[str, Sequence[CanonicalImpression]],
    ) -> list[dict[str, Any]]:
        nodes: dict[str, dict[str, Any]] = {}

        def add(graph_id: str, label: str, **values: Any) -> None:
            nodes[graph_id] = {
                ":ID": graph_id,
                "nodeId:String": values.get("node_id", ""),
                "userId:String": values.get("user_id", ""),
                "itemId:String": values.get("item_id", ""),
                "productType:String": values.get("product_type", ""),
                "price:Double": values.get("price", ""),
                "quality:Double": values.get("quality", ""),
                "utility:Double": values.get("utility", ""),
                "emotionality:Double": values.get("emotionality", ""),
                "priceSensitivity:Double": values.get(
                    "price_sensitivity", ""
                ),
                "spendingPower:Double": values.get("spending_power", ""),
                "latentVector:String": "",
                ":LABEL": label,
            }

        for category in sorted(
            {
                category
                for item in self.dataset.items.values()
                for category in item.categories
            }
        ):
            add(f"category:{category}", "Category", node_id=category)
        for item in self.dataset.items.values():
            add(
                f"item:{item.item_id}",
                "Item",
                node_id=item.item_id,
                item_id=item.item_id,
                product_type=item.product_type,
                price=item.price,
                quality=item.attributes.get("quality", ""),
                utility=item.attributes.get("utility", ""),
                emotionality=item.attributes.get("emotionality", ""),
            )
            character = item.attributes.get("character")
            event_id = item.attributes.get("event_id")
            if character:
                add(
                    f"character:{character}",
                    "Character",
                    node_id=str(character),
                )
            if event_id:
                add(f"event:{event_id}", "Event", node_id=str(event_id))
        for user_id in grouped:
            user = self.dataset.users[user_id]
            actor_id = self._pseudonym("user", user_id)
            add(
                f"user:{actor_id}",
                "User",
                user_id=actor_id,
                price_sensitivity=user.price_sensitivity,
                spending_power=(
                    user.budget_reference
                    if user.budget_reference is not None
                    else ""
                ),
            )
        return list(nodes.values())

    def _neptune_edges(
        self,
        grouped: Mapping[str, Sequence[CanonicalImpression]],
    ) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []

        def add(
            start: str,
            end: str,
            relation: str,
            *,
            timestamp: str = "",
            session_id: str = "",
            weight: float = 1.0,
            state: str = "",
            next_state: str = "",
        ) -> None:
            raw = f"{start}|{end}|{relation}|{timestamp}|{session_id}"
            edge_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            edges.append(
                {
                    ":ID": f"edge-{edge_id}",
                    ":START_ID": start,
                    ":END_ID": end,
                    ":TYPE": relation,
                    "timestamp:DateTime": timestamp,
                    "weight:Double": weight,
                    "sessionId:String": session_id,
                    "state:String": state,
                    "nextState:String": next_state,
                    "synthetic:Bool": "false",
                }
            )

        for item in self.dataset.items.values():
            item_node = f"item:{item.item_id}"
            for category in item.categories:
                add(item_node, f"category:{category}", "IN_CATEGORY")
            character = item.attributes.get("character")
            event_id = item.attributes.get("event_id")
            if character:
                add(item_node, f"character:{character}", "TARGETS")
            if event_id:
                add(item_node, f"event:{event_id}", "AVAILABLE_IN")
            for component in item.components:
                add(
                    item_node,
                    f"item:{component.item_id}",
                    "CONTAINS",
                    weight=float(component.quantity),
                )

        for user_id, impressions in grouped.items():
            actor_id = self._pseudonym("user", user_id)
            user_node = f"user:{actor_id}"
            for impression in impressions:
                session_id = self._pseudonym(
                    "session", impression.session_id
                )
                item_node = f"item:{impression.item_id}"
                add(
                    user_node,
                    item_node,
                    "VIEWED",
                    timestamp=impression.timestamp.isoformat(),
                    session_id=session_id,
                    weight=0.2,
                )
                state, actions = impression.action_path()
                for offset, action in enumerate(actions, start=1):
                    transition = DEFAULT_ACTION_GRAPH.transition(state, action)
                    add(
                        user_node,
                        item_node,
                        action,
                        timestamp=(
                            impression.timestamp
                            + timedelta(seconds=offset * 2)
                        ).isoformat(),
                        session_id=session_id,
                        weight=self._action_edge_weight(action),
                        state=state,
                        next_state=transition.next_state,
                    )
                    state = transition.next_state
        return edges

    @staticmethod
    def _action_edge_weight(action: str) -> float:
        return {
            "CLICK": 0.20,
            "START_PURCHASE": 0.45,
            "CONFIRM_PURCHASE": 0.70,
            "PAYMENT_SUCCESS": 1.00,
            "INSUFFICIENT_CURRENCY": 0.35,
            "OPEN_TOP_UP": 0.55,
            "TOP_UP_SUCCESS": 0.80,
            "PAYMENT_FAILED": -0.45,
            "CANCEL": -0.35,
            "CANCEL_TOP_UP": -0.45,
            "BACK": -0.15,
            "BACK_TO_ITEM": -0.10,
            "SKIP": -0.25,
            "EXIT": -0.30,
            "PURCHASE_NOW": 0.55,
        }.get(action, 0.0)

    @staticmethod
    def _write_neptune_nodes(
        path: Path, rows: Sequence[Mapping[str, Any]]
    ) -> None:
        fieldnames = (
            ":ID",
            "nodeId:String",
            "userId:String",
            "itemId:String",
            "productType:String",
            "price:Double",
            "quality:Double",
            "utility:Double",
            "emotionality:Double",
            "priceSensitivity:Double",
            "spendingPower:Double",
            "latentVector:String",
            ":LABEL",
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_neptune_edges(
        path: Path, rows: Sequence[Mapping[str, Any]]
    ) -> None:
        fieldnames = (
            ":ID",
            ":START_ID",
            ":END_ID",
            ":TYPE",
            "timestamp:DateTime",
            "weight:Double",
            "sessionId:String",
            "state:String",
            "nextState:String",
            "synthetic:Bool",
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
