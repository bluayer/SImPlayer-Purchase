from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(value: str | None) -> datetime:
    if not value:
        return utc_now()
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    persona_summary: str = ""
    pickiness: float = 0.5
    price_sensitivity: float = 0.5
    category_preferences: Mapping[str, float] = field(default_factory=dict)
    engagement: float = 0.5
    variety: float = 0.5

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UserProfile":
        return cls(
            user_id=str(payload["user_id"]),
            persona_summary=str(payload.get("persona_summary", "")),
            pickiness=float(payload.get("pickiness", 0.5)),
            price_sensitivity=float(payload.get("price_sensitivity", 0.5)),
            category_preferences={
                str(key): float(value)
                for key, value in payload.get("category_preferences", {}).items()
            },
            engagement=float(payload.get("engagement", 0.5)),
            variety=float(payload.get("variety", 0.5)),
        )


@dataclass(frozen=True)
class ProductComponent:
    item_id: str
    quantity: int = 1

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("product component item_id cannot be empty")
        if self.quantity < 1:
            raise ValueError("product component quantity must be positive")

    @classmethod
    def from_value(cls, value: Any) -> "ProductComponent":
        if isinstance(value, str):
            return cls(item_id=value)
        if not isinstance(value, Mapping):
            raise ValueError("product component must be an item ID or object")
        item_id = value.get("item_id", value.get("product_id"))
        if not item_id:
            raise ValueError("product component requires item_id or product_id")
        quantity = int(value.get("quantity", 1))
        if quantity < 1:
            raise ValueError("product component quantity must be positive")
        return cls(item_id=str(item_id), quantity=quantity)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProductNeedProfile:
    rational: float = 0.5
    emotional: float = 0.5
    rational_aspects: tuple[str, ...] = ()
    emotional_aspects: tuple[str, ...] = ()
    source: str = "inferred"

    def __post_init__(self) -> None:
        if not 0.0 <= self.rational <= 1.0:
            raise ValueError("rational need weight must be between 0 and 1")
        if not 0.0 <= self.emotional <= 1.0:
            raise ValueError("emotional need weight must be between 0 and 1")
        if self.rational + self.emotional <= 0.0:
            raise ValueError("at least one need weight must be positive")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> "ProductNeedProfile | None":
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise ValueError("need_profile must be an object")
        return cls(
            rational=float(payload.get("rational", 0.5)),
            emotional=float(payload.get("emotional", 0.5)),
            rational_aspects=tuple(
                str(value) for value in payload.get("rational_aspects", ())
            ),
            emotional_aspects=tuple(
                str(value) for value in payload.get("emotional_aspects", ())
            ),
            source=str(payload.get("source", "catalog")),
        )

    def to_dict(self) -> dict[str, Any]:
        total = self.rational + self.emotional
        return {
            **asdict(self),
            "rational": self.rational / total,
            "emotional": self.emotional / total,
            "dominant_need": (
                "rational"
                if self.rational >= self.emotional + 0.15
                else "emotional"
                if self.emotional >= self.rational + 0.15
                else "mixed"
            ),
        }


@dataclass(frozen=True)
class Item:
    item_id: str
    product_type: str = "item"
    categories: tuple[str, ...] = ()
    price: float = 0.0
    discount_rate: float = 0.0
    components: tuple[ProductComponent, ...] = ()
    need_profile: ProductNeedProfile | None = None
    eligible: bool = True
    already_owned: bool = False
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.product_type not in {"item", "bundle"}:
            raise ValueError("product_type must be 'item' or 'bundle'")
        if self.price < 0.0:
            raise ValueError("product price cannot be negative")
        if not 0.0 <= self.discount_rate <= 1.0:
            raise ValueError("discount_rate must be between 0 and 1")
        if any(component.item_id == self.item_id for component in self.components):
            raise ValueError("a product cannot contain itself")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Item":
        item_id = payload.get("item_id", payload.get("product_id"))
        if not item_id:
            raise ValueError("target product requires item_id or product_id")
        return cls(
            item_id=str(item_id),
            product_type=str(payload.get("product_type", "item")).lower(),
            categories=tuple(str(value) for value in payload.get("categories", ())),
            price=float(payload.get("price", 0.0)),
            discount_rate=float(payload.get("discount_rate", 0.0)),
            components=tuple(
                ProductComponent.from_value(value)
                for value in payload.get("components", ())
            ),
            need_profile=ProductNeedProfile.from_dict(
                payload.get("need_profile")
            ),
            eligible=bool(payload.get("eligible", True)),
            already_owned=bool(payload.get("already_owned", False)),
            attributes=dict(payload.get("attributes", {})),
        )


@dataclass(frozen=True)
class ProductScenario:
    price_override: float | None = None
    discount_rate_override: float | None = None
    add_categories: tuple[str, ...] = ()
    remove_categories: tuple[str, ...] = ()
    attribute_overrides: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ProductScenario":
        value = payload or {}
        if not isinstance(value, Mapping):
            raise ValueError("product_scenario must be an object")
        price = value.get("price_override")
        discount = value.get("discount_rate_override")
        scenario = cls(
            price_override=float(price) if price is not None else None,
            discount_rate_override=(
                float(discount) if discount is not None else None
            ),
            add_categories=tuple(
                str(category) for category in value.get("add_categories", ())
            ),
            remove_categories=tuple(
                str(category) for category in value.get("remove_categories", ())
            ),
            attribute_overrides=dict(value.get("attribute_overrides", {})),
        )
        if scenario.price_override is not None and scenario.price_override < 0.0:
            raise ValueError("price_override cannot be negative")
        if (
            scenario.discount_rate_override is not None
            and not 0.0 <= scenario.discount_rate_override <= 1.0
        ):
            raise ValueError("discount_rate_override must be between 0 and 1")
        return scenario

    def apply(self, product: Item) -> Item:
        removed = set(self.remove_categories)
        categories = tuple(
            dict.fromkeys(
                category
                for category in (*product.categories, *self.add_categories)
                if category not in removed
            )
        )
        attributes = {
            **dict(product.attributes),
            **dict(self.attribute_overrides),
        }
        if self.add_categories or self.remove_categories:
            attributes["scenario_base_categories"] = list(product.categories)
            attributes["scenario_categories_overridden"] = True
        if self.price_override is not None:
            attributes["scenario_base_price"] = product.price
            attributes["scenario_price_overridden"] = True
        if self.discount_rate_override is not None:
            attributes["scenario_base_discount_rate"] = product.discount_rate
            attributes["scenario_discount_overridden"] = True
        return replace(
            product,
            categories=categories,
            price=(
                self.price_override
                if self.price_override is not None
                else product.price
            ),
            discount_rate=(
                self.discount_rate_override
                if self.discount_rate_override is not None
                else product.discount_rate
            ),
            attributes=attributes,
        )


@dataclass(frozen=True)
class ExposureContext:
    surface: str = "store_home"
    session_fatigue: float = 0.0
    budget_reference: float | None = None
    timestamp: datetime = field(default_factory=utc_now)
    features: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExposureContext":
        budget = payload.get("budget_reference")
        return cls(
            surface=str(payload.get("surface", "store_home")),
            session_fatigue=float(payload.get("session_fatigue", 0.0)),
            budget_reference=float(budget) if budget is not None else None,
            timestamp=parse_timestamp(payload.get("timestamp")),
            features={
                str(key): float(value)
                for key, value in payload.get("features", {}).items()
            },
        )


@dataclass(frozen=True)
class GameStateSnapshot:
    currency_balance: float | None = None
    progression_need: float = 0.5
    recent_failure_intensity: float = 0.0
    inventory_overlap: float = 0.0
    event_urgency: float = 0.0
    purchase_cooldown: float = 0.0
    current_goals: tuple[str, ...] = ()
    owned_item_ids: tuple[str, ...] = ()
    features: Mapping[str, float] = field(default_factory=dict)
    provided_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.currency_balance is not None and self.currency_balance < 0.0:
            raise ValueError("currency_balance cannot be negative")
        for name in (
            "progression_need",
            "recent_failure_intensity",
            "inventory_overlap",
            "event_urgency",
            "purchase_cooldown",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        context_features: Mapping[str, float] | None = None,
    ) -> "GameStateSnapshot":
        value = payload or {}
        if not isinstance(value, Mapping):
            raise ValueError("game_state must be an object")
        fallbacks = dict(context_features or {})
        recognized = {
            "currency_balance",
            "progression_need",
            "recent_failure_intensity",
            "inventory_overlap",
            "event_urgency",
            "purchase_cooldown",
            "current_goals",
            "owned_item_ids",
        }
        provided_fields = tuple(
            sorted(
                recognized.intersection(value)
                | recognized.intersection(fallbacks)
            )
        )

        def signal(name: str, default: float) -> float:
            return float(value.get(name, fallbacks.get(name, default)))

        balance = value.get(
            "currency_balance",
            fallbacks.get("currency_balance"),
        )
        return cls(
            currency_balance=float(balance) if balance is not None else None,
            progression_need=signal("progression_need", 0.5),
            recent_failure_intensity=signal(
                "recent_failure_intensity",
                0.0,
            ),
            inventory_overlap=signal("inventory_overlap", 0.0),
            event_urgency=signal("event_urgency", 0.0),
            purchase_cooldown=signal("purchase_cooldown", 0.0),
            current_goals=tuple(
                str(item) for item in value.get("current_goals", ())
            ),
            owned_item_ids=tuple(
                str(item) for item in value.get("owned_item_ids", ())
            ),
            features={
                str(key): float(item)
                for key, item in value.get("features", {}).items()
            },
            provided_fields=provided_fields,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BehaviorEvent:
    event_type: str
    timestamp: datetime
    item_id: str | None = None
    categories: tuple[str, ...] = ()
    rating: float | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BehaviorEvent":
        rating = payload.get("rating")
        return cls(
            event_type=str(payload["event_type"]),
            timestamp=parse_timestamp(payload.get("timestamp")),
            item_id=str(payload["item_id"]) if payload.get("item_id") else None,
            categories=tuple(str(value) for value in payload.get("categories", ())),
            rating=float(rating) if rating is not None else None,
        )


@dataclass(frozen=True)
class ObservedStateTransition:
    state: str
    action: str
    next_state: str
    timestamp: datetime
    item_id: str | None = None
    categories: tuple[str, ...] = ()
    surface: str = ""
    price_budget_ratio: float | None = None
    session_fatigue: float | None = None
    outcome: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObservedStateTransition":
        price_budget_ratio = payload.get("price_budget_ratio")
        session_fatigue = payload.get("session_fatigue")
        return cls(
            state=str(payload["state"]).upper(),
            action=str(payload["action"]).upper(),
            next_state=str(payload["next_state"]).upper(),
            timestamp=parse_timestamp(payload.get("timestamp")),
            item_id=str(payload["item_id"]) if payload.get("item_id") else None,
            categories=tuple(str(value) for value in payload.get("categories", ())),
            surface=str(payload.get("surface", "")),
            price_budget_ratio=(
                float(price_budget_ratio)
                if price_budget_ratio is not None
                else None
            ),
            session_fatigue=(
                float(session_fatigue) if session_fatigue is not None else None
            ),
            outcome=str(payload.get("outcome", "")),
        )


ALLOWED_OBSERVATION_SOURCES = frozenset(
    {
        "external_observation",
        "historical_import",
        "experiment_observation",
    }
)


@dataclass(frozen=True)
class ObservationBatch:
    user_id: str
    session_id: str
    events: tuple[BehaviorEvent, ...]
    source: str = "external_observation"
    page_id: str | None = None
    recommended_item_ids: tuple[str, ...] = ()
    feeling: str = ""
    review: str = ""
    transitions: tuple[ObservedStateTransition, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObservationBatch":
        source = str(payload.get("source", "external_observation"))
        if source not in ALLOWED_OBSERVATION_SOURCES:
            allowed = ", ".join(sorted(ALLOWED_OBSERVATION_SOURCES))
            raise ValueError(
                f"observation source must be externally observed ({allowed}); got {source!r}"
            )
        events = tuple(
            BehaviorEvent.from_dict(value) for value in payload.get("events", ())
        )
        if not events:
            raise ValueError("observation batch requires at least one event")
        if any(event.event_type.lower() == "add_to_cart" for event in events):
            raise ValueError("game purchase observations do not support add_to_cart")
        transitions = tuple(
            ObservedStateTransition.from_dict(value)
            for value in payload.get("transitions", ())
        )
        if any(
            transition.state == "CART"
            or transition.next_state == "CART"
            or transition.action
            in {"ADD_TO_CART", "CHECKOUT", "REMOVE", "CONTINUE"}
            for transition in transitions
        ):
            raise ValueError("game purchase observations do not support cart transitions")
        return cls(
            user_id=str(payload["user_id"]),
            session_id=str(payload["session_id"]),
            events=events,
            source=source,
            page_id=str(payload["page_id"]) if payload.get("page_id") else None,
            recommended_item_ids=tuple(
                str(value) for value in payload.get("recommended_item_ids", ())
            ),
            feeling=str(payload.get("feeling", "")),
            review=str(payload.get("review", "")),
            transitions=transitions,
        )


@dataclass(frozen=True)
class MemoryDocument:
    content: str
    relevance: float = 0.5
    namespace: str = ""
    source_query: str = ""
    observed_at: datetime | None = None
    kind: str = "memory"


@dataclass(frozen=True)
class EpisodicMemoryEvidence:
    queries: tuple[str, ...] = ()
    documents: tuple[MemoryDocument, ...] = ()
    interactions: tuple[BehaviorEvent, ...] = ()
    transitions: tuple[ObservedStateTransition, ...] = ()

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(document.content for document in self.documents)


@dataclass(frozen=True)
class ObservationReceipt:
    user_id: str
    session_id: str
    event_count: int
    long_term_record_count: int
    source: str
    reflection: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KnowledgeGraphEvidence:
    item_shared_paths: float = 0.0
    item_source_self_paths: float = 0.0
    item_target_self_paths: float = 0.0
    user_shared_paths: float = 0.0
    user_self_paths: float = 0.0
    user_target_self_paths: float = 0.0
    precomputed_affinity: float | None = None
    precomputed_confidence: float = 0.0
    meta_path_scores: Mapping[str, float] = field(default_factory=dict)
    retrieved_evidence: tuple[str, ...] = ()
    retrieval_support: float = 0.0
    retrieval_coverage: float = 0.0
    retrieval_consistency: float = 0.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "KnowledgeGraphEvidence":
        affinity = payload.get("precomputed_affinity")
        return cls(
            item_shared_paths=float(payload.get("item_shared_paths", 0.0)),
            item_source_self_paths=float(payload.get("item_source_self_paths", 0.0)),
            item_target_self_paths=float(payload.get("item_target_self_paths", 0.0)),
            user_shared_paths=float(payload.get("user_shared_paths", 0.0)),
            user_self_paths=float(payload.get("user_self_paths", 0.0)),
            user_target_self_paths=float(payload.get("user_target_self_paths", 0.0)),
            precomputed_affinity=float(affinity) if affinity is not None else None,
            precomputed_confidence=float(payload.get("precomputed_confidence", 0.0)),
            meta_path_scores={
                str(key): float(value)
                for key, value in payload.get("meta_path_scores", {}).items()
            },
            retrieved_evidence=tuple(
                str(value) for value in payload.get("retrieved_evidence", ())
            ),
            retrieval_support=float(payload.get("retrieval_support", 0.0)),
            retrieval_coverage=float(payload.get("retrieval_coverage", 0.0)),
            retrieval_consistency=float(
                payload.get("retrieval_consistency", 0.0)
            ),
        )


@dataclass(frozen=True)
class AgentAssessment:
    likelihood: float = 0.5
    relative_preference_score: float | None = None
    rollout_probability: float | None = None
    action_distributions: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    decision_state: Mapping[str, float] = field(default_factory=dict)
    intention_distribution: Mapping[str, float] = field(default_factory=dict)
    counterfactual_checks: Mapping[str, bool] = field(default_factory=dict)
    validator_adjusted: bool = False
    commitment_adjusted: bool = False
    counterfactual_adjusted: bool = False
    transition_grounding_strength: float = 0.0
    confidence: float = 0.0
    rollout_confidence: float | None = None
    reasons: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "AgentAssessment":
        if not payload:
            return cls()
        relative_preference = payload.get("relative_preference_score")
        rollout_probability = payload.get("rollout_probability")
        return cls(
            likelihood=float(payload.get("likelihood", 0.5)),
            relative_preference_score=(
                float(relative_preference)
                if relative_preference is not None
                else None
            ),
            rollout_probability=(
                float(rollout_probability)
                if rollout_probability is not None
                else None
            ),
            action_distributions={
                str(state): {
                    str(action): float(probability)
                    for action, probability in distribution.items()
                }
                for state, distribution in payload.get(
                    "action_distributions", {}
                ).items()
            },
            decision_state={
                str(key): float(value)
                for key, value in payload.get("decision_state", {}).items()
            },
            intention_distribution={
                str(key): float(value)
                for key, value in payload.get(
                    "intention_distribution",
                    {},
                ).items()
            },
            counterfactual_checks={
                str(key): bool(value)
                for key, value in payload.get(
                    "counterfactual_checks",
                    {},
                ).items()
            },
            validator_adjusted=bool(payload.get("validator_adjusted", False)),
            commitment_adjusted=bool(
                payload.get("commitment_adjusted", False)
            ),
            counterfactual_adjusted=bool(
                payload.get("counterfactual_adjusted", False)
            ),
            transition_grounding_strength=float(
                payload.get("transition_grounding_strength", 0.0)
            ),
            confidence=float(payload.get("confidence", 0.0)),
            rollout_confidence=(
                float(payload["rollout_confidence"])
                if payload.get("rollout_confidence") is not None
                else None
            ),
            reasons=tuple(str(value) for value in payload.get("reasons", ())),
            contradictions=tuple(str(value) for value in payload.get("contradictions", ())),
        )


@dataclass(frozen=True)
class SimulationRequest:
    user: UserProfile
    item: Item
    context: ExposureContext
    game_state: GameStateSnapshot = field(default_factory=GameStateSnapshot)
    interactions: tuple[BehaviorEvent, ...] = ()
    kg_evidence: KnowledgeGraphEvidence = field(default_factory=KnowledgeGraphEvidence)
    base_model_probability: float | None = None
    agent_assessment: AgentAssessment | None = None
    request_id: str | None = None
    memory_session_id: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SimulationRequest":
        item_payload = payload.get("target_product", payload.get("item"))
        if not isinstance(item_payload, Mapping):
            raise ValueError("request requires item or target_product")
        item = ProductScenario.from_dict(payload.get("product_scenario")).apply(
            Item.from_dict(item_payload)
        )
        context_payload = payload.get(
            "exposure_scenario",
            payload.get("context", {}),
        )
        if not isinstance(context_payload, Mapping):
            raise ValueError("context or exposure_scenario must be an object")
        context = ExposureContext.from_dict(context_payload)
        game_state = GameStateSnapshot.from_dict(
            payload.get("game_state"),
            context_features=context.features,
        )
        if item.item_id in game_state.owned_item_ids and not item.already_owned:
            item = replace(item, already_owned=True)
        return cls(
            user=UserProfile.from_dict(payload["user"]),
            item=item,
            context=context,
            game_state=game_state,
            interactions=tuple(
                BehaviorEvent.from_dict(value) for value in payload.get("interactions", ())
            ),
            kg_evidence=KnowledgeGraphEvidence.from_dict(payload.get("kg_evidence", {})),
            base_model_probability=(
                float(payload["base_model_probability"])
                if payload.get("base_model_probability") is not None
                else None
            ),
            agent_assessment=(
                AgentAssessment.from_dict(payload.get("agent_assessment"))
                if payload.get("agent_assessment") is not None
                else None
            ),
            request_id=str(payload["request_id"]) if payload.get("request_id") else None,
            memory_session_id=(
                str(payload["memory_session_id"])
                if payload.get("memory_session_id")
                else None
            ),
        )


@dataclass(frozen=True)
class SimulationResult:
    # Backward-compatible alias for scalar_purchase_probability.
    probability: float
    confidence: float
    eligible: bool
    is_calibrated: bool
    components: Mapping[str, float]
    reasons: Sequence[str]
    contradictions: Sequence[str]
    model_version: str
    calibration_version: str | None
    scalar_purchase_probability: float | None = None
    trajectory_purchase_probability: float | None = None
    action_distributions: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    likely_trajectories: Sequence[Mapping[str, Any]] = ()
    decision_state: Mapping[str, float] = field(default_factory=dict)
    intention_distribution: Mapping[str, float] = field(default_factory=dict)
    action_graph_id: str | None = None
    action_graph_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["scalar_purchase_probability"] is None:
            payload["scalar_purchase_probability"] = self.probability
        return payload
