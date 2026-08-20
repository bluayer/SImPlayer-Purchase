from __future__ import annotations

import json
import os
import time
from typing import Sequence

from .evaluation_trace import (
    TraceEvents,
    invoke_with_transient_retries,
    sanitized_error,
    strands_result_metrics,
)
from .model_factory import build_strands_model
from .models import ObservationBatch, SimulationRequest
from .product_needs import resolve_product_need_profile


SELF_ASK_SYSTEM_PROMPT = """You generate retrieval questions for episodic memory.
Given a game-store user and a target item, produce concise follow-up questions
that retrieve concrete past interactions. Include both supporting and opposing
evidence. Include one question about time-sensitive functional need satisfaction
and one about stable aesthetic, identity, collection, enjoyment, or social
preference when relevant. A bundle may mix both motivations, so inspect its
components instead of treating bundle as a motivation. Never ask for model
scores, predictions, labels, or inferred sensitive traits. Return at most three questions.
Ignore legacy add_to_cart or cart tokens because this game store has no cart action.
"""

REFLECTION_SYSTEM_PROMPT = """You reflect on externally observed game-store behavior.
Summarize what the observed action and optional rating/feeling imply, what remains
uncertain, and at least one plausible alternative cause. Do not invent actions.
Do not mention or infer a purchase prediction, probability, label, or recommendation
decision. This reflection may become future episodic evidence, so stay factual and
calibrated.
"""


class DeterministicSelfAskQueryPlanner:
    def plan(
        self,
        request: SimulationRequest,
        initial_query: str,
    ) -> tuple[str, ...]:
        categories = ", ".join(request.item.categories) or "이 상품군"
        need_profile = resolve_product_need_profile(request.item).to_dict()
        price_ratio = (
            request.item.price / request.context.budget_reference
            if request.context.budget_reference
            else None
        )
        ratio_text = f"{price_ratio:.2f}" if price_ratio is not None else "unknown"
        return (
            (
                f"{categories}의 기능적 필요를 최근 충족했거나 다시 필요해진 "
                f"시점의 실제 행동은 무엇인가? rational weight "
                f"{need_profile['rational']:.2f}"
            ),
            (
                f"{categories}와 관련된 색상·스타일·수집·정체성 등 장기적으로 "
                f"반복된 취향과 반대 사례는 무엇인가? emotional weight "
                f"{need_profile['emotional']:.2f}"
            ),
            (
                f"surface {request.context.surface}, price/budget {ratio_text}, "
                f"fatigue {request.context.session_fatigue:.2f}, progression need "
                f"{request.game_state.progression_need:.2f}와 유사한 상황에서 "
                "구매한 사례와 구매하지 않은 사례는 각각 무엇인가?"
            ),
        )


class StrandsSelfAskQueryPlanner:
    def __init__(
        self,
        model_id: str | None = None,
        region_name: str | None = None,
        fallback: DeterministicSelfAskQueryPlanner | None = None,
        trace_events: TraceEvents | None = None,
    ) -> None:
        self.model_id = model_id or os.environ["BEDROCK_MODEL_ID"]
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.fallback = fallback or DeterministicSelfAskQueryPlanner()
        self.trace_events = trace_events

    def plan(
        self,
        request: SimulationRequest,
        initial_query: str,
    ) -> tuple[str, ...]:
        from pydantic import BaseModel, ConfigDict, Field
        from strands import Agent

        class QueryOutput(BaseModel):
            model_config = ConfigDict(extra="forbid")

            questions: list[str] = Field(min_length=1, max_length=3)

        started = time.monotonic()
        try:
            model = build_strands_model(
                model_id=self.model_id,
                region_name=self.region_name,
                max_tokens=int(
                    os.getenv(
                        "PURCHASE_BEHAVIOR_SELF_ASK_MAX_TOKENS",
                        "500",
                    )
                ),
            )
            agent = Agent(
                model=model,
                system_prompt=SELF_ASK_SYSTEM_PROMPT,
                tools=[],
                callback_handler=None,
            )
            result, attempts = invoke_with_transient_retries(
                lambda: agent(
                    json.dumps(
                        self._payload(request, initial_query),
                        ensure_ascii=False,
                    ),
                    structured_output_model=QueryOutput,
                ),
            )
            output = result.structured_output
            if output is None:
                raise ValueError("Strands returned no self-ask questions")
            questions = _dedupe_questions(output.questions)
            self._trace(
                {
                    "stage": "self_ask",
                    "duration_seconds": round(time.monotonic() - started, 6),
                    "initial_query": initial_query,
                    "questions": list(questions),
                    "metrics": strands_result_metrics(result),
                    "fallback": False,
                    "attempts": attempts,
                }
            )
            return questions
        except Exception as exc:
            questions = self.fallback.plan(request, initial_query)
            self._trace(
                {
                    "stage": "self_ask",
                    "duration_seconds": round(time.monotonic() - started, 6),
                    "initial_query": initial_query,
                    "questions": list(questions),
                    "error": sanitized_error(exc),
                    "fallback": True,
                }
            )
            return questions

    def _trace(self, event: dict[str, object]) -> None:
        if self.trace_events is not None:
            self.trace_events.append(event)

    @staticmethod
    def _payload(
        request: SimulationRequest,
        initial_query: str,
    ) -> dict[str, object]:
        need_profile = resolve_product_need_profile(request.item)
        return {
            "initial_question": initial_query,
            "persona": {
                "summary": request.user.persona_summary,
                "pickiness": request.user.pickiness,
                "price_sensitivity": request.user.price_sensitivity,
                "category_preferences": dict(request.user.category_preferences),
            },
            "target_item": {
                "item_id": request.item.item_id,
                "product_type": request.item.product_type,
                "categories": list(request.item.categories),
                "price": request.item.price,
                "discount_rate": request.item.discount_rate,
                "components": [
                    component.to_dict()
                    for component in request.item.components
                ],
                "need_profile": need_profile.to_dict(),
            },
            "surface": request.context.surface,
            "game_state": request.game_state.to_dict(),
        }


class DeterministicReflectionProvider:
    def reflect(self, batch: ObservationBatch) -> str:
        event_types = ", ".join(event.event_type for event in batch.events)
        item_ids = ", ".join(
            event.item_id for event in batch.events if event.item_id
        ) or "상품 미지정"
        feeling = f" 관측된 감정: {batch.feeling}." if batch.feeling else ""
        return (
            f"외부에서 관측된 행동은 {event_types}, 대상은 {item_ids}이다."
            f"{feeling} 이 행동은 선호 신호일 수 있지만 가격, 노출 위치, "
            "일시적 과업 같은 대안 원인도 배제할 수 없다."
        )


class StrandsReflectionProvider:
    def __init__(
        self,
        model_id: str | None = None,
        region_name: str | None = None,
        fallback: DeterministicReflectionProvider | None = None,
    ) -> None:
        self.model_id = model_id or os.environ["BEDROCK_MODEL_ID"]
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.fallback = fallback or DeterministicReflectionProvider()

    def reflect(self, batch: ObservationBatch) -> str:
        from pydantic import BaseModel, ConfigDict, Field
        from strands import Agent

        class ReflectionOutput(BaseModel):
            model_config = ConfigDict(extra="forbid")

            reflection: str = Field(min_length=1, max_length=1200)

        try:
            model = build_strands_model(
                model_id=self.model_id,
                region_name=self.region_name,
                max_tokens=int(
                    os.getenv(
                        "PURCHASE_BEHAVIOR_REFLECTION_MAX_TOKENS",
                        "700",
                    )
                ),
            )
            agent = Agent(
                model=model,
                system_prompt=REFLECTION_SYSTEM_PROMPT,
                tools=[],
                callback_handler=None,
            )
            result = agent(
                json.dumps(self._payload(batch), ensure_ascii=False),
                structured_output_model=ReflectionOutput,
            )
            output = result.structured_output
            if output is None:
                raise ValueError("Strands returned no episodic reflection")
            return output.reflection.strip()
        except Exception:
            return self.fallback.reflect(batch)

    @staticmethod
    def _payload(batch: ObservationBatch) -> dict[str, object]:
        return {
            "source": batch.source,
            "page_id": batch.page_id,
            "recommended_item_ids": list(batch.recommended_item_ids),
            "feeling": batch.feeling,
            "review": batch.review,
            "events": [
                {
                    "event_type": event.event_type,
                    "timestamp": event.timestamp.isoformat(),
                    "item_id": event.item_id,
                    "categories": list(event.categories),
                    "rating": event.rating,
                }
                for event in batch.events
            ],
        }


def _dedupe_questions(values: Sequence[str]) -> tuple[str, ...]:
    questions: list[str] = []
    seen: set[str] = set()
    for value in values:
        question = " ".join(str(value).split()).strip()
        normalized = question.lower()
        if not question or normalized in seen:
            continue
        seen.add(normalized)
        questions.append(question)
    return tuple(questions[:3])
