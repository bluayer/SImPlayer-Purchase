from __future__ import annotations

from typing import Any, Mapping

from .models import ObservationBatch, SimulationRequest
from .service import BehaviorSimulationService


def handle_request(
    payload: Mapping[str, Any],
    *,
    service: BehaviorSimulationService,
) -> dict[str, Any]:
    """Handle the stable prototype operations shared by local and AgentCore."""
    operation = str(payload.get("operation", "simulate"))

    if operation in {"record_observations", "initialize_memory"}:
        observation_payload = payload.get("observation", payload)
        if not isinstance(observation_payload, Mapping):
            return {
                "accepted": False,
                "error": {
                    "type": "validation_error",
                    "message": "observation must be an object",
                },
            }
        try:
            batch = ObservationBatch.from_dict(observation_payload)
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "accepted": False,
                "error": {
                    "type": "validation_error",
                    "message": str(exc),
                },
            }
        return service.record_observations(batch).to_dict()

    if operation == "evaluate_snapshot":
        request_payload = payload.get("request", payload)
        if not isinstance(request_payload, Mapping):
            raise ValueError("evaluate_snapshot request must be an object")
        request = SimulationRequest.from_dict(request_payload)
        return service.evaluate_snapshot(request).to_dict()

    if operation != "simulate":
        raise ValueError(f"unsupported operation: {operation}")

    request_payload = payload.get("request", payload)
    if not isinstance(request_payload, Mapping):
        raise ValueError("simulation request must be an object")
    request = SimulationRequest.from_dict(request_payload)
    return service.simulate(request).to_dict()
