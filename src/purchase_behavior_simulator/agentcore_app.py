from __future__ import annotations

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from purchase_behavior_simulator.bootstrap import build_service
from purchase_behavior_simulator.runtime_api import handle_request


app = BedrockAgentCoreApp()
service = build_service()

@app.entrypoint
def simulate_purchase_behavior(payload, context=None):
    return handle_request(payload, service=service)


if __name__ == "__main__":
    app.run()
