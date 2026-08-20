# SimPlayer Purchase Action graph 구성

Action graph는 게임 안에서 관측되는 화면·이벤트 상태와 사용자 action의 전이 경로를 선언한다. 모델 prompt, action 확률 정규화, trajectory 열거와 terminal 구매확률 계산이 같은 JSON을 사용한다.

기본 graph는 [`game-store-purchase.json`](../src/purchase_behavior_simulator/action_graphs/game-store-purchase.json)이다.

## 가장 빠른 변경 절차

모든 명령은 저장소 루트에서 실행한다.

1. 기본 파일을 같은 package 디렉터리에 복사한다.

```bash
cp src/purchase_behavior_simulator/action_graphs/game-store-purchase.json \
  src/purchase_behavior_simulator/action_graphs/my-game-store.json
```

2. `graph_id`, `version`, 상태와 전이를 변경한다.
3. 로컬 simulation으로 graph 로딩과 action 확률합을 확인한다.

```bash
PURCHASE_BEHAVIOR_ACTION_GRAPH=my-game-store.json \
  purchase-behavior-simulator simulate examples/request.json
```

응답의 `action_graph_id`, `action_graph_version`, `action_distributions`, `likely_trajectories`가 변경한 graph를 사용해야 한다.

4. AgentCore Runtime에서도 사용할 경우 [`agentcore.json`](../deployment/agentcore/agentcore/agentcore.json)의 `PURCHASE_BEHAVIOR_ACTION_GRAPH` 값을 같은 파일명으로 바꾸고 표준 배포 명령을 실행한다.

```bash
PYTHONPATH=src python scripts/deploy_agentcore.py \
  --target default \
  --smoke
```

`action_graphs/` 디렉터리의 JSON은 Python wheel과 AgentCore CodeZip에 포함된다. 저장소 밖의 절대 경로는 로컬 실험에는 사용할 수 있지만 Runtime 배포물에는 자동 포함되지 않는다.

## JSON 필드

| 필드 | 역할 |
| --- | --- |
| `graph_id` | 응답과 trace에서 graph를 구분하는 안정적인 ID |
| `version` | graph 계약 버전 |
| `default_initial_state` | surface별 override가 없을 때 시작할 상태 |
| `surface_initial_states` | `surface` 값을 시작 상태로 매핑 |
| `state_output_fields` | structured actor output에서 사용할 상태별 필드명 |
| `terminal_outcomes` | terminal state를 `purchase`, `exit` 같은 결과로 매핑 |
| `ineligible_policy` | 이미 보유한 비반복 상품 등 구매 불가 case의 행동분포 |
| `max_depth` | trajectory 열거의 최대 transition 수 |
| `transitions` | `state`, `action`, `next_state` 전이 목록 |

최소 전이 예시는 다음과 같다.

```json
{
  "graph_id": "direct_purchase",
  "version": "1",
  "default_initial_state": "OFFER",
  "state_output_fields": {
    "OFFER": "offer"
  },
  "terminal_outcomes": {
    "PURCHASED": "purchase",
    "DISMISSED": "exit"
  },
  "max_depth": 1,
  "transitions": [
    {
      "state": "OFFER",
      "action": "PURCHASE",
      "next_state": "PURCHASED"
    },
    {
      "state": "OFFER",
      "action": "DISMISS",
      "next_state": "DISMISSED"
    }
  ]
}
```

## 자동 검증

Graph 로딩 시 다음 조건을 검사한다.

- `(state, action)` 조합이 중복되지 않는다.
- `surface_initial_states`가 존재하는 상태를 가리킨다.
- 상태별 structured output 필드명이 중복되지 않는다.
- `ineligible_policy`가 해당 상태에 존재하는 action만 사용한다.
- `max_depth`가 1 이상이다.
- 각 상태의 최종 action 확률합이 1.0이다.

전체 계약 테스트:

```bash
PYTHONPATH=src python -m unittest \
  tests.test_action_rollout \
  tests.test_strands_assessment \
  tests.test_service \
  -v
```

## 기본 graph와 다른 UX

Generic actor와 rollout engine은 새 상태와 action을 JSON에서 읽는다. 다만 현재 selection-to-commitment 보정은 기본 게임 상점 UX의 `ITEM_EXPOSURE`와 `ITEM_DETAIL` 의미에 맞춰져 있다. 완전히 다른 UX에서 동일한 보정이 필요하면 해당 graph용 commitment policy adapter를 별도로 구현하고 평가해야 한다. Adapter가 없어도 graph 기반 action 생성과 trajectory 계산은 동작하지만 기본 상점 전용 보정을 자동으로 재사용하지 않는다.

## Future plan: 시간

현재 graph는 행동 순서와 terminal outcome만 결정한다. 체류시간, 다음 행동까지의 지연과 timeout은 예측하지 않으며 `expected_duration_seconds`는 `null`이다. 실제 telemetry가 확보되면 action probability와 분리된 timing model을 연결한다.
