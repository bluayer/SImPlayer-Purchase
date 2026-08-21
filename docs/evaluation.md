# SimPlayer Purchase 평가 결과

> 이 결과는 가정 기반 synthetic 데이터의 긴 구매 행동 경로를 대상으로 수행한 graph v3 200건 탐색 평가다. 실제 플레이어 구매 성능이나 production CVR을 의미하지 않는다.

평가일: 2026-08-21

## 결과 요약

| 평가 대상 | 생성된 관측 행동 | Simulator 기대 | 차이 |
| --- | --: | --: | --: |
| 전체 구매 - scalar probability | 11건 | 11.12건 | +0.12건 |
| 전체 구매 - trajectory probability | 11건 | 11.97건 | +0.97건 |
| 노출 상태의 `CLICK` | 30건 | 37.28건 | +7.28건 |
| 전체 경로 길이 평균 | 1.68단계 | 1.70단계 | +0.03단계 |
| 실행 안정성 | 200건 | 200건 성공 | fallback 0 |

Scalar와 trajectory 구매 기대 건수는 생성된 구매 11건과 각각 0.12건, 0.97건 차이다. 평균 경로 길이도 0.03단계 차이다. 코호트 전체 구매량과 경로 길이 규모는 가깝게 생성됐다.

사용자 action 구성에는 차이가 남아 있다. `SKIP`을 17.93건 많이, `EXIT`를 19.72건 적게 생성했다. 구매 과정에서는 `START_PURCHASE`를 4.42건, `CONFIRM_PURCHASE`를 6.86건 적게 생성했다.

## 평가 단위

| 항목                                       |    값 |
| ------------------------------------------ | ----: |
| Synthetic 사용자                           |  20명 |
| 상품 노출 case                             | 200건 |
| 생성된 구매 행동                           |  11건 |
| 생성된 관측 경로 종류                      |  15종 |
| 관측 경로 최대 길이                        | 7단계 |
| 4단계 이상 경로                            |  18건 |
| 최근 16 transition에 과거 구매가 있는 case |  77건 |

200명의 서로 다른 사용자를 뜻하지 않는다. 20명의 synthetic 사용자에게 시간순으로 발생한 상품 노출 case 200개다.

`생성된 관측 행동`은 synthetic labeler가 상태별 확률에서 한 번 생성한 UI·transaction event다. Simulator 기대 건수는 200개 case의 action 확률을 더한 값이며 특정 case를 해당 action으로 분류했다는 뜻이 아니다.

## 핵심 사용자 Action gap

| 상태 / Action                              |  관측 |     기대 |     차이 |
| ------------------------------------------ | ----: | -------: | -------: |
| `ITEM_EXPOSURE / CLICK`                    |  30건 |  37.28건 |  +7.28건 |
| `ITEM_EXPOSURE / SKIP`                     | 102건 | 119.93건 | +17.93건 |
| `ITEM_EXPOSURE / EXIT`                     |  87건 |  67.28건 | -19.72건 |
| `ITEM_EXPOSURE / PURCHASE_NOW`             |   2건 |   7.76건 |  +5.76건 |
| `ITEM_DETAIL / START_PURCHASE`             |  22건 |  17.58건 |  -4.42건 |
| `ITEM_DETAIL / BACK`                       |  39건 |  50.50건 | +11.50건 |
| `PURCHASE_CONFIRMATION / CONFIRM_PURCHASE` |  20건 |  13.14건 |  -6.86건 |
| `PURCHASE_CONFIRMATION / CANCEL`           |  12건 |  12.58건 |  +0.58건 |

Graph v3에서 `SKIP`은 현재 offer만 닫고 store surface에 남는 행동, `EXIT`는 store surface를 떠나는 행동이다. 상세 화면과 구매 확인 화면을 떠나는 행동은 각각 `BACK`, `CANCEL`로 기록한다. 이 정의로 조정 전 존재하던 상세·확인 상태의 직접 `EXIT` 기대 건수는 0이 됐다.

## 두 구매확률

### Scalar probability

과거 행동, persona, 현재 게임 상태, 상품 맥락, Memory/KG evidence, LLM 판단과 trajectory 신호를 결합한 종합 점수다.

```text
200건 확률 합계: 11.12건
생성된 구매 행동: 11건
차이: +0.12건
```

### Trajectory probability

Action graph에서 구매 terminal state에 도착하는 모든 경로의 확률을 합한 값이다. 구매 시작, 확인과 결제 성공 단계를 통과해야 한다.

```text
200건 확률 합계: 11.97건
생성된 구매 행동: 11건
차이: +0.97건
```

프로토타입 단계에서는 두 후보 점수를 모두 공개한다. 실제 게임 telemetry를 연결한 뒤 어떤 점수가 코호트 구매량과 개별 행동 경로를 더 안정적으로 재현하는지 판단한다.

## Natural suite와 Path-coverage suite

공식 action gap은 자연분포 protocol에서 사용자 선택 action을 비교한다.

- `CLICK`, `SKIP`, `EXIT`, `PURCHASE_NOW`
- `START_PURCHASE`, `BACK`
- `CONFIRM_PURCHASE`, `CANCEL`
- 충전 화면을 열거나 돌아가는 사용자 action

`PAYMENT_SUCCESS`, `PAYMENT_FAILED`, `INSUFFICIENT_CURRENCY`, `TOP_UP_SUCCESS`는 사용자 선택이 아니라 시스템 결과다. 특히 결제 실패 빈도는 이 프로토타입의 핵심 성능 목표로 사용하지 않는다. 이 경로들은 [`artifacts/dataset/path-coverage-protocol`](../artifacts/dataset/path-coverage-protocol)에서 경로가 정상 처리되는지만 확인하며, 해당 suite의 빈도 지표는 성능 수치로 해석하지 않는다.

## History 입력

`SimulationRequest.interactions`는 optional이다. 제공하면 현재 요청 시점 직전의 recent-session hint로 사용한다. 이번 protocol은 최대 16개의 관측 transition을 제공했으며 77개 case에는 과거 결제 성공이 포함됐다.

배포 환경의 장기 이력은 request에 모두 넣지 않는다.

- Amazon Bedrock AgentCore Memory: 관측 transition과 episode
- Amazon Neptune: 사용자와 상품 사이의 action edge 및 상품 관계

Production export는 전체 관측 경로를 두 저장소에 보존한다. Request history가 비어 있어도 Runtime은 Memory와 Neptune evidence로 장기 이력을 보완한다.

## 실행 검증

- 200/200 case 성공
- 최종 fallback 0
- Schema success 100%
- 모든 결과에 graph v3의 6개 non-terminal state action distribution 포함
- 모든 state의 action 확률합 1.0
- 평균 latency 34.59초, p95 58.11초

## 데이터 한계

1. 모든 사용자, 상품, 게임 상태와 행동은 명시적인 synthetic 가정으로 생성됐다.
2. 생성된 구매 11건은 실제 게임의 구매 식별력을 판단하기에는 여전히 작은 표본이다.
3. `SKIP`과 `EXIT`의 차이는 실제 게임 instrumentation과 맞춰야 한다.
4. 희소 결제·충전 경로는 natural suite의 빈도 성능이 아니라 path-coverage suite로 검증한다.
5. 실제 게임 데이터 연결 후에는 action 이름, initial state, 경로 빈도와 평가 기준을 다시 구성해야 한다.

## 산출물

- `artifacts/dataset/data-generation`: state-rich long-path 데이터 생성 명세
- `artifacts/dataset/protocol`: 500건 natural protocol
- `artifacts/dataset/path-coverage-protocol`: 희소 경로 기능 검증 protocol
- `artifacts/evaluation/current/results/summary.json`: 현재 graph v3 200건 결과

Raw prediction과 observable trace는 `artifacts/evaluation/runs/`에 보관하며 Git에는 포함하지 않는다.

## 재실행

먼저 1건 smoke를 실행한다.

```bash
export AWS_REGION='us-east-1'
export PURCHASE_BEHAVIOR_MODEL_ID='openai.gpt-5.6-sol'

aws sts get-caller-identity

PYTHONPATH=src python scripts/run_full_suite.py \
  --protocol-dir artifacts/dataset/protocol \
  --output-dir artifacts/evaluation/runs/current \
  --model-id "$PURCHASE_BEHAVIOR_MODEL_ID" \
  --limit 1 \
  --workers 1 \
  --fallback-retries 2
```

Smoke가 성공하면 같은 output directory로 필요한 case 수까지 확장한다. 완료한 case는 checkpoint에서 재사용한다.

```bash
PYTHONPATH=src python scripts/run_full_suite.py \
  --protocol-dir artifacts/dataset/protocol \
  --output-dir artifacts/evaluation/runs/current \
  --model-id "$PURCHASE_BEHAVIOR_MODEL_ID" \
  --limit 200 \
  --workers 1 \
  --confirm-model-cost \
  --fallback-retries 2
```
