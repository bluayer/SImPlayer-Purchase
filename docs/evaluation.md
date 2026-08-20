# SimPlayer Purchase 평가 결과

> 이 결과는 가정 기반 synthetic 데이터에서 기술 프로토타입의 행동 생성 특성을 확인한 것이다. 실제 플레이어 구매 성능이나 production CVR을 의미하지 않는다.

평가일: 2026-08-21

## 결과 요약

| 평가 대상 | 생성된 관측 행동 | Simulator 기대 | 차이 |
| --- | --: | --: | --: |
| 전체 구매 - scalar probability | 10건 | 15.08건 | +5.08건 |
| 전체 구매 - trajectory probability | 10건 | 11.80건 | +1.80건 |
| 노출 후 `CLICK` | 30건 | 40.50건 | +10.50건 |
| 실행 안정성 | 200건 | 200건 성공 | fallback 0 |

Trajectory 구매확률은 생성된 구매 행동 10건에 대해 11.80건의 기대 구매량을 생성했다. Scalar 구매확률은 같은 case에서 15.08건을 생성했다. 어떤 구매점수가 실제 게임 행동을 더 안정적으로 재현하는지는 아직 결정하지 않았으므로 두 값을 모두 공개 응답과 평가에 유지한다.

`CLICK`은 생성된 클릭 행동 30건보다 10.50건 많은 40.50건을 기대했다. 현재 결과는 구매량 시뮬레이션과 비교해 클릭 행동을 더 자주 생성하는 경향을 보여준다.

## 평가 단위

평가셋은 20명의 synthetic 사용자에게 발생한 200개의 상품 노출 case로 구성된다. 200명의 서로 다른 사용자를 뜻하지 않는다.

| 항목                     |    값 |
| ------------------------ | ----: |
| Synthetic 사용자         |  20명 |
| 상품 노출 case           | 200건 |
| 생성된 구매 행동         |  10건 |
| 생성된 비구매 행동       | 190건 |
| 생성된 구매 행동 비율    |  5.0% |
| 노출 행동 비교 가능 case | 186건 |
| 생성된 `CLICK`           |  30건 |

이 문서의 데이터는 전부 synthetic이다. `생성된 관측 행동`은 synthetic labeler가 각 노출에서 확률적으로 한 번 생성한 구매·비구매와 클릭·비클릭 행동을 뜻한다.

Simulator 기대 건수는 case별 확률을 더한 값이다. 예를 들어 200개의 trajectory 구매확률 합계가 `11.80`이면, 같은 조건을 반복했을 때 평균적으로 약 11.80건의 구매 행동을 생성한다는 뜻이다. 특정 12개 case를 구매로 분류했다는 뜻은 아니다.

## 두 구매확률

### Scalar probability

과거 행동, persona, 상품 맥락, 현재 게임 상태, LLM 판단과 trajectory 신호를 결합한 종합 구매점수다.

```text
200건 확률 합계: 15.08건
생성된 구매 행동: 10건
건수 차이:        +5.08건
비율:             7.54% vs 5.00%, +2.54%p
```

### Trajectory probability

Action graph의 행동분포에서 terminal purchase 상태에 도착하는 모든 경로의 확률을 더한 값이다.

```text
P(PURCHASE_NOW)
+ P(CLICK) × P(PURCHASE | 다음 구매 단계)
```

```text
200건 확률 합계: 11.80건
생성된 구매 행동: 10건
건수 차이:        +1.80건
비율:             5.90% vs 5.00%, +0.90%p
```

Scalar와 trajectory의 차이는 결함으로 단정하지 않는다. 프로토타입 단계에서는 두 후보 점수를 함께 관찰하며, 실제 게임 telemetry가 연결된 뒤 최종 출력 계약을 결정한다.

## Action gap

행동 평가는 생성된 관측 행동 건수와 Simulator 기대 건수의 차이를 우선한다.

### 전체 구매

| 점수       | 생성된 관측 행동 | 기대 건수 | 건수 차이 | 비율 차이 |
| ---------- | ---------------: | --------: | --------: | --------: |
| Scalar     |             10건 |   15.08건 |   +5.08건 |   +2.54%p |
| Trajectory |             10건 |   11.80건 |   +1.80건 |   +0.90%p |

### 노출 후 클릭

노출 행동을 직접 비교할 수 있는 case는 186건이다.

| Action | 생성된 관측 행동 | 기대 건수 | 건수 차이 | 관측 행동 비율 | 기대 비율 | 비율 차이 |
| --- | --: | --: | --: | --: | --: | --: |
| `CLICK` | 30건 | 40.50건 | +10.50건 | 16.13% | 21.78% | +5.65%p |

클릭 차이가 Simulator와 synthetic data 중 어느 쪽의 문제인지는 실제 게임 UX와 telemetry 없이 구분할 수 없다. 실제 데이터를 연결할 때 노출·클릭 정의와 action graph를 함께 맞춰야 한다.

## 상세 화면 평가 제외

상세 화면은 게임 상품 구매에서 항상 존재하는 단계가 아니다. 현재 synthetic protocol은 클릭 이후 행동을 `ITEM_DETAIL`의 `PURCHASE/BACK/EXIT`로 매핑하지만, 이 구조가 실제 게임의 구매 UX를 대표한다는 근거는 없다.

따라서 상세 화면 지표와 synthetic label이 독립적으로 관측하지 않는 `EXIT`, `PURCHASE_NOW`의 개별 적중률은 핵심 결과에서 제외한다.

## 실행 검증

- 200/200 case 성공
- 최종 fallback 0
- Schema success 100%
- 모든 결과에 scalar/trajectory 구매확률, action distribution, graph ID/version과 trajectory 포함
- 모든 state의 action 확률합 1.0

실행 중 provider connection fallback이 한 case에서 발생했지만 checkpoint harness가 case ID를 기록했다. 해당 case만 재시도해 정상 결과로 교체했으며 최종 report에는 fallback 결과가 남아 있지 않다.

## 데이터 한계

1. 모든 사용자, 상품, 상태와 행동은 명시적인 synthetic 가정으로 생성됐다.
2. 생성된 관측 행동은 확률에서 한 번 생성되므로 labeling seed에 따라 달라질 수 있다.
3. 사용자는 20명이며 여러 상품 노출이 같은 사용자에게 반복된다.
4. Synthetic action schema는 실제 게임의 구매 UX로 검증되지 않았다.
5. 실제 게임 데이터 연결 후에는 action graph, 생성 분포와 평가 기준을 다시 구성해야 한다.

## 산출물

- `artifacts/dataset/data-generation`: 생성과 protocol 품질 report
- `artifacts/dataset/protocol`: blind cases, bootstrap과 inference-isolated answer key
- `artifacts/evaluation/current/results/summary.json`: 현재 200건 결과와 핵심 action gap

Raw prediction과 observable trace는 `artifacts/evaluation/runs/`에 보관하며 Git에는 포함하지 않는다.

## 재실행

공개 compact report와 같은 model/region을 사용할 수 있는지 확인한 뒤 먼저 1건 smoke를 실행한다.

```bash
export AWS_REGION='us-east-1'
export PURCHASE_BEHAVIOR_MODEL_ID='openai.gpt-5.6-sol'

aws sts get-caller-identity

PYTHONPATH=src python scripts/run_full_suite.py \
  --protocol-dir artifacts/dataset/protocol \
  --output-dir artifacts/evaluation/runs/current-200 \
  --model-id "$PURCHASE_BEHAVIOR_MODEL_ID" \
  --limit 1 \
  --workers 1 \
  --fallback-retries 1
```

Smoke가 성공하면 같은 output directory로 200건까지 확장한다.

```bash
PYTHONPATH=src python scripts/run_full_suite.py \
  --protocol-dir artifacts/dataset/protocol \
  --output-dir artifacts/evaluation/runs/current-200 \
  --model-id "$PURCHASE_BEHAVIOR_MODEL_ID" \
  --limit 200 \
  --workers 1 \
  --confirm-model-cost \
  --fallback-retries 1
```

실행 중에는 `predictions.partial.jsonl`에 case별로 fsync한다. 같은 명령을 다시 실행하면 완료한 case를 재사용한다. 실패 또는 fallback case는 한 번 자동 재시도하며, 재시도 후에도 남아 있으면 `retry-case-ids.txt`로 확인할 수 있다.
