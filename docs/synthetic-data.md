# SimPlayer Purchase Synthetic 데이터 생성

## 목적

이 데이터 파이프라인은 사용자 이력과 현재 게임 상태에 따라 구매 관련 행동이 달라지는지를 검증하기 위한 synthetic dataset을 만든다. 평가 대상 LLM은 label 생성에 참여하지 않으며, scenario 생성과 action sampling은 서로 다른 session과 seed로 실행한다.

Synthetic 결과는 실제 운영 CVR을 대신하지 않는다. 모든 사용자·세션·게임 상태 분포는 코드에 명시한 생성 가정이며, 실제 게임 telemetry에서 관측한 값으로 표현하지 않는다.

## 데이터 흐름

```text
explicit synthetic assumptions
  -> label-free users/items/scenarios
  -> isolated stateful labeler
  -> time-ordered GameStateSnapshot + observed actions
  -> state-rich holdout protocol
  -> paired state counterfactual protocol
```

### 1. Generation session

`purchase_behavior_simulator.synthetic`은 구매 label을 만들지 않는다.

| 파일 | 내용 |
| --- | --- |
| `users.jsonl` | persona, latent preference와 초기 state 조건 |
| `items.jsonl` | 상품 category, 가격, 할인, utility와 emotionality |
| `scenarios.jsonl` | 노출 시각, surface, session 상태, progression/failure/urgency와 current goals |
| `oracle/oracle.jsonl` | oracle version과 protected latent shock. 구매확률과 label은 없음 |
| `neptune/nodes.csv` | 정적 graph node |
| `neptune/static_edges.csv` | 정적 graph edge |
| `manifest.json` | generation session, 설정, 생성 가정과 SHA-256 |

Generation 단계가 만드는 상태는 label에 의존하지 않는 외생 상태다. `currency_balance`, `owned_item_ids`와 `purchase_cooldown`처럼 이전 구매 결과에 따라 달라지는 상태는 이 단계에서 미리 만들지 않는다.

### 2. Labeling session

`purchase_behavior_simulator.synthetic_labeling`은 scenario를 시간순으로 처리하며 사용자별 동적 state를 유지한다.

```text
노출 직전 balance / inventory / last purchase
  -> GameStateSnapshot
  -> inference-isolated oracle probability
  -> CLICK / PURCHASE sampling
  -> balance 차감, inventory 추가, cooldown 시작
  -> 다음 노출
```

따라서 구매한 비소모성 상품은 이후 `owned_item_ids`에 나타나고, 구매 직후에는 `purchase_cooldown`이 높아진다. Label과 이후 state가 서로 다른 hidden trajectory를 사용하지 않는다.

출력 `impressions.jsonl`의 `game_state`에는 다음 필드가 모두 포함된다.

| 필드 | 생성 방식 |
| --- | --- |
| `currency_balance` | 사용자 spending scale, 초기 multiplier와 시간 경과 refill 가정으로 생성 |
| `progression_need` | surface와 session context에서 생성하고 이전 utility 구매의 relief를 반영 |
| `recent_failure_intensity` | failure-recovery surface, progression need와 noise로 생성 |
| `inventory_overlap` | labeler가 유지하는 실제 synthetic inventory와 target category에서 계산 |
| `event_urgency` | 21일 campaign 진행 시점과 event boost 가정으로 생성 |
| `purchase_cooldown` | 마지막 synthetic 구매 이후 시간과 72시간 감쇠 가정으로 계산 |
| `current_goals` | 선호 category, active character와 현재 progression context에서 생성 |
| `owned_item_ids` | 초기 inventory와 앞서 생성된 구매 행동을 시간순으로 누적 |

## 생성 가정

기본 가정은 `SyntheticAssumptions`에 직접 정의한다.

| 항목                                        |               기본 가정 |
| ------------------------------------------- | ----------------------: |
| 할인 상품 비율                              |                     30% |
| 할인율 후보                                 | 10%, 15%, 20%, 30%, 40% |
| 재방문 플레이어 비율                        |                     65% |
| 세션 내 상품 조회 수 중앙값 / 상위 10% 경계 |                 10 / 30 |
| 세션 길이 중앙값 / 상위 10% 경계            |             15분 / 60분 |
| event boost 발생 비율                       |                     10% |
| 구매 cooldown 감쇠 시간                     |                  72시간 |
| 초기 보유 상품 수 중앙값 / 상위 10% 경계    |                  4 / 12 |

이 값은 실제 게임에서 관측한 수치가 아니다. 현재 게임 상태에 따라 simulator의 판단이 기대한 방향으로 달라지는지를 검증하기 위한 시나리오 범위다. 실제 게임 데이터가 준비되면 같은 필드의 비식별 집계로 가정을 교체하고 protocol을 다시 생성해야 한다.

## 실행

```bash
purchase-behavior-simulator generate-synthetic \
  --output generated/scenarios-current \
  --users 20 \
  --items 60 \
  --impressions 4000 \
  --days 90 \
  --seed 20260820

purchase-behavior-simulator label-synthetic \
  --input generated/scenarios-current \
  --output generated/labeled-current \
  --seed 20260821
```

State-rich 평가 protocol은 coverage gate를 반드시 켠다.

```bash
PYTHONPATH=src python scripts/prepare_dataset_eval.py \
  --canonical-dir generated/labeled-current \
  --output-dir generated/state-rich-protocol \
  --users 20 \
  --cases-per-user 25 \
  --history-fraction 0.5 \
  --require-game-state
```

동일한 사용자·상품·이력을 고정하고 state 한 항목만 바꾸는 paired protocol도 생성한다.

```bash
PYTHONPATH=src python scripts/prepare_state_counterfactual_eval.py \
  --protocol-dir generated/state-rich-protocol \
  --output-dir generated/state-counterfactual \
  --base-cases 10
```

## 품질 게이트

Model-backed 평가 전에 다음 조건을 모두 확인한다.

1. Generation과 labeling session ID가 다르다.
2. Generation 이후 파일 hash가 바뀌면 labeling이 실패한다.
3. Oracle spec에는 구매확률과 추출된 행동 label이 없다.
4. 8개 핵심 `GameStateSnapshot` field coverage가 100%다.
5. 수치형 state가 constant가 아니며 inventory와 goals가 실제로 변한다.
6. 가격 상승, balance 감소, need 해소, urgency 제거, ownership 추가와 cooldown 증가는 oracle 구매확률을 높이지 않는다.
7. 평가에 사용하는 200건 prefix가 20명을 포함하고 자연 구매율을 유지한다.
8. Answer key는 blind request에 포함되지 않는다.

현재 Git의 state-rich artifact는 4,000 impressions에서 만들었다.

| 항목                                   |                    결과 |
| -------------------------------------- | ----------------------: |
| 관측 구매율 / expected 구매율          |         4.425% / 5.258% |
| GameState coverage                     |      8개 필드 모두 100% |
| Oracle counterfactual direction        |      6개 항목 모두 100% |
| 재현용 전체 protocol                   |             500건, 20명 |
| 현재 평가 대상                         |  200건, 20명, 구매 10건 |
| Paired audit                           | 10 base cases, 59 pairs |
| 데이터 준비 단계의 AWS 또는 model 호출 |                       0 |

이 protocol의 200건 prefix를 사용한 model-backed 결과는 [평가 문서](evaluation.md)에 기록한다. 데이터 준비 단계의 품질 gate와 simulator 성능 평가는 서로 분리한다.
