# SimPlayer Purchase

> **Prototype status:** SimPlayer Purchase는 사용자 profile, 과거 관측 행동과 현재 게임 상태를 바탕으로 게임 상품 구매 관련 행동 경로를 시뮬레이션하는 기술 프로토타입이다. 실제 게임 telemetry로 보정된 production 구매 예측 모델이나 매출 보장 시스템이 아니다.

SimUSER의 persona·memory 기반 사용자 simulation에서 영감을 받아, target product와 노출 맥락에 대한 다음 행동과 가능한 행동 경로를 확률분포로 생성한다.

주요 출력은 다음과 같다.

- Scalar 구매확률과 trajectory 구매확률
- 각 게임 상태에서 가능한 다음 action의 확률분포
- 확률이 높은 최대 5개의 행동 trajectory
- 판단 근거, 반대 근거, confidence, observable structured trace

SimPlayer Purchase는 상품을 선택하거나 정렬하는 recommender가 아니다. 호출자가 지정한 상품에 사용자가 어떻게 반응할 가능성이 있는지를 탐색한다.

## 현재 구현 상태

| 항목                             |                     결과 |
| -------------------------------- | -----------------------: |
| State-rich synthetic impressions |                    4,000 |
| GameState coverage               |            8개 필드 100% |
| State-rich long-path protocol    |                    500건 |
| 현재 model-backed 평가 범위      |                    200건 |
| Model-backed 평가                | 200/200 성공, fallback 0 |
| 생성된 구매 행동                 |              2/50, 4.00% |
| Scalar 기대 구매량               | 생성된 행동 대비 +0.82건 |
| Trajectory 기대 구매량           | 생성된 행동 대비 +1.06건 |
| 관측 / 기대 경로 길이 평균       |              1.76 / 1.75 |

현재 코드는 선호와 구매 실행을 분리하는 decision process를 포함한다. 명시적인 생성 가정, 동적 inventory/balance/cooldown, 긴 관측 경로와 최근 16개 transition history는 [Synthetic 데이터 문서](docs/synthetic-data.md)에 설명한다. 현재 graph v3 200건 결과와 지표 해석은 [평가 문서](docs/evaluation.md)를 따른다.

## 기술 스택

| 영역 | 기술 | 사용하는 이유 |
| --- | --- | --- |
| Managed serving | Amazon Bedrock AgentCore Runtime | Python simulation application을 VPC/IAM 경계에서 실행한다. Foundation model 자체를 호스팅하는 서비스는 아니다. |
| Agent orchestration | Strands Agents | probability actor, action actor와 critic을 structured output으로 호출한다. |
| Foundation model 호출 | Strands `OpenAIResponsesModel` / `BedrockModel` adapter | 배포 환경에서 허용된 model ID를 선택한다. 모델은 AgentCore Runtime 안에 포함되지 않으며 Git에도 고정하지 않는다. |
| Memory | Amazon Bedrock AgentCore Memory | 외부에서 관측된 사용자 행동과 episodic evidence를 actor별 namespace로 관리한다. |
| Graph evidence | Amazon Neptune Serverless | 사용자·상품·category·character·event·bundle 관계를 PathSim evidence로 검색한다. |
| Domain core | Python 3.14, Pydantic | 로컬과 AgentCore Runtime에서 같은 요청·응답 및 action graph 계약을 사용한다. |
| Infrastructure | AgentCore CDK, AWS CDK, TypeScript | Runtime, Memory, evaluator와 IAM/VPC 설정을 재배포 가능한 선언으로 관리한다. |

이 문서에서 `Runtime`은 별도 수식어가 없으면 **Amazon Bedrock AgentCore Runtime**을 뜻한다. AgentCore Runtime은 SimPlayer Purchase의 Python orchestration application을 실행하며 foundation model은 Strands adapter를 통해 별도로 호출한다. 로컬 demo는 AgentCore Runtime endpoint를 호출하지 않고 같은 Python service를 process 안에서 실행한다. 세부 경계와 선택 이유는 [아키텍처 문서](docs/architecture.md)를 따른다.

## 구매 판단 흐름

```text
persona + 과거 행동 + 현재 게임 상태 + 상품
  -> 현재 need, selection, feasibility, urgency, uncertainty, hesitation
  -> BUY_NOW / EXPLORE / DEFER / REJECT 의도 비교
  -> 기존 action schema로 commitment 반영
  -> 가격 상승, 필요 해소, 긴급성 제거, 보유 상태 counterfactual 검사
  -> 노출 / 상세 / 구매 확인 / 결제 / 잔액 부족 / 충전 상태별 행동분포
```

`DEFER`는 사용자 의도를 설명하기 위한 내부 상태이며 공개 action에는 추가되지 않는다. 노출 화면에서는 주로 `SKIP`, 상세 화면에서는 `BACK`으로 표현된다. Memory는 유사한 구매 사례만 찾지 않고 구매하지 않은 사례도 함께 제공한다.

## 공개 응답

`simulate`와 `evaluate_snapshot`은 두 구매점수와 행동분포를 같은 응답으로 반환한다.

```json
{
  "probability": 0.081,
  "scalar_purchase_probability": 0.081,
  "trajectory_purchase_probability": 0.064,
  "action_distributions": {
    "ITEM_EXPOSURE": {
      "CLICK": 0.31,
      "SKIP": 0.57,
      "EXIT": 0.06,
      "PURCHASE_NOW": 0.06
    },
    "PURCHASE_CONFIRMATION": {
      "CONFIRM_PURCHASE": 0.42,
      "CANCEL": 0.58
    }
  },
  "likely_trajectories": [
    {
      "probability": 0.57,
      "states": ["ITEM_EXPOSURE", "EXITED"],
      "actions": ["SKIP"],
      "terminal_outcome": "exit",
      "expected_duration_seconds": null
    }
  ],
  "action_graph_id": "game_store_purchase",
  "action_graph_version": "3"
}
```

`probability`는 기존 호출 호환성을 위한 scalar 구매확률 alias다. 프로토타입 단계에서는 scalar와 trajectory 중 하나를 제거하지 않고 둘 다 평가한다.

## Action graph

Action graph는 게임 안에서 관측되는 화면·이벤트 상태와 사용자 action의 전이 경로다. 기본 graph는 [`src/purchase_behavior_simulator/action_graphs/game-store-purchase.json`](src/purchase_behavior_simulator/action_graphs/game-store-purchase.json)에 있다. `state`, `action`, `next_state`, terminal outcome을 JSON에 추가하면 정규화, 경로 열거와 terminal 구매확률 계산이 같은 정의를 사용한다.

다른 graph를 사용하려면 JSON을 같은 `action_graphs/` package 디렉터리에 추가하고 환경변수에 파일명 또는 파일 경로를 지정한다. 이 디렉터리의 JSON은 Python wheel과 AgentCore CodeZip에 함께 포함된다.

```bash
export PURCHASE_BEHAVIOR_ACTION_GRAPH=game-store-purchase.json
```

현재 프로토타입은 행동 순서와 terminal outcome만 시뮬레이션하며 체류시간이나 다음 행동까지의 지연시간은 예측하지 않는다. 응답의 `expected_duration_seconds`는 `null`이다. 시간 모델은 실제 telemetry가 확보된 뒤 transition별 체류시간과 timeout 분포를 별도로 연결하는 future plan이다.

## 저장소 구조

```text
src/purchase_behavior_simulator/  최종 simulator 코드
  action_graphs/                  배포 artifact에 포함되는 게임별 graph
scripts/                          데이터 준비와 full-suite 실행 명령
tests/                            AWS 연결 없는 단위 테스트
examples/                         요청과 dataset adapter 예제
docs/                             아키텍처, 평가, 데이터, 배포 문서
deployment/agentcore/             AgentCore Runtime, Memory, evaluator
deployment/neptune/               Neptune 배포 템플릿
artifacts/dataset/                state-rich 500건 protocol과 데이터 생성 명세
artifacts/evaluation/current/     현재 graph v3 200건 평가 결과
```

개발 실험은 `internal/`에 격리하며 Git 저장소에는 포함하지 않는다.

## 환경 구성 요약

| 사용 범위 | 필요한 준비 |
| --- | --- |
| 로컬 demo | Python 3.14 |
| Model-backed 평가 | AWS credential과 대상 region의 model access |
| AgentCore 배포 | AWS CLI v2, Node.js 20+, AgentCore CLI 0.27.0, uv, private VPC, Neptune |
| 실제 데이터 연결 | schema mapping, 가명화 salt, 필드 allow-list와 보존 정책 |

설치부터 E2E 확인까지의 순서는 [환경 구성 가이드](docs/getting-started.md)에 정리한다.

## 설치

아래 명령은 별도 표시가 없으면 모두 저장소 루트에서 실행한다. Python 3.14를 사용한다.

```bash
python3.14 -m venv .venv314
source .venv314/bin/activate
python -m pip install -e src
```

## 로컬 실행

```bash
purchase-behavior-simulator simulate examples/request.json
```

기본 로컬 구성은 외부 모델을 호출하지 않는다. 요청에 포함된 structured assessment를 사용하거나 중립값으로 동작한다.

`game_state`에는 현재 재화, progression need, 최근 실패 강도, 보유 중복, 이벤트 긴급성, 최근 구매 직후의 cooldown, 현재 목표와 보유 상품을 넣을 수 있다. 생략하면 보수적인 기본값을 사용한다.

공개 Python API:

```python
from purchase_behavior_simulator import (
    BehaviorSimulationService,
    SimulationRequest,
    SimulationResult,
)
```

## 테스트

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Synthetic 데이터

scenario 생성과 label 생성은 서로 다른 session과 seed로 분리한다.

```bash
purchase-behavior-simulator generate-synthetic \
  --output generated/scenarios-current \
  --users 20 \
  --items 60 \
  --impressions 4000 \
  --days 90 \
  --seed 20260822

purchase-behavior-simulator label-synthetic \
  --input generated/scenarios-current \
  --output generated/labeled-current \
  --seed 20260823

PYTHONPATH=src python scripts/prepare_dataset_eval.py \
  --canonical-dir generated/labeled-current \
  --output-dir generated/state-rich-protocol \
  --coverage-output-dir generated/path-coverage-protocol \
  --coverage-cases 100 \
  --users 20 \
  --cases-per-user 25 \
  --history-limit 16 \
  --seed 20260824 \
  --require-game-state
```

자세한 격리 원칙과 schema는 [Synthetic 데이터 문서](docs/synthetic-data.md)를 참고한다.

## Full Suite

Git에는 state-rich long-path 500건 natural protocol, 별도 path-coverage protocol과 데이터 품질 report를 포함한다. Answer key는 추론 입력에서 분리하며, 현재 공식 model-backed 평가는 natural protocol의 200건 prefix를 사용한다.

먼저 사용할 AWS profile/region과 model ID를 확인하고 1건 smoke를 실행한다. 공개 compact report는 `us-east-1`의 `openai.gpt-5.6-sol`로 생성했지만, 해당 model이 대상 계정에서 활성화돼 있어야 한다.

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

Smoke가 성공하면 같은 output directory와 명령을 사용해 필요한 case 수까지 확장한다. 완료한 첫 case는 checkpoint에서 재사용한다.

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

현재 Git에 포함된 compact report는 graph v3 200건 탐색 실행 결과다. Simulation 단계는 `predictions.partial.jsonl`에 case별로 fsync하며 같은 명령을 다시 실행하면 완료한 case를 건너뛰고 이어서 실행한다. `--fallback-retries 2`는 실패 또는 fallback case가 있을 때만 해당 case를 최대 두 번 더 호출하고 report를 자동 재집계한다. 다른 model이나 region을 사용하면 동일 수치를 기대할 수 없다.

## AgentCore 배포

배포에는 AWS CLI credential, private VPC subnet, Runtime security group, 사용할 수 있는 model access와 기존 Neptune cluster가 필요하다. 신규 cluster는 [`deployment/neptune/neptune-serverless.yaml`](deployment/neptune/neptune-serverless.yaml)로 먼저 생성한다. AgentCore 배포 스크립트가 Neptune cluster 자체를 만들지는 않는다.

최초 한 번 local AWS 설정을 만든다. 이 파일은 Git에 포함되지 않는다.

```bash
cp deployment/agentcore/deployment.example.json \
  deployment/agentcore/deployment.local.json
```

계정, 사용할 수 있는 model ID, subnet, security group, Neptune endpoint와 cluster ID를 실제 대상 환경에 맞게 입력한다. 다음 명령은 리소스를 변경하지 않고 계정 일치와 AgentCore schema를 먼저 검사한다.

```bash
PYTHONPATH=src python scripts/deploy_agentcore.py --validate-only
```

검증 후 배포한다. AgentCore, Memory, evaluator와 Neptune은 사용량에 따라 AWS 비용이 발생하며, Neptune deletion protection은 기본 활성이다.

```bash
PYTHONPATH=src python scripts/deploy_agentcore.py \
  --target default \
  --smoke
```

배포 리소스:

- Runtime: `PurchaseBehaviorSimulator`
- Memory: `PurchaseBehaviorSimulatorMemory`
- optional evaluator: `PurchaseBehaviorRuntimeHealth`
- optional online evaluation: `RuntimeHealthOptional`

continuous evaluation과 Runtime OTel은 기본 비활성이다. 배포 환경과 E2E 검증 절차는 [배포 문서](docs/deployment.md)를 따른다.

이 명령은 새 Runtime role과 Memory ID를 자동 연결하고, 생성된 Memory strategy ID와 local AWS 설정을 배포 구간에만 주입한 뒤 source config를 원상복구한다. 현재 AWS credential의 계정이 local 설정과 다르면 배포 전에 중단한다. 마지막 smoke는 observation write, 같은 Memory session read, live Neptune query와 실제 모델 simulation을 모두 확인한다.

## 데이터 초기화

AgentCore Runtime 배포는 빈 Memory와 빈 Neptune graph를 사용할 수 있는 상태로 만든다. 실제 사용자 이력과 상품 관계를 사용하려면 배포 후 production artifact를 별도로 적재해야 한다.

먼저 실제 export를 canonical dataset으로 변환하고, 평가 label과 model prediction을 제외한 가명화 artifact를 만든다.

```bash
PYTHONPATH=src python scripts/configure_dataset.py \
  --config '<mapping-config.json>' \
  --output-dir generated/service-canonical

PURCHASE_BEHAVIOR_IDENTITY_SALT='<value-from-approved-secret-store>' \
PYTHONPATH=src python scripts/prepare_dataset_production.py \
  --canonical-dir generated/service-canonical \
  --output-dir generated/service-production \
  --as-of '<ISO-8601-timestamp>'
```

AWS에 쓰기 전에 모든 Memory payload와 Neptune CSV를 로컬에서 검증한다.

```bash
PYTHONPATH=src python scripts/bootstrap_production_data.py \
  --artifact-dir generated/service-production \
  --dry-run
```

Runtime, Memory와 Neptune 배포가 끝난 뒤 원격 초기화를 실행한다.

```bash
PYTHONPATH=src python scripts/bootstrap_production_data.py \
  --artifact-dir generated/service-production \
  --target default \
  --confirm-write
```

이 명령은 private S3 bucket과 Neptune loader stack을 준비하고, graph CSV load 완료를 기다린 뒤 `memory_import.jsonl`을 Runtime의 `record_observations` operation으로 적재한다. 마지막 canary는 imported session과 다른 새 session에서 long-term Memory record를 검색하고, 방금 적재한 Neptune graph evidence를 사용한 실제 `simulate`를 호출한다. 같은 명령을 다시 실행하면 artifact fingerprint와 checkpoint를 확인해 완료 단계를 건너뛴다.

배포된 Runtime을 직접 호출할 때는 raw `SimulationRequest` JSON을 그대로 사용할 수 있다.

```bash
PYTHONPATH=src python scripts/invoke_agentcore_runtime.py \
  examples/request.json \
  --target default
```

새 관측 구간은 이전 artifact의 `as_of`를 다음 artifact의 `--since`로 사용해 증분 반영한다. 같은 identity salt를 계속 사용해야 동일 사용자가 같은 pseudonym을 유지한다.

```bash
PURCHASE_BEHAVIOR_IDENTITY_SALT='<same-secret>' \
PYTHONPATH=src python scripts/prepare_dataset_production.py \
  --canonical-dir generated/service-canonical \
  --output-dir generated/service-production-delta \
  --since '<previous-as-of>' \
  --as-of '<new-as-of>'

PYTHONPATH=src python scripts/bootstrap_production_data.py \
  --artifact-dir generated/service-production-delta \
  --target default \
  --confirm-write
```

## 데이터 취급

Git에 포함된 평가 protocol과 smoke fixture는 synthetic 데이터다. Runtime은 자동 PII 탐지나 redaction을 제공하지 않으므로 실제 운영 데이터 연결 전에는 가명화·필드 allow-list·보존 정책을 별도로 적용해야 한다.

## 문서

- [문서 안내](docs/README.md)
- [환경 구성 가이드](docs/getting-started.md)
- [아키텍처](docs/architecture.md)
- [Action graph 구성](docs/action-graphs.md)
- [평가 지표와 현재 결과](docs/evaluation.md)
- [Synthetic 데이터](docs/synthetic-data.md)
- [배포와 운영](docs/deployment.md)
- [문제 해결](docs/troubleshooting.md)

## Git 저장소 범위

저장소에는 최종 소스, 테스트, 예제, 문서, 배포 선언과 state-rich 평가 protocol을 포함한다. 개발 실험, virtual environment, CDK/AgentCore 생성물과 model trace 원본은 `.gitignore`로 제외한다.

## License

라이선스 조건은 [MIT License](LICENSE)를 따른다.
