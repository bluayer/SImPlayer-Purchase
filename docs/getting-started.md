# SimPlayer Purchase 환경 구성

> SimPlayer Purchase는 기술 프로토타입이다. 로컬 demo는 외부 모델 없이 구조와 응답 계약을 확인하고, model-backed 평가는 별도의 AWS credential과 모델 접근 권한을 사용한다.

이 문서에서 `Runtime`은 **Amazon Bedrock AgentCore Runtime**을 뜻한다. AgentCore Runtime은 SimPlayer Purchase의 Python application을 실행하고, application이 Strands adapter를 통해 foundation model과 AgentCore Memory, Neptune을 호출한다. 로컬 demo는 AgentCore Runtime endpoint를 사용하지 않는다.

필요한 환경은 사용할 범위에 따라 달라진다.

별도 표시가 없으면 모든 명령은 저장소 루트에서 실행한다. 로컬 demo만 확인하려면 1장까지만, model-backed 평가까지 확인하려면 2장까지, AWS Runtime을 배포하려면 3장까지 진행한다.

| 사용 범위 | 필요한 셋업 |
| --- | --- |
| 로컬 deterministic demo | Git, Python 3.14 |
| Model-backed full suite | 위 항목 + AWS credential, 대상 region의 model access |
| AgentCore Runtime 배포 | 위 항목 + AWS CLI v2, Node.js 20+, AgentCore CLI, uv, private VPC, Neptune |
| 실제 운영 데이터 연결 | 위 항목 + schema mapping, 가명화 salt, 필드 allow-list와 보존 정책 |

## 1. 공통 설치

검증한 주요 toolchain은 다음과 같다.

| 도구          |                  검증 버전 |
| ------------- | -------------------------: |
| Python        |                     3.14.5 |
| AWS CLI       |                     2.33.9 |
| Node.js       |  20 이상, 검증 버전 23.9.0 |
| npm           |                      9.6.1 |
| AgentCore CLI |                     0.27.0 |
| uv            |                      0.7.8 |
| AWS CDK CLI   | 2.1126.0 (저장소 lockfile) |

AgentCore CLI 0.27.0은 Node.js 20 이상을 요구한다.

```bash
npm install -g @aws/agentcore@0.27.0
agentcore --version
```

Python package를 설치한다.

```bash
python3.14 -m venv .venv314
source .venv314/bin/activate
python -m pip install -e src
```

AgentCore evaluator를 bundle하는 CDK synth에는 `uv`가 필요하다. Runtime 배포까지 진행할 때만 설치한다.

```bash
python -m pip install 'uv==0.7.8'
uv --version
```

설치 확인:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
purchase-behavior-simulator simulate examples/request.json
```

로컬 demo는 외부 모델이나 AWS 리소스를 호출하지 않는다.

`examples/request.json`은 현재 게임 상태 입력을 포함한다.

| `game_state` 필드          | 의미                                       |
| -------------------------- | ------------------------------------------ |
| `currency_balance`         | 현재 구매에 사용할 수 있는 재화 또는 예산  |
| `progression_need`         | 현재 progression 문제를 해결할 필요의 강도 |
| `recent_failure_intensity` | 최근 실패가 현재 행동에 미치는 강도        |
| `inventory_overlap`        | target product와 보유 상품의 중복 정도     |
| `event_urgency`            | 이벤트 종료 등 지금 행동해야 할 이유       |
| `purchase_cooldown`        | 최근 구매 직후 추가 구매를 미루는 정도     |
| `current_goals`            | 현재 세션 또는 progression 목표            |
| `owned_item_ids`           | 이미 보유한 상품 ID                        |

이 입력은 persona를 대체하지 않는다. Persona는 장기 성향을, `game_state`는 지금 구매를 실행하거나 미루는 조건을 표현한다.

기본 action graph는 `src/purchase_behavior_simulator/action_graphs/game-store-purchase.json`에 있다. 다른 게임 UX를 검증할 때는 같은 package 디렉터리에 graph JSON을 추가한다. 필드 정의, 검증 규칙과 Runtime 적용 절차는 [action-graphs.md](action-graphs.md)를 따른다.

```bash
export PURCHASE_BEHAVIOR_ACTION_GRAPH='<action-graph-file-name.json>'
purchase-behavior-simulator simulate examples/request.json
```

현재는 시간 모델을 구성하지 않는다. Runtime은 행동 순서만 반환하고 `expected_duration_seconds`는 `null`로 유지한다. Transition timing은 실제 telemetry를 연결하는 후속 단계에서 추가한다.

명령이 실패하면 [troubleshooting.md](troubleshooting.md)에서 Python 환경, model access, checkpoint와 배포 네트워크 항목을 확인한다.

## 2. 모델 기반 Full Suite

다음 항목이 필요하다.

1. AWS SDK credential chain에서 읽을 수 있는 credential
2. 실행 region에서 사용할 수 있는 model ID
3. 해당 model의 inference 권한
4. OpenAI 계열 model을 사용할 경우 Bedrock Mantle 접근 권한

먼저 credential과 account를 확인한다.

```bash
export AWS_REGION='us-east-1'
export PURCHASE_BEHAVIOR_MODEL_ID='openai.gpt-5.6-sol'

aws sts get-caller-identity
```

공개 compact report는 `us-east-1`의 `openai.gpt-5.6-sol`로 생성했다. 같은 model과 region을 사용할 수 없으면 대상 계정에서 허용된 model ID로 바꿀 수 있지만 동일 결과를 기대하지 않는다.

먼저 1건 smoke를 실행한다. 이 명령부터 유료 model 호출이 발생할 수 있다.

```bash
PYTHONPATH=src python scripts/run_full_suite.py \
  --protocol-dir artifacts/dataset/protocol \
  --output-dir artifacts/evaluation/runs/current-200 \
  --model-id "$PURCHASE_BEHAVIOR_MODEL_ID" \
  --limit 1 \
  --workers 1 \
  --fallback-retries 1
```

다음 파일이 생성되면 simulation, action 집계와 stage report가 모두 완료된 것이다.

```text
artifacts/evaluation/runs/current-200/
  simulation/report.json
  simulation/predictions.jsonl
  action-metrics/report.json
  action-metrics/stage-report-advisory.json
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

완료된 첫 case는 checkpoint에서 재사용한다. `--confirm-model-cost`는 200건 model 호출 비용을 확인했다는 명시적 표시다. `--fallback-retries 1`은 실패 또는 fallback case가 있을 때만 해당 case를 한 번 더 호출하고 main result를 교체한 뒤 report를 다시 만든다. 재시도 후에도 정화되지 않은 case가 있으면 output root에 `retry-case-ids.txt`가 남는다.

현재 공개 결과는 이 200건 실행을 기준으로 한다. 실행 결과와 해석 범위는 [evaluation.md](evaluation.md)에 기록돼 있다.

## 3. AgentCore Runtime 배포

### AWS 배포 권한

배포에 사용하는 role 또는 user는 최소한 다음 리소스를 생성·갱신할 수 있어야 한다.

- CloudFormation stack
- AgentCore Runtime, Memory, evaluator와 online-evaluation config
- Runtime execution role과 필요한 IAM policy, `iam:PassRole`
- Lambda와 CloudWatch Logs
- EC2 security group과 VPC endpoint
- Neptune cluster와 instance
- S3 bucket과 bulk-loader Lambda를 사용할 경우 관련 loader 리소스
- Production bootstrap을 사용할 경우 S3 object write, Lambda invoke, Neptune cluster role 연결, route table과 VPC endpoint 조회·갱신

장기 access key를 저장소나 `deployment.local.json`에 넣지 않는다. AWS profile, SSO 또는 workload role처럼 SDK credential chain이 읽을 수 있는 방식을 사용한다.

### 네트워크와 모델

다음 값을 준비한다.

- AWS account ID와 region
- VPC ID
- 서로 다른 Availability Zone의 private subnet 두 개 이상
- Runtime security group
- 대상 account와 region에서 사용할 수 있는 model ID
- Neptune endpoint와 cluster ID

Runtime security group은 Neptune security group의 TCP 8182 ingress source가 된다. Private subnet에서 AWS service로 나갈 수 있도록 VPC endpoint 또는 승인된 NAT 경로가 필요하다. Template은 Bedrock Runtime과 AgentCore data-plane endpoint를 기본 생성하며, Bedrock Mantle endpoint는 사용하는 model과 subnet 지원 여부에 따라 명시적으로 활성화한다.

### Neptune

기존 Neptune cluster를 사용할 수 있다. 신규 cluster가 필요하면 [deployment.md](deployment.md)의 `Neptune 준비` 절차로 먼저 배포한다.

신규 template의 기본값:

- IAM database authentication
- storage encryption
- 7일 backup retention
- deletion protection
- CloudWatch audit-log export

CloudFormation output의 `RuntimeEndpoint`, `ClusterId`, `ClientSecurityGroupId`가 AgentCore local config에 필요하다.

### AgentCore 로컬 설정

실제 AWS 값은 Git에 넣지 않는 ignored 파일에만 기록한다.

```bash
cp deployment/agentcore/deployment.example.json \
  deployment/agentcore/deployment.local.json
```

| 필드                 | 값                                               |
| -------------------- | ------------------------------------------------ |
| `account`            | 배포 대상 12자리 AWS account ID                  |
| `region`             | Runtime, Memory, Neptune과 model을 사용할 region |
| `model_id`           | 대상 account에서 inference가 허용된 model ID     |
| `subnet_ids`         | Runtime용 private subnet IDs                     |
| `security_group_ids` | Runtime에 연결할 security group IDs              |
| `neptune_endpoint`   | 포트가 제외된 Neptune hostname                   |
| `neptune_cluster_id` | Neptune DB cluster identifier                    |

CDK dependency를 lockfile대로 설치한다.

```bash
(
  cd deployment/agentcore/agentcore/cdk
  npm ci
)
```

리소스를 변경하기 전에 account와 schema를 검증한다.

```bash
PYTHONPATH=src python scripts/deploy_agentcore.py --validate-only
```

배포와 E2E smoke:

```bash
PYTHONPATH=src python scripts/deploy_agentcore.py \
  --target default \
  --smoke
```

smoke는 dummy observation write, 같은 Memory session read, live Neptune query와 model-backed simulation을 확인한다.

## 4. 실제 운영 데이터 연결

Runtime은 자동 PII 탐지나 redaction을 제공하지 않는다. 실제 로그를 연결하기 전에 다음을 별도로 적용해야 한다.

1. 입력 schema와 field mapping 승인
2. 직접 식별자 제거 및 user/session ID 가명화
3. 허용 필드 allow-list
4. 보존 기간과 삭제 절차
5. 평가 데이터와 production import 권한 분리
6. answer key와 model inference 입력 경로 분리

외부 데이터를 canonical schema로 변환한다.

저장소의 작은 mapping 예제로 adapter 동작부터 확인할 수 있다. 이 예제는 8개 state field의 mapping과 protocol gate를 확인하는 synthetic fixture이며 성능 판단용 데이터는 아니다.

```bash
PYTHONPATH=src python scripts/configure_dataset.py \
  --config examples/dataset_adapter/config.json \
  --output-dir generated/adapter-smoke
```

예제의 8개 game-state mapping이 protocol gate까지 이어지는지도 확인할 수 있다.

```bash
PYTHONPATH=src python scripts/prepare_dataset_eval.py \
  --canonical-dir generated/adapter-smoke \
  --output-dir generated/adapter-smoke-evaluation \
  --users 2 \
  --cases-per-user 2 \
  --history-fraction 0.5 \
  --require-game-state
```

실제 데이터에서는 [`examples/dataset_adapter/config.json`](../examples/dataset_adapter/config.json)을 복사해 입력 파일과 column mapping을 변경한다. `game_state` mapping에는 `currency_balance`, `progression_need`, `recent_failure_intensity`, `inventory_overlap`, `event_urgency`, `purchase_cooldown`, `current_goals`, `owned_item_ids`를 모두 연결해야 `--require-game-state` gate를 통과할 수 있다.

```bash
PYTHONPATH=src python scripts/configure_dataset.py \
  --config '<mapping-config.json>' \
  --output-dir generated/service-canonical
```

Evaluation protocol 생성:

```bash
PYTHONPATH=src python scripts/prepare_dataset_eval.py \
  --canonical-dir generated/service-canonical \
  --output-dir generated/service-evaluation \
  --require-game-state
```

`--require-game-state`는 8개 핵심 state field의 coverage가 100%가 아니거나 수치형 state가 모두 같은 값이면 model 호출 전에 중단한다.

Production import를 만들 때는 안정적으로 보관한 HMAC salt를 환경변수로 주입한다. Salt 값은 Git, shell script 또는 artifact에 기록하지 않는다.

```bash
PURCHASE_BEHAVIOR_IDENTITY_SALT='<value-from-approved-secret-store>' \
PYTHONPATH=src python scripts/prepare_dataset_production.py \
  --canonical-dir generated/service-canonical \
  --output-dir generated/service-production \
  --as-of '<ISO-8601-timestamp>'
```

이 명령은 artifact만 만들며 AWS에 배포하지 않는다. 생성물을 검토한 뒤 Memory import와 Neptune loader를 별도 승인 절차로 실행한다.

### 로컬 bootstrap 검증

AWS 환경이 없어도 예제 CSV에서 production artifact 생성과 bootstrap 계약을 끝까지 확인할 수 있다. 예제 source는 synthetic fixture이므로 로컬 smoke에서만 `--allow-synthetic`을 사용한다.

```bash
PYTHONPATH=src python scripts/configure_dataset.py \
  --config examples/dataset_adapter/config.json \
  --output-dir generated/adapter-bootstrap-canonical

PURCHASE_BEHAVIOR_IDENTITY_SALT='local-test-only-not-production' \
PYTHONPATH=src python scripts/prepare_dataset_production.py \
  --canonical-dir generated/adapter-bootstrap-canonical \
  --output-dir generated/adapter-bootstrap-production \
  --as-of '2026-01-07T00:00:00Z' \
  --allow-synthetic

PYTHONPATH=src python scripts/bootstrap_production_data.py \
  --artifact-dir generated/adapter-bootstrap-production \
  --dry-run
```

마지막 출력의 `validated=true`, `aws_called=false`와 Memory batch, Neptune node/edge 수를 확인한다. Dry-run은 모든 Memory row를 실제 parser로 읽고 CSV schema와 manifest count를 대조하지만 AWS API는 호출하지 않는다.

### 배포 환경 bootstrap

AgentCore Runtime, Memory와 Neptune을 먼저 배포하고 dummy smoke를 통과시킨다. 그다음 실제 production artifact를 적재한다.

```bash
PYTHONPATH=src python scripts/bootstrap_production_data.py \
  --artifact-dir generated/service-production \
  --target default \
  --confirm-write
```

이 명령은 Neptune S3 upload/load, AgentCore Memory import와 imported-data canary를 순서대로 실행한다. Canary는 imported session과 다른 새 session에서 long-term Memory record를 검색하고 Neptune retrieval support가 0보다 큰지 확인한다. Long-term indexing이 끝나지 않았으면 제한된 시간 동안 polling한다.

기본 state 파일은 artifact 아래 `.bootstrap/<target>/state.json`이며 같은 명령으로 안전하게 재개한다. Artifact 간 순서는 상위 output 디렉터리의 `.bootstrap/<target>/lineage.json`에 기록된다. 다른 artifact에 기존 state를 재사용하면 fingerprint 불일치로 중단한다.

### 증분 관측 반영

첫 적재는 `--since`가 없는 snapshot artifact를 사용한다. 이후에는 이전 artifact의 `as_of`와 정확히 같은 값을 `--since`로 사용한다.

```bash
PURCHASE_BEHAVIOR_IDENTITY_SALT='<same-approved-secret>' \
PYTHONPATH=src python scripts/prepare_dataset_production.py \
  --canonical-dir generated/service-canonical \
  --output-dir generated/service-production-delta-2026-08-22 \
  --since '2026-08-21T00:00:00Z' \
  --as-of '2026-08-22T00:00:00Z'

PYTHONPATH=src python scripts/bootstrap_production_data.py \
  --artifact-dir generated/service-production-delta-2026-08-22 \
  --target default \
  --confirm-write
```

증분 artifact는 `since < timestamp <= as_of`의 관측만 포함한다. 동일 HMAC salt를 계속 사용해야 user/session pseudonym이 유지된다. Lineage가 없는 증분 artifact, 시간 구간이 끊긴 artifact와 과거 snapshot 재적재는 시작 전에 거부된다.

Neptune 증분 load는 stable node/edge ID를 사용한다. 이미 존재하는 ID는 중복 생성하지 않고 새 사용자·상품·interaction edge를 추가한다. 기존 node property 자체를 바꿔야 하는 catalog migration은 append-only 증분 범위가 아니므로 별도 graph rebuild로 수행한다.

### 배포된 Runtime 직접 호출

`examples/request.json`은 raw `SimulationRequest`다. Client가 자동으로 `simulate` operation으로 감싼다.

```bash
PYTHONPATH=src python scripts/invoke_agentcore_runtime.py \
  examples/request.json \
  --target default
```

결과를 파일로 보존할 때:

```bash
PYTHONPATH=src python scripts/invoke_agentcore_runtime.py \
  examples/request.json \
  --target default \
  --output generated/runtime-response.json
```

`record_observations`처럼 operation이 포함된 JSON도 같은 client로 전송할 수 있다. Client는 local deployment config의 account guard, CloudFormation Runtime output과 `READY` 상태를 확인한 뒤 호출한다.

기본 bucket 이름은 account, region과 target으로 안정적으로 생성한다. 조직의 bucket naming 정책이 있으면 `--data-bucket-name`, checkpoint를 별도 보관하려면 `--state-dir`을 지정한다.

```bash
PYTHONPATH=src python scripts/bootstrap_production_data.py \
  --artifact-dir generated/service-production \
  --target default \
  --data-bucket-name '<approved-private-bucket-name>' \
  --state-dir generated/bootstrap-state/default \
  --confirm-write
```

## 5. 선택형 Custom Evaluation

Evaluator resource는 함께 배포되지만 continuous evaluation과 Runtime OTel은 기본 비활성이다. 활성화 전에는 sampling, CloudWatch 보존 기간과 추가 비용을 검토한다.

On-demand 사용에는 evaluator ID, Runtime ID 또는 log group, traced session ID가 필요하다.

```bash
PYTHONPATH=src python scripts/run_agentcore_custom_evaluation.py --help
```

## 6. 구성 완료 확인

다음 항목이 통과하면 기본 셋업이 완료된 것이다.

1. Python unit tests 통과
2. `examples/request.json` 로컬 simulation 성공
3. Model-backed 평가를 사용할 경우 1건 smoke 성공
4. 200건 평가 완료 후 `retry-case-ids.txt`가 없음
5. `aws sts get-caller-identity`의 account가 local config와 일치
6. `deploy_agentcore.py --validate-only` 통과
7. Neptune stack output과 local config 일치
8. AgentCore Runtime 상태 `READY`
9. `deploy_agentcore.py --smoke` 완료
10. `bootstrap_production_data.py --dry-run` 통과
11. Production bootstrap 후 Neptune load와 Memory batch 완료
12. Imported-data canary가 새 session에서 long-term Memory와 Neptune evidence를 확인
13. `invoke_agentcore_runtime.py`로 독립적인 Runtime 호출 확인
14. 실제 데이터 사용 전 가명화·allow-list·보존 정책 승인
