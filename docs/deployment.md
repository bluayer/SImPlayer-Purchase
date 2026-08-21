# SimPlayer Purchase 배포와 운영

이 문서는 SimPlayer Purchase의 AWS 배포 경계와 E2E 검증 절차를 설명한다. 배포 대상은 foundation model 자체가 아니라 Python orchestration application을 실행하는 **Amazon Bedrock AgentCore Runtime**이다. 이후 `Runtime`은 이 AgentCore Runtime을 뜻한다. 배포 결과는 추천, 랭킹 또는 production-calibrated CVR API가 아니라 기술 검증용 행동 시뮬레이터다.

모든 명령은 저장소 루트에서 실행한다. 다른 디렉터리에서 실행해야 하는 명령은 subshell로 표시한다.

## 리소스

| 리소스 | 선언 이름 | 역할 |
| --- | --- | --- |
| AgentCore Runtime | `PurchaseBehaviorSimulator` | simulation 요청 처리 |
| AgentCore Memory | `PurchaseBehaviorSimulatorMemory` | 관측 episode와 transition 저장 |
| Code evaluator | `PurchaseBehaviorRuntimeHealth` | error, 5xx, fallback span 탐지 |
| Online evaluation | `RuntimeHealthOptional` | opt-in 1% 운영 평가 |
| Neptune | `purchase-behavior-neptune` stack | 사용자 행동과 상품 관계 검색 |

Runtime과 Memory의 AWS ID는 배포 시 생성된다. 애플리케이션은 CLI/CDK가 주입하는 환경변수로 ID를 발견하며, 문서나 코드에 물리 ID를 고정하지 않는다.

## 소스 위치

```text
deployment/agentcore/
  deployment.example.json        Git에 포함되는 환경 설정 예제
  deployment.local.json          실제 계정/VPC 설정, Git 제외
  agentcore/agentcore.json       AgentCore 선언
  agentcore/aws-targets.json     안전한 placeholder target
  agentcore/cdk/                 생성된 CDK project
  evaluators/runtime-health/     optional custom evaluator

deployment/neptune/
  neptune-serverless.yaml
  neptune-bulk-loader.yaml
  runtime-data-access-policy.yaml
```

`agentcore/agentcore.json`이 Runtime, Memory, evaluator의 논리적 source of truth다. 계정, subnet, security group과 Neptune 물리 endpoint는 ignored `deployment.local.json`에서 읽어 배포 중에만 주입한다. 모델 ID 역시 대상 계정에서 사용 권한이 있는 값으로 local config에 지정한다. 생성된 `cdk/lib/agentcore-stack.ts`를 직접 편집하지 않는다.

## 배포 전 검증

Python 3.14 환경과 `uv 0.7.8`에서 실행한다. `uv`는 evaluator Lambda dependency bundle에 사용된다.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/deploy_agentcore.py --validate-only

(
  cd deployment/agentcore/agentcore/cdk
  npm ci
  npm run build
  npm run test
  npm run cdk -- synth
)
```

현재 저장소는 AgentCore CLI `0.27.0`과 lockfile의 AWS CDK CLI `2.1126.0`으로 검증했다. `npm ci`는 저장소의 lockfile을 그대로 사용하므로 개발 시점과 같은 CDK dependency tree를 재현한다.

## Neptune 준비

AgentCore 배포 스크립트는 Runtime과 Memory를 배포하지만 Neptune cluster 자체는 생성하지 않는다. 대상 VPC에 이미 Neptune cluster가 있으면 endpoint와 cluster ID를 `deployment.local.json`에 사용한다.

신규 private cluster가 필요하면 먼저 다음 template을 배포한다.

```bash
aws cloudformation deploy \
  --template-file deployment/neptune/neptune-serverless.yaml \
  --stack-name purchase-behavior-neptune \
  --parameter-overrides \
    VpcId='<vpc-id>' \
    SubnetIds='<private-subnet-a>,<private-subnet-b>'
```

기본값은 storage encryption, IAM database authentication, 7일 backup, deletion protection과 audit-log export를 활성화한다. 사용자 관리형 KMS key가 필요하면 `KmsKeyId`도 지정한다. NAT 없이 OpenAI 계열 Bedrock Mantle model을 사용한다면 지원되는 subnet과 함께 `CreateBedrockMantleEndpoint=true`를 지정해야 한다.

CloudFormation output의 `RuntimeEndpoint`, `ClusterId`, `ClientSecurityGroupId`를 local deployment config에 입력한다. 이 template과 AgentCore 리소스는 AWS 비용을 발생시킨다. 삭제 전에는 deletion protection을 명시적으로 해제하고 snapshot 보존 정책을 검토해야 한다.

Output 확인:

```bash
aws cloudformation describe-stacks \
  --stack-name purchase-behavior-neptune \
  --query 'Stacks[0].Outputs' \
  --output table
```

## AgentCore 배포

```bash
cp deployment/agentcore/deployment.example.json \
  deployment/agentcore/deployment.local.json
```

복사한 파일에 대상 AWS 환경과 해당 계정에서 사용할 수 있는 model ID를 입력한다. 배포 스크립트는 STS account와 설정 파일의 account가 다르면 리소스를 변경하기 전에 실패한다.

리소스를 변경하지 않는 사전 검증:

```bash
PYTHONPATH=src python scripts/deploy_agentcore.py --validate-only
```

```bash
PYTHONPATH=src python scripts/deploy_agentcore.py \
  --target default \
  --smoke
```

리소스 이름을 변경하면 CloudFormation 관점에서 새 Runtime 또는 Memory가 생성될 수 있다. 배포 스크립트는 다음 순서로 이를 처리한다.

1. local config와 현재 AWS credential의 account가 같은지 확인한다.
2. account, region, VPC와 Neptune 값을 배포 구간의 임시 config에 주입한다.
3. 기존 target이 있으면 현재 Memory의 active EPISODIC/SEMANTIC strategy ID를 조회한다.
4. AgentCore Runtime, Memory와 evaluator를 배포한다.
5. Memory가 새로 생성돼 strategy ID가 달라졌으면 새 ID로 Runtime을 한 번 더 갱신한다.
6. 원본 config를 byte 단위로 복원해 물리 ID가 Git source에 남지 않게 한다.
7. 새 Runtime role과 Memory ID로 target별 최소권한 stack을 생성 또는 갱신한다.
8. Runtime `READY`를 기다린 후 observation write/read와 model smoke를 수행한다.

target 이름은 AgentCore stack과 data-access stack 이름에 모두 포함된다. 예를 들어 `--target default`는 `AgentCore-BehaviorSim-default`와 `BehaviorSim-default-runtime-data-access`를 관리한다.

다른 action graph를 Runtime에 배포하려면 graph JSON을 package의 `action_graphs/`에 추가하고 `agentcore/agentcore.json`의 `PURCHASE_BEHAVIOR_ACTION_GRAPH` 값을 변경한다. 상세 절차는 [action-graphs.md](action-graphs.md)를 따른다.

## E2E Smoke

실제 사용자 이력 대신 `artifacts/smoke/current/request.json`의 dummy fixture를 사용한다.

1. Runtime 상태가 `READY`인지 확인한다.
2. 매 실행마다 고유한 dummy actor와 Runtime session ID를 생성한다.
3. dummy observation을 `record_observations`로 기록한다.
4. 같은 Memory session을 지정해 `simulate`를 호출한다.
5. 기록한 행동 수만큼 `episodic_memory_events`가 반환되는지 확인한다.
6. live Neptune query와 actor/critic model simulation이 HTTP 200으로 끝나는지 확인한다.
7. SDK connect/read timeout과 최대 재시도 횟수 안에 끝나지 않으면 배포를 실패시킨다.

smoke는 production 데이터를 쓰지 않으며, dummy actor/session은 별도의 namespace를 사용한다.

## Production 데이터 초기화

`deploy_agentcore.py --smoke`가 성공해도 Memory와 Neptune에는 smoke용 dummy record 외에 실제 dataset이 없다. 실제 이력과 상품 관계를 사용하려면 label-free production artifact를 별도로 준비하고 적재한다.

로컬에서 artifact를 검증한다. 이 명령은 AWS를 호출하지 않는다.

```bash
PYTHONPATH=src python scripts/bootstrap_production_data.py \
  --artifact-dir generated/service-production \
  --dry-run
```

검증 후 원격 data plane 변경을 명시적으로 승인한다.

```bash
PYTHONPATH=src python scripts/bootstrap_production_data.py \
  --artifact-dir generated/service-production \
  --target default \
  --confirm-write
```

명령은 다음 순서로 실행한다.

1. Manifest가 answer key, oracle probability와 model prediction을 제외하는지 확인한다.
2. 모든 `memory_import.jsonl` row를 `ObservationBatch` 계약으로 검증한다.
3. Neptune node/edge CSV header와 manifest 행 수를 대조한다.
4. target 전용 private S3 bucket과 bulk-loader stack을 생성 또는 갱신한다.
5. 필요한 경우 private subnet route table에 S3 gateway endpoint를 생성한다.
6. Neptune load role을 cluster에 연결하고 artifact fingerprint별 S3 prefix를 load한다.
7. Runtime의 `record_observations` operation으로 Memory batch를 순차 적재한다.
8. imported actor를 새 session에서 호출해 long-term Memory record와 Neptune graph evidence가 실제 응답에 사용되는지 확인한다.

기본 checkpoint는 `generated/service-production/.bootstrap/default/state.json`에 저장된다. 같은 artifact와 target으로 재실행하면 완료된 Neptune load와 Memory batch를 건너뛴다. Artifact 간 lineage는 output parent의 `.bootstrap/default/lineage.json`에 저장되며 snapshot 이후 연속된 incremental artifact만 허용한다. Loader가 terminal failure로 끝나 새 job이 필요할 때만 `--restart-neptune-load`를 추가한다.

Checkpoint에는 bucket, loader job과 canary 식별자가 들어가므로 Git에 포함하지 않는다. 저장소의 `.gitignore`는 `.bootstrap/` 디렉터리를 제외한다.

Bootstrap 실행 principal에는 Runtime 배포 권한과 별도로 다음 data initialization 권한이 필요하다.

- loader CloudFormation stack 생성·갱신
- private S3 bucket 생성과 CSV object write
- `ec2:DescribeRouteTables`, `ec2:DescribeVpcEndpoints`와 필요한 route association 갱신
- Neptune cluster 조회와 load role 연결
- loader Lambda invoke
- AgentCore Runtime invoke

기본 bucket 이름을 사용할 수 없는 환경에서는 `--data-bucket-name`으로 승인된 private bucket 이름을 지정한다. Loader template은 bucket encryption, public-access block, versioning과 retain policy를 적용한다.

```bash
PYTHONPATH=src python scripts/bootstrap_production_data.py \
  --artifact-dir generated/service-production \
  --target default \
  --confirm-write \
  --restart-neptune-load
```

Memory long-term record write는 deterministic request identifier와 client token을 사용하지만, short-term event import는 network timeout 직후 결과를 확인할 수 없는 at-least-once 경계다. Checkpoint 파일을 삭제하거나 다른 위치로 옮기지 않고 동일 명령으로 재개한다.

증분 관측은 이전 `as_of`를 다음 artifact의 exclusive `--since`로 사용한다. 동일 identity salt를 유지하며 새 observation과 stable-ID graph row만 append한다. Existing Neptune node property 변경은 append-only load로 갱신되지 않으므로 catalog schema/property migration은 별도 graph rebuild 절차로 처리한다.

## Runtime 직접 호출

배포 스크립트 내부 smoke와 별도로 application integration에서 사용할 수 있는 client가 있다.

```bash
PYTHONPATH=src python scripts/invoke_agentcore_runtime.py \
  examples/request.json \
  --target default
```

Client는 target stack에서 Runtime ARN을 찾고 `READY` 상태와 AWS account를 확인한다. `--runtime-arn`으로 명시적 ARN을 넘길 수도 있고 `--output`으로 JSON 응답을 저장할 수 있다.

## 검증된 bootstrap E2E

2026-08-21에 격리된 synthetic fixture로 다음 경로를 실제 AWS에서 검증했다.

- Initial snapshot: Neptune 16 node/40 edge load, Memory 12 batch/22 events/36 long-term records
- New-session canary: long-term Memory 11 records와 Neptune retrieval support 확인
- Incremental lineage: snapshot 6 batch 뒤 연속 delta 6 batch를 같은 lineage에 추가
- Incremental canary: 누적 long-term Memory와 새 graph interaction 확인
- Standalone client: scalar/trajectory 확률, action distributions와 trajectories 응답 확인

이 검증은 데이터 적재와 연동 경로의 동작을 확인한 것이며 실제 게임 데이터의 성능을 의미하지 않는다.

## Memory 계약

Memory 쓰기와 simulation은 분리된 operation이다.

```text
record_observations
  -> source allow-list 검증
  -> deterministic reflection 생성
  -> observed transition 생성
  -> observation/reflection/transition record 저장

simulate
  -> self-ask query
  -> semantic search + recent records
  -> deterministic reranking
  -> actor/grounding/critic
  -> 행동분포와 trajectory 반환
```

허용 source는 `external_observation`, `historical_import`, `experiment_observation`이다. 모델 prediction, synthetic label, counterfactual 결과는 Memory에 기록하지 않는다.

`SimulationRequest.interactions`는 optional recent-session hint다. Production bootstrap은 노출, 상세 열기, 구매 시작, 확인, 결제 성공·실패, 잔액 부족과 충전 action의 전체 `state/action/next_state` transition을 AgentCore Memory에 적재한다. Request history가 비어 있어도 long-term Memory retrieval이 장기 관측 경로를 보완한다.

observation ingestion은 안정적인 write path를 위해 기본적으로 deterministic reflection을 사용한다. LLM reflection이 필요한 실험에서만 `PURCHASE_BEHAVIOR_REFLECTION_MODE=llm`으로 명시적으로 활성화한다.

## Neptune

- IAM database authentication을 사용한다.
- storage encryption과 7일 backup retention을 적용한다.
- deletion protection과 CloudWatch audit-log export는 기본 활성이다.
- `KmsKeyId`를 지정하면 사용자 관리형 KMS key를 사용하고, 비워 두면 AWS managed encryption을 사용한다.
- Runtime role에는 필요한 read와 검증된 record write 권한만 부여한다.
- `bedrock-mantle:CallWithBearerToken`은 resource-level scope를 지원하지 않아 유일하게 `Resource: '*'`를 사용하며 `SHORT_TERM` token 조건으로 제한한다. 실제 inference 생성 권한은 대상 계정의 `project/default` ARN으로 제한한다.
- bulk loader는 별도 stack과 S3 prefix를 사용한다.
- Production export는 관측된 각 action을 별도 user-item edge로 저장하고 `state`와 `nextState`를 함께 기록한다.
- bootstrap loader bucket은 encryption, public-access block과 versioning을 사용한다.
- private VPC에 S3 gateway endpoint가 없으면 loader stack이 대상 subnet route table에 생성한다.
- synthetic graph는 운영 graph와 namespace 및 load manifest를 분리한다.

## 선택형 Custom Evaluation

Custom Evaluation은 offline full suite를 대체하지 않는다.

- 기본 상태: evaluator resource 선언, online evaluation 비활성
- Runtime OTel: 기본 비활성
- sampling: 활성화 시 1%
- 판단 범위: error status, exception, HTTP 5xx, explicit fallback
- 제외 범위: prompt/response 본문, answer key, private chain-of-thought

on-demand 실행:

```bash
PYTHONPATH=src python scripts/run_agentcore_custom_evaluation.py --help
```

평가 결과는 scoring이나 Memory에 되먹임하지 않는다.

## 배포 확인 기준

- Python 단위 및 계약 테스트 통과
- AgentCore schema validation 통과
- TypeScript build와 CDK synth 통과
- Runtime `READY`
- dummy observation write와 동일 Memory session read
- live Neptune query와 실제 model `simulate` HTTP 200
- production artifact dry-run과 bootstrap canary 통과
- `simulate` 응답에 scalar/trajectory 구매확률, action distribution, likely trajectory 포함
- local model-backed graph v3 long-path evaluation 200/200 성공, final fallback 0

소스나 action graph가 변경된 뒤에는 과거 배포 상태를 근거로 삼지 않는다. 최종 artifact는 반드시 다음 명령으로 대상 계정에서 다시 검증한다.

```bash
PYTHONPATH=src python scripts/deploy_agentcore.py \
  --target default \
  --smoke
```
