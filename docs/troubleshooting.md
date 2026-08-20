# SimPlayer Purchase 문제 해결

모든 명령은 저장소 루트에서 실행한다.

## `python3.14`를 찾을 수 없음

이 저장소의 Python package는 `>=3.14,<3.15`를 요구한다.

```bash
python3.14 --version
```

Python 3.14 설치 후 virtual environment를 다시 만든다.

```bash
python3.14 -m venv .venv314
source .venv314/bin/activate
python -m pip install -e src
```

## Package 또는 module을 찾을 수 없음

Virtual environment 활성화와 editable install을 확인한다.

```bash
source .venv314/bin/activate
python -m pip show purchase-behavior-simulator
purchase-behavior-simulator simulate examples/request.json
```

Repository script는 문서처럼 `PYTHONPATH=src`를 붙여 실행한다.

## Model-backed 평가가 시작되지 않음

먼저 credential, region과 model ID를 확인한다.

```bash
aws sts get-caller-identity
printf '%s\n' "$AWS_REGION"
printf '%s\n' "$PURCHASE_BEHAVIOR_MODEL_ID"
```

`AccessDenied`이면 대상 account/region의 model access와 inference policy를 확인한다. OpenAI 계열 model은 Bedrock Mantle 접근 권한도 필요하다. `ConnectionError` 또는 timeout이면 같은 output directory로 명령을 다시 실행해 완료 checkpoint부터 재개한다.

## 평가 checkpoint가 선택한 case와 맞지 않음

하나의 output directory에는 같은 protocol, model과 case 범위를 사용한다. 이미 200건을 실행한 디렉터리를 `--limit 1` 같은 더 작은 범위에 재사용하면 checkpoint 외 case 오류가 발생한다.

- 1건에서 200건으로 늘릴 때: 같은 output directory 사용
- 다른 model, protocol 또는 더 작은 범위를 실행할 때: 새 output directory 사용

## `retry-case-ids.txt`가 남음

`--fallback-retries 1`을 사용하면 실패 또는 fallback case를 한 번 자동 재호출하고 main prediction을 교체한다. 파일이 여전히 남아 있으면 transient provider 문제가 반복된 것이다.

같은 명령에서 `--fallback-retries` 값을 늘려 재실행할 수 있지만 추가 model 비용이 발생한다. 반복 실패 case는 파일의 case ID와 `simulation/predictions.jsonl`의 `error`, observable trace를 확인한다.

## Custom action graph가 로드되지 않음

파일명만 지정할 때는 JSON이 package 디렉터리에 있어야 한다.

```bash
ls src/purchase_behavior_simulator/action_graphs/

PURCHASE_BEHAVIOR_ACTION_GRAPH=my-game-store.json \
  purchase-behavior-simulator simulate examples/request.json
```

응답의 `action_graph_id`와 `action_graph_version`을 확인한다. Runtime 배포에서는 `deployment/agentcore/agentcore/agentcore.json`의 동일 환경변수도 변경해야 한다.

## `deploy_agentcore.py --validate-only`가 실패함

다음을 확인한다.

1. `deployment/agentcore/deployment.local.json` 존재
2. 설정 account와 `aws sts get-caller-identity` account 일치
3. model ID가 placeholder가 아님
4. subnet/security group이 대상 region에 존재
5. Neptune endpoint와 cluster ID가 CloudFormation output과 일치

이 명령은 STS를 조회하지만 Runtime 리소스를 변경하지 않는다.

## Runtime smoke가 Neptune 단계에서 실패함

- Runtime security group이 Neptune client security group인지 확인한다.
- Neptune security group의 TCP 8182 ingress source를 확인한다.
- Runtime private subnet에서 필요한 AWS service endpoint 또는 승인된 NAT 경로가 있는지 확인한다.
- Neptune IAM database authentication과 Runtime role의 data-access policy를 확인한다.

수정 후 표준 배포 명령을 다시 실행한다.

```bash
PYTHONPATH=src python scripts/deploy_agentcore.py \
  --target default \
  --smoke
```

## Production bootstrap dry-run이 실패함

- `manifest.json`, `memory_import.jsonl`, `catalog_items.jsonl`, `neptune/nodes.csv`, `neptune/edges.csv`가 같은 artifact 디렉터리에 있는지 확인한다.
- Manifest의 `contains_answer_key`, `contains_oracle_probability`, `contains_model_predictions`가 모두 `false`여야 한다.
- 실제 데이터 export에서는 `--allow-synthetic`을 사용하지 않는다.
- `memory_import.jsonl`의 source는 `historical_import`이어야 한다.

## Neptune bootstrap load가 실패함

- Loader stack의 S3 bucket, Lambda와 Neptune load role이 생성됐는지 확인한다.
- Neptune cluster에 load role이 연결됐는지 확인한다.
- Private subnet route table에 S3 gateway endpoint가 있는지 확인한다.
- Loader Lambda가 사용하는 security group이 Neptune TCP 8182 ingress source인지 확인한다.
- 실패한 load ID와 상태는 artifact 아래 `.bootstrap/<target>/state.json`에 남는다.

원인을 고친 뒤 새 Neptune load job을 시작한다.

```bash
PYTHONPATH=src python scripts/bootstrap_production_data.py \
  --artifact-dir generated/service-production \
  --target default \
  --confirm-write \
  --restart-neptune-load
```

`Couldn't find the aws credential for iam_role_arn`이 나오면 cluster의 `AssociatedRoles`에서 loader role 상태가 `ACTIVE`인지 확인한다. Bootstrap 명령은 role이 ACTIVE가 될 때까지 기다린 뒤 load를 시작한다.

## 증분 artifact가 lineage 오류로 중단됨

- 첫 artifact는 `--since` 없는 snapshot이어야 한다.
- 다음 artifact의 `--since`는 이전 성공 artifact의 `as_of`와 정확히 같아야 한다.
- Snapshot과 delta에 동일한 identity salt를 사용한다.
- 다른 환경이나 새 graph를 초기화할 때만 별도의 `--lineage-path`를 사용한다.

## Memory bootstrap 중 연결이 끊김

같은 artifact, target과 state 경로로 동일 명령을 다시 실행한다. 완료된 Memory batch는 checkpoint에서 건너뛴다.

```bash
PYTHONPATH=src python scripts/bootstrap_production_data.py \
  --artifact-dir generated/service-production \
  --target default \
  --confirm-write
```

Checkpoint를 삭제하면 이미 성공한 short-term event가 다시 기록될 수 있다. Artifact fingerprint가 다른 state 파일은 자동으로 거부된다.

## 삭제 또는 재배포가 막힘

Neptune deletion protection은 기본 활성이다. Stack 삭제 전 protection 해제와 snapshot 보존 여부를 먼저 결정한다. Runtime 또는 Memory 이름을 바꾸면 새 리소스가 생성될 수 있으므로 [deployment.md](deployment.md)의 재배포 절차를 따른다.
