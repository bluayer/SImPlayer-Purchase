# SimPlayer Purchase AgentCore 배포

이 디렉터리는 SimPlayer Purchase 기술 프로토타입의 AgentCore Runtime, Memory와 optional evaluator 선언을 담는다.

아래 명령은 모두 저장소 루트에서 실행한다.

## 환경 설정

Git에 포함된 `agentcore/agentcore.json`과 `agentcore/aws-targets.json`에는 실제 AWS 계정, VPC와 endpoint를 넣지 않는다.

```bash
cp deployment/agentcore/deployment.example.json \
  deployment/agentcore/deployment.local.json
```

`deployment.local.json`에 대상 account, region, 사용할 수 있는 model ID, subnet, security group, Neptune endpoint와 cluster ID를 입력한다. 이 파일은 `.gitignore` 대상이다.

리소스를 변경하지 않고 local 설정과 schema를 먼저 검사할 수 있다.

```bash
PYTHONPATH=src python scripts/deploy_agentcore.py --validate-only
```

## 배포

저장소 루트에서 다음 명령만 사용한다.

```bash
PYTHONPATH=src python scripts/deploy_agentcore.py \
  --target default \
  --smoke
```

`agentcore deploy`를 직접 실행하면 local 환경 설정, Memory strategy 연결, data-access IAM과 E2E smoke orchestration을 건너뛰므로 지원되는 배포 경로가 아니다.

상세 절차와 검증 범위는 [배포 문서](../../docs/deployment.md)에서 확인한다.
