# SimPlayer Purchase CDK 프로젝트

AgentCore CLI가 관리하는 CDK 프로젝트다. `@aws/agentcore-cdk` L3 construct로 `PurchaseBehaviorSimulator` Runtime, Memory와 평가 리소스를 AWS에 배포한다.

## 구조

- `bin/cdk.ts`: `agentcore/` 설정을 읽고 deployment target별 stack을 생성한다.
- `lib/cdk-stack.ts`: `AgentCoreApplication` L3 construct를 감싸는 `AgentCoreStack`을 정의한다.
- `test/cdk.test.ts`: stack synthesis 계약을 검증한다.

## 검증 명령

저장소 루트에서 실행한다.

```bash
(
  cd deployment/agentcore/agentcore/cdk
  npm ci
  npm run build
  npm run test
  npm run cdk -- synth
)
```

## 배포

이 디렉터리에서 `cdk deploy` 또는 `agentcore deploy`를 직접 실행하지 않는다. 저장소 루트의 배포 스크립트가 local AWS 설정, Memory strategy, data-access IAM과 E2E smoke를 함께 처리한다.

저장소 루트에서 실행한다.

```bash
PYTHONPATH=src python scripts/deploy_agentcore.py \
  --target default \
  --smoke
```

전체 절차는 [배포 문서](../../../../docs/deployment.md)에서 확인한다.
