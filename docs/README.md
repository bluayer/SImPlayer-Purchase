# SimPlayer Purchase 문서

> **Prototype status:** 이 문서는 실제 게임 telemetry로 보정된 production 구매 예측 모델이 아니라, 게임 상태와 사용자 이력을 사용해 구매 관련 행동 경로를 생성하는 기술 프로토타입을 설명한다.

문서에서 `Runtime`은 별도 수식어가 없으면 **Amazon Bedrock AgentCore Runtime**을 뜻한다. AgentCore Runtime은 foundation model이 아니라 SimPlayer Purchase의 Python orchestration application을 실행하는 관리형 serving 환경이다. 로컬 실행은 이 endpoint를 호출하지 않고 같은 domain service를 process 안에서 실행한다.

| 문서 | 설명 |
| --- | --- |
| [getting-started.md](getting-started.md) | 로컬 실행, model 평가, AWS 배포, Memory·Neptune 데이터 초기화 |
| [architecture.md](architecture.md) | 기술 스택의 역할과 선택 이유, 시스템 경계, 요청 흐름, scoring과 Memory/KG 설계 |
| [action-graphs.md](action-graphs.md) | 게임별 행동·이벤트 경로 추가와 Runtime 적용 |
| [evaluation.md](evaluation.md) | Graph v3 구매 경로 200건 탐색 평가의 사용자 action별 관측·기대 건수 차이 |
| [synthetic-data.md](synthetic-data.md) | 가정 기반 game state, scenario/label 격리와 품질 gate |
| [deployment.md](deployment.md) | AgentCore, Memory, Neptune, evaluator 배포와 E2E 검증 |
| [code-map.md](code-map.md) | 핵심 코드의 책임과 추천 탐색 순서 |
| [troubleshooting.md](troubleshooting.md) | 설치, model 평가, checkpoint와 배포 오류 해결 |

문서의 purchase probability는 SimPlayer Purchase가 생성하는 여러 출력 중 terminal propensity를 뜻하며, 프로젝트 자체를 CVR 예측 모델로 정의하지 않는다.
