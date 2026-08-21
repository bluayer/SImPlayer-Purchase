# SimPlayer Purchase 코드 맵

최종 동작을 이해할 때는 아래 순서로 읽는 것이 가장 짧다.

## 1. 요청과 응답

- `models.py`: `SimulationRequest`, 두 구매확률·행동분포·trajectory를 반환하는 `SimulationResult`와 domain value object
- `runtime_api.py`: 공개 operation인 `simulate`, `record_observations`, `evaluate_snapshot` routing
- `service.py`: 한 요청의 Memory, KG, agent, scoring orchestration

`evaluate_snapshot`은 inference-isolated answer key를 보지 않는 read-only 평가 operation이다. 일반 호출자는 `simulate`를 사용한다.

## 2. 행동 생성

- `action_rollout.py`: 설정 기반 action graph, deterministic transition과 terminal purchase path 합산. Timing metadata는 future timing model을 위한 확장 자리만 제공
- `action_graphs/`: package와 AgentCore artifact에 포함되는 게임 UX별 state/action/next-state와 terminal outcome 정의
- `decision_process.py`: 현재 게임 상태 해석, selection/commitment 분리, 내부 의도와 counterfactual 검사
- `strands_assessment.py`: probability actor, action actor, independent critic
- `hybrid_assessment.py`: probability와 action 평가의 병렬 결합
- `product_needs.py`: rational/emotional need profile

LLM은 action graph에 선언된 state별 조건부 행동분포와 현재 의도 가설을 제안한다. 결정론적 decision process가 이력을 현재 상태와 결합하고 bounded commitment gate를 적용한다. 존재하지 않는 action과 transition은 graph validation과 rollout engine이 차단한다. 기본 상점 UX가 아닌 graph에는 해당 UX의 commitment policy adapter를 별도로 연결할 수 있다.

## 3. 최종 확률 결합

- `scoring.py`: base, context, episodic, KG, actor, rollout score 결합
- `service.py`: eligibility short circuit와 최종 response 구성

Scalar와 trajectory 구매확률은 서로 다른 후보 출력이며 `SimulationResult`에 함께 포함된다. 이 값을 상품 ranking API나 production CVR로 해석하지 않는다.

## 4. Memory와 Graph

- `episodic_memory.py`: 전체 action path의 observation/transition/reflection 직렬화, reranking과 구매/비구매 대조 evidence 선택
- `episodic_reasoning.py`: self-ask query와 reflection
- `agentcore_memory.py`: AgentCore Memory data-plane adapter
- `neptune_graph.py`: 모든 관측 action edge와 상품 관계를 사용하는 PathSim-style graph evidence adapter

`SimulationRequest.interactions`는 optional recent-session hint다. 장기 이력은 AgentCore Memory와 Neptune에서 검색한다. Prediction과 counterfactual은 두 저장소에 쓰지 않고 외부에서 관측된 `state/action/next_state` 사실만 기록한다.

## 5. 데이터와 평가

- `dataset_adapter.py`: 외부 schema 정규화, temporal split, production export
- `synthetic.py`: 명시적 가정으로 label-free 사용자·상품과 외생 game-state scenario 생성
- `synthetic_labeling.py`: 시간순 balance/inventory/cooldown과 관측 action sampling
- `synthetic_oracle.py`: state-aware inference-isolated label oracle
- `state_counterfactuals.py`: 동일 case의 state perturbation pair 생성
- `evaluation.py`: holdout protocol과 공통 확률 지표
- `next_action_evaluation.py`: state/action 및 trajectory 평가
- `evaluation_summary.py`: full-suite 실행 안정성, token, latency 요약
- `production_bootstrap.py`: production artifact 검증, Memory checkpoint import와 canary 계약

Full-suite 실행 진입점은 `scripts/run_full_suite.py` 하나다. 세부 단계 스크립트는 이 명령이 호출한다.

## 6. 배포

- `agentcore_app.py`: AgentCore Runtime entrypoint
- `bootstrap.py`: production service composition
- `deployment/agentcore/agentcore/agentcore.json`: Runtime, Memory, evaluator 선언
- `deployment/neptune/`: graph infrastructure
- `scripts/deploy_agentcore.py`: ignored local AWS 설정과 생성된 Memory strategy ID 임시 주입, account guard, AgentCore, target별 data-access IAM, READY와 live smoke orchestration
- `scripts/bootstrap_production_data.py`: private S3 upload, Neptune bulk load, Memory import, checkpoint/resume와 imported-data canary
- `scripts/invoke_agentcore_runtime.py`: 배포된 AgentCore Runtime을 raw request JSON으로 직접 호출하는 client

개발 실험은 `internal/`에 격리하며 Git 저장소와 최종 Runtime에서 제외한다.
