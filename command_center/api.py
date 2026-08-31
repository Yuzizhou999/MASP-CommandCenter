from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent_protocol import AgentBudgets
from .agent_run_manager import TERMINAL_AGENT_RUN_STATUSES, AgentRunManager
from .approvals import ApprovalStore
from .audit import AuditStore
from .auth import (
    OperatorIdentity,
    is_protected,
    operator_dependency,
    token_matches,
)
from .benchmark import BenchmarkRunner
from .clarifications import ClarificationResolver, ClarificationStore
from .contracts import (
    AgentConversationMemory,
    AgentRunCreateRequest,
    AgentRunRecord,
    AgentRunResumeRequest,
    ApprovalDecision,
    ApprovalRequest,
    BenchmarkRequest,
    ChatRequest,
    ChatResponse,
    ComparisonRequest,
    ComparisonResult,
    DatasetExportRequest,
    DeadlockInjectionRequest,
    DispatchIntent,
    FaultInjectionRequest,
    IncidentApprovalRequest,
    IncidentRecord,
    IncidentType,
    IncidentWhatIfRequest,
    IntentType,
    ModelEvaluationRequest,
    PlanExplanationReport,
    PlanExplanationRequest,
    ResourceBlockDraft,
    SimulationRequest,
    SimulationSummary,
    WorkstationInjectionRequest,
)
from .dataset_exports import DatasetExporter
from .dispatch_workflow import DispatchWorkflowService
from .engine_adapter import EngineVersionError, MaspAdapter
from .explanations import PlanExplanationService
from .incidents import IncidentService, IncidentStore
from .intent_store import IntentStore
from .knowledge import KnowledgeBase
from .llm_provider import create_llm_provider
from .logging_setup import configure_logging, get_logger
from .model_evaluation import ModelSafetyEvaluator
from .orchestrator import DispatchOrchestrator
from .scenario_drafts import ScenarioDraftConflict, ScenarioDraftStore
from .settings import Settings

settings = Settings.load()
engine = MaspAdapter(settings)
audit = AuditStore(settings.data_dir / "audit.jsonl")
approvals = ApprovalStore(settings.data_dir / "approvals.json")
intents = IntentStore(settings.data_dir / "committed-intents.json")
knowledge = KnowledgeBase(settings.root / "knowledge")
provider = create_llm_provider(settings)
clarification_store = ClarificationStore(settings.data_dir / "clarifications.json")
clarification_resolver = ClarificationResolver(clarification_store, engine)
orchestrator = DispatchOrchestrator(
    engine=engine,
    provider=provider,
    knowledge=knowledge,
    audit=audit,
    clarifications=clarification_resolver,
    runtime_mode=settings.agent_runtime_mode,
    budgets=AgentBudgets(
        maxDecisions=settings.agent_max_decisions,
        maxToolCalls=settings.agent_max_tool_calls,
        maxRepairAttempts=settings.agent_max_repair_attempts,
        maxTotalTokens=settings.agent_max_total_tokens,
        maxEstimatedCostUsd=settings.agent_max_estimated_cost_usd,
        maxLatencyMs=settings.agent_max_latency_ms,
        maxSteps=settings.agent_max_steps,
    ),
)
dispatch_workflow = DispatchWorkflowService(
    engine=engine,
    approvals=approvals,
    intents=intents,
    audit=audit,
)
agent_runs = AgentRunManager(
    settings.data_dir / "agent-runs.json",
    orchestrator=orchestrator,
    provider=provider,
    workflow=dispatch_workflow,
)
incident_store = IncidentStore(settings.data_dir / "incidents.json")
incident_service = IncidentService(
    store=incident_store,
    engine=engine,
    provider=provider,
    knowledge=knowledge,
    audit=audit,
)
scenario_drafts = ScenarioDraftStore(settings.data_dir, engine, audit)
benchmarks = BenchmarkRunner(settings.data_dir, engine, audit)
model_evaluations = ModelSafetyEvaluator(
    settings.data_dir,
    suite_path=settings.root / "evals" / "model-safety-v1.json",
    provider=provider,
    knowledge=knowledge,
    audit=audit,
)
dataset_exports = DatasetExporter(
    settings.data_dir,
    engine=engine,
    audit=audit,
    approvals=approvals,
    intents=intents,
    incidents=incident_store,
)
plan_explanations = PlanExplanationService(
    engine=engine,
    provider=provider,
    audit=audit,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    logger = get_logger("api")
    logger.info(
        "command center starting",
        extra={
            "environment": settings.app_env,
            "runtimeMode": settings.agent_runtime_mode,
            "llmProvider": settings.llm_provider,
            "apiTokenEnabled": settings.api_token is not None,
        },
    )
    agent_runs.start()
    try:
        yield
    finally:
        agent_runs.shutdown()
        logger.info("command center stopped")


app = FastAPI(
    title="保利智仓·灵枢 API",
    version="0.1.0",
    description="大模型调度智能体、MASP数字孪生和确定性安全边界。",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)


current_operator = Depends(operator_dependency(settings))


@app.middleware("http")
async def enforce_api_token(request: Request, call_next):
    """配置 token 后，拦下所有未携带有效 token 的变更类请求。

    身份覆盖由 current_operator 依赖负责，这里只负责准入。两者分开是因为
    路由目前直接挂在 app 上而不是 APIRouter，无法用 router 级 dependencies。
    """
    if is_protected(request.method, request.url.path) and not token_matches(
        settings, request.headers.get("authorization")
    ):
        return JSONResponse(
            status_code=401,
            content={"detail": "缺少或无效的 Authorization: Bearer <token>。"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


@app.get("/api/health")
def health() -> dict[str, Any]:
    status = engine.engine_status()
    model_status = provider.status()
    return {
        "status": "ok" if status["allowed"] else "degraded",
        "environment": settings.app_env,
        "engine": status,
        "model": model_status,
        "agentRuntime": {
            "mode": settings.agent_runtime_mode,
            "strategy": (
                "ACTION_PROTOCOL_LOOP"
                if settings.agent_runtime_mode == "loop"
                else "LINEAR_PIPELINE"
            ),
            "budgets": {
                "maxDecisions": settings.agent_max_decisions,
                "maxToolCalls": settings.agent_max_tool_calls,
                "maxRepairAttempts": settings.agent_max_repair_attempts,
                "maxTotalTokens": settings.agent_max_total_tokens,
                "maxEstimatedCostUsd": settings.agent_max_estimated_cost_usd,
                "maxLatencyMs": settings.agent_max_latency_ms,
                "maxSteps": settings.agent_max_steps,
            },
            "storageNamespace": settings.data_dir.name,
        },
        "agentPolicy": engine.agent_model_status().model_dump(
            by_alias=True, mode="json"
        ),
        "safety": {
            "mode": "simulation-only",
            "fieldExecutionEnabled": False,
            "approvalBoundaryEnabled": True,
            # 如实暴露鉴权状态：未配置 token 时任何能访问端口的人都能提交
            # 变更并自称任意审批人身份，这一点不应被隐藏。
            "apiTokenEnabled": settings.api_token is not None,
            "approverIdentityTrusted": settings.api_token is not None,
        },
    }


@app.get("/api/v1/agent-policy")
def agent_policy_status() -> dict[str, Any]:
    return engine.agent_model_status().model_dump(by_alias=True, mode="json")


@app.post("/api/v1/evaluations/benchmarks")
async def run_benchmark(request: BenchmarkRequest) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(benchmarks.run, request)
    except (ValueError, KeyError, EngineVersionError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/v1/evaluations/benchmarks")
def list_benchmarks() -> list[dict[str, Any]]:
    return benchmarks.list()


@app.get("/api/v1/evaluations/benchmarks/{benchmark_id}")
def benchmark_detail(benchmark_id: str) -> dict[str, Any]:
    try:
        return benchmarks.get(benchmark_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/v1/evaluations/model-safety")
async def run_model_safety_evaluation(
    request: ModelEvaluationRequest,
) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(model_evaluations.run, request)
    except (ValueError, KeyError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/v1/evaluations/model-safety")
def list_model_safety_evaluations() -> list[dict[str, Any]]:
    return model_evaluations.list()


@app.get("/api/v1/evaluations/model-safety/{evaluation_id}")
def model_safety_evaluation_detail(evaluation_id: str) -> dict[str, Any]:
    try:
        return model_evaluations.get(evaluation_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/v1/dataset-exports")
async def create_dataset_export(request: DatasetExportRequest) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(dataset_exports.create, request)
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/v1/dataset-exports")
def list_dataset_exports() -> list[dict[str, Any]]:
    return dataset_exports.list()


@app.get("/api/v1/dataset-exports/{export_id}")
def dataset_export_detail(export_id: str) -> dict[str, Any]:
    try:
        return dataset_exports.get(export_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/v1/dataset-exports/{export_id}/download")
def download_dataset_export(export_id: str) -> FileResponse:
    try:
        path = dataset_exports.bundle_path(export_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return FileResponse(
        path,
        media_type="application/zip",
        filename=f"{export_id}.zip",
    )


@app.get("/api/v1/scenarios")
def list_scenarios() -> list[dict[str, Any]]:
    return engine.scenarios()


@app.post("/api/v1/scenario-drafts")
def create_scenario_draft(
    document: dict[str, Any],
    requested_by: str = Query(default="demo-operator", alias="requestedBy"),
) -> dict[str, Any]:
    try:
        return scenario_drafts.create(document, requested_by)
    except ScenarioDraftConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/v1/scenario-drafts/from-runtime")
def create_scenario_draft_from_runtime(
    scenario_id: str = Query(alias="scenarioId"),
    package_id: str | None = Query(default=None, alias="packageId"),
    version: str = Query(default="0.1.0"),
    requested_by: str = Query(default="demo-operator", alias="requestedBy"),
) -> dict[str, Any]:
    try:
        return scenario_drafts.create_from_runtime(
            scenario_id,
            package_id or f"{scenario_id}-draft",
            version,
            requested_by,
        )
    except ScenarioDraftConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (ValueError, KeyError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/v1/scenario-drafts")
def list_scenario_drafts() -> list[dict[str, Any]]:
    return scenario_drafts.list()


@app.get("/api/v1/scenario-drafts/{package_id}")
def get_scenario_draft(package_id: str) -> dict[str, Any]:
    try:
        return scenario_drafts.get(package_id)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.put("/api/v1/scenario-drafts/{package_id}")
def update_scenario_draft(
    package_id: str,
    document: dict[str, Any],
    expected_revision: int = Query(alias="expectedRevision", ge=1),
    requested_by: str = Query(default="demo-operator", alias="requestedBy"),
) -> dict[str, Any]:
    try:
        return scenario_drafts.update(
            package_id, document, expected_revision, requested_by
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ScenarioDraftConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/v1/scenario-drafts/{package_id}/validate")
def validate_scenario_draft(
    package_id: str,
    requested_by: str = Query(default="demo-operator", alias="requestedBy"),
) -> dict[str, Any]:
    try:
        return scenario_drafts.validate(package_id, requested_by)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/v1/scenario-drafts/{package_id}/generate-tasks")
def generate_scenario_draft_tasks(
    package_id: str,
    generation: dict[str, Any],
    expected_revision: int = Query(alias="expectedRevision", ge=1),
    requested_by: str = Query(default="demo-operator", alias="requestedBy"),
) -> dict[str, Any]:
    try:
        return scenario_drafts.generate_tasks(
            package_id, generation, expected_revision, requested_by
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ScenarioDraftConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/v1/scenario-drafts/{package_id}/compile")
def compile_scenario_draft(
    package_id: str,
    requested_by: str = Query(default="demo-operator", alias="requestedBy"),
) -> dict[str, Any]:
    try:
        result = scenario_drafts.compile(package_id, requested_by)
        if not result["compiled"]:
            raise HTTPException(status_code=422, detail=result["validation"])
        return result
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/v1/scenario-drafts/{package_id}/publish")
def publish_scenario_draft(
    package_id: str,
    requested_by: str = Query(default="demo-operator", alias="requestedBy"),
) -> dict[str, Any]:
    try:
        return scenario_drafts.publish(package_id, requested_by)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ScenarioDraftConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/v1/world/snapshot")
def world_snapshot(
    scenario_id: str = Query(default="interactive-multi-fleet", alias="scenarioId"),
) -> dict[str, Any]:
    try:
        return engine.world_snapshot(scenario_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/v1/map")
def map_model() -> dict[str, Any]:
    return engine.map_model()


@app.post(
    "/api/v1/agent/chat", response_model=ChatResponse, response_model_by_alias=True
)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return orchestrator.chat(request)
    except (ValueError, KeyError, EngineVersionError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post(
    "/api/v1/agent/runs",
    response_model=AgentRunRecord,
    response_model_by_alias=True,
    status_code=202,
)
def create_agent_run(
    request: AgentRunCreateRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> AgentRunRecord:
    try:
        return agent_runs.create(request, idempotency_key=idempotency_key)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get(
    "/api/v1/agent/runs/{run_id}",
    response_model=AgentRunRecord,
    response_model_by_alias=True,
)
def get_agent_run(run_id: str) -> AgentRunRecord:
    try:
        return agent_runs.get(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/v1/agent/runs/{run_id}/events")
async def stream_agent_run_events(
    run_id: str,
    last_event_id: int | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    try:
        agent_runs.get(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    async def event_stream():
        cursor = last_event_id or 0
        idle_cycles = 0
        while True:
            events = agent_runs.events_after(run_id, cursor)
            for event in events:
                cursor = int(event["eventId"])
                yield (
                    f"id: {cursor}\n"
                    "event: agent_run\n"
                    f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                )
            record = agent_runs.get(run_id)
            if record.status in TERMINAL_AGENT_RUN_STATUSES and not events:
                break
            idle_cycles += 1
            if idle_cycles % 60 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(
    "/api/v1/agent/runs/{run_id}/resume",
    response_model=AgentRunRecord,
    response_model_by_alias=True,
)
def resume_agent_run(
    run_id: str,
    decision: AgentRunResumeRequest,
    operator: OperatorIdentity = current_operator,
) -> AgentRunRecord:
    # 与审批决策同理：暂停中的 R3 run 由谁批准必须由服务端裁定。
    decided = decision.model_copy(
        update={"decided_by": operator.resolve(decision.decided_by)}
    )
    try:
        return agent_runs.resume(run_id, decided)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post(
    "/api/v1/agent/runs/{run_id}/cancel",
    response_model=AgentRunRecord,
    response_model_by_alias=True,
)
def cancel_agent_run(run_id: str) -> AgentRunRecord:
    try:
        return agent_runs.cancel(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/v1/agent/tools")
def agent_tools() -> list[dict[str, Any]]:
    return orchestrator.tool_catalog()


@app.get(
    "/api/v1/agent/memory/{conversation_id}",
    response_model=AgentConversationMemory,
    response_model_by_alias=True,
)
def agent_memory(conversation_id: str) -> AgentConversationMemory:
    memory = orchestrator.memory.get(conversation_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="当前会话没有可召回记忆")
    return memory


@app.get("/api/v1/agent/metrics")
def agent_metrics(
    recent_limit: int = Query(default=20, ge=1, le=100, alias="recentLimit"),
) -> dict[str, Any]:
    return orchestrator.observability.summary(recent_limit=recent_limit)


@app.post("/api/v1/intents/validate")
def validate_intent(
    intent: DispatchIntent,
    scenario_id: str = Query(default="interactive-multi-fleet", alias="scenarioId"),
) -> dict[str, Any]:
    return engine.validate_intent(intent, scenario_id).model_dump(
        by_alias=True, mode="json"
    )


@app.post(
    "/api/v1/simulations",
    response_model=SimulationSummary,
    response_model_by_alias=True,
)
async def simulate(request: SimulationRequest) -> SimulationSummary:
    try:
        return await asyncio.to_thread(dispatch_workflow.simulate, request)
    except (ValueError, KeyError, EngineVersionError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get(
    "/api/v1/simulations",
    response_model=list[SimulationSummary],
    response_model_by_alias=True,
)
def list_simulations() -> list[SimulationSummary]:
    return engine.list_runs()


@app.get("/api/v1/simulations/{run_id}")
def simulation_detail(run_id: str) -> dict[str, Any]:
    try:
        return engine.get_run_detail(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post(
    "/api/v1/simulations/{run_id}/explain",
    response_model=PlanExplanationReport,
    response_model_by_alias=True,
)
async def explain_simulation_plan(
    run_id: str, request: PlanExplanationRequest
) -> PlanExplanationReport:
    try:
        return await asyncio.to_thread(plan_explanations.explain, run_id, request)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post(
    "/api/v1/simulations/compare",
    response_model=ComparisonResult,
    response_model_by_alias=True,
)
def compare(request: ComparisonRequest) -> ComparisonResult:
    try:
        return engine.compare(request.run_ids)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/v1/knowledge/search")
def search_knowledge(q: str = Query(min_length=1)) -> list[dict[str, Any]]:
    return [row.model_dump(by_alias=True, mode="json") for row in knowledge.search(q)]


@app.get("/api/v1/knowledge/stats")
def knowledge_stats() -> dict[str, object]:
    return knowledge.stats()


@app.get(
    "/api/v1/incidents",
    response_model=list[IncidentRecord],
    response_model_by_alias=True,
)
def list_incidents() -> list[IncidentRecord]:
    return incident_store.list()


@app.post(
    "/api/v1/incidents/inject",
    response_model=IncidentRecord,
    response_model_by_alias=True,
)
def inject_incident(request: FaultInjectionRequest) -> IncidentRecord:
    try:
        return incident_service.inject_vehicle_fault(request)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, EngineVersionError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post(
    "/api/v1/incidents/inject/workstation",
    response_model=IncidentRecord,
    response_model_by_alias=True,
)
def inject_workstation_incident(
    request: WorkstationInjectionRequest,
) -> IncidentRecord:
    try:
        return incident_service.inject_workstation_outage(request)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, EngineVersionError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post(
    "/api/v1/incidents/inject/deadlock",
    response_model=IncidentRecord,
    response_model_by_alias=True,
)
def inject_deadlock_incident(request: DeadlockInjectionRequest) -> IncidentRecord:
    try:
        return incident_service.inject_deadlock(request)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, EngineVersionError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get(
    "/api/v1/incidents/{incident_id}",
    response_model=IncidentRecord,
    response_model_by_alias=True,
)
def incident_detail(incident_id: str) -> IncidentRecord:
    try:
        return incident_store.get(incident_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post(
    "/api/v1/incidents/{incident_id}/diagnose",
    response_model=IncidentRecord,
    response_model_by_alias=True,
)
async def diagnose_incident(
    incident_id: str,
    requested_by: str = Query(default="demo-operator", alias="requestedBy"),
) -> IncidentRecord:
    try:
        return await asyncio.to_thread(
            incident_service.diagnose,
            incident_id,
            requested_by,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post(
    "/api/v1/incidents/{incident_id}/what-if",
    response_model=IncidentRecord,
    response_model_by_alias=True,
)
async def incident_what_if(
    incident_id: str,
    request: IncidentWhatIfRequest,
) -> IncidentRecord:
    try:
        return await asyncio.to_thread(
            incident_service.run_what_if,
            incident_id,
            request.mode,
            request.requested_by,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, EngineVersionError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post(
    "/api/v1/incidents/{incident_id}/approvals",
    response_model=ApprovalRequest,
    response_model_by_alias=True,
)
def create_incident_approval(
    incident_id: str,
    request: IncidentApprovalRequest,
) -> ApprovalRequest:
    try:
        incident = incident_store.get(incident_id)
        existing_id = incident.approval_ids.get(request.mode.value)
        if existing_id:
            return approvals.get(existing_id)
        run_id = incident.what_if_run_ids.get(request.mode.value)
        if run_id is None:
            raise ValueError("处置方案尚未完成 What-if 推演，不能提交审批。")
        run = engine.get_run(run_id)
        if run.status != "COMPLETED":
            raise ValueError("处置方案推演未成功，不能提交审批。")
        revision = engine.world_revision(incident.scenario_id)
        if incident.incident_type is IncidentType.WORKSTATION_DISABLED:
            intent = DispatchIntent(
                intentType=IntentType.BLOCK_RESOURCE,
                requestedBy=request.requested_by,
                basedOnWorldRevision=revision,
                reason=f"异常 {incident_id} 的 {request.mode.value} 处置申请",
                resourceBlock=ResourceBlockDraft(
                    resourceIds=incident.resource_ids,
                    startMs=incident.fault_at_ms,
                    endMs=incident.fault_at_ms + incident.recovery_duration_ms,
                    reason=f"工位 {incident.workstation_id or incident.location_node_id} 停用处置",
                ),
            )
        else:
            intent = DispatchIntent(
                intentType=(
                    IntentType.REQUEST_RECOVERY
                    if incident.incident_type is IncidentType.DEADLOCK_RISK
                    else IntentType.REPORT_VEHICLE_FAULT
                ),
                requestedBy=request.requested_by,
                basedOnWorldRevision=revision,
                reason=f"异常 {incident_id} 的 {request.mode.value} 处置申请",
                query=f"incident={incident_id}; mode={request.mode.value}; run={run_id}",
            )
        validation = engine.validate_intent(intent, incident.scenario_id)
        if not validation.valid or not validation.approval_required:
            raise ValueError("处置意图未通过高风险审批策略校验。")
        approval = approvals.create(intent, validation, [run_id])
        incident_service.link_approval(
            incident_id, request.mode, approval.approval_id, request.requested_by
        )
        audit.append(
            trace_id=intent.intent_id,
            event_type="APPROVAL_CREATED",
            actor=request.requested_by,
            payload=approval.model_dump(by_alias=True, mode="json"),
        )
        return approval
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, EngineVersionError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/v1/incidents/{incident_id}/report")
def incident_report(incident_id: str) -> dict[str, Any]:
    try:
        incident = incident_store.get(incident_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    what_if_runs = []
    for mode, run_id in incident.what_if_run_ids.items():
        try:
            summary = engine.get_run(run_id)
        except KeyError:
            continue
        what_if_runs.append(
            {
                "mode": mode,
                "run": summary.model_dump(by_alias=True, mode="json"),
            }
        )
    return {
        "title": f"异常诊断报告 {incident.incident_id}",
        "mode": "simulation-only",
        "incident": incident.model_dump(by_alias=True, mode="json"),
        "whatIfRuns": what_if_runs,
        "safetyNotice": (
            "AI 仅解释已提供证据，工位封锁、车辆隔离和倒退恢复均为 "
            "R3 高风险仿真候选，未连接或控制真实设备。"
        ),
    }


@app.post(
    "/api/v1/approvals", response_model=ApprovalRequest, response_model_by_alias=True
)
def create_approval(
    intent: DispatchIntent,
    scenario_id: str = Query(default="interactive-multi-fleet", alias="scenarioId"),
    run_id: list[str] | None = Query(default=None, alias="runId"),
) -> ApprovalRequest:
    try:
        return dispatch_workflow.create_approval(
            intent,
            scenario_id=scenario_id,
            run_ids=run_id or [],
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, EngineVersionError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get(
    "/api/v1/approvals",
    response_model=list[ApprovalRequest],
    response_model_by_alias=True,
)
def list_approvals() -> list[ApprovalRequest]:
    return approvals.list()


@app.post(
    "/api/v1/approvals/{approval_id}/decision",
    response_model=ApprovalRequest,
    response_model_by_alias=True,
)
def decide_approval(
    approval_id: str,
    decision: ApprovalDecision,
    operator: OperatorIdentity = current_operator,
) -> ApprovalRequest:
    # 审批人身份由服务端裁定：已认证时忽略客户端提交的 decidedBy，
    # 否则任何能访问端口的人都能以任意身份批准 R3 操作。
    decided = decision.model_copy(
        update={"decided_by": operator.resolve(decision.decided_by)}
    )
    try:
        return dispatch_workflow.decide_approval(
            approval_id, decided, authenticated=operator.authenticated
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/v1/intents/{intent_id}/commit")
def commit_intent(
    intent_id: str,
    intent: DispatchIntent,
    scenario_id: str = Query(default="interactive-multi-fleet", alias="scenarioId"),
    approval_id: str | None = Query(default=None, alias="approvalId"),
) -> dict[str, Any]:
    if intent.intent_id != intent_id:
        raise HTTPException(status_code=422, detail="路径和请求体的intentId不一致。")
    try:
        return dispatch_workflow.commit(
            intent,
            scenario_id=scenario_id,
            approval_id=approval_id,
        )
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/v1/intents/committed")
def committed_intents() -> list[dict[str, Any]]:
    return intents.list()


@app.get("/api/v1/audit")
def latest_audit(limit: int = Query(default=50, ge=1, le=200)) -> list[dict[str, Any]]:
    return [row.model_dump(by_alias=True, mode="json") for row in audit.latest(limit)]


@app.get("/api/v1/reports/shift")
def shift_report() -> dict[str, Any]:
    runs = engine.list_runs()
    successful = [row for row in runs if row.status == "COMPLETED"]
    approvals_rows = approvals.list()
    return {
        "title": "灵枢仿真运营报告",
        "mode": "simulation-only",
        "runCount": len(runs),
        "successfulRunCount": len(successful),
        "approvalCount": len(approvals_rows),
        "pendingApprovalCount": sum(
            row.status.value == "PENDING" for row in approvals_rows
        ),
        "latestRun": (
            successful[0].model_dump(by_alias=True, mode="json") if successful else None
        ),
        "notice": "报告中的所有数值均来自MASP仿真结果，不代表真实生产收益。",
    }


frontend_dist = settings.root / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def frontend(full_path: str) -> FileResponse:
        candidate = frontend_dist / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(frontend_dist / "index.html")
