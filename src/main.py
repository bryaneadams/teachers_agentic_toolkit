from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field

from config import WorkflowConfig, load_config
from crews.workflow_crews import (
    DEFAULT_ANALYSIS_AGENT_ID,
    DEFAULT_ANALYSIS_CREW_ID,
    DEFAULT_DECK_PLANNING_CREW_ID,
    CrewRegistry,
    build_crew_registry,
)
from models import Document, DocumentAnalysis, SlideDeck, WorkflowResult
from tools.document_ingest import load_document
from tools.image_assignment import attach_images
from tools.pptx_writer import write_pptx


class WorkflowState(BaseModel):
    """Mutable state shared by tools during one workflow run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    document_path: str | None = None
    output_path: str | None = None
    slide_count: int | None = None
    question_count: int | None = None
    template_path: str | None = None
    image_dir: str | None = None
    config_path: str | None = None
    llm_profile: str | None = None
    analysis_agent_id: str = DEFAULT_ANALYSIS_AGENT_ID
    analysis_agent_goal: str | None = None
    analysis_agent_backstory: str | None = None
    analysis_crew_id: str = DEFAULT_ANALYSIS_CREW_ID
    deck_planning_crew_id: str = DEFAULT_DECK_PLANNING_CREW_ID
    config: WorkflowConfig | None = None
    crew_registry: CrewRegistry | None = None
    document: Document | None = None
    analysis: DocumentAnalysis | None = None
    deck: SlideDeck | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


# Backward-compatible alias for callers that imported the old state name.
SlideMakerState = WorkflowState


@dataclass(frozen=True)
class ToolDefinition:
    """UI-friendly metadata for a standalone executable tool."""

    id: str
    label: str
    description: str
    input_key: str | None
    output_key: str
    requires: tuple[str, ...] = ()
    enabled: bool = True


class Tool:
    """Standalone executable unit.

    Tools own the domain work. They receive explicit state plus one optional
    input value, and return a value the engine stores in workflow context.
    """

    definition: ToolDefinition

    @property
    def id(self) -> str:
        return self.definition.id

    def execute(self, state: WorkflowState, input_value: Any = None) -> Any:
        raise NotImplementedError


class IngestDocumentTool(Tool):
    definition = ToolDefinition(
        id="ingest_document",
        label="Ingest document",
        description="Load runtime config and parse the source document.",
        input_key=None,
        output_key="document",
    )

    def execute(self, state: WorkflowState, input_value: Any = None) -> Document:
        if state.document_path is None:
            raise RuntimeError("Document path is required to ingest a document.")

        state.config = load_config(state.config_path)
        if state.crew_registry is None:
            state.crew_registry = build_crew_registry(state.config)
        else:
            state.crew_registry.config = state.config
        document = load_document(state.document_path, image_dir=state.image_dir)
        state.document = document
        return document


class AnalyzeDocumentTool(Tool):
    definition = ToolDefinition(
        id="analyze_document",
        label="Analyze document",
        description="Analyze source material and extract teaching concepts.",
        input_key="document",
        output_key="analysis",
        requires=("document",),
    )

    def execute(
        self, state: WorkflowState, input_value: Document | None = None
    ) -> DocumentAnalysis:
        if input_value is None:
            raise RuntimeError("Document was not loaded before analysis.")
        if state.config is None or state.crew_registry is None:
            raise RuntimeError(
                "Config and crew registry must be loaded before analysis."
            )

        crew = state.crew_registry.get(state.analysis_crew_id, state.config)
        analysis = crew.execute(
            "analyze_document",
            document=input_value,
            llm_profile=state.llm_profile,
            analysis_agent_id=state.analysis_agent_id,
            analysis_agent_goal=state.analysis_agent_goal,
            analysis_agent_backstory=state.analysis_agent_backstory,
        )
        state.analysis = analysis
        return analysis


class CreateSlideDeckTool(Tool):
    definition = ToolDefinition(
        id="create_slide_deck",
        label="Create slide deck",
        description="Turn the analysis into a slide deck plan.",
        input_key="analysis",
        output_key="deck",
        requires=("analysis", "document"),
    )

    def execute(
        self, state: WorkflowState, input_value: DocumentAnalysis | None = None
    ) -> SlideDeck:
        if state.document is None:
            raise RuntimeError("Document was not loaded before slide planning.")
        if input_value is None:
            raise RuntimeError(
                "Document analysis was not created before slide planning."
            )
        if state.config is None or state.crew_registry is None:
            raise RuntimeError(
                "Config and crew registry must be loaded before planning."
            )
        if state.slide_count is None:
            raise RuntimeError("Slide count is required before slide planning.")

        crew = state.crew_registry.get(state.deck_planning_crew_id, state.config)
        deck = crew.execute(
            "create_slide_deck",
            document=state.document,
            analysis=input_value,
            slide_count=state.slide_count,
            question_count=state.question_count or 0,
            llm_profile=state.llm_profile,
        )
        state.deck = deck
        return deck


class AttachDocumentImagesTool(Tool):
    definition = ToolDefinition(
        id="attach_document_images",
        label="Attach document images",
        description="Match parsed document images to planned slides.",
        input_key="deck",
        output_key="deck",
        requires=("deck", "document"),
    )

    def execute(
        self, state: WorkflowState, input_value: SlideDeck | None = None
    ) -> SlideDeck:
        if state.document is None:
            raise RuntimeError("Document was not loaded before image assignment.")
        if input_value is None:
            raise RuntimeError("Slide deck was not created before image assignment.")

        deck_with_images = attach_images(input_value, state.document.images)
        state.deck = deck_with_images
        return deck_with_images


class RenderPowerPointTool(Tool):
    definition = ToolDefinition(
        id="render_powerpoint",
        label="Render PowerPoint",
        description="Write the final PowerPoint file.",
        input_key="deck",
        output_key="result",
        requires=("deck",),
    )

    def execute(
        self, state: WorkflowState, input_value: SlideDeck | None = None
    ) -> WorkflowResult:
        if input_value is None:
            raise RuntimeError("Slide deck was not created before rendering.")
        if state.output_path is None:
            raise RuntimeError("Output path is required before rendering.")

        output, render_warnings = write_pptx(
            input_value,
            state.output_path,
            template_path=state.template_path,
        )

        state.warnings.extend(render_warnings)

        return WorkflowResult(
            output_path=output,
            deck=input_value,
            warnings=tuple(state.warnings),
        )


class ToolRegistry:
    """Registry for discovering and retrieving executable tools."""

    def __init__(self, tools: Sequence[Tool] = ()):
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.id in self._tools:
            raise ValueError(f"Tool already registered: {tool.id}")
        self._tools[tool.id] = tool

    def get(self, tool_id: str) -> Tool:
        try:
            return self._tools[tool_id]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {tool_id}") from exc

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool.definition for tool in self._tools.values())

    def ids(self) -> tuple[str, ...]:
        return tuple(self._tools)


@dataclass(frozen=True)
class Workflow:
    """Ordered list of tool IDs that a UI can display, persist, and reorder."""

    id: str
    label: str
    tool_ids: tuple[str, ...]


class WorkflowEngine:
    """Orchestrates workflows without owning tool implementation details."""

    def __init__(self, state: WorkflowState, registry: ToolRegistry):
        self.state = state
        self.registry = registry

    def run(self, workflow: Workflow) -> dict[str, Any]:
        context: dict[str, Any] = {}

        for tool_id in workflow.tool_ids:
            tool = self.registry.get(tool_id)
            definition = tool.definition

            if not definition.enabled:
                continue

            missing = [key for key in definition.requires if key not in context]
            if missing:
                raise RuntimeError(
                    f"Tool '{definition.id}' requires missing outputs: {missing}"
                )

            input_value = (
                context.get(definition.input_key) if definition.input_key else None
            )
            output = tool.execute(self.state, input_value)
            context[definition.output_key] = output

        return context


DEFAULT_WORKFLOW_TOOL_IDS = (
    "ingest_document",
    "analyze_document",
    "create_slide_deck",
    "attach_document_images",
    "render_powerpoint",
)

DEFAULT_WORKFLOW = Workflow(
    id="slide_deck_default",
    label="Default slide deck workflow",
    tool_ids=DEFAULT_WORKFLOW_TOOL_IDS,
)

# Backward-compatible alias for callers that were already passing an order.
DEFAULT_WORKFLOW_ORDER = DEFAULT_WORKFLOW_TOOL_IDS


def build_tool_registry() -> ToolRegistry:
    """Create the default tool registry.

    This is the integration point for UI discovery: call `definitions()` to
    render available tools, then persist selected tool IDs in a workflow.
    """

    return ToolRegistry(
        (
            IngestDocumentTool(),
            AnalyzeDocumentTool(),
            CreateSlideDeckTool(),
            AttachDocumentImagesTool(),
            RenderPowerPointTool(),
        )
    )


def run(
    document_path: str | Path | None = None,
    output_path: str | Path | None = None,
    slide_count: int | None = None,
    question_count: int | None = None,
    template_path: str | Path | None = None,
    image_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    llm_profile: str | None = None,
    analysis_agent_id: str = DEFAULT_ANALYSIS_AGENT_ID,
    analysis_agent_goal: str | None = None,
    analysis_agent_backstory: str | None = None,
    analysis_crew_id: str = DEFAULT_ANALYSIS_CREW_ID,
    deck_planning_crew_id: str = DEFAULT_DECK_PLANNING_CREW_ID,
    crew_registry: CrewRegistry | None = None,
    workflow_order: str | Sequence[str] | None = None,
    workflow: Workflow | None = None,
) -> WorkflowResult | dict[str, Any]:
    """Run a workflow and return the final result or execution context."""

    state = WorkflowState(
        document_path=str(document_path) if document_path else None,
        output_path=str(output_path) if output_path else None,
        slide_count=slide_count,
        question_count=question_count,
        template_path=str(template_path) if template_path else None,
        image_dir=str(image_dir) if image_dir else None,
        config_path=str(config_path) if config_path else None,
        llm_profile=llm_profile,
        analysis_agent_id=analysis_agent_id,
        analysis_agent_goal=analysis_agent_goal,
        analysis_agent_backstory=analysis_agent_backstory,
        analysis_crew_id=analysis_crew_id,
        deck_planning_crew_id=deck_planning_crew_id,
        crew_registry=crew_registry or build_crew_registry(),
    )

    registry = build_tool_registry()
    engine = WorkflowEngine(state, registry)
    selected_workflow = workflow or DEFAULT_WORKFLOW
    if workflow_order is not None:
        selected_workflow = Workflow(
            id="custom_slide_deck_workflow",
            label="Custom slide deck workflow",
            tool_ids=_normalize_workflow_order(workflow_order),
        )

    context = engine.run(selected_workflow)

    return context.get("result", context)


def _normalize_workflow_order(workflow_order: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize one tool ID or a sequence of tool IDs."""

    if isinstance(workflow_order, str):
        return (workflow_order,)
    return tuple(workflow_order)


def kickoff() -> WorkflowResult:
    """Run the default slide-maker workflow."""

    return cast(
        WorkflowResult,
        run(
            document_path="examples/document.md",
            output_path="out/deck.pptx",
            slide_count=6,
            question_count=3,
            config_path="examples/config.json",
        ),
    )


if __name__ == "__main__":
    kickoff()
