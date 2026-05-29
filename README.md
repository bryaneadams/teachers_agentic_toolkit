# Teachers Agentic Toolkit

This project is structured so a UI can discover executable tools, discover crew workflows, and compose an ordered workflow from selected tool IDs.

Core concepts:

- `Tool`: standalone executable unit.
- `Workflow`: ordered list of tool IDs.
- `WorkflowEngine`: orchestrator only.
- `ToolRegistry`: tool discovery and lookup.
- `WorkflowCrew`: standalone CrewAI workflow with named operations.
- `CrewRegistry`: crew discovery and lookup.

## Directory Tree

```text
.
├── README.md
├── requirements.txt
└── src
    ├── config.py
    ├── errors.py
    ├── main.py
    ├── models.py
    ├── crews
    │   ├── __init.py__
    │   ├── workflow_crews.py
    │   └── config
    │       ├── agents.yaml
    │       └── tasks.yaml
    └── tools
        ├── __init__.py
        ├── context_chunking.py
        ├── document_ingest.py
        ├── image_assignment.py
        └── pptx_writer.py
```

## Add A New Tool

Tools live as executable classes in `src/main.py`. The domain implementation can call code from `src/tools`, but the UI-facing executable unit is a `Tool` subclass.

There are two tool layers:

- `src/tools/`: implementation modules and domain helpers such as document ingestion, image assignment, context chunking, and PowerPoint writing.
- `Tool` classes in `src/main.py`: UI-facing workflow units with IDs, labels, inputs, outputs, requirements, and an `execute()` method.

Do not put UI workflow metadata in `src/tools/`. Keep that directory focused on reusable implementation code, then wrap that implementation with a `Tool` class when it needs to be selectable in a workflow.

1. Create a class that extends `Tool`.
2. Add a `ToolDefinition` with stable IDs and context keys.
3. Implement `execute(state, input_value)`.
4. Register it in `build_tool_registry()`.

Example:

```python
class GenerateReportTool(Tool):
    definition = ToolDefinition(
        id="generate_report",
        label="Generate report",
        description="Create a written report from document analysis.",
        input_key="analysis",
        output_key="report",
        requires=("analysis", "document"),
    )

    def execute(
        self, state: WorkflowState, input_value: DocumentAnalysis | None = None
    ) -> str:
        if input_value is None:
            raise RuntimeError("Document analysis is required before report generation.")

        # Call report-generation code here.
        report = f"# {state.document.title}\n\n{input_value.summary}"
        return report
```

In `definition` note that there is validation, specifically `input_key="analysis"` is the input which creates a new key `output_key="report"`, but this tool requires `requires=("analysis", "document")`. This will help ensure proper ordering of tools.

Then register it:

```python
def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        (
            IngestDocumentTool(),
            AnalyzeDocumentTool(),
            CreateSlideDeckTool(),
            AttachDocumentImagesTool(),
            RenderPowerPointTool(),
            GenerateReportTool(),
        )
    )
```

The important fields are:

- `id`: stable ID the UI stores in a workflow.
- `input_key`: context value this tool receives.
- `output_key`: context value this tool writes.
- `requires`: context values that must exist before the tool runs.

## Add A New Crew

Crews live in `src/crews/workflow_crews.py`. A crew is a reusable CrewAI workflow with one or more named operations.

1. Create a class that extends `WorkflowCrew`.
2. Add a `CrewDefinition`.
3. Implement one or more operation methods.
4. Register the crew in `build_crew_registry()`.
5. Add or reuse agent/task config in `src/crews/config/agents.yaml` and `src/crews/config/tasks.yaml`.

Example:

```python
class ReportWritingCrew(WorkflowCrew):
    definition = CrewDefinition(
        id="report_writing",
        label="Report writing",
        description="Generate a report from document analysis.",
        operations=("generate_report",),
        agent_ids=("report_writer",),
        task_ids=("generate_report",),
    )

    def generate_report(
        self,
        document: Document,
        analysis: DocumentAnalysis,
        llm_profile: str | None = None,
    ) -> str:
        # Build CrewAI agents/tasks, call crew.kickoff(), parse result.
        ...
```

Then register it:

```python
def build_crew_registry(config: WorkflowConfig | None = None) -> CrewRegistry:
    return CrewRegistry(
        (
            DocumentAnalysisCrew,
            SlideDeckPlanningCrew,
            DocumentSlidePlanningCrew,
            ReportWritingCrew,
        ),
        config=config,
    )
```

Current registered crews:

- `document_analysis`: runs `analyze_document`.
- `slide_deck_planning`: runs `create_slide_deck`.
- `document_slide_planning`: compatibility crew that chains analysis and slide planning.

Crew workflows require CrewAI and an LLM. They fail with `OrchestrationError` if CrewAI or the LLM call is unavailable.

## Compose A Workflow

A workflow is an ordered list of tool IDs. The engine executes tools in that order and passes values through a shared context using each tool's `input_key`, `output_key`, and `requires`.

`run()` inputs default to `None` so a workflow only needs the values required by the tools it actually runs. If the workflow includes `render_powerpoint`, `run()` returns a `WorkflowResult`. If the workflow stops earlier, `run()` returns the execution context dictionary.

Ingest-only workflow:

```python
from main import run

context = run(
    document_path="path/to/document.md",
    workflow_order=("ingest_document",),
)

document = context["document"]
```

For one tool, either pass a string or a one-item tuple with a trailing comma:

```python
workflow_order="ingest_document"
workflow_order=("ingest_document",)
```

Default slide deck workflow:

```python
from main import run

result = run(
    document_path="path/to/document.md",
    output_path="out/deck.pptx",
    slide_count=6,
    question_count=3,
)
```

Custom ordered tool workflow:

```python
from main import run

result = run(
    document_path="path/to/document.md",
    output_path="out/deck.pptx",
    slide_count=6,
    question_count=3,
    workflow_order=(
        "ingest_document",
        "analyze_document",
        "create_slide_deck",
        "attach_document_images",
        "render_powerpoint",
    ),
)
```

Equivalent explicit `Workflow` object:

```python
from main import Workflow, run

workflow = Workflow(
    id="deck_without_images",
    label="Deck without images",
    tool_ids=(
        "ingest_document",
        "analyze_document",
        "create_slide_deck",
        "render_powerpoint",
    ),
)

result = run(
    document_path="path/to/document.md",
    output_path="out/deck.pptx",
    slide_count=6,
    question_count=3,
    workflow=workflow,
)
```

## Select Crews For Tools

Some tools delegate LLM work to crew workflows. For the default slide deck path:

- `analyze_document` uses `analysis_crew_id`.
- `analyze_document` uses `analysis_agent_id` to choose the agent profile inside `agents.yaml`.
- `create_slide_deck` uses `deck_planning_crew_id`.
- `create_slide_deck` uses `slide_design_agent_id` to choose the slide designer profile inside `agents.yaml`.

Example:

```python
from main import run

result = run(
    document_path="path/to/document.md",
    output_path="out/deck.pptx",
    slide_count=6,
    question_count=3,
    analysis_crew_id="document_analysis",
    analysis_agent_id="instructional_content_analyst",
    deck_planning_crew_id="slide_deck_planning",
    slide_design_agent_id="slide_designer",
)
```

The default analysis agent is `instructional_content_analyst`. For a more general analysis that is not slide-deck specific, use:

```python
context = run(
    document_path="path/to/document.md",
    workflow_order=("ingest_document", "analyze_document"),
    analysis_agent_id="general_document_analyst",
)
```

For a UI-defined analyst, use `custom_document_analyst` and pass the goal and backstory from UI fields:

```python
context = run(
    document_path="path/to/document.md",
    workflow_order=("ingest_document", "analyze_document"),
    analysis_agent_id="custom_document_analyst",
    analysis_agent_goal="Make two funny puns based on the document.",
    analysis_agent_backstory="You are a math nerd who writes concise math jokes from source material.",
)
```

The user does not need to know internal output fields such as `summary`. The analysis task maps the requested output into the `DocumentAnalysis` schema. The `custom_document_analyst` profile lives in `src/crews/config/agents.yaml` and uses `{analysis_agent_goal}` and `{analysis_agent_backstory}` placeholders. To add another fixed analysis profile, add an entry to `agents.yaml` and pass that YAML key as `analysis_agent_id`.

For a UI-defined slide designer, use `custom_slide_designer` and pass the goal and backstory from UI fields:

```python
result = run(
    document_path="path/to/document.md",
    output_path="out/deck.pptx",
    slide_count=6,
    question_count=2,
    workflow_order=(
        "ingest_document",
        "analyze_document",
        "create_slide_deck",
        "render_powerpoint",
    ),
    slide_design_agent_id="custom_slide_designer",
    slide_design_agent_goal="Create funny, fast-paced review slides with one joke per slide.",
    slide_design_agent_backstory="You are a math teacher who uses playful slide titles and short punchy examples.",
)
```

The user does not need to know the slide JSON fields. The slide planning task maps the requested style, tone, structure, and constraints into the `SlideDeck` schema. The `custom_slide_designer` profile lives in `src/crews/config/agents.yaml` and uses `{slide_design_agent_goal}` and `{slide_design_agent_backstory}` placeholders.

For a future report workflow, the ordered tools might look like:

```python
workflow_order = (
    "ingest_document",
    "analyze_document",
    "generate_report",
)
```

In that shape, `analyze_document` still writes `analysis` to context, and `generate_report` can consume the same reusable `analysis` output.

## Discover Tools And Crews For A UI

Use registries to populate UI pickers.

```python
from main import build_tool_registry
from crews.workflow_crews import build_crew_registry

tool_registry = build_tool_registry()
crew_registry = build_crew_registry()

tools = tool_registry.definitions()
crews = crew_registry.definitions()
```

Each definition contains stable IDs, labels, descriptions, and dependency metadata that can be displayed or stored by a UI.
