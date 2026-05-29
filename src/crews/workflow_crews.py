from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from config import WorkflowConfig
from errors import OrchestrationError
from models import Document, DocumentAnalysis, ExampleQuestion, Slide, SlideDeck
from tools.context_chunking import chunk_text_for_llm

DEFAULT_ANALYSIS_AGENT_ID = "instructional_content_analyst"
CUSTOM_ANALYSIS_AGENT_ID = "custom_document_analyst"


@dataclass(frozen=True)
class CrewDefinition:
    """UI-friendly metadata for a crew and its callable operations."""

    id: str
    label: str
    description: str
    operations: tuple[str, ...]
    agent_ids: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()
    enabled: bool = True


class WorkflowCrew:
    """Base class for standalone crew workflows."""

    definition: CrewDefinition

    def __init__(self, config: WorkflowConfig):
        self.config = config

    @property
    def id(self) -> str:
        return self.definition.id

    def execute(self, operation: str, **inputs: Any) -> Any:
        """Run a named crew operation with keyword arguments.

        Args:
            operation (str): Operation name declared on the crew definition.
            **inputs (Any): Keyword arguments passed to the matching method.

        Raises:
            KeyError: If the requested operation is not declared by this crew.

        Returns:
            Any: The result returned by the operation method.
        """
        if operation not in self.definition.operations:
            raise KeyError(f"Crew '{self.id}' does not support operation: {operation}")
        handler = getattr(self, operation)
        return handler(**inputs)

    def _crewai_llm(self, llm_profile: str | None):
        """Build a CrewAI LLM instance from the configured profile.

        Args:
            llm_profile (str | None): Named LLM profile to use, or None for the default profile.
                Defaults to None.

        Returns:
            LLM: Configured CrewAI LLM instance.
        """
        from crewai import LLM

        cfg = self.config.llm(llm_profile)
        kwargs = {
            "model": _crewai_model_name(cfg.provider, cfg.model),
            "temperature": cfg.temperature,
        }
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        api_key = self.config.api_key_for(llm_profile)
        if api_key:
            kwargs["api_key"] = api_key
        kwargs.update(cfg.extra)
        return LLM(**kwargs)

    def _build_agent(
        self,
        agent_id: str,
        llm,
        agent_inputs: dict[str, str] | None = None,
    ):
        """Create one CrewAI agent from `agents.yaml`.

        Args:
            agent_id (str): Agent ID from `agents.yaml`.
            llm (crewai.LLM): LLM instance to attach to the agent.
            agent_inputs (dict[str, str] | None, optional): Values used to format agent profile placeholders.
                Defaults to None.

        Returns:
            crewai.Agent: Configured CrewAI agent.
        """
        from crewai import Agent

        agent_config = _agent_config(agent_id)
        agent_inputs = agent_inputs or {}
        return Agent(
            role=_format_agent_field(agent_config["role"], agent_inputs, agent_id),
            goal=_format_agent_field(agent_config["goal"], agent_inputs, agent_id),
            backstory=_format_agent_field(
                agent_config["backstory"], agent_inputs, agent_id
            ),
            llm=llm,
            verbose=self.config.orchestration.verbose,
        )


class DocumentAnalysisCrew(WorkflowCrew):
    """Crew workflow for reusable document analysis."""

    definition = CrewDefinition(
        id="document_analysis",
        label="Document analysis",
        description="Analyze source material into reusable teaching concepts.",
        operations=("analyze_document",),
        agent_ids=(
            DEFAULT_ANALYSIS_AGENT_ID,
            "general_document_analyst",
            CUSTOM_ANALYSIS_AGENT_ID,
        ),
        task_ids=("analyze_document",),
    )

    def analyze_document(
        self,
        document: Document,
        llm_profile: str | None = None,
        analysis_agent_id: str = DEFAULT_ANALYSIS_AGENT_ID,
        analysis_agent_goal: str | None = None,
        analysis_agent_backstory: str | None = None,
    ) -> DocumentAnalysis:
        """Extract reusable analysis from document text.

        Args:
            document (Document): Source document to analyze.
            llm_profile (str | None, optional): Named LLM profile to use, or None for the default profile.
                Defaults to None.
            analysis_agent_id (str, optional): Agent profile ID from `agents.yaml`.
                Defaults to "instructional_content_analyst".
            analysis_agent_goal (str | None, optional): Goal text for the customizable analysis agent.
                Defaults to None.
            analysis_agent_backstory (str | None, optional): Backstory text for the customizable analysis agent.
                Defaults to None.

        Returns:
            DocumentAnalysis: Parsed document analysis.
        """
        from crewai import Crew, Process, Task

        task_config = _load_yaml_config("tasks.yaml")
        chunked_context = chunk_text_for_llm(
            document.text,
            max_tokens=self.config.orchestration.max_context_tokens,
        )
        llm = self._crewai_llm(llm_profile)
        agent_inputs = _analysis_agent_inputs(
            analysis_agent_id=analysis_agent_id,
            analysis_agent_goal=analysis_agent_goal,
            analysis_agent_backstory=analysis_agent_backstory,
        )
        analyst = self._build_agent(analysis_agent_id, llm, agent_inputs)
        task_template = task_config["analyze_document"]
        task = Task(
            description=task_template["description"].format(
                analysis_instructions=_analysis_instructions(
                    analysis_agent_id=analysis_agent_id,
                    analysis_agent_goal=analysis_agent_goal,
                    analysis_agent_backstory=analysis_agent_backstory,
                ),
                document_title=document.title,
                document_text=chunked_context.text,
            ),
            expected_output=task_template["expected_output"],
            agent=analyst,
            context=[],
        )
        crew = Crew(
            agents=[analyst],
            tasks=[task],
            process=Process.sequential,
            verbose=self.config.orchestration.verbose,
        )
        try:
            result = crew.kickoff()
            return replace(_analysis_from_json(str(result)), used_llm=True)
        except Exception as exc:
            raise OrchestrationError(f"CrewAI analysis workflow failed: {exc}") from exc


class SlideDeckPlanningCrew(WorkflowCrew):
    """Crew workflow for creating a slide deck from an existing analysis."""

    definition = CrewDefinition(
        id="slide_deck_planning",
        label="Slide deck planning",
        description="Create a classroom slide deck from document analysis.",
        operations=("create_slide_deck",),
        agent_ids=("slide_designer",),
        task_ids=("create_slide_plan",),
    )

    def create_slide_deck(
        self,
        document: Document,
        analysis: DocumentAnalysis,
        slide_count: int,
        question_count: int,
        llm_profile: str | None = None,
    ) -> SlideDeck:
        """Create a slide deck from a document and prior analysis.

        Args:
            document (Document): Source document to plan from.
            analysis (DocumentAnalysis): Previously generated document analysis.
            slide_count (int): Number of slides to create.
            question_count (int): Number of question slides to create.
            llm_profile (str | None, optional): Named LLM profile to use, or None for the default profile.
                Defaults to None.

        Returns:
            SlideDeck: Parsed slide deck.
        """
        from crewai import Crew, Process, Task

        task_config = _load_yaml_config("tasks.yaml")
        chunked_context = chunk_text_for_llm(
            document.text,
            max_tokens=self.config.orchestration.max_context_tokens,
        )

        llm = self._crewai_llm(llm_profile)
        designer = self._build_agent("slide_designer", llm)
        task_template = task_config["create_slide_plan"]
        task = Task(
            description=task_template["description"].format(
                slide_count=slide_count,
                question_count=question_count,
                document_title=document.title,
                document_analysis=_analysis_to_json(analysis),
                document_text=chunked_context.text,
            ),
            expected_output=task_template["expected_output"],
            agent=designer,
            context=[],
        )
        crew = Crew(
            agents=[designer],
            tasks=[task],
            process=Process.sequential,
            verbose=self.config.orchestration.verbose,
        )
        try:
            result = crew.kickoff()
            deck = replace(
                _deck_from_json(
                    str(result),
                    fallback_title=document.title,
                    slide_count=slide_count,
                    question_count=question_count,
                ),
                used_llm=True,
            )
            return replace(
                deck,
                warnings=deck.warnings
                + (
                    _context_warning(
                        document.text, self.config.orchestration.max_context_tokens
                    ),
                ),
            )
        except Exception as exc:
            raise OrchestrationError(
                f"CrewAI slide deck planning workflow failed: {exc}"
            ) from exc


class DocumentSlidePlanningCrew(WorkflowCrew):
    """Compatibility crew that chains document analysis and slide planning."""

    definition = CrewDefinition(
        id="document_slide_planning",
        label="Document slide planning",
        description="Analyze source material and plan a classroom slide deck.",
        operations=("plan", "analyze_document", "create_slide_deck"),
        agent_ids=(
            DEFAULT_ANALYSIS_AGENT_ID,
            "general_document_analyst",
            CUSTOM_ANALYSIS_AGENT_ID,
            "slide_designer",
        ),
        task_ids=("analyze_document", "create_slide_plan"),
    )

    def plan(
        self,
        document: Document,
        slide_count: int,
        question_count: int,
        llm_profile: str | None = None,
        analysis_agent_id: str = DEFAULT_ANALYSIS_AGENT_ID,
        analysis_agent_goal: str | None = None,
        analysis_agent_backstory: str | None = None,
    ) -> SlideDeck:
        """Analyze a document and create a slide deck.

        Args:
            document (Document): Source document to process.
            slide_count (int): Number of slides to create.
            question_count (int): Number of question slides to create.
            llm_profile (str | None, optional): Named LLM profile to use, or None for the default profile.
                Defaults to None.
            analysis_agent_id (str, optional): Agent profile ID from `agents.yaml`.
                Defaults to "instructional_content_analyst".
            analysis_agent_goal (str | None, optional): Goal text for the customizable analysis agent.
                Defaults to None.
            analysis_agent_backstory (str | None, optional): Backstory text for the customizable analysis agent.
                Defaults to None.

        Returns:
            SlideDeck: Planned slide deck.
        """
        analysis = self.analyze_document(
            document,
            llm_profile=llm_profile,
            analysis_agent_id=analysis_agent_id,
            analysis_agent_goal=analysis_agent_goal,
            analysis_agent_backstory=analysis_agent_backstory,
        )
        return self.create_slide_deck(
            document=document,
            analysis=analysis,
            slide_count=slide_count,
            question_count=question_count,
            llm_profile=llm_profile,
        )

    def analyze_document(
        self,
        document: Document,
        llm_profile: str | None = None,
        analysis_agent_id: str = DEFAULT_ANALYSIS_AGENT_ID,
        analysis_agent_goal: str | None = None,
        analysis_agent_backstory: str | None = None,
    ) -> DocumentAnalysis:
        """Analyze a document using the document analysis crew.

        Args:
            document (Document): Source document to analyze.
            llm_profile (str | None, optional): Named LLM profile to use, or None for the default profile.
                Defaults to None.
            analysis_agent_id (str, optional): Agent profile ID from `agents.yaml`.
                Defaults to "instructional_content_analyst".
            analysis_agent_goal (str | None, optional): Goal text for the customizable analysis agent.
                Defaults to None.
            analysis_agent_backstory (str | None, optional): Backstory text for the customizable analysis agent.
                Defaults to None.

        Returns:
            DocumentAnalysis: Parsed document analysis.
        """
        return DocumentAnalysisCrew(self.config).analyze_document(
            document=document,
            llm_profile=llm_profile,
            analysis_agent_id=analysis_agent_id,
            analysis_agent_goal=analysis_agent_goal,
            analysis_agent_backstory=analysis_agent_backstory,
        )

    def create_slide_deck(
        self,
        document: Document,
        analysis: DocumentAnalysis,
        slide_count: int,
        question_count: int,
        llm_profile: str | None = None,
    ) -> SlideDeck:
        """Create a slide deck using the slide deck planning crew.

        Args:
            document (Document): Source document to plan from.
            analysis (DocumentAnalysis): Previously generated document analysis.
            slide_count (int): Number of slides to create.
            question_count (int): Number of question slides to create.
            llm_profile (str | None, optional): Named LLM profile to use, or None for the default profile.
                Defaults to None.

        Returns:
            SlideDeck: Planned slide deck.
        """
        return SlideDeckPlanningCrew(self.config).create_slide_deck(
            document=document,
            analysis=analysis,
            slide_count=slide_count,
            question_count=question_count,
            llm_profile=llm_profile,
        )


# Backward-compatible alias for existing imports or callers.
SlideDeckCrew = DocumentSlidePlanningCrew


class CrewRegistry:
    """Registry for discovering and retrieving executable crews."""

    def __init__(
        self,
        crew_factories: Sequence[type[WorkflowCrew]] = (),
        config: WorkflowConfig | None = None,
    ):
        """Create a crew registry.

        Args:
            crew_factories (Sequence[type[WorkflowCrew]], optional): Crew classes to register.
                Defaults to ().
            config (WorkflowConfig | None, optional): Default config to use when instantiating crews.
                Defaults to None.
        """
        self.config = config
        self._crew_factories: dict[str, type[WorkflowCrew]] = {}
        for crew_factory in crew_factories:
            self.register(crew_factory)

    def register(self, crew_factory: type[WorkflowCrew]) -> None:
        """Register a crew class by its definition ID.

        Args:
            crew_factory (type[WorkflowCrew]): Crew class to register.

        Raises:
            ValueError: If the crew ID is already registered.
        """
        crew_id = crew_factory.definition.id
        if crew_id in self._crew_factories:
            raise ValueError(f"Crew already registered: {crew_id}")
        self._crew_factories[crew_id] = crew_factory

    def get(self, crew_id: str, config: WorkflowConfig | None = None) -> WorkflowCrew:
        """Instantiate a registered crew.

        Args:
            crew_id (str): Registered crew ID.
            config (WorkflowConfig | None, optional): Config to pass to the crew, or the registry default.
                Defaults to None.

        Raises:
            KeyError: If the crew ID is not registered.
            RuntimeError: If no config is available for crew instantiation.

        Returns:
            WorkflowCrew: Instantiated crew.
        """
        try:
            crew_factory = self._crew_factories[crew_id]
        except KeyError as exc:
            raise KeyError(f"Unknown crew: {crew_id}") from exc

        selected_config = config or self.config
        if selected_config is None:
            raise RuntimeError("A WorkflowConfig is required to instantiate crews.")
        return crew_factory(selected_config)

    def definitions(self) -> tuple[CrewDefinition, ...]:
        """Return the registered crew definitions.

        Returns:
            tuple[CrewDefinition, ...]: Registered crew metadata.
        """
        return tuple(
            crew_factory.definition for crew_factory in self._crew_factories.values()
        )

    def ids(self) -> tuple[str, ...]:
        """Return the registered crew IDs.

        Returns:
            tuple[str, ...]: Registered crew IDs.
        """
        return tuple(self._crew_factories)


DEFAULT_ANALYSIS_CREW_ID = DocumentAnalysisCrew.definition.id
DEFAULT_DECK_PLANNING_CREW_ID = SlideDeckPlanningCrew.definition.id
DEFAULT_CREW_ID = DocumentSlidePlanningCrew.definition.id


def build_crew_registry(config: WorkflowConfig | None = None) -> CrewRegistry:
    """Create the default crew registry used by the workflow engine and UI.

    Args:
        config (WorkflowConfig | None, optional): Config to bind to the registry.
            Defaults to None.

    Returns:
        CrewRegistry: Registry containing the built-in crew workflows.
    """

    return CrewRegistry(
        (DocumentAnalysisCrew, SlideDeckPlanningCrew, DocumentSlidePlanningCrew),
        config=config,
    )


def _deck_from_json(
    raw: str,
    fallback_title: str,
    slide_count: int | None = None,
    question_count: int = 0,
) -> SlideDeck:
    """Parse a CrewAI JSON response into a `SlideDeck`.

    Args:
        raw (str): Raw CrewAI response, optionally wrapped with extra text.
        fallback_title (str): Title to use if the response does not include one.
        slide_count (int | None, optional): Expected total slide count.
            Defaults to None.
        question_count (int, optional): Expected number of final question
            slides. Defaults to 0.

    Raises:
        OrchestrationError: If no JSON object can be found in the response.
        OrchestrationError: If the parsed response contains no slides.

    Returns:
        SlideDeck: Sanitized deck with question slides normalized.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise OrchestrationError("CrewAI did not return JSON.")
    data = json.loads(raw[start : end + 1])
    slides = tuple(
        Slide(
            title=_plain_slide_text(str(item["title"])),
            bullets=tuple(
                _plain_slide_text(str(bullet)) for bullet in item.get("bullets", [])
            ),
            speaker_notes=_plain_slide_text(str(item.get("speaker_notes", ""))),
        )
        for item in data.get("slides", [])
    )
    questions = tuple(
        ExampleQuestion(
            prompt=_plain_slide_text(str(item["prompt"])),
            answer=_plain_slide_text(str(item.get("answer", ""))),
        )
        for item in data.get("questions", [])
    )
    slides = _ensure_question_slides(slides, questions, slide_count, question_count)
    if not slides:
        raise OrchestrationError("CrewAI returned no slides.")
    return SlideDeck(
        title=_plain_slide_text(str(data.get("title") or fallback_title)),
        slides=slides,
        questions=questions,
        used_llm=True,
    )


def _analysis_from_json(raw: str) -> DocumentAnalysis:
    """Parse a CrewAI JSON response into a `DocumentAnalysis`.

    Args:
        raw (str): Raw CrewAI response, optionally wrapped with extra text.

    Raises:
        OrchestrationError: If no JSON object can be found in the response.

    Returns:
        DocumentAnalysis: Sanitized document analysis marked as LLM-generated.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise OrchestrationError("CrewAI did not return analysis JSON.")
    data = json.loads(raw[start : end + 1])
    return DocumentAnalysis(
        summary=_plain_slide_text(str(data.get("summary", ""))),
        key_concepts=tuple(
            _plain_slide_text(str(item)) for item in data.get("key_concepts", [])
        ),
        key_terms=tuple(
            _plain_slide_text(str(item)) for item in data.get("key_terms", [])
        ),
        example_questions=tuple(
            _plain_slide_text(str(item)) for item in data.get("example_questions", [])
        ),
        cautions=tuple(str(item) for item in data.get("cautions", [])),
        used_llm=True,
    )


def _load_yaml_config(file_name: str) -> dict:
    """Load a CrewAI YAML config file from this crew's config directory.

    Args:
        file_name (str): Name of the YAML file under `config/`.

    Raises:
        OrchestrationError: If PyYAML is not installed.
        OrchestrationError: If the YAML file does not contain a mapping.

    Returns:
        dict: Parsed YAML mapping.
    """
    path = Path(__file__).parent / "config" / file_name
    try:
        import yaml
    except ImportError as exc:
        raise OrchestrationError(
            "PyYAML is required to load CrewAI YAML config files."
        ) from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise OrchestrationError(f"CrewAI config must be a mapping: {path}")
    return data


def _agent_config(agent_id: str) -> dict:
    """Load one agent profile from `agents.yaml`.

    Args:
        agent_id (str): Agent profile ID to load.

    Raises:
        OrchestrationError: If the agent profile does not exist.

    Returns:
        dict: Agent configuration mapping.
    """
    agent_config = _load_yaml_config("agents.yaml")
    try:
        return agent_config[agent_id]
    except KeyError as exc:
        available = ", ".join(sorted(agent_config))
        raise OrchestrationError(
            f"Unknown agent profile '{agent_id}'. Available profiles: {available}"
        ) from exc


def _analysis_agent_inputs(
    analysis_agent_id: str,
    analysis_agent_goal: str | None,
    analysis_agent_backstory: str | None,
) -> dict[str, str]:
    """Build placeholder values for analysis agent profiles.

    Args:
        analysis_agent_id (str): Selected analysis agent profile ID.
        analysis_agent_goal (str | None): Goal text from the UI.
        analysis_agent_backstory (str | None): Backstory text from the UI.

    Raises:
        OrchestrationError: If the custom analysis agent is selected without required text.

    Returns:
        dict[str, str]: Placeholder values for formatting agent profile fields.
    """
    if analysis_agent_id == CUSTOM_ANALYSIS_AGENT_ID and (
        not analysis_agent_goal or not analysis_agent_backstory
    ):
        raise OrchestrationError(
            "custom_document_analyst requires analysis_agent_goal and analysis_agent_backstory."
        )

    return {
        "analysis_agent_goal": analysis_agent_goal or "",
        "analysis_agent_backstory": analysis_agent_backstory or "",
    }


def _analysis_instructions(
    analysis_agent_id: str,
    analysis_agent_goal: str | None,
    analysis_agent_backstory: str | None,
) -> str:
    """Build task instructions for the selected analysis agent.

    Args:
        analysis_agent_id (str): Selected analysis agent profile ID.
        analysis_agent_goal (str | None): Goal text from the UI.
        analysis_agent_backstory (str | None): Backstory text from the UI.

    Returns:
        str: Instructions inserted into the analysis task prompt.
    """
    if analysis_agent_id == CUSTOM_ANALYSIS_AGENT_ID:
        return (
            f"Goal: {analysis_agent_goal}\n"
            f"Backstory: {analysis_agent_backstory}\n"
            "Follow this custom analyst profile while still returning the required JSON schema."
        )
    if analysis_agent_id == DEFAULT_ANALYSIS_AGENT_ID:
        return "Extract teachable concepts that can support instructional workflows."
    return "Follow the selected analyst profile while returning the required JSON schema."


def _format_agent_field(value: str, inputs: dict[str, str], agent_id: str) -> str:
    """Format one agent profile field with runtime values.

    Args:
        value (str): Agent profile field value from YAML.
        inputs (dict[str, str]): Runtime placeholder values.
        agent_id (str): Agent profile ID being formatted.

    Raises:
        OrchestrationError: If the YAML field references an unknown placeholder.

    Returns:
        str: Formatted agent profile field.
    """
    try:
        return value.format(**inputs)
    except KeyError as exc:
        raise OrchestrationError(
            f"Agent profile '{agent_id}' references unknown placeholder: {exc}"
        ) from exc


def _crewai_model_name(provider: str, model: str) -> str:
    """Format a configured provider/model pair for CrewAI's LLM wrapper.

    Args:
        provider (str): Provider namespace from config.
        model (str): Provider model name from config.

    Returns:
        str: CrewAI-compatible model identifier.
    """
    normalized = provider.lower().strip()
    if normalized in {"openai", "azure"}:
        return model
    if normalized == "gemini":
        normalized = "google"
    return f"{normalized}/{model}"


def _context_warning(text: str, max_tokens: int) -> str:
    """Build a warning describing the LLM context size used for a document.

    Args:
        text (str): Source document text.
        max_tokens (int): Maximum token budget for LLM context chunking.

    Returns:
        str: Human-readable context chunk and token estimate warning.
    """
    chunked = chunk_text_for_llm(text, max_tokens=max_tokens)
    return (
        f"LLM context built from {chunked.chunk_count} chunk(s); "
        f"approx {chunked.estimated_tokens} tokens before compression."
    )


def _analysis_to_json(analysis: DocumentAnalysis) -> str:
    """Serialize document analysis for insertion into a CrewAI task prompt.

    Args:
        analysis (DocumentAnalysis): Analysis to pass to the slide planner.

    Returns:
        str: ASCII JSON representation of the analysis.
    """
    return json.dumps(
        {
            "summary": analysis.summary,
            "key_concepts": list(analysis.key_concepts),
            "key_terms": list(analysis.key_terms),
            "example_questions": list(analysis.example_questions),
            "cautions": list(analysis.cautions),
        },
        ensure_ascii=True,
    )


def _ensure_question_slides(
    slides: tuple["Slide", ...],
    questions: tuple["ExampleQuestion", ...],
    slide_count: int | None,
    question_count: int,
) -> tuple["Slide", ...]:
    """Replace the final slide slots with normalized question slides.

    Args:
        slides (tuple[Slide, ...]): Parsed slides from the CrewAI response.
        questions (tuple[ExampleQuestion, ...]): Parsed question objects.
        slide_count (int | None): Requested total slide count, if known.
        question_count (int): Requested number of question slides.

    Returns:
        tuple[Slide, ...]: Slides trimmed to the requested total with question
            slides at the end.
    """
    if question_count <= 0 or not questions:
        return slides

    total = slide_count or len(slides)
    question_total = min(question_count, len(questions), total)
    content_total = max(total - question_total, 0)
    content_slides = slides[:content_total]
    question_slides = tuple(
        Slide(
            title=f"Question {index}",
            bullets=(question.prompt, f"Answer: {question.answer}"),
            speaker_notes=question.answer,
        )
        for index, question in enumerate(questions[:question_total], start=1)
    )
    return (content_slides + question_slides)[:total]


def _plain_slide_text(text: str) -> str:
    """Strip Markdown and bullet prefixes from model-generated slide text.

    Args:
        text (str): Raw text from a model.

    Returns:
        str: Plain text suitable for PowerPoint rendering.
    """
    clean = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)
    clean = re.sub(r"[*_`~]+", "", clean)
    clean = re.sub(r"^\s*(?:[-*+]|\d+[.)]|\u2022)\s+", "", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()
