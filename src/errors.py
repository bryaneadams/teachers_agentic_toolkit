class SlideMakerError(Exception):
    """Base error for expected workflow failures."""


class InputValidationError(SlideMakerError):
    """Raised when workflow inputs are invalid."""


class RenderingError(SlideMakerError):
    """Raised when PowerPoint rendering fails."""


class OrchestrationError(SlideMakerError):
    """Raised when the CrewAI orchestration layer fails."""
