from pydantic import BaseModel, Field


class DraftingPhase(BaseModel):
    """Structured planning phase before execution."""

    analysis: str = Field(..., description="Analysis of the request. What is the goal?")
    hypothesis: str = Field(..., description="What do you expect to find?")
    strategy: list[str] = Field(..., description="Sequential steps (max 5).")
    risk_assessment: str = Field(
        ..., description="Potential pitfalls (e.g., permissions)."
    )
