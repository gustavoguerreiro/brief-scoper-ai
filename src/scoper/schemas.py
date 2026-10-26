from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Risk(BaseModel):
    description: str = Field(min_length=1)
    severity: Literal["low", "medium", "high"]


class Estimate(BaseModel):
    low: float = Field(ge=0)
    high: float = Field(ge=0)
    unit: Literal["hours", "usd"]
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ordered(self) -> "Estimate":
        if self.high < self.low:
            raise ValueError("estimate.high must be >= estimate.low")
        return self


class ScopeOutput(BaseModel):
    summary: str = Field(min_length=1)
    open_questions: list[str]
    risks: list[Risk]
    estimate: Estimate | None
    insufficient_info: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]


class ScopeRequest(BaseModel):
    brief: str = Field(min_length=20, max_length=20_000)


class ScopeResponse(BaseModel):
    run_id: str
    output: ScopeOutput
    cost_usd: float
    llm_calls: int
    latency_ms: float
