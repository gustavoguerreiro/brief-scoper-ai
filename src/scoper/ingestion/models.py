"""Brief file format: YAML front matter + four markdown sections (see data/briefs/_TEMPLATE.md)."""

import re
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

REQUIRED_SECTIONS = ("context", "scope", "constraints", "outcome")


class BriefDoc(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    title: str = Field(min_length=1)
    industry: str = Field(min_length=1)
    project_type: Literal[
        "web-app",
        "mobile-app",
        "integration",
        "data-platform",
        "internal-tool",
        "ecommerce",
        "other",
    ]
    budget_band: Literal["<10k", "10k-50k", "50k-100k", "100k-250k", "250k+"]
    outcome: Literal["delivered", "won", "lost", "cancelled"]
    actual_hours: float | None = Field(default=None, ge=0)
    actual_cost: float | None = Field(default=None, ge=0)
    visibility: Literal["team", "private"] = "team"
    source: Literal["real", "synthetic"] = "real"
    sections: dict[str, str]
    raw_text: str

    @field_validator("id", "title", "industry", mode="before")
    @classmethod
    def _to_str(cls, v):
        return str(v).strip() if v is not None else v

    @field_validator("industry")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.lower()

    def meta(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "industry": self.industry,
            "project_type": self.project_type,
            "budget_band": self.budget_band,
            "outcome": self.outcome,
            "actual_hours": self.actual_hours,
            "actual_cost": self.actual_cost,
            "visibility": self.visibility,
        }


class BriefParseError(ValueError):
    pass


_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)
_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def parse_brief(text: str) -> BriefDoc:
    m = _FRONT.match(text.replace("\r\n", "\n"))
    if not m:
        raise BriefParseError("missing YAML front matter (--- ... ---) at top of file")
    try:
        front = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        raise BriefParseError(f"invalid YAML front matter: {e}") from e
    if not isinstance(front, dict):
        raise BriefParseError("front matter must be key: value pairs")
    body = m.group(2)
    heads = list(_SECTION.finditer(body))
    sections = {}
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        sections[h.group(1).strip().lower()] = body[h.end() : end].strip()
    missing = [s for s in REQUIRED_SECTIONS if not sections.get(s)]
    if missing:
        raise BriefParseError(f"missing or empty sections: {', '.join(missing)}")
    try:
        return BriefDoc(**front, sections=sections, raw_text=body.strip())
    except Exception as e:  # pydantic ValidationError: keep message short and readable
        raise BriefParseError(_short(e)) from e


def _short(e: Exception) -> str:
    errs = getattr(e, "errors", None)
    if not errs:
        return str(e)
    return "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in errs())
