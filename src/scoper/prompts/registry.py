"""Versioned prompts stored as files: prompts/<name>/<version>.md plus prompts/active.json.

Why files in Git: a prompt change becomes a reviewable diff, promotion is a one-line change to
active.json, rollback is `git revert`, and every eval run records exactly which version it used.

Placeholders use <<name>> so JSON braces in a prompt never need escaping.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"
ACTIVE_PATH = PROMPTS_DIR / "active.json"
_PLACEHOLDER = re.compile(r"<<(\w+)>>")
_VERSION = re.compile(r"^v\d+$")
_FRONT = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)


class PromptError(ValueError):
    pass


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str
    notes: str = ""

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:12]

    @property
    def id(self) -> str:
        """Short identifier stored with every eval run, e.g. scope_grounded@v1#a1b2c3d4e5f6."""
        return f"{self.name}@{self.version}#{self.sha}"

    @property
    def placeholders(self) -> set[str]:
        return set(_PLACEHOLDER.findall(self.text))

    def render(self, **values: str) -> str:
        missing, extra = self.placeholders - values.keys(), values.keys() - self.placeholders
        if missing or extra:
            raise PromptError(
                f"{self.id}: missing placeholders {sorted(missing)}, unexpected {sorted(extra)}"
            )
        return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), self.text)


def _parse(name: str, version: str, raw: str) -> Prompt:
    m = _FRONT.match(raw.replace("\r\n", "\n"))
    if not m:
        raise PromptError(f"{name}/{version}: missing front matter")
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    if meta.get("version", "").strip() != version:
        raise PromptError(f"{name}/{version}: front-matter version does not match file name")
    return Prompt(name, version, m.group(2).strip("\n"), meta.get("notes", "").strip())


def list_versions(name: str) -> list[str]:
    d = PROMPTS_DIR / name
    found = [p.stem for p in d.glob("v*.md") if _VERSION.match(p.stem)]
    return sorted(found, key=lambda v: int(v[1:]))


def active_versions() -> dict[str, str]:
    return json.loads(ACTIVE_PATH.read_text(encoding="utf-8"))


def get_prompt(name: str, version: str | None = None) -> Prompt:
    """Resolve: explicit version > env SCOPER_PROMPT_<NAME> (for experiments) > active.json."""
    version = version or os.getenv(f"SCOPER_PROMPT_{name.upper()}") or active_versions().get(name)
    if not version:
        raise PromptError(f"no active version for prompt {name!r}")
    path = PROMPTS_DIR / name / f"{version}.md"
    if not path.exists():
        raise PromptError(f"{name}/{version} not found (have {list_versions(name)})")
    return _parse(name, version, path.read_text(encoding="utf-8"))


def prompt_ids(*names: str) -> dict[str, str]:
    return {n: get_prompt(n).id for n in names}


HASHES_PATH = PROMPTS_DIR / "hashes.json"


def all_versions() -> list[tuple[str, str]]:
    return [
        (d.name, v)
        for d in sorted(PROMPTS_DIR.iterdir())
        if d.is_dir()
        for v in list_versions(d.name)
    ]


def recorded_hashes() -> dict[str, str]:
    return json.loads(HASHES_PATH.read_text(encoding="utf-8")) if HASHES_PATH.exists() else {}


def freeze() -> list[str]:
    """Record the hash of every NEW version. Existing entries are never overwritten:
    a version is immutable once frozen (tests fail if its text changes)."""
    hashes, added = recorded_hashes(), []
    for name, version in all_versions():
        key = f"{name}/{version}"
        if key not in hashes:
            hashes[key] = get_prompt(name, version).sha
            added.append(key)
    text = json.dumps(dict(sorted(hashes.items())), indent=1) + "\n"
    HASHES_PATH.write_text(text, encoding="utf-8", newline="\n")
    return added


if __name__ == "__main__":
    print("frozen:", freeze() or "nothing new")
