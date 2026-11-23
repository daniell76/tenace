from __future__ import annotations

from collections.abc import Iterable, Mapping
import os
from pathlib import Path
import re
import threading
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from tenace.layer1_strategist.conventions import ConventionCard

BUILTIN_BIDDING_SKILLS_DIR: Path = Path(__file__).resolve().parent / "bidding"
DEFAULT_BIDDING_SYSTEM_ID: str = "sayc"

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_FRONTMATTER_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$")
_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", flags=re.MULTILINE)

__all__ = [
    "BUILTIN_BIDDING_SKILLS_DIR",
    "DEFAULT_BIDDING_SYSTEM_ID",
    "BiddingSkill",
    "BiddingSkillMeta",
    "SkillNotFoundError",
    "SkillRegistry",
    "default_bidding_system_from_env",
    "default_skill_registry",
    "load_bidding_skill",
    "parse_skill_markdown",
    "parse_skill_metadata",
    "render_skill_markdown",
    "validate_skill_slug",
]


class SkillNotFoundError(KeyError):
    """Raised when a valid bidding-skill identifier cannot be found in the registry."""


def validate_skill_slug(raw_id: str) -> str:
    """Validate and normalize a bidding-system slug ID (safe against path traversal)."""
    if not isinstance(raw_id, str):
        raise ValueError("Skill id must be a string")
    slug = raw_id.strip().lower()
    if not slug or not _SLUG_RE.match(slug):
        raise ValueError(
            f"Invalid bidding skill id {raw_id!r}: must match ^[a-z0-9][a-z0-9_-]{{0,63}}$"
        )
    return slug


def _unquote_scalar(raw_val: str) -> str:
    val = raw_val.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
        inner = val[1:-1]
        if val[0] == '"':
            inner = (
                inner.replace('\\"', '"')
                .replace("\\n", "\n")
                .replace("\\\\", "\\")
            )
        return inner.strip()
    return val


def _split_frontmatter(markdown_text: str) -> tuple[dict[str, str], str]:
    """Extract strict single-level YAML-style frontmatter (`--- ... ---`) and body."""
    if not isinstance(markdown_text, str):
        raise ValueError("SKILL.md content must be a string")
    normalized = markdown_text.replace("\r\n", "\n").lstrip("\ufeff")
    lines = normalized.split("\n")
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must begin with YAML frontmatter delimited by '---'")

    closing_idx: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            closing_idx = i
            break
    if closing_idx is None:
        raise ValueError("Unclosed YAML frontmatter: missing closing '---' line")

    meta: dict[str, str] = {}
    for line_no, raw_line in enumerate(lines[1:closing_idx], start=2):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _FRONTMATTER_LINE_RE.match(stripped)
        if match is None:
            raise ValueError(
                f"Invalid frontmatter entry on line {line_no}: {raw_line!r}"
            )
        key = match.group(1).lower()
        if key in meta:
            raise ValueError(f"Duplicate frontmatter key {key!r} on line {line_no}")
        meta[key] = _unquote_scalar(match.group(2))

    body = "\n".join(lines[closing_idx + 1 :]).strip()
    return meta, body


def _read_frontmatter_only(path: Path) -> dict[str, str]:
    """Read only up to the closing frontmatter line for fast metadata indexing."""
    header_lines: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        first = fh.readline()
        if first.lstrip("\ufeff").strip() != "---":
            raise ValueError(f"{path}: missing opening '---' frontmatter delimiter")
        header_lines.append("---")
        for raw_line in fh:
            stripped = raw_line.rstrip("\r\n")
            header_lines.append(stripped)
            if stripped.strip() == "---":
                break
        else:
            raise ValueError(f"{path}: unclosed YAML frontmatter")
    meta, _ = _split_frontmatter("\n".join(header_lines))
    return meta


def _parse_meta_dict(raw: dict[str, str], *, is_builtin: bool) -> BiddingSkillMeta:
    raw_id = raw.get("id", "")
    slug = validate_skill_slug(raw_id)
    name = raw.get("name", "").strip()
    if not name:
        raise ValueError(f"Skill {slug!r} frontmatter requires a non-empty 'name'")
    description = raw.get("description", "").strip()
    if not description:
        raise ValueError(
            f"Skill {slug!r} frontmatter requires a non-empty 'description'"
        )
    version = raw.get("version", "1.0").strip() or "1.0"
    nt_range = raw.get("nt_range", "").strip() or None
    slam_control = raw.get("slam_control", "").strip() or None
    opening_style = raw.get("opening_style", "").strip() or None
    return BiddingSkillMeta(
        id=slug,
        name=name,
        version=version,
        description=description,
        nt_range=nt_range,
        slam_control=slam_control,
        opening_style=opening_style,
        is_builtin=is_builtin,
    )


def _extract_sections(body: str) -> tuple[str, str, str]:
    """Split Markdown body into `(summary_rules, detailed_rules, calibration_notes)`."""
    text = body.strip()
    if not text:
        raise ValueError("SKILL.md must define non-empty 'Summary Rules' body text")

    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return text, "", ""

    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    for idx, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections.append((heading, content))

    summary_parts: list[str] = []
    detailed_parts: list[str] = []
    calibration_parts: list[str] = []
    for heading, content in sections:
        if not content:
            continue
        heading_key = heading.lower()
        if heading_key in ("summary rules", "summary", "rules"):
            summary_parts.append(content)
        elif heading_key in ("calibration notes", "calibration overrides", "calibration"):
            calibration_parts.append(content)
        else:
            detailed_parts.append(f"### {heading}\n{content}")

    if not summary_parts:
        if preamble:
            summary_parts.append(preamble)
        else:
            raise ValueError(
                "SKILL.md must define a non-empty '## Summary Rules' section"
            )

    summary_rules = "\n\n".join(summary_parts).strip()
    detailed_rules = "\n\n".join(detailed_parts).strip()
    calibration_notes = "\n\n".join(calibration_parts).strip()
    if not summary_rules:
        raise ValueError("SKILL.md '## Summary Rules' section cannot be empty")
    return summary_rules, detailed_rules, calibration_notes


class BiddingSkillMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    version: str = "1.0"
    description: str
    nt_range: str | None = None
    slam_control: str | None = None
    opening_style: str | None = None
    is_builtin: bool = False

    @field_validator("id")
    @classmethod
    def _check_slug(cls, value: str) -> str:
        return validate_skill_slug(value)


class BiddingSkill(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    meta: BiddingSkillMeta
    summary_rules: str
    detailed_rules: str = ""
    calibration_notes: str = ""
    source_path: Path | None = Field(default=None)

    def to_convention_card(self) -> ConventionCard:
        from tenace.layer1_strategist.conventions import ConventionCard

        return ConventionCard(
            id=self.meta.id,
            name=self.meta.name,
            summary_rules=self.summary_rules,
            detailed_rules=self.detailed_rules,
            calibration_notes=self.calibration_notes,
        )


def _yaml_scalar(value: str) -> str:
    if any(ch in value for ch in (":", "#", '"', "'", "\n")) or value.strip() != value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    return value


def render_skill_markdown(
    meta: BiddingSkillMeta,
    *,
    summary_rules: str,
    detailed_rules: str = "",
    calibration_notes: str = "",
) -> str:
    """Serialize a bidding-system skill cleanly to `SKILL.md` markdown with YAML frontmatter."""
    fm_lines = [
        "---",
        f"id: {validate_skill_slug(meta.id)}",
        f"name: {_yaml_scalar(meta.name.strip())}",
        f"version: {_yaml_scalar(meta.version.strip() or '1.0')}",
        f"description: {_yaml_scalar(meta.description.strip())}",
    ]
    if meta.nt_range:
        fm_lines.append(f"nt_range: {_yaml_scalar(meta.nt_range.strip())}")
    if meta.slam_control:
        fm_lines.append(f"slam_control: {_yaml_scalar(meta.slam_control.strip())}")
    if meta.opening_style:
        fm_lines.append(f"opening_style: {_yaml_scalar(meta.opening_style.strip())}")
    fm_lines.append("---")

    parts = [
        "\n".join(fm_lines),
        f"## Summary Rules\n\n{summary_rules.strip()}",
    ]
    ref_body = detailed_rules.strip()
    if ref_body:
        # Convert stored `### <Heading>` blocks back into top-level `## <Heading>` sections.
        sub_blocks = re.split(r"(?m)^###\s+", ref_body)
        if len(sub_blocks) > 1 and not sub_blocks[0].strip():
            for blk in sub_blocks[1:]:
                first_nl = blk.find("\n")
                if first_nl == -1:
                    continue
                heading = blk[:first_nl].strip()
                content = blk[first_nl + 1 :].strip()
                if heading and content:
                    parts.append(f"## {heading}\n\n{content}")
        else:
            parts.append(f"## Reference Guide\n\n{ref_body}")
    cal_body = calibration_notes.strip()
    if cal_body:
        parts.append(f"## Calibration Notes\n\n{cal_body}")
    return "\n\n".join(parts) + "\n"


def parse_skill_metadata(
    markdown_text: str, *, is_builtin: bool = False
) -> BiddingSkillMeta:
    raw, _ = _split_frontmatter(markdown_text)
    return _parse_meta_dict(raw, is_builtin=is_builtin)


def parse_skill_markdown(
    markdown_text: str,
    *,
    source_path: Path | None = None,
    is_builtin: bool = False,
) -> BiddingSkill:
    raw, body = _split_frontmatter(markdown_text)
    meta = _parse_meta_dict(raw, is_builtin=is_builtin)
    summary_rules, detailed_rules, calibration_notes = _extract_sections(body)
    return BiddingSkill(
        meta=meta,
        summary_rules=summary_rules,
        detailed_rules=detailed_rules,
        calibration_notes=calibration_notes,
        source_path=source_path,
    )


class SkillRegistry:
    """Progressive, mtime-aware dynamic registry for bidding-system `SKILL.md` skills."""

    def __init__(
        self,
        *,
        builtin_dir: Path | None = None,
        extra_dirs: Iterable[Path | str] | None = None,
    ) -> None:
        self.builtin_dir: Path = (
            Path(builtin_dir).resolve()
            if builtin_dir is not None
            else BUILTIN_BIDDING_SKILLS_DIR
        )
        self.extra_dirs: tuple[Path, ...] = tuple(
            Path(d).expanduser().resolve() for d in (extra_dirs or ())
        )
        self._lock = threading.Lock()
        self._cache: dict[Path, tuple[int, BiddingSkill]] = {}

    def _iter_skill_files(self) -> list[tuple[Path, bool]]:
        found: list[tuple[Path, bool]] = []
        roots: list[tuple[Path, bool]] = [(self.builtin_dir, True)]
        roots.extend((root, False) for root in self.extra_dirs)
        for root, is_builtin in roots:
            if not root.is_dir():
                continue
            for entry in sorted(root.iterdir(), key=lambda p: p.name):
                if entry.is_dir():
                    candidate = entry / "SKILL.md"
                    if candidate.is_file():
                        found.append((candidate.resolve(), is_builtin))
                elif entry.is_file() and entry.name.upper() == "SKILL.MD":
                    found.append((entry.resolve(), is_builtin))
        return found

    def _index(self) -> dict[str, tuple[Path, bool, BiddingSkillMeta]]:
        index: dict[str, tuple[Path, bool, BiddingSkillMeta]] = {}
        for path, is_builtin in self._iter_skill_files():
            raw = _read_frontmatter_only(path)
            meta = _parse_meta_dict(raw, is_builtin=is_builtin)
            # Later directories override earlier entries sharing the same id.
            index[meta.id] = (path, is_builtin, meta)
        return index

    def _load_file(self, path: Path, *, is_builtin: bool) -> BiddingSkill:
        resolved = path.resolve()
        if not resolved.is_file():
            raise SkillNotFoundError(f"Skill file does not exist: {path}")
        mtime_ns = resolved.stat().st_mtime_ns
        with self._lock:
            cached = self._cache.get(resolved)
            if cached is not None and cached[0] == mtime_ns:
                return cached[1]
        skill = parse_skill_markdown(
            resolved.read_text(encoding="utf-8"),
            source_path=resolved,
            is_builtin=is_builtin,
        )
        with self._lock:
            self._cache[resolved] = (mtime_ns, skill)
        return skill

    def list_skills(self) -> list[BiddingSkillMeta]:
        """Return lightweight frontmatter metadata for all discovered skills (`id` sorted)."""
        index = self._index()
        return [index[k][2] for k in sorted(index.keys())]

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def load(
        self,
        skill_id_or_path: str,
        *,
        allow_file_path: bool = False,
    ) -> BiddingSkill:
        """Dynamically load a skill by registered slug ID or (local-only) `SKILL.md` path."""
        if not isinstance(skill_id_or_path, str) or not skill_id_or_path.strip():
            raise ValueError("Bidding skill identifier must be a non-blank string")
        raw = skill_id_or_path.strip()

        if allow_file_path:
            maybe_path = Path(raw).expanduser()
            looks_like_path = (
                os.sep in raw
                or (os.altsep is not None and os.altsep in raw)
                or raw.startswith(".")
                or raw.lower().endswith(".md")
                or maybe_path.exists()
            )
            if looks_like_path:
                target = maybe_path / "SKILL.md" if maybe_path.is_dir() else maybe_path
                if not target.is_file():
                    raise SkillNotFoundError(
                        f"Bidding skill file not found: {target}"
                    )
                return self._load_file(target, is_builtin=False)

        slug = validate_skill_slug(raw)
        index = self._index()
        entry = index.get(slug)
        if entry is None:
            available = ", ".join(sorted(index.keys())) or "(none)"
            raise SkillNotFoundError(
                f"Unknown bidding skill {slug!r}. Available: {available}"
            )
        path, is_builtin, _meta = entry
        return self._load_file(path, is_builtin=is_builtin)


def _extra_dirs_from_env(env: Mapping[str, str] | None = None) -> list[Path]:
    mapping = os.environ if env is None else env
    raw = mapping.get("TENACE_SKILLS_DIR", "").strip()
    if not raw:
        return []
    return [Path(part.strip()) for part in raw.split(os.pathsep) if part.strip()]


def default_skill_registry(
    env: Mapping[str, str] | None = None,
    *,
    extra_dirs: Iterable[Path | str] | None = None,
) -> SkillRegistry:
    dirs: list[Path | str] = list(_extra_dirs_from_env(env))
    if extra_dirs is not None:
        dirs.extend(extra_dirs)
    return SkillRegistry(extra_dirs=dirs)


def default_bidding_system_from_env(env: Mapping[str, str] | None = None) -> str:
    mapping = os.environ if env is None else env
    raw = mapping.get("TENACE_BIDDING_SYSTEM", "").strip()
    return raw if raw else DEFAULT_BIDDING_SYSTEM_ID


def load_bidding_skill(
    skill_id_or_path: str = DEFAULT_BIDDING_SYSTEM_ID,
    *,
    extra_dirs: Iterable[Path | str] | None = None,
    allow_file_path: bool = False,
) -> BiddingSkill:
    registry = default_skill_registry(extra_dirs=extra_dirs)
    return registry.load(skill_id_or_path, allow_file_path=allow_file_path)
