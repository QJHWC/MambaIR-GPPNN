"""Repository-level checks that do not require network access."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]

INLINE_LINK_RE = re.compile(r"!?\[[^\]]*]\(\s*(?:<([^>]+)>|([^\s)]+))")
REFERENCE_LINK_RE = re.compile(
    r"^\s*\[(?!\^)[^\]]+]:\s*(?:<([^>]+)>|(\S+))",
    re.MULTILINE,
)
PLACEHOLDER_RE = re.compile(
    r"your[-_ ]?(?:name|email|repo|repository|username|docs[-_ ]?url)"
    r"|github\.com/(?:your[-_ ]?|username)"
    r"|https?://example\.com"
    r"|<\s*(?:author|email|username|repository[-_ ]?url|your[-_ ][^>]+)\s*>",
    re.IGNORECASE,
)

IGNORED_PATH_SAMPLES = [
    "__pycache__/module.cpython-310.pyc",
    ".pytest_cache/v/cache/nodeids",
    ".ruff_cache/content",
    ".venv/pyvenv.cfg",
    ".idea/workspace.xml",
    ".vscode/settings.json",
    ".claude/settings.local.json",
    ".cargo/bin/rustup",
    "photo/dataset/example.png",
    "datasets/example.tif",
    "data/raw/example.tif",
    "checkpoints/run/model.pth",
    "checkpoints_optimized/run/model.pth",
    "logs/events.out.tfevents.123",
    "results/prediction.png",
    "outputs/prediction.png",
    "runs/experiment/metrics.json",
    "wandb/run/files/config.yaml",
]

TRACKED_PATH_SAMPLES = [
    "data/photo_dataloader.py",
    "models/mambair_gppnn.py",
]

FORBIDDEN_TRACKED_PARTS = {
    ".cargo",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "logs",
    "outputs",
    "photo",
    "results",
    "runs",
    "wandb",
}

FORBIDDEN_TRACKED_SUFFIXES = {
    ".ckpt",
    ".log",
    ".onnx",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".safetensors",
}


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run Git against the repository with stable text decoding."""

    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def tracked_files() -> list[PurePosixPath]:
    output = git("ls-files", "-z").stdout
    return [PurePosixPath(item) for item in output.split("\0") if item]


@pytest.mark.parametrize("path", IGNORED_PATH_SAMPLES)
def test_generated_paths_are_ignored(path: str) -> None:
    result = git("check-ignore", "--no-index", "--quiet", "--", path, check=False)
    assert result.returncode == 0, f"{path!r} should be ignored"


@pytest.mark.parametrize("path", TRACKED_PATH_SAMPLES)
def test_source_paths_are_not_ignored(path: str) -> None:
    result = git("check-ignore", "--no-index", "--quiet", "--", path, check=False)
    assert result.returncode == 1, f"{path!r} should remain trackable"


def test_no_ignored_or_generated_files_are_tracked() -> None:
    ignored_tracked = git("ls-files", "-ci", "--exclude-standard").stdout.splitlines()
    assert not ignored_tracked, f"ignored files are tracked: {ignored_tracked}"

    forbidden: list[str] = []
    for path in tracked_files():
        lower_parts = {part.lower() for part in path.parts}
        top = path.parts[0].lower()
        if (
            lower_parts & FORBIDDEN_TRACKED_PARTS
            or top.startswith("checkpoints")
            or top.startswith("logs_")
            or top.startswith("results_")
            or path.suffix.lower() in FORBIDDEN_TRACKED_SUFFIXES
            or path.name.startswith("events.out.tfevents.")
        ):
            forbidden.append(path.as_posix())

    assert not forbidden, f"generated artifacts are tracked: {forbidden}"


def test_no_lfs_rules_or_pointer_files() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").lower()
    for rule in ("filter=lfs", "diff=lfs", "merge=lfs"):
        assert rule not in attributes

    pointers: list[str] = []
    signature = b"version https://git-lfs.github.com/spec/v1"
    for path in tracked_files():
        disk_path = ROOT.joinpath(*path.parts)
        if disk_path.is_file() and disk_path.read_bytes()[:200].startswith(signature):
            pointers.append(path.as_posix())

    assert not pointers, f"Git LFS pointer files are tracked: {pointers}"


def test_tracked_files_stay_below_github_size_limit() -> None:
    oversized: list[str] = []
    limit = 10 * 1024 * 1024
    for path in tracked_files():
        disk_path = ROOT.joinpath(*path.parts)
        if disk_path.is_file() and disk_path.stat().st_size > limit:
            oversized.append(path.as_posix())

    assert not oversized, f"tracked files larger than 10 MiB: {oversized}"


def _local_link_target(markdown_file: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().strip("<>")
    if not target or target.startswith(("#", "//")):
        return None

    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return None

    decoded_path = unquote(parsed.path)
    if not decoded_path:
        return None

    if decoded_path.startswith("/"):
        return ROOT / decoded_path.lstrip("/")
    return markdown_file.parent / decoded_path


def test_local_markdown_links_resolve() -> None:
    broken: list[str] = []
    for markdown_file in ROOT.rglob("*.md"):
        if ".git" in markdown_file.parts:
            continue

        text = markdown_file.read_text(encoding="utf-8")
        matches = list(INLINE_LINK_RE.finditer(text)) + list(REFERENCE_LINK_RE.finditer(text))
        for match in matches:
            raw_target = match.group(1) or match.group(2)
            target = _local_link_target(markdown_file, raw_target)
            if target is not None and not target.resolve().exists():
                relative_doc = markdown_file.relative_to(ROOT).as_posix()
                broken.append(f"{relative_doc}: {raw_target}")

    assert not broken, "broken local Markdown links:\n" + "\n".join(broken)


def test_readme_has_no_template_placeholders() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    placeholders = sorted({match.group(0) for match in PLACEHOLDER_RE.finditer(readme)})
    assert not placeholders, f"README placeholders remain: {placeholders}"


def test_mit_license_is_consistent() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2025 Qin Jiahong" in license_text
    assert "Permission is hereby granted, free of charge" in license_text
    assert re.search(r"\bMIT\b", readme)
    assert "License-Apache" not in readme
    assert "License%20Apache" not in readme

    citation = ROOT / "CITATION.cff"
    if citation.exists():
        citation_text = citation.read_text(encoding="utf-8")
        assert re.search(r"(?mi)^license:\s*[\"']?MIT[\"']?\s*$", citation_text)


def test_shell_scripts_are_lf_and_executable() -> None:
    mode_output = git("ls-files", "-s", "--", "*.sh").stdout.splitlines()
    modes = {line.split(maxsplit=3)[3]: line.split(maxsplit=1)[0] for line in mode_output}

    failures: list[str] = []
    for path in tracked_files():
        if path.suffix != ".sh":
            continue

        data = ROOT.joinpath(*path.parts).read_bytes()
        if b"\r\n" in data:
            failures.append(f"{path.as_posix()}: contains CRLF")
        if modes.get(path.as_posix()) != "100755":
            failures.append(f"{path.as_posix()}: Git mode is not executable")

    assert not failures, "shell script hygiene failures:\n" + "\n".join(failures)
