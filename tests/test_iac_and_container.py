from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TERRAFORM_BIN = shutil.which("terraform")


def _run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


@pytest.fixture(scope="module")
def terraform_workspace(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, str]]:
    if TERRAFORM_BIN is None:
        pytest.skip("terraform CLI is not installed")

    base_dir = tmp_path_factory.mktemp("tf_iac")
    plugin_cache = base_dir / "plugin_cache"
    plugin_cache.mkdir(parents=True, exist_ok=True)

    tf_src = REPO_ROOT / "terraform"
    assert tf_src.is_dir(), "Missing terraform/ directory at repository root"

    tf_copy = base_dir / "terraform"
    shutil.copytree(
        tf_src,
        tf_copy,
        ignore=shutil.ignore_patterns(".terraform", ".terraform.lock.hcl", "local", "*.tfstate*"),
    )

    env = dict(os.environ)
    env["TF_PLUGIN_CACHE_DIR"] = str(plugin_cache)
    env["TF_IN_AUTOMATION"] = "1"
    return tf_copy, env


def _run_terraform(
    chdir: Path,
    *args: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    assert TERRAFORM_BIN is not None, "terraform CLI is required"
    return subprocess.run(
        [TERRAFORM_BIN, f"-chdir={chdir}", *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_gitignore_protects_local_tfvars_backends_state_and_plugins() -> None:
    must_ignore = [
        "terraform/local/bootstrap/terraform.tfvars",
        "terraform/local/dev/terraform.tfvars",
        "terraform/local/dev/backend.tfbackend",
        "terraform/local/prod/terraform.tfvars",
        "terraform/local/prod/backend.tfbackend",
        "terraform/environments/dev/terraform.tfstate",
        "terraform/environments/dev/terraform.tfstate.backup",
        "terraform/environments/dev/secret.tfvars",
        "terraform/environments/dev/secret.tfvars.json",
        "terraform/environments/dev/custom.tfbackend",
        "terraform/environments/dev/.terraform/providers/registry.terraform.io/lock",
        "terraform/environments/dev/.terraform.lock.hcl",
        "crash.log",
        "crash.20261112.log",
    ]
    for rel_path in must_ignore:
        res = _run_git("check-ignore", "-q", rel_path, check=False)
        assert res.returncode == 0, f"Expected .gitignore to ignore {rel_path!r}"

    must_not_ignore = [
        "terraform/bootstrap/terraform.tfvars.example",
        "terraform/environments/dev/terraform.tfvars.example",
        "terraform/environments/dev/backend.tfbackend.example",
        "terraform/environments/prod/terraform.tfvars.example",
        "terraform/environments/prod/backend.tfbackend.example",
    ]
    for rel_path in must_not_ignore:
        res = _run_git("check-ignore", "-q", rel_path, check=False)
        assert res.returncode == 1, f"Expected .gitignore NOT to ignore {rel_path!r}"


def test_dockerfile_and_dockerignore_cloudrun_hygiene() -> None:
    dockerfile_path = REPO_ROOT / "Dockerfile"
    dockerignore_path = REPO_ROOT / ".dockerignore"
    assert dockerfile_path.is_file()
    assert dockerignore_path.is_file()

    dockerfile = dockerfile_path.read_text(encoding="utf-8")
    assert "FROM python:3.13-slim" in dockerfile
    assert "procps" in dockerfile
    assert "PORT=8080" in dockerfile
    assert "EXPOSE 8080" in dockerfile
    assert re.search(r"^USER\s+(?!root\b)\S+", dockerfile, re.MULTILINE)
    assert 'CMD ["tenace-serve"]' in dockerfile

    dockerignore_lines = {
        line.strip().rstrip("/")
        for line in dockerignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    for expected in (".git", "terraform", "tests", ".venv"):
        assert expected in dockerignore_lines, f"Missing {expected!r} in .dockerignore"


def test_terraform_directory_structure_and_docs() -> None:
    expected_files = [
        "terraform/README.md",
        "terraform/bootstrap/versions.tf",
        "terraform/bootstrap/variables.tf",
        "terraform/bootstrap/main.tf",
        "terraform/bootstrap/outputs.tf",
        "terraform/bootstrap/terraform.tfvars.example",
        "terraform/bootstrap/tests/bootstrap.tftest.hcl",
        "terraform/modules/tenace_cloudrun/versions.tf",
        "terraform/modules/tenace_cloudrun/variables.tf",
        "terraform/modules/tenace_cloudrun/main.tf",
        "terraform/modules/tenace_cloudrun/outputs.tf",
        "terraform/modules/tenace_cloudrun/tests/cloudrun.tftest.hcl",
        "terraform/environments/dev/backend.tf",
        "terraform/environments/dev/versions.tf",
        "terraform/environments/dev/variables.tf",
        "terraform/environments/dev/main.tf",
        "terraform/environments/dev/outputs.tf",
        "terraform/environments/dev/backend.tfbackend.example",
        "terraform/environments/dev/terraform.tfvars.example",
        "terraform/environments/prod/backend.tf",
        "terraform/environments/prod/versions.tf",
        "terraform/environments/prod/variables.tf",
        "terraform/environments/prod/main.tf",
        "terraform/environments/prod/outputs.tf",
        "terraform/environments/prod/backend.tfbackend.example",
        "terraform/environments/prod/terraform.tfvars.example",
    ]
    for rel_path in expected_files:
        assert (REPO_ROOT / rel_path).is_file(), f"Missing expected file: {rel_path}"

    for env_name in ("dev", "prod"):
        backend_hcl = (
            REPO_ROOT / "terraform" / "environments" / env_name / "backend.tf"
        ).read_text(encoding="utf-8")
        assert 'backend "gcs"' in backend_hcl

    dev_vars = (REPO_ROOT / "terraform/environments/dev/variables.tf").read_text(encoding="utf-8")
    prod_vars = (REPO_ROOT / "terraform/environments/prod/variables.tf").read_text(encoding="utf-8")
    mod_vars = (REPO_ROOT / "terraform/modules/tenace_cloudrun/variables.tf").read_text(
        encoding="utf-8"
    )
    assert re.search(
        r'variable\s+"job_args"\s*\{[^}]*default\s*=\s*\["--boards",\s*"4"\]',
        mod_vars,
        re.DOTALL,
    )
    assert re.search(
        r'variable\s+"job_args"\s*\{[^}]*default\s*=\s*\["--boards",\s*"4"\]',
        dev_vars,
        re.DOTALL,
    )
    assert re.search(
        r'variable\s+"job_args"\s*\{[^}]*default\s*=\s*\["--boards",\s*"8"\]',
        prod_vars,
        re.DOTALL,
    )
    assert re.search(
        r'variable\s+"deletion_protection"\s*\{[^}]*default\s*=\s*false',
        dev_vars,
        re.DOTALL,
    )
    assert re.search(
        r'variable\s+"deletion_protection"\s*\{[^}]*default\s*=\s*true',
        prod_vars,
        re.DOTALL,
    )
    assert re.search(
        r'variable\s+"max_instance_count"\s*\{[^}]*default\s*=\s*2\b',
        dev_vars,
        re.DOTALL,
    )
    assert re.search(
        r'variable\s+"max_instance_count"\s*\{[^}]*default\s*=\s*10\b',
        prod_vars,
        re.DOTALL,
    )

    tf_readme = (REPO_ROOT / "terraform/README.md").read_text(encoding="utf-8")
    root_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for doc_text in (tf_readme, root_readme):
        assert "your-gcp-project-id" in doc_text
        assert "gcloud builds submit" in doc_text
        assert "/health" in doc_text
        assert "/v1/step" in doc_text
        assert "/v1/match" in doc_text
        assert "gcloud run jobs execute" in doc_text
    assert "terraform/local/" in tf_readme
    assert "-backend-config=" in tf_readme


def test_terraform_fmt_check() -> None:
    assert TERRAFORM_BIN is not None, "terraform CLI is required"
    tf_dir = REPO_ROOT / "terraform"
    assert tf_dir.is_dir(), "Missing terraform/ directory"
    res = subprocess.run(
        [TERRAFORM_BIN, "fmt", "-check", "-recursive", str(tf_dir)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert res.returncode == 0, (
        f"terraform fmt -check failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )


@pytest.mark.parametrize(
    "stack_rel",
    [
        "bootstrap",
        "environments/dev",
        "environments/prod",
    ],
)
def test_terraform_validate_all_stacks(
    terraform_workspace: tuple[Path, dict[str, str]],
    stack_rel: str,
) -> None:
    tf_copy, env = terraform_workspace
    stack_dir = tf_copy / stack_rel

    init_res = _run_terraform(stack_dir, "init", "-backend=false", "-input=false", env=env)
    assert init_res.returncode == 0, (
        f"terraform init failed for {stack_rel}:\n{init_res.stdout}\n{init_res.stderr}"
    )

    val_res = _run_terraform(stack_dir, "validate", env=env)
    assert val_res.returncode == 0, (
        f"terraform validate failed for {stack_rel}:\n{val_res.stdout}\n{val_res.stderr}"
    )


@pytest.mark.parametrize(
    "stack_rel",
    [
        "bootstrap",
        "modules/tenace_cloudrun",
    ],
)
def test_terraform_native_tests(
    terraform_workspace: tuple[Path, dict[str, str]],
    stack_rel: str,
) -> None:
    tf_copy, env = terraform_workspace
    stack_dir = tf_copy / stack_rel

    init_res = _run_terraform(stack_dir, "init", "-backend=false", "-input=false", env=env)
    assert init_res.returncode == 0, (
        f"terraform init failed for {stack_rel}:\n{init_res.stdout}\n{init_res.stderr}"
    )

    test_res = _run_terraform(stack_dir, "test", env=env)
    assert test_res.returncode == 0, (
        f"terraform test failed for {stack_rel}:\n{test_res.stdout}\n{test_res.stderr}"
    )
    assert "0 failed" in test_res.stdout


def test_tracked_files_contain_no_sensitive_identifiers() -> None:
    tracked_files = [
        line.strip()
        for line in _run_git("ls-files").stdout.splitlines()
        if line.strip()
    ]
    assert tracked_files, "Expected tracked files in git repository"

    for rel_path in tracked_files:
        assert not rel_path.startswith("terraform/local/"), (
            f"Local Terraform file must never be tracked by git: {rel_path}"
        )

    status_lines = _run_git("status", "--porcelain").stdout.splitlines()
    for line in status_lines:
        assert "terraform/local" not in line, (
            f"terraform/local/ must be completely ignored by git status, got: {line}"
        )

    dynamic_forbidden_literals: set[str] = set()
    home_str = str(Path.home())
    if len(home_str) >= 6:
        dynamic_forbidden_literals.add(home_str)
    home_parent = str(Path.home().parent)
    if len(home_parent) >= 8 and home_parent not in {"/home", "/Users"}:
        dynamic_forbidden_literals.add(home_parent)
    home_user = Path.home().name
    if len(home_user) >= 4 and home_user.lower() not in {"root", "runner", "ubuntu", "user"}:
        dynamic_forbidden_literals.add(home_user)

    local_tf_dir = REPO_ROOT / "terraform" / "local"
    if local_tf_dir.is_dir():
        for local_file in local_tf_dir.rglob("*"):
            if not local_file.is_file():
                continue
            local_text = local_file.read_text(encoding="utf-8")
            for match in re.finditer(
                r'^\s*(?:project_id|tfstate_bucket_name|bucket)\s*=\s*"([^"]+)"',
                local_text,
                re.MULTILINE,
            ):
                val = match.group(1).strip()
                if val and "your-gcp-project-id" not in val:
                    dynamic_forbidden_literals.add(val)

    forbidden_regexes = [
        (re.compile(r"\b[a-z][a-z0-9-]{3,23}-\d{6}\b"), "numbered GCP project ID"),
        (re.compile(r"\b[a-z0-9.-]+\.goog\b", re.IGNORECASE), "internal .goog domain"),
        (re.compile(r"/" + r"google/(?:data|src)/"), "internal /google/{data,src}/ path"),
    ]

    for rel_path in tracked_files:
        file_path = REPO_ROOT / rel_path
        if not file_path.is_file():
            continue
        content = file_path.read_text(encoding="utf-8")
        for literal in dynamic_forbidden_literals:
            assert literal not in content, (
                f"Tracked file {rel_path} contains forbidden local identifier"
            )
        for pattern, description in forbidden_regexes:
            match = pattern.search(content)
            assert match is None, (
                f"Tracked file {rel_path} contains {description}: {match.group(0)!r}"
            )


def test_code_and_docs_contain_no_outdated_gemini_2_5_or_gemini_3_0_models() -> None:
    outdated_model_re = re.compile(r"gemini-(?:2\.5|3-)", re.IGNORECASE)
    checked_paths = [
        REPO_ROOT / "README.md",
        *sorted((REPO_ROOT / "src").rglob("*")),
        *sorted((REPO_ROOT / "tests").rglob("*")),
        *sorted((REPO_ROOT / "terraform").rglob("*")),
    ]
    for path in checked_paths:
        if not path.is_file():
            continue
        rel_parts = path.relative_to(REPO_ROOT).parts
        if any(part in {".terraform", "__pycache__", "local"} for part in rel_parts):
            continue
        if path.name == "test_iac_and_container.py":
            continue
        text = path.read_text(encoding="utf-8")
        match = outdated_model_re.search(text)
        assert match is None, (
            f"Outdated Gemini model reference {match.group(0)!r} found in "
            f"{path.relative_to(REPO_ROOT)}"
        )
