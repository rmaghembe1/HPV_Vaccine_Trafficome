#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required for the Phase 3D2F metadata audit."
    ) from exc


ROOT = Path("/mnt/d/HPV_Vaccine_Trafficome_Project_PRISTINE")
EXPECTED_HEAD = "4c0fee0b52387e1d92efb5ec9d669399a7ceb92c"
EXPECTED_TREE = "6da1fb6add196e8501b75149ef76ad638a93720f"
EXPECTED_COMMIT_COUNT = 16
EXPECTED_ROOT_COUNT = 1

MIXED_COMMITS = {
    "e801ba349610af9f67f9bc3ee9b32d35e9923885",
    "1b3a8e8f96fcc2eeca6a662a1385b9964d3ea923",
    "24d4f5b35c751ca665e872c299090f99fcaa9992",
}

RAW_REL = Path(
    "03_data_raw/hpv_specific/fiji_nct02276521/"
    "NCOMMS-24-64334A_HPV_collated_antibody_feature_data.xlsx"
)
EXPECTED_RAW_MD5 = "e42173e1d8297cd64420fd9682c42674"
SOURCE_DOI = "10.5281/zenodo.14848069"
PLANNED_REPOSITORY_URL = (
    "https://github.com/rmaghembe1/HPV_Vaccine_Trafficome"
)
PLANNED_RELEASE_TAG = "v0.1.0-fiji-systems-serology"

PHASE_D_DECISION = (
    ROOT / "08_results/tables/"
    "phase3D2D_hpv_repository_identity_rebuild_decision.tsv"
)
PHASE_E_DECISION = (
    ROOT / "08_results/tables/"
    "phase3D2E_hpv_dual_license_decision.tsv"
)

CITATION_FILE = ROOT / "CITATION.cff"
ENVIRONMENT_FILE = ROOT / "environment.yml"
REQUIREMENTS_FILE = ROOT / "requirements.txt"
DEPENDENCY_FILE = (
    ROOT / "08_results/tables/"
    "phase3D2D_hpv_dependency_inventory.tsv"
)

SCRIPT_REL = Path(
    "06_scripts/python/"
    "phase3D2F_hpv_identity_precommit_audit.py"
)
REPORT_FILE = (
    ROOT / "02_dataset_audit/project_identity/"
    "phase3D2F_hpv_identity_precommit_audit_report.md"
)
MANIFEST_FILE = (
    ROOT / "08_results/tables/"
    "phase3D2F_hpv_precommit_staging_manifest.tsv"
)
DECISION_FILE = (
    ROOT / "08_results/tables/"
    "phase3D2F_hpv_identity_precommit_audit_decision.tsv"
)

SUCCESS = "READY_FOR_HPV_IDENTITY_COMMIT"
FAILURE = "PHASE3D2F_HPV_IDENTITY_PRECOMMIT_REPAIR_REQUIRED"

FORBIDDEN_TOKENS = (
    "Response_" + "to_Editor",
    "Scientific_" + "Reports_revision",
    "af37a99c-" + "209f-4c31-acd3-1893c74da95d",
    "VTrafficome_" + "manuscript",
    "github.com/rmaghembe1/" + "Vaccine_Trafficome",
)

EXPECTED_CHANGE_PATHS = {
    "README.md",
    "02_dataset_audit/project_identity/"
    "phase3D2D_hpv_repository_identity_rebuild_report.md",
    "02_dataset_audit/project_identity/"
    "phase3D2E_hpv_dual_license_report.md",
    "02_dataset_audit/project_identity/"
    "phase3D2F_hpv_identity_precommit_audit_report.md",
    "06_scripts/python/"
    "phase3D2D_hpv_repository_identity_rebuild.py",
    "06_scripts/python/"
    "phase3D2E_hpv_dual_license_implementation.py",
    "06_scripts/python/"
    "phase3D2F_hpv_identity_precommit_audit.py",
    "08_results/tables/"
    "phase3D2D_hpv_dependency_inventory.tsv",
    "08_results/tables/"
    "phase3D2D_hpv_identity_file_manifest.tsv",
    "08_results/tables/"
    "phase3D2D_hpv_repository_identity_rebuild_decision.tsv",
    "08_results/tables/"
    "phase3D2E_hpv_dual_license_decision.tsv",
    "08_results/tables/"
    "phase3D2E_hpv_dual_license_file_manifest.tsv",
    "08_results/tables/"
    "phase3D2F_hpv_identity_precommit_audit_decision.tsv",
    "08_results/tables/"
    "phase3D2F_hpv_precommit_staging_manifest.tsv",
    "CITATION.cff",
    "DATA_AVAILABILITY.md",
    "LICENSE",
    "LICENSE-CONTENT.md",
    "RELEASE_NOTES_v0.1.0-fiji-systems-serology.md",
    "REPOSITORY_METADATA.md",
    "REPOSITORY_SCOPE.md",
    "environment.yml",
    "requirements.txt",
}


def run(
    command: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            "Command failed: "
            + " ".join(command)
            + "\n"
            + result.stderr.strip()
        )
    return result


def git(*args: str, check: bool = True) -> str:
    return run(["git", *args], check=check).stdout


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_tsv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"status": "NO_ROWS"}]

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_single_tsv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one data row in {path}")
    return rows[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remove_top_level_yaml_block(
    text: str,
    key: str,
) -> tuple[str, bool]:
    lines = text.splitlines()
    target = f"{key}:"
    output: list[str] = []
    removed = False
    index = 0

    while index < len(lines):
        if lines[index] == target:
            removed = True
            index += 1
            while index < len(lines):
                line = lines[index]
                if line and not line[0].isspace():
                    break
                index += 1
            continue

        output.append(lines[index])
        index += 1

    return "\n".join(output).rstrip() + "\n", removed


def status_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in git(
        "status",
        "--porcelain",
        "--untracked-files=all",
    ).splitlines():
        if len(line) < 4:
            continue
        status = line[:2]
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        mapping[path] = status
    return mapping


def validate_citation(
    failures: list[str],
    warnings: list[str],
) -> None:
    text = CITATION_FILE.read_text(encoding="utf-8")

    repaired, removed = remove_top_level_yaml_block(
        text,
        "identifiers",
    )
    if removed:
        write_text(CITATION_FILE, repaired)
        warnings.append(
            "Removed the source-workbook DOI from top-level CFF "
            "identifiers so it is not represented as an identifier "
            "of the software repository."
        )

    try:
        data = yaml.safe_load(
            CITATION_FILE.read_text(encoding="utf-8")
        )
    except Exception as exc:
        failures.append(f"CITATION.cff YAML parsing failed: {exc}")
        return

    if not isinstance(data, dict):
        failures.append("CITATION.cff did not parse to a mapping.")
        return

    expected = {
        "cff-version": "1.2.0",
        "title": "HPV Vaccine Trafficome: Fiji systems-serology analysis",
        "type": "software",
        "license": "MIT",
        "version": "0.1.0",
        "repository-code": PLANNED_REPOSITORY_URL,
    }

    for key, value in expected.items():
        if str(data.get(key, "")) != value:
            failures.append(
                f"CITATION.cff {key!r} is {data.get(key)!r}, "
                f"expected {value!r}."
            )

    authors = data.get("authors")
    if not isinstance(authors, list) or len(authors) != 1:
        failures.append(
            "CITATION.cff must contain exactly one repository author."
        )
    else:
        author = authors[0]
        if not isinstance(author, dict):
            failures.append("CITATION.cff author record is malformed.")
        else:
            if author.get("family-names") != "Maghembe":
                failures.append("CITATION.cff family name is incorrect.")
            if author.get("given-names") != "Reuben S.":
                failures.append("CITATION.cff given names are incorrect.")

    if "identifiers" in data:
        failures.append(
            "CITATION.cff still contains top-level identifiers."
        )

    if SOURCE_DOI not in (
        ROOT / "DATA_AVAILABILITY.md"
    ).read_text(encoding="utf-8"):
        failures.append(
            "The source dataset DOI is absent from DATA_AVAILABILITY.md."
        )


def validate_environment(failures: list[str]) -> None:
    try:
        environment = yaml.safe_load(
            ENVIRONMENT_FILE.read_text(encoding="utf-8")
        )
    except Exception as exc:
        failures.append(f"environment.yml parsing failed: {exc}")
        return

    if not isinstance(environment, dict):
        failures.append("environment.yml did not parse to a mapping.")
        return

    if environment.get("name") != "hpv-vaccine-trafficome":
        failures.append("environment.yml has the wrong environment name.")

    dependencies = environment.get("dependencies")
    if not isinstance(dependencies, list):
        failures.append("environment.yml dependencies are malformed.")


def expected_requirements() -> set[str]:
    rows: list[dict[str, str]] = []
    with DEPENDENCY_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    values = set()
    for row in rows:
        if (
            row.get("classification") == "THIRD_PARTY"
            and row.get("installed") == "True"
            and row.get("package_name")
            and row.get("installed_version")
        ):
            values.add(
                f"{row['package_name']}=={row['installed_version']}"
            )
    return values


def validate_requirements(failures: list[str]) -> None:
    observed = {
        line.strip()
        for line in REQUIREMENTS_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    expected = expected_requirements()
    if observed != expected:
        failures.append(
            "requirements.txt does not match the resolved "
            "third-party dependency inventory."
        )


def validate_python_scripts(failures: list[str]) -> None:
    script_paths = [
        ROOT / line.strip()
        for line in git("ls-files", "*.py").splitlines()
        if line.strip()
    ]

    for relative in (
        "06_scripts/python/"
        "phase3D2D_hpv_repository_identity_rebuild.py",
        "06_scripts/python/"
        "phase3D2E_hpv_dual_license_implementation.py",
        SCRIPT_REL.as_posix(),
    ):
        path = ROOT / relative
        if path.exists() and path not in script_paths:
            script_paths.append(path)

    for path in sorted(script_paths):
        relative = path.relative_to(ROOT).as_posix()
        try:
            source = path.read_text(
                encoding="utf-8",
                errors="strict",
            )
            compile(
                source,
                relative,
                "exec",
                dont_inherit=True,
            )
        except Exception as exc:
            failures.append(
                f"Python syntax failure in {relative}: {exc}"
            )


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []
    checks: list[tuple[str, bool, str]] = []

    root = git("rev-parse", "--show-toplevel").strip()
    head = git("rev-parse", "HEAD").strip()
    tree = git("rev-parse", "HEAD^{tree}").strip()
    branch = git("branch", "--show-current").strip()
    remotes = [
        line for line in git("remote").splitlines() if line.strip()
    ]
    tags = [
        line for line in git("tag", "--list").splitlines() if line.strip()
    ]
    commit_count = int(git("rev-list", "--count", "HEAD").strip())
    root_count = len(
        [
            line
            for line in git(
                "rev-list",
                "--max-parents=0",
                "HEAD",
            ).splitlines()
            if line.strip()
        ]
    )

    gate_values = [
        ("active root", root == str(ROOT), root),
        ("HEAD checkpoint", head == EXPECTED_HEAD, head),
        ("tree checkpoint", tree == EXPECTED_TREE, tree),
        ("branch main", branch == "main", branch),
        ("no remotes", len(remotes) == 0, str(len(remotes))),
        ("no tags", len(tags) == 0, str(len(tags))),
        (
            "reachable commit count",
            commit_count == EXPECTED_COMMIT_COUNT,
            str(commit_count),
        ),
        (
            "root commit count",
            root_count == EXPECTED_ROOT_COUNT,
            str(root_count),
        ),
    ]

    for label, passed, observed in gate_values:
        checks.append((label, passed, observed))
        if not passed:
            failures.append(f"{label} failed: observed {observed}")

    phase_d = read_single_tsv(PHASE_D_DECISION)
    phase_e = read_single_tsv(PHASE_E_DECISION)

    if (
        phase_d.get("decision")
        != "READY_FOR_LICENSE_SELECTION_AND_HPV_IDENTITY_COMMIT"
    ):
        failures.append("Phase 3D2D decision is not the expected state.")
    if phase_d.get("license_selected") != "True":
        failures.append("Phase 3D2D does not record license_selected=True.")
    if (
        phase_e.get("decision")
        != "READY_FOR_HPV_IDENTITY_PRECOMMIT_AUDIT"
    ):
        failures.append("Phase 3D2E decision is not the expected state.")
    if phase_e.get("validation_failure_count") != "0":
        failures.append("Phase 3D2E records validation failures.")

    raw_path = ROOT / RAW_REL
    if not raw_path.exists():
        failures.append("The Fiji source workbook is missing.")
    else:
        raw_md5 = md5_file(raw_path)
        if raw_md5 != EXPECTED_RAW_MD5:
            failures.append("The Fiji source workbook MD5 does not match.")

    ignore_result = run(
        ["git", "check-ignore", "-q", RAW_REL.as_posix()],
        check=False,
    )
    if ignore_result.returncode != 0:
        failures.append("The Fiji source workbook is not ignored by Git.")

    tracked_raw = git(
        "ls-files",
        "--error-unmatch",
        RAW_REL.as_posix(),
        check=False,
    )
    if tracked_raw.strip():
        failures.append("The Fiji source workbook is tracked by Git.")

    validate_citation(failures, warnings)
    validate_environment(failures)
    validate_requirements(failures)
    validate_python_scripts(failures)

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    content_license_text = (
        ROOT / "LICENSE-CONTENT.md"
    ).read_text(encoding="utf-8")
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")

    content_checks = {
        "MIT licence heading": license_text.startswith("MIT License\n"),
        "MIT copyright": (
            "Copyright (c) 2026 Reuben S. Maghembe" in license_text
        ),
        "CC SPDX identifier": (
            "SPDX-License-Identifier: CC-BY-4.0"
            in content_license_text
        ),
        "third-party data not relicensed": (
            "not relicensed" in content_license_text
        ),
        "single README licensing section": (
            readme_text.count("## Licensing") == 1
        ),
        "README evidence boundary": (
            "does **not** directly measure intracellular" in readme_text
        ),
        "README project identity": (
            "# HPV Vaccine Trafficome Project" in readme_text
        ),
        "README trial identifier": "NCT02276521" in readme_text,
        "README source DOI": SOURCE_DOI in readme_text,
    }

    for label, passed in content_checks.items():
        checks.append((label, passed, "True" if passed else "False"))
        if not passed:
            failures.append(label)

    # Scan every file planned for the identity commit. Detection strings
    # are assembled from fragments in source code so the audit machinery
    # itself does not introduce literal cross-project identifiers.
    text_files_to_scan = [
        ROOT / path
        for path in EXPECTED_CHANGE_PATHS
        if (ROOT / path).is_file()
    ]

    contamination_hits: list[str] = []
    for path in text_files_to_scan:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in FORBIDDEN_TOKENS:
            if token in text:
                contamination_hits.append(
                    f"{path.relative_to(ROOT).as_posix()}: {token}"
                )

    if contamination_hits:
        failures.append(
            "Cross-project identifiers detected: "
            + "; ".join(contamination_hits)
        )

    foreign_rows: list[dict[str, object]] = []
    for commit in sorted(MIXED_COMMITS):
        present_result = run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False,
        )
        present = present_result.returncode == 0
        foreign_rows.append(
            {
                "commit_sha": commit,
                "object_present": present,
            }
        )
        if present:
            failures.append(
                f"Foreign Git commit object is present: {commit}"
            )

    fsck_result = run(
        ["git", "fsck", "--full", "--no-reflogs"],
        check=False,
    )
    fsck_text = (
        fsck_result.stdout.rstrip()
        + "\n"
        + fsck_result.stderr.rstrip()
    ).strip()
    unreachable_count = len(
        re.findall(
            r"^(?:unreachable|dangling) "
            r"(?:commit|tree|blob|tag) ",
            fsck_text,
            flags=re.MULTILINE,
        )
    )
    if fsck_result.returncode != 0:
        failures.append(
            f"git fsck returned status {fsck_result.returncode}."
        )
    if unreachable_count != 0:
        failures.append(
            f"git fsck found {unreachable_count} unreachable objects."
        )

    diff_check = run(
        ["git", "diff", "--check"],
        check=False,
    )
    if diff_check.returncode != 0:
        failures.append(
            "git diff --check failed: "
            + (
                diff_check.stdout.strip()
                or diff_check.stderr.strip()
            )
        )

    # Write provisional outputs before checking the final change set.
    decision = FAILURE if failures else SUCCESS

    report_lines = [
        "# Phase 3D2F HPV identity precommit audit",
        "",
        f"**Decision: {decision}**",
        "",
        "## Repository checkpoint",
        "",
        f"- Root: `{root}`",
        f"- HEAD: `{head}`",
        f"- Tree: `{tree}`",
        f"- Branch: `{branch}`",
        f"- Reachable commits: {commit_count}",
        f"- Root commits: {root_count}",
        f"- Remotes: {len(remotes)}",
        f"- Tags: {len(tags)}",
        "",
        "## Checks",
        "",
    ]

    report_lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: {label} "
        f"(observed: `{observed}`)"
        for label, passed, observed in checks
    )

    report_lines.extend(
        [
            "",
            "## Metadata correction",
            "",
            (
                "- The source-workbook DOI is retained in "
                "`DATA_AVAILABILITY.md` and `README.md`, but is not "
                "represented as a top-level identifier of the software "
                "in `CITATION.cff`."
            ),
            "",
            "## Git object audit",
            "",
            f"- `git fsck` status: {fsck_result.returncode}",
            f"- Unreachable objects: {unreachable_count}",
        ]
    )

    for row in foreign_rows:
        report_lines.append(
            f"- Foreign object `{row['commit_sha']}` present: "
            f"{row['object_present']}"
        )

    if warnings:
        report_lines.extend(["", "## Warnings", ""])
        report_lines.extend(f"- {warning}" for warning in warnings)

    if failures:
        report_lines.extend(["", "## Failures", ""])
        report_lines.extend(f"- {failure}" for failure in failures)

    write_text(REPORT_FILE, "\n".join(report_lines))

    # Create placeholders so the complete audit output set is visible to
    # Git status before the exact precommit path inventory is evaluated.
    write_text(MANIFEST_FILE, "status\nPENDING")
    write_text(DECISION_FILE, "decision\nPENDING")

    current_status = status_map()
    observed_paths = set(current_status)
    missing_expected = EXPECTED_CHANGE_PATHS - observed_paths
    unexpected_paths = observed_paths - EXPECTED_CHANGE_PATHS

    if missing_expected:
        failures.append(
            "Expected precommit paths are absent from Git status: "
            + "; ".join(sorted(missing_expected))
        )
    if unexpected_paths:
        failures.append(
            "Unexpected precommit paths are present: "
            + "; ".join(sorted(unexpected_paths))
        )

    final_decision = FAILURE if failures else SUCCESS
    decision_row = {
        "decision": final_decision,
        "active_root": root,
        "head_sha": head,
        "tree_sha": tree,
        "branch": branch,
        "reachable_commit_count": commit_count,
        "root_commit_count": root_count,
        "remote_count": len(remotes),
        "tag_count": len(tags),
        "working_tree_path_count": len(observed_paths),
        "expected_path_count": len(EXPECTED_CHANGE_PATHS),
        "missing_expected_path_count": len(missing_expected),
        "unexpected_path_count": len(unexpected_paths),
        "foreign_object_count": sum(
            bool(row["object_present"]) for row in foreign_rows
        ),
        "unreachable_object_count": unreachable_count,
        "git_fsck_status": fsck_result.returncode,
        "raw_workbook_md5": (
            md5_file(raw_path) if raw_path.exists() else ""
        ),
        "software_license": "MIT",
        "content_license": "CC-BY-4.0",
        "planned_repository_url": PLANNED_REPOSITORY_URL,
        "planned_release_tag": PLANNED_RELEASE_TAG,
        "validation_failure_count": len(failures),
        "validation_warning_count": len(warnings),
        "validation_failures": ("; ".join(failures) or "NONE"),
        "validation_warnings": ("; ".join(warnings) or "NONE"),
    }
    write_tsv(DECISION_FILE, [decision_row])

    manifest_rows: list[dict[str, object]] = []
    for path_text in sorted(observed_paths):
        path = ROOT / path_text
        is_self_manifest = path.resolve() == MANIFEST_FILE.resolve()
        manifest_rows.append(
            {
                "git_status": current_status[path_text],
                "relative_path": path_text,
                "exists": path.exists(),
                "file_size_bytes": (
                    ""
                    if is_self_manifest
                    else (
                        path.stat().st_size if path.is_file() else 0
                    )
                ),
                "sha256": (
                    ""
                    if is_self_manifest
                    else (
                        sha256_file(path) if path.is_file() else ""
                    )
                ),
                "manifest_note": (
                    "SELF_MANIFEST_HASH_OMITTED"
                    if is_self_manifest
                    else ""
                ),
                "expected_for_identity_commit": (
                    path_text in EXPECTED_CHANGE_PATHS
                ),
            }
        )

    write_tsv(MANIFEST_FILE, manifest_rows)

    if failures:
        # Refresh the report with any change-set failures detected after
        # the provisional report was written.
        report_text = REPORT_FILE.read_text(encoding="utf-8")
        report_text = re.sub(
            r"\*\*Decision: .*?\*\*",
            f"**Decision: {FAILURE}**",
            report_text,
            count=1,
        )
        additional = [
            "",
            "## Final change-set failures",
            "",
            *[f"- {failure}" for failure in failures],
        ]
        write_text(
            REPORT_FILE,
            report_text.rstrip() + "\n" + "\n".join(additional),
        )

    print("===== PHASE 3D2F COMPLETE =====")
    print("Decision:", final_decision)
    print("Working-tree paths:", len(observed_paths))
    print("Expected paths:", len(EXPECTED_CHANGE_PATHS))
    print("Missing expected paths:", len(missing_expected))
    print("Unexpected paths:", len(unexpected_paths))
    print("Foreign objects:", sum(
        bool(row["object_present"]) for row in foreign_rows
    ))
    print("Unreachable objects:", unreachable_count)
    print("Validation failures:", len(failures))
    print("Validation warnings:", len(warnings))
    print("Decision file:", DECISION_FILE)
    print("Staging manifest:", MANIFEST_FILE)
    print("Report:", REPORT_FILE)

    if failures:
        print()
        print("Validation failures:")
        for failure in failures:
            print("-", failure)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
