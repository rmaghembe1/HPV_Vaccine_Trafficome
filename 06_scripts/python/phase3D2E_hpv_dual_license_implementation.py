#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path


ROOT = Path("/mnt/d/HPV_Vaccine_Trafficome_Project_PRISTINE")
EXPECTED_HEAD = "4c0fee0b52387e1d92efb5ec9d669399a7ceb92c"
EXPECTED_DECISION = "READY_FOR_LICENSE_SELECTION_AND_HPV_IDENTITY_COMMIT"

MIT_FILE = ROOT / "LICENSE"
CONTENT_FILE = ROOT / "LICENSE-CONTENT.md"
README_FILE = ROOT / "README.md"
SCOPE_FILE = ROOT / "REPOSITORY_SCOPE.md"
DATA_FILE = ROOT / "DATA_AVAILABILITY.md"
CITATION_FILE = ROOT / "CITATION.cff"
METADATA_FILE = ROOT / "REPOSITORY_METADATA.md"
RELEASE_FILE = ROOT / "RELEASE_NOTES_v0.1.0-fiji-systems-serology.md"

PHASE_D_DECISION = (
    ROOT / "08_results/tables/"
    "phase3D2D_hpv_repository_identity_rebuild_decision.tsv"
)
PHASE_E_MANIFEST = (
    ROOT / "08_results/tables/"
    "phase3D2E_hpv_dual_license_file_manifest.tsv"
)
PHASE_E_DECISION = (
    ROOT / "08_results/tables/"
    "phase3D2E_hpv_dual_license_decision.tsv"
)
PHASE_E_REPORT = (
    ROOT / "02_dataset_audit/project_identity/"
    "phase3D2E_hpv_dual_license_report.md"
)

RAW_REL = (
    "03_data_raw/hpv_specific/fiji_nct02276521/"
    "NCOMMS-24-64334A_HPV_collated_antibody_feature_data.xlsx"
)
EXPECTED_RAW_MD5 = "e42173e1d8297cd64420fd9682c42674"

SUCCESS = "READY_FOR_HPV_IDENTITY_PRECOMMIT_AUDIT"
FAILURE = "PHASE3D2E_HPV_DUAL_LICENSE_REPAIR_REQUIRED"

MIT_TEXT = """MIT License

Copyright (c) 2026 Reuben S. Maghembe

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

CONTENT_TEXT = """# Content licence

Copyright (c) 2026 Reuben S. Maghembe

Unless a file or directory states otherwise, the original documentation,
figures, figure legends, and derived analytical tables created for this
repository are licensed under the Creative Commons Attribution 4.0
International licence (CC BY 4.0).

SPDX-License-Identifier: CC-BY-4.0

Canonical licence:
https://creativecommons.org/licenses/by/4.0/

Legal code:
https://creativecommons.org/licenses/by/4.0/legalcode

## Attribution

When reusing covered material, provide appropriate credit to Reuben S.
Maghembe, identify this repository, link to the CC BY 4.0 licence, and
indicate whether changes were made.

## Scope exclusions

This content licence does not relicense:

- source datasets or deposited workbooks obtained from third parties;
- third-party publications, trademarks, logos, or quoted material;
- material whose file or directory contains a different licence notice;
- participant-level information or any rights not owned by the repository
  author.

The Fiji source workbook remains governed by the terms attached to its
original Zenodo deposit and associated source publications.

Software, scripts, and software documentation are licensed separately under
the MIT License in `LICENSE`.
"""

README_LICENSE = """## Licensing

This repository uses scope-based dual licensing:

- software, scripts, and software documentation are licensed under the
  **MIT License**; see `LICENSE`;
- original documentation, figures, figure legends, and derived analytical
  tables are licensed under **Creative Commons Attribution 4.0 International
  (CC BY 4.0)**; see `LICENSE-CONTENT.md`;
- source datasets, deposited workbooks, third-party publications, and other
  third-party material retain their original terms and are not relicensed by
  this repository.

Copyright (c) 2026 Reuben S. Maghembe.
"""

SCOPE_LICENSE = """## Licensing boundary

Code and software components are licensed under MIT. Original repository
documentation, figures, figure legends, and derived analytical tables are
licensed under CC BY 4.0 unless otherwise stated. Source datasets and
third-party material retain their original terms and are not relicensed.
"""

DATA_LICENSE = """## Licensing of derived outputs

Original derived tables, figures, figure legends, and repository
documentation are available under CC BY 4.0 unless otherwise stated.
Software and scripts are available under MIT. The source workbook and other
third-party materials retain their original terms and are not relicensed by
this repository.
"""

METADATA_LICENSE = """## Licensing

- code and software: `MIT`
- original documentation, figures, figure legends, and derived tables:
  `CC-BY-4.0`
- source data and third-party material: original terms retained
"""

RELEASE_LICENSE = """## Licensing

- Software and scripts: MIT License.
- Original documentation, figures, figure legends, and derived analytical
  tables: CC BY 4.0.
- Source data and third-party material: original terms retained.
"""


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def replace_section(text: str, heading: str, replacement: str) -> str:
    lines = text.splitlines()
    try:
        start = lines.index(heading)
    except ValueError:
        return text.rstrip() + "\n\n" + replacement.rstrip() + "\n"

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break

    new_lines = lines[:start] + replacement.rstrip().splitlines() + lines[end:]
    return "\n".join(new_lines).rstrip() + "\n"


def append_section_once(path: Path, heading: str, section: str) -> None:
    text = path.read_text(encoding="utf-8")
    if heading in text:
        text = replace_section(text, heading, section)
    else:
        text = text.rstrip() + "\n\n" + section.rstrip() + "\n"
    write_text(path, text)


def read_decision() -> str:
    with PHASE_D_DECISION.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1:
        return ""
    return rows[0].get("decision", "")


def update_phase_d_decision() -> None:
    with PHASE_D_DECISION.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    row = rows[0]
    row["license_selected"] = "True"
    fieldnames = list(row)
    with PHASE_D_DECISION.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    failures: list[str] = []

    root = run_git("rev-parse", "--show-toplevel").strip()
    head = run_git("rev-parse", "HEAD").strip()
    branch = run_git("branch", "--show-current").strip()
    remotes = [line for line in run_git("remote").splitlines() if line.strip()]
    tags = [line for line in run_git("tag", "--list").splitlines() if line.strip()]

    if root != str(ROOT):
        failures.append("Active root is not the pristine HPV repository.")
    if head != EXPECTED_HEAD:
        failures.append("HEAD differs from the verified HPV checkpoint.")
    if branch != "main":
        failures.append("Active branch is not main.")
    if remotes:
        failures.append("A remote is already configured.")
    if tags:
        failures.append("A tag already exists.")
    if read_decision() != EXPECTED_DECISION:
        failures.append("Phase 3D2D decision is not the expected success state.")

    raw_path = ROOT / RAW_REL
    if not raw_path.exists():
        failures.append("The local Fiji workbook is missing.")
    elif md5_file(raw_path) != EXPECTED_RAW_MD5:
        failures.append("The local Fiji workbook MD5 does not match.")

    if failures:
        print("STOP: Phase 3D2E safety gate failed.")
        for failure in failures:
            print("-", failure)
        raise SystemExit(1)

    write_text(MIT_FILE, MIT_TEXT)
    write_text(CONTENT_FILE, CONTENT_TEXT)

    readme = README_FILE.read_text(encoding="utf-8")

    for heading in (
        "## Licensing",
        "## Licence",
        "## License",
    ):
        if heading in readme:
            readme = replace_section(
                readme,
                heading,
                README_LICENSE,
            )
            break
    else:
        readme = (
            readme.rstrip()
            + "\n\n"
            + README_LICENSE.rstrip()
            + "\n"
        )

    write_text(README_FILE, readme)

    append_section_once(SCOPE_FILE, "## Licensing boundary", SCOPE_LICENSE)
    append_section_once(DATA_FILE, "## Licensing of derived outputs", DATA_LICENSE)
    append_section_once(METADATA_FILE, "## Licensing", METADATA_LICENSE)
    append_section_once(RELEASE_FILE, "## Licensing", RELEASE_LICENSE)

    citation = CITATION_FILE.read_text(encoding="utf-8")
    if "\nlicense:" not in citation:
        citation = citation.replace(
            "type: software\n",
            "type: software\nlicense: MIT\n",
            1,
        )
    write_text(CITATION_FILE, citation)

    update_phase_d_decision()

    license_files = [
        MIT_FILE,
        CONTENT_FILE,
        README_FILE,
        SCOPE_FILE,
        DATA_FILE,
        CITATION_FILE,
        METADATA_FILE,
        RELEASE_FILE,
        PHASE_D_DECISION,
    ]

    checks = {
        "MIT file exists": MIT_FILE.exists(),
        "CC content file exists": CONTENT_FILE.exists(),
        "MIT SPDX meaning recorded": "MIT License" in MIT_FILE.read_text(encoding="utf-8"),
        "CC SPDX identifier recorded": "CC-BY-4.0" in CONTENT_FILE.read_text(encoding="utf-8"),
        "README records MIT": "MIT License" in README_FILE.read_text(encoding="utf-8"),
        "README records CC BY 4.0": "CC BY 4.0" in README_FILE.read_text(encoding="utf-8"),
        "CITATION records software licence": "license: MIT" in CITATION_FILE.read_text(encoding="utf-8"),
        "source data not relicensed": (
            "not relicensed"
            in CONTENT_FILE.read_text(encoding="utf-8")
            or "does not relicense"
            in CONTENT_FILE.read_text(encoding="utf-8")
        ),
        "no remote configured": len(remotes) == 0,
        "no tag exists": len(tags) == 0,
        "raw workbook checksum verified": md5_file(raw_path) == EXPECTED_RAW_MD5,
    }

    for label, passed in checks.items():
        if not passed:
            failures.append(label)

    manifest_rows = []
    for path in license_files:
        manifest_rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    PHASE_E_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with PHASE_E_MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "file_size_bytes", "sha256"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    decision = FAILURE if failures else SUCCESS
    decision_row = {
        "decision": decision,
        "head_sha": head,
        "branch": branch,
        "remote_count": len(remotes),
        "tag_count": len(tags),
        "software_license": "MIT",
        "content_license": "CC-BY-4.0",
        "source_data_relicensed": "False",
        "copyright_holder": "Reuben S. Maghembe",
        "copyright_year": "2026",
        "validation_failure_count": len(failures),
        "validation_failures": ("; ".join(failures) or "NONE"),
    }

    with PHASE_E_DECISION.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(decision_row),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(decision_row)

    report_lines = [
        "# Phase 3D2E HPV dual-licensing implementation",
        "",
        f"**Decision: {decision}**",
        "",
        "- Software and scripts: MIT",
        "- Original documentation, figures, figure legends, and derived tables: CC BY 4.0",
        "- Source data and third-party material: original terms retained",
        "- Copyright: 2026 Reuben S. Maghembe",
        "",
        "## Validation",
        "",
    ]
    report_lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: {label}"
        for label, passed in checks.items()
    )
    if failures:
        report_lines.extend(["", "## Failures", ""])
        report_lines.extend(f"- {failure}" for failure in failures)
    write_text(PHASE_E_REPORT, "\n".join(report_lines))

    print("===== PHASE 3D2E COMPLETE =====")
    print("Decision:", decision)
    print("Software licence: MIT")
    print("Content licence: CC-BY-4.0")
    print("Source data relicensed: False")
    print("Validation failures:", len(failures))
    print("Decision file:", PHASE_E_DECISION)
    print("Report:", PHASE_E_REPORT)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
