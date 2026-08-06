#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path('/mnt/d/HPV_Vaccine_Trafficome_Project_PRISTINE')
EXPECTED_HEAD = '4c0fee0b52387e1d92efb5ec9d669399a7ceb92c'
EXPECTED_TREE = '6da1fb6add196e8501b75149ef76ad638a93720f'
PLANNED_REPO = 'rmaghembe1/HPV_Vaccine_Trafficome'
PLANNED_URL = f'https://github.com/{PLANNED_REPO}'
PLANNED_TAG = 'v0.1.0-fiji-systems-serology'
TRIAL_ID = 'NCT02276521'
ZENODO_RECORD = '14848069'
ZENODO_DOI = '10.5281/zenodo.14848069'
PUBLIC_WORKBOOK = 'NCOMMS-24-64334A HPV collated antibody feature data.xlsx'
RAW_REL = Path('03_data_raw/hpv_specific/fiji_nct02276521/NCOMMS-24-64334A_HPV_collated_antibody_feature_data.xlsx')
EXPECTED_MD5 = 'e42173e1d8297cd64420fd9682c42674'
SCRIPT_REL = Path('06_scripts/python/phase3D2D_hpv_repository_identity_rebuild.py')
TABLE_DIR = ROOT / '08_results' / 'tables'
AUDIT_DIR = ROOT / '02_dataset_audit' / 'project_identity'
DEP_TSV = TABLE_DIR / 'phase3D2D_hpv_dependency_inventory.tsv'
MANIFEST_TSV = TABLE_DIR / 'phase3D2D_hpv_identity_file_manifest.tsv'
DECISION_TSV = TABLE_DIR / 'phase3D2D_hpv_repository_identity_rebuild_decision.tsv'
REPORT_MD = AUDIT_DIR / 'phase3D2D_hpv_repository_identity_rebuild_report.md'
LOG_REL = Path('10_reproducibility/logs/phase3D2D_hpv_repository_identity_rebuild.log')
SUCCESS = 'READY_FOR_LICENSE_SELECTION_AND_HPV_IDENTITY_COMMIT'
FAILURE = 'PHASE3D2D_HPV_REPOSITORY_IDENTITY_REPAIR_REQUIRED'

PACKAGE_OVERRIDES = {
    'PIL': 'Pillow', 'sklearn': 'scikit-learn', 'yaml': 'PyYAML',
    'cv2': 'opencv-python', 'Bio': 'biopython', 'docx': 'python-docx',
    'bs4': 'beautifulsoup4', 'dateutil': 'python-dateutil',
}

README = f'''# HPV Vaccine Trafficome Project

## Project identity

This is an independent HPV-vaccine computational systems-immunology repository. The current release is centered on the Fiji HPV systems-serology study associated with ClinicalTrials.gov identifier `{TRIAL_ID}`.

It is separate from the earlier general cross-vaccine Vaccine Trafficome project and from that project's manuscript, journal correspondence, figures, and submission history.

## Current analytical scope

The Fiji analysis includes 80 participants, 7,360 participant-antigen-feature observations, and 92 antigen-feature analysis rules. It evaluates primary bivalent HPV vaccine induction, long-term immunity after quadrivalent HPV vaccination, heterologous bivalent HPV vaccine recall, HPV16/18 vaccine-target responses, cross-reactive HPV31/33/45/52/58 responses, antibody abundance, subclass organization, Fc-gamma-receptor binding, neutralization, antibody-dependent cellular phagocytosis, previous-dose effects, persistence, primary-versus-recall contrasts, and multivariate immune-state architecture.

## Evidence boundary

The Fiji resource is a systems-serology dataset. It measures downstream antibody magnitude, breadth, subclass organization, Fc-receptor engagement, and functional activity. It does **not** directly measure intracellular endocytosis, endosomal routing, lysosomal processing, or antigen-presentation kinetics. Trafficome-related interpretation is therefore mechanistic and hypothesis-generating rather than a claim of direct intracellular-trafficking measurement.

## Principal findings

- Primary immunization generated strong HPV16/18 IgG, IgG1, IgA1, neutralizing-antibody, and phagocytic responses.
- Heterologous recall produced broader systems-serology remodeling, especially in subclass and Fc-receptor features.
- Neutralization and phagocytosis remained functionally coupled after dose adjustment.
- Cross-reactive serological breadth extended beyond HPV16/18.
- Two reproducible continuous immune-state axes captured recall breadth and HPV16/18 abundance-effector organization.
- A stable raw two-cluster solution was predominantly aligned with primary-versus-recall context and is not interpreted as an intrinsic biological subtype.
- Previous-dose effects were comparatively sparse.
- Bovine papillomavirus control behavior was not uniformly inert and was evaluated through calibration and sensitivity analyses.

## Data limitations

The deposited workbook does not provide participant-level BMI, age, sex, or ethnicity variables suitable for the covariate analyses considered here. Those attributes are not inferred or reconstructed.

## Repository organization

- `02_dataset_audit/`: metadata, quality-control, and analytical decisions
- `03_data_raw/`: local source data excluded from Git
- `05_gene_sets_and_modules/`: mechanistic module definitions
- `06_scripts/`: reproducible analysis and figure-generation code
- `08_results/`: processed tables, model outputs, and figure source data
- `09_figures/`: publication-quality PNG, TIFF, and editable SVG figures
- `10_reproducibility/`: logs and reproducibility records

## Reproducibility checkpoint

The verified HPV-only analytical lineage ends at `{EXPECTED_HEAD}`. The local source workbook is excluded from Git and has verified MD5 `{EXPECTED_MD5}`.

## Public data

The source workbook is publicly available from Zenodo record `{ZENODO_RECORD}` with DOI `{ZENODO_DOI}`. The deposited filename is `{PUBLIC_WORKBOOK}`.

## Licence

A repository licence has not yet been selected. Independent public publication is blocked until that decision is recorded.
'''

SCOPE = '''# Repository scope and project boundaries

## Included

This repository contains the Fiji HPV systems-serology analysis, including metadata audit, normalization, descriptive immune landscapes, mixed-effects models, sensitivity analyses, cross-reactive breadth, neutralization-phagocytosis coupling, multivariate immune-state analysis, source-data tables, and publication-grade figures.

## Excluded

The active repository excludes unrelated cross-vaccine analyses, unrelated manuscripts, journal revision correspondence, submission packages, legacy figures from another project, controlled-access data, and identifiable participant information.

## Interpretation boundary

Systems serology provides downstream functional evidence about antibody and Fc-mediated immune communication. It is not a direct assay of intracellular cargo trafficking. Claims about antigen uptake, endolysosomal processing, antigen presentation, Tfh-B-cell coupling, or metabolic regulation must remain mechanistic interpretations unless supported by additional molecular data.

## Manuscript status

No standalone HPV manuscript is represented as submitted or accepted here. A new HPV-focused manuscript will be developed independently from the completed Fiji analysis and any future HPV molecular extensions.
'''

DATA = f'''# Data availability

The Fiji HPV systems-serology workbook is publicly available from Zenodo record `{ZENODO_RECORD}` with DOI `{ZENODO_DOI}`. The deposited filename is `{PUBLIC_WORKBOOK}`, and the associated trial identifier is `{TRIAL_ID}`.

The local analysis copy is stored at `{RAW_REL.as_posix()}` and has verified MD5 `{EXPECTED_MD5}`. It is excluded from Git. Users should retrieve the workbook from Zenodo.

The repository contains derived metadata, processed matrices, model results, evidence-synthesis tables, figure source data, publication-quality figures, and reproducibility records. No attempt should be made to identify participants or infer unavailable demographic attributes.

The planned independent code repository is `{PLANNED_URL}`. That URL should be treated as active only after publication and verification.
'''

CITATION = f'''cff-version: 1.2.0
message: "Please cite this computational repository and the associated source-data publications when reusing the analysis."
title: "HPV Vaccine Trafficome: Fiji systems-serology analysis"
type: software
authors:
  - family-names: "Maghembe"
    given-names: "Reuben S."
version: "0.1.0"
repository-code: "{PLANNED_URL}"
abstract: >-
  Reproducible computational analysis of primary induction, long-term persistence
  and heterologous recall in the Fiji HPV systems-serology study, including
  antibody magnitude, subclass, Fc-receptor engagement, neutralization,
  phagocytosis, cross-reactive breadth, functional coupling, and multivariate
  immune-state analyses.
keywords:
  - "HPV vaccine"
  - "systems serology"
  - "neutralizing antibodies"
  - "antibody-dependent cellular phagocytosis"
  - "Fc receptors"
  - "heterologous recall"
  - "immune memory"
  - "computational immunology"
identifiers:
  - type: doi
    value: "{ZENODO_DOI}"
    description: "DOI of the publicly deposited source systems-serology workbook"
'''

METADATA = f'''# Planned repository metadata

- owner/repository: `{PLANNED_REPO}`
- planned URL: `{PLANNED_URL}`
- default branch: `main`
- visibility: public
- planned first independent release tag: `{PLANNED_TAG}`
- source analytical checkpoint: `{EXPECTED_HEAD}`

## Proposed description

HPV-anchored computational systems-immunology analysis of primary induction, long-term persistence, and heterologous recall in the Fiji HPV systems-serology study.

## Proposed topics

`hpv`, `hpv-vaccine`, `systems-serology`, `computational-immunology`, `vaccine-immunology`, `antibody`, `fc-receptor`, `neutralization`, `phagocytosis`, `immune-memory`, `reproducible-research`, `public-data`
'''

RELEASE_NOTES = f'''# Fiji HPV systems-serology analysis v0.1.0

This is the planned first independent release of the Fiji HPV systems-serology analysis within the HPV Vaccine Trafficome repository.

Included components comprise source-workbook and metadata audit, normalization, primary/persistence/recall contrasts, previous-dose comparisons, mixed-model convergence and floor-sensitivity analyses, bovine papillomavirus calibration, cross-reactive breadth, neutralization-phagocytosis coupling, standardized PCA, stability and clustering evaluation, continuous immune-state synthesis, publication-quality figures, source-data tables, and QA records.

The analytical figure package was finalized at `{EXPECTED_HEAD}`. Source data are available from Zenodo record `{ZENODO_RECORD}`, DOI `{ZENODO_DOI}`, trial `{TRIAL_ID}`. The raw workbook is not redistributed through Git.

This release provides systems-serology evidence and does not claim direct measurement of intracellular antigen trafficking.
'''


def run_git(*args: str, allow_failure: bool = False) -> str:
    p = subprocess.run(['git', *args], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode and not allow_failure:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return '' if p.returncode else p.stdout


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + '\n', encoding='utf-8')


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{'status': 'NO_ROWS'}]
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', encoding='utf-8', newline='') as h:
        w = csv.DictWriter(h, fieldnames=fields, delimiter='\t', lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def status_paths() -> set[str]:
    out: set[str] = set()
    for line in run_git('status', '--porcelain', '--untracked-files=all').splitlines():
        if len(line) >= 4:
            value = line[3:]
            if ' -> ' in value:
                value = value.split(' -> ', 1)[1]
            out.add(value)
    return out


def dependency_rows() -> tuple[list[dict[str, object]], list[str]]:
    tracked = [ROOT / x for x in run_git('ls-files', '*.py').splitlines() if x.strip()]
    current = ROOT / SCRIPT_REL
    if current.exists() and current not in tracked:
        tracked.append(current)
    module_files: dict[str, set[str]] = {}
    parse_failures: list[str] = []
    for path in tracked:
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding='utf-8', errors='replace'), filename=rel)
        except SyntaxError as e:
            parse_failures.append(f'{rel}:{e.lineno}:{e.msg}')
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_files.setdefault(alias.name.split('.')[0], set()).add(rel)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                module_files.setdefault(node.module.split('.')[0], set()).add(rel)
    stdlib = set(getattr(sys, 'stdlib_module_names', set()))
    local = {p.stem for p in ROOT.rglob('*.py') if '.git' not in p.parts}
    mapping = importlib.metadata.packages_distributions()
    rows: list[dict[str, object]] = []
    for module in sorted(module_files):
        if module in stdlib:
            cls, package, version, installed = 'STANDARD_LIBRARY', '', '', True
        elif module in local:
            cls, package, version, installed = 'LOCAL_PROJECT_MODULE', '', '', True
        else:
            cls = 'THIRD_PARTY'
            candidates = mapping.get(module, [])
            package = PACKAGE_OVERRIDES.get(module) or (sorted(candidates)[0] if candidates else module)
            try:
                version, installed = importlib.metadata.version(package), True
            except importlib.metadata.PackageNotFoundError:
                version, installed = '', False
        rows.append({
            'import_module': module, 'classification': cls, 'package_name': package,
            'installed_version': version, 'installed': installed,
            'source_file_count': len(module_files[module]),
            'source_files': '; '.join(sorted(module_files[module])),
        })
    return rows, parse_failures


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    warnings: list[str] = []
    root = run_git('rev-parse', '--show-toplevel').strip()
    head = run_git('rev-parse', 'HEAD').strip()
    tree = run_git('rev-parse', 'HEAD^{tree}').strip()
    branch = run_git('branch', '--show-current').strip()
    remotes = [x for x in run_git('remote').splitlines() if x.strip()]
    initial = status_paths()
    if root != str(ROOT): failures.append('Active Git root is not the pristine HPV repository.')
    if head != EXPECTED_HEAD: failures.append('HEAD differs from the verified HPV analytical checkpoint.')
    if tree != EXPECTED_TREE: failures.append('Tree differs from the verified HPV analytical tree.')
    if branch != 'main': failures.append('Active branch is not main.')
    if remotes: failures.append('A Git remote is already configured.')
    unexpected_initial = initial - {SCRIPT_REL.as_posix()}
    if unexpected_initial:
        failures.append('Unexpected pre-existing changes: ' + '; '.join(sorted(unexpected_initial)))
    raw_path = ROOT / RAW_REL
    raw_md5 = digest(raw_path, 'md5') if raw_path.exists() else ''
    if not raw_path.exists(): failures.append('The local Fiji source workbook is missing.')
    elif raw_md5 != EXPECTED_MD5: failures.append('The Fiji source workbook MD5 does not match.')
    ignored = subprocess.run(['git', 'check-ignore', '-q', RAW_REL.as_posix()], cwd=ROOT).returncode == 0
    if not ignored: failures.append('The local Fiji source workbook is not ignored by Git.')
    if failures:
        print('STOP: Identity rebuild safety gate failed.')
        for item in failures: print('-', item)
        raise SystemExit(1)

    files = {
        ROOT / 'README.md': README,
        ROOT / 'REPOSITORY_SCOPE.md': SCOPE,
        ROOT / 'DATA_AVAILABILITY.md': DATA,
        ROOT / 'CITATION.cff': CITATION,
        ROOT / 'REPOSITORY_METADATA.md': METADATA,
        ROOT / 'RELEASE_NOTES_v0.1.0-fiji-systems-serology.md': RELEASE_NOTES,
    }
    for path, text in files.items(): write_text(path, text)

    deps, parse_failures = dependency_rows()
    if parse_failures: failures.append('Python parse failures: ' + '; '.join(parse_failures))
    third = [r for r in deps if r['classification'] == 'THIRD_PARTY']
    unresolved = sorted({str(r['package_name']) for r in third if not r['installed']})
    if unresolved: warnings.append('Unresolved packages: ' + '; '.join(unresolved))
    requirements = sorted({f"{r['package_name']}=={r['installed_version']}" for r in third if r['installed'] and r['package_name'] and r['installed_version']}, key=str.lower)
    if not requirements: failures.append('No third-party requirements were resolved.')
    write_text(ROOT / 'requirements.txt', '\n'.join(requirements))
    env = ['name: hpv-vaccine-trafficome', 'channels:', '  - conda-forge', 'dependencies:', f'  - python={sys.version_info.major}.{sys.version_info.minor}', '  - pip', '  - pip:', *[f'      - {x}' for x in requirements]]
    write_text(ROOT / 'environment.yml', '\n'.join(env))
    write_tsv(DEP_TSV, deps)

    identity_paths = [*files.keys(), ROOT / 'requirements.txt', ROOT / 'environment.yml', DEP_TSV]
    manifest = [{
        'relative_path': p.relative_to(ROOT).as_posix(), 'exists': p.exists(),
        'file_size_bytes': p.stat().st_size if p.exists() else 0,
        'sha256': digest(p, 'sha256') if p.exists() else '',
        'role': 'dependency_audit' if p == DEP_TSV else 'repository_identity',
    } for p in identity_paths]
    write_tsv(MANIFEST_TSV, manifest)

    readme_text = (ROOT / 'README.md').read_text(encoding='utf-8')
    for token in ['# HPV Vaccine Trafficome Project', TRIAL_ID, ZENODO_DOI, '80 participants', '7,360 participant', '92 antigen-feature', 'does **not** directly measure intracellular', EXPECTED_HEAD, 'licence has not yet been selected']:
        if token not in readme_text: failures.append('README missing required token: ' + token)
    forbidden = ['Response_' + 'to_Editor', 'Scientific_' + 'Reports_revision', 'af37a99c-' + '209f-4c31-acd3-1893c74da95d', 'VTrafficome_' + 'manuscript', 'github.com/rmaghembe1/' + 'Vaccine_Trafficome']
    for path in files:
        content = path.read_text(encoding='utf-8', errors='replace')
        for token in forbidden:
            if token in content: failures.append(f'{path.name} contains cross-project token: {token}')
    if 'cff-version: 1.2.0' not in (ROOT / 'CITATION.cff').read_text(encoding='utf-8'):
        failures.append('CITATION.cff does not declare version 1.2.0.')

    decision = FAILURE if failures else SUCCESS
    row = {
        'decision': decision, 'active_root': str(ROOT), 'head_sha': head,
        'tree_before_identity_rebuild': tree, 'branch': branch,
        'remote_count': len(remotes), 'python_version': platform.python_version(),
        'python_script_count': len([p for p in run_git('ls-files', '*.py').splitlines() if p.strip()]) + 1,
        'import_module_count': len(deps), 'third_party_package_count': len(third),
        'resolved_requirement_count': len(requirements), 'unresolved_package_count': len(unresolved),
        'python_parse_failure_count': len(parse_failures), 'identity_file_count': len(manifest),
        'raw_workbook_md5': raw_md5, 'planned_repository': PLANNED_REPO,
        'planned_release_tag': PLANNED_TAG, 'license_selected': False,
        'validation_failure_count': len(failures), 'validation_warning_count': len(warnings),
        'validation_failures': '; '.join(failures), 'validation_warnings': '; '.join(warnings),
    }
    write_tsv(DECISION_TSV, [row])
    report = [
        '# Phase 3D2D HPV repository identity rebuild', '', '## Decision', '', f'**{decision}**', '',
        '## Repository identity', '', f'- Local root: `{ROOT}`', f'- Analytical checkpoint: `{head}`',
        f'- Planned repository: `{PLANNED_REPO}`', f'- Planned release tag: `{PLANNED_TAG}`',
        '- Current release focus: Fiji HPV systems serology.',
        '- Direct intracellular-trafficking measurement is not claimed.', '',
        '## Identity files', '', *[f'- `{p.relative_to(ROOT).as_posix()}`' for p in identity_paths], '',
        '## Dependency reconstruction', '', f'- Third-party packages: {len(third)}',
        f'- Resolved requirements: {len(requirements)}', f'- Unresolved packages: {len(unresolved)}', '',
        '## Publication blocker', '', 'No software licence has yet been selected. The independent public repository must not be published until that choice is recorded.'
    ]
    if failures: report += ['', '## Validation failures', '', *['- ' + x for x in failures]]
    if warnings: report += ['', '## Validation warnings', '', *['- ' + x for x in warnings]]
    write_text(REPORT_MD, '\n'.join(report))

    allowed = {
        'README.md', 'REPOSITORY_SCOPE.md', 'DATA_AVAILABILITY.md', 'CITATION.cff',
        'REPOSITORY_METADATA.md', 'RELEASE_NOTES_v0.1.0-fiji-systems-serology.md',
        'requirements.txt', 'environment.yml', SCRIPT_REL.as_posix(),
        DEP_TSV.relative_to(ROOT).as_posix(), MANIFEST_TSV.relative_to(ROOT).as_posix(),
        DECISION_TSV.relative_to(ROOT).as_posix(), REPORT_MD.relative_to(ROOT).as_posix(),
        LOG_REL.as_posix(),
    }
    unexpected = status_paths() - allowed
    if unexpected:
        failures.append('Unexpected files changed: ' + '; '.join(sorted(unexpected)))
        row['decision'] = FAILURE
        row['validation_failure_count'] = len(failures)
        row['validation_failures'] = '; '.join(failures)
        write_tsv(DECISION_TSV, [row])
        decision = FAILURE

    print('===== PHASE 3D2D COMPLETE =====')
    print('Decision:', decision)
    print('Active root:', ROOT)
    print('Analytical checkpoint:', head)
    print('Third-party packages:', len(third))
    print('Resolved requirements:', len(requirements))
    print('Unresolved packages:', len(unresolved))
    print('Identity files:', len(manifest))
    print('Planned repository:', PLANNED_REPO)
    print('Planned release tag:', PLANNED_TAG)
    print('Licence selected:', False)
    print('Report:', REPORT_MD)
    if failures:
        for item in failures: print('-', item)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
