# Validation and Reproducibility

NAVAL-SEM is designed for transparent, reproducible SEM workflows and keeps the analysis fully local by default.

## Current validation approach

The project validates its outputs against established SEM references and release checks, including:

- lavaan (R) benchmark comparisons
- semopy (Python) benchmark comparisons
- SmartPLS and alternative SEM tool references
- fit-index checks for CB-SEM (CFI, RMSEA, SRMR, χ²/df)
- bootstrap, HTMT, mediation, and fsQCA verification across the release suite

The release process also enforces version consistency across `pyproject.toml`, `CITATION.cff`, `app/version.py`, and `uv.lock` to prevent mismatched build metadata.

## Reproducibility goals

NAVAL-SEM supports reproducible workflows through:

- Offline execution and localhost-only networking
- Export to R (`lavaan`, `seminr`), Python (`semopy`), and JASP-ready code
- Versioned GitHub releases and DOI-backed citation metadata
- Optional Bitcoin timestamp provenance for analysis fingerprints
- Source-available repository visibility and transparent build scripts

## Reporting recommendation

The project is stable and release-driven rather than pre-release. For publication-critical work, researchers should still independently validate their chosen models and cite the exact NAVAL-SEM version used.
