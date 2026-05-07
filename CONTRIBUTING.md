# Contributing to NAVAL-SEM

Thank you for your interest in NAVAL-SEM.

NAVAL-SEM is an offline desktop application for Structural Equation Modelling (SEM), supporting both PLS-SEM and CB-SEM workflows with a visual model builder and reproducible code export.

The project is currently maintained by a single independent developer and contributions, bug reports, and feedback are welcome.

---

# Ways to contribute

You can help by:

- Reporting bugs
- Suggesting features
- Improving documentation
- Testing releases on different operating systems
- Improving UI/UX
- Validating statistical outputs
- Contributing tutorials or demo datasets

---

# Before opening issues

Please:

- Check existing issues first
- Include screenshots or logs when possible
- Mention:
  - operating system
  - NAVAL-SEM version
  - dataset type (CSV / Excel / SPSS)

---

# Development setup

Clone the repository:

    git clone https://github.com/navalsingh9/naval-sem.git
    cd naval-sem

Create virtual environment:

    python -m venv .venv

Activate environment

Windows:

    .venv\Scripts\activate

Linux/macOS:

    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

Run locally:

    python launcher.py

---

# Build instructions

Windows:

    build_windows.bat

Linux:

    bash build_linux.sh

macOS:

    bash build_macos.sh

---

# Coding guidelines

Please try to:

- Keep code readable and modular
- Avoid unnecessary dependencies
- Preserve offline-first behaviour
- Keep UI responsive
- Add comments where statistical logic may not be obvious

---

# Statistical validation

SEM implementations can differ subtly across tools.

If contributing to:
- estimation
- fit statistics
- bootstrapping
- HTMT
- missing data handling

please include references, equations, papers, or comparisons where possible.

---

# Security

NAVAL-SEM is designed as an offline-first desktop application.

Please responsibly disclose any security-related concerns instead of posting them publicly.

See:

SECURITY.md

---

# Project status

NAVAL-SEM is under active development and APIs/UI behaviour may evolve rapidly during early releases.

Feedback from:
- PhD scholars
- researchers
- faculty
- quantitative analysts
- SEM practitioners

is especially valuable.

---

Thank you for helping improve accessible SEM tooling.
