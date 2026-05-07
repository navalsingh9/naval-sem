# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| v0.2.x  | ✅ Yes     |
| < v0.2  | ❌ No      |

## Reporting a Vulnerability

NAVAL-SEM takes security seriously. If you discover a vulnerability, please **do not** open a public GitHub issue.

Instead, report it privately using one of these methods:

**Option 1 — GitHub Private Vulnerability Reporting (preferred)**
Use the **Report a vulnerability** button on the [Security Advisories](https://github.com/navalsingh9/naval-sem/security/advisories) page.

**Option 2 — Feedback Form**
Submit via our feedback form: https://forms.gle/N4AmCkJyCK6HHsZz8
Mark it as a security report in the submission type field.

## What to Include

Please provide as much of the following as possible:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Your suggested fix (if any)
- Your OS and NAVAL-SEM version

## Response Timeline

- **Acknowledgement**: within 48 hours
- **Assessment**: within 7 days
- **Fix or mitigation**: within 30 days depending on severity

## Scope

NAVAL-SEM runs entirely offline on your local machine. It does not transmit any data externally. The attack surface is limited to:

- The local FastAPI server running on 127.0.0.1
- File parsing (CSV, Excel, SPSS)
- The PyWebView browser window

## Out of Scope

- Issues in third-party dependencies (report these to the respective maintainers)
- Bugs unrelated to security (use the feedback form instead)

---

Thank you for helping keep NAVAL-SEM safe for researchers everywhere.
