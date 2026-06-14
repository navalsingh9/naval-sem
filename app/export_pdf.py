"""
export_pdf.py  —  NAVAL-SEM  PDF Report Generator
==================================================
Generates a ReportLab A4 PDF from the /export/pdf payload.

Payload keys (all optional except snap):
  snap         {runId, ts, algo, bsN, miss, fname, cmb, syntax,
                analysisType, n_obs, n_params}
  results      ModelResult dict
  mga          MGAResult dict
  htmt         HTMTResult dict
  predictive   PredictResult dict
  diagram_png  base64-encoded PNG of path diagram
  analyst      {name, email, org}
  note         analyst note string
"""

from __future__ import annotations

import base64
import html
import io
import math
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

_PKG_DIR = Path(__file__).resolve().parent
_BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", _PKG_DIR))
_FONT_DIRS = [
    _BUNDLE_DIR / "fonts",
    _PKG_DIR / "fonts",
    _PKG_DIR.parent / "fonts",
    Path.cwd() / "fonts",
]
# ── ReportLab ────────────────────────────────────────────────────────────────
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    FrameBreak,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

# ── Register DejaVu (Unicode, supports Greek/arrows/check marks) ─────────────
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


_FONTS_REGISTERED = False

_UNICODE_ASCII = {
    "✓": "OK",  "✗": "--",  "α": "alpha",  "β": "beta",  "Δ": "D",
    "χ²": "chi2",  "√": "sqrt",  "·": ".",  "\u2013": "-",  "\u2019": "'",
    "≥": ">=",  "≤": "<=",  "→": "->",
}

def _safe_text(s: str) -> str:
    """Replace known non-base14 glyphs when DejaVu fonts are unavailable."""
    if _FONTS_REGISTERED:
        return s
    for uni, asc in _UNICODE_ASCII.items():
        s = s.replace(uni, asc)
    return s


_FONT = "DV"          # will fall back to Helvetica if registration fails
_FONT_BOLD = "DV-Bold"
_FONT_MONO = "DV-Mono"


def _register_fonts() -> None:
    """Register bundled DejaVu fonts when present, otherwise use built-ins."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return

    font_dir = next(
        (
            d for d in _FONT_DIRS
            if (d / "DejaVuSans.ttf").exists()
            and (d / "DejaVuSans-Bold.ttf").exists()
        ),
        None,
    )
    if font_dir is None:
        return

    try:
        registered = set(pdfmetrics.getRegisteredFontNames())
        mono_path = font_dir / "DejaVuSansMono.ttf"
        if not mono_path.exists():
            mono_path = font_dir / "DejaVuSans.ttf"
        if _FONT not in registered:
            pdfmetrics.registerFont(TTFont(_FONT, str(font_dir / "DejaVuSans.ttf")))
        if _FONT_BOLD not in registered:
            pdfmetrics.registerFont(TTFont(_FONT_BOLD, str(font_dir / "DejaVuSans-Bold.ttf")))
        if _FONT_MONO not in registered:
            pdfmetrics.registerFont(TTFont(_FONT_MONO, str(mono_path)))
        pdfmetrics.registerFontFamily("DV", normal=_FONT, bold=_FONT_BOLD)
        _FONTS_REGISTERED = True
    except Exception:
        _FONTS_REGISTERED = False


def _first_present(mapping: dict, *keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _xml_escape(value: Any) -> str:
    return html.escape(str(value), quote=False)

# ── Colour palette ────────────────────────────────────────────────────────────
_ACCENT    = colors.HexColor("#2F5DD3")
_GREEN     = colors.HexColor("#1B8A5A")
_AMBER     = colors.HexColor("#B8860B")
_RED       = colors.HexColor("#C0392B")
_PURPLE    = colors.HexColor("#6B3FA0")
_TEXT_DARK = colors.HexColor("#1A1D23")
_TEXT_MED  = colors.HexColor("#3A3D45")
_TEXT_MUTE = colors.HexColor("#6B7280")
_LINE      = colors.HexColor("#D1D5DB")
_BG_SOFT   = colors.HexColor("#F3F4F6")
_BG_HEAD   = colors.HexColor("#E8EBF0")
_BG_ACCENT = colors.HexColor("#EEF2FF")
_WHITE     = colors.white

# ── Page geometry ─────────────────────────────────────────────────────────────
_PW, _PH   = A4
_ML = _MR  = 18 * mm
_MT = _MB  = 16 * mm
_CW        = _PW - _ML - _MR   # content width  ≈ 174 mm


# ═════════════════════════════════════════════════════════════════════════════
#  Styles
# ═════════════════════════════════════════════════════════════════════════════

def _build_styles() -> dict:
    _register_fonts()
    f = _FONT if _FONTS_REGISTERED else "Helvetica"
    fb = _FONT_BOLD if _FONTS_REGISTERED else "Helvetica-Bold"
    fm = _FONT_MONO if _FONTS_REGISTERED else "Courier"

    def ps(name, **kw) -> ParagraphStyle:
        base = kw.pop("parent", None)
        if base:
            s = ParagraphStyle(name, parent=base, **kw)
        else:
            s = ParagraphStyle(name, **kw)
        return s

    return {
        "ReportTitle": ps("ReportTitle",
            fontName=fb, fontSize=18, textColor=_ACCENT,
            spaceAfter=1, leading=22),
        "ReportSubtitle": ps("ReportSubtitle",
            fontName=f, fontSize=9, textColor=_TEXT_MUTE,
            spaceAfter=2, leading=12),
        "RunMeta": ps("RunMeta",
            fontName=fm, fontSize=7.5, textColor=_TEXT_MUTE,
            leading=11, spaceAfter=1),
        "SectionTitle": ps("SectionTitle",
            fontName=fb, fontSize=9, textColor=_ACCENT,
            spaceBefore=10, spaceAfter=4, leading=12,
            borderPadding=(0, 0, 2, 0)),
        "Body": ps("Body",
            fontName=f, fontSize=8.5, textColor=_TEXT_DARK,
            leading=13, spaceAfter=3),
        "Small": ps("Small",
            fontName=f, fontSize=7.5, textColor=_TEXT_MED,
            leading=11),
        "Muted": ps("Muted",
            fontName=f, fontSize=7.5, textColor=_TEXT_MUTE,
            leading=10, spaceAfter=1),
        "Mono": ps("Mono",
            fontName=fm, fontSize=7.5, textColor=_TEXT_DARK,
            leading=11, spaceAfter=2),
        "MonoBlock": ps("MonoBlock",
            fontName=fm, fontSize=7, textColor=_TEXT_MED,
            leading=10, spaceAfter=0,
            leftIndent=4, backColor=_BG_SOFT,
            borderPadding=(4, 6, 4, 6)),
        "Note": ps("Note",
            fontName=f, fontSize=8, textColor=_TEXT_MED,
            leading=12, leftIndent=6, spaceAfter=4,
            borderWidth=0, borderColor=_ACCENT,
            borderPadding=(0, 0, 0, 8)),
        "TH": ps("TH",
            fontName=fb, fontSize=7.5, textColor=_TEXT_DARK,
            alignment=TA_CENTER, leading=10),
        "THL": ps("THL",
            fontName=fb, fontSize=7.5, textColor=_TEXT_DARK,
            alignment=TA_LEFT, leading=10),
        "TC": ps("TC",
            fontName=f, fontSize=7.5, textColor=_TEXT_DARK,
            alignment=TA_CENTER, leading=10),
        "TCL": ps("TCL",
            fontName=f, fontSize=7.5, textColor=_TEXT_DARK,
            alignment=TA_LEFT, leading=10),
        "TCMono": ps("TCMono",
            fontName=fm, fontSize=7, textColor=_TEXT_DARK,
            alignment=TA_CENTER, leading=10),
        "Footer": ps("Footer",
            fontName=f, fontSize=6.5, textColor=_TEXT_MUTE,
            alignment=TA_CENTER, leading=8),
        "ColHeader": ps("ColHeader",
            fontName=fb, fontSize=8, textColor=_TEXT_DARK,
            leading=11, spaceAfter=2),
        "Italic": ps("Italic",
            fontName=f, fontSize=8, textColor=_TEXT_MUTE,
            leading=11, spaceAfter=2),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Formatting helpers
# ═════════════════════════════════════════════════════════════════════════════

def _fmt(v, dp: int = 3, na: str = "—") -> str:
    if v is None:
        return na
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return na
        return f"{f:.{dp}f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_p(v, na: str = "—") -> str:
    if v is None:
        return na
    try:
        p = float(v)
        if math.isnan(p) or math.isinf(p):
            return na
        if p < 0.001:
            return "< .001"
        if p < 0.01:
            return f"{p:.3f}"
        return f"{p:.3f}"
    except (TypeError, ValueError):
        return str(v)


def _sig_stars(p) -> str:
    if p is None:
        return ""
    try:
        pv = float(p)
        if pv < 0.001:
            return "***"
        if pv < 0.01:
            return "**"
        if pv < 0.05:
            return "*"
        return ""
    except (TypeError, ValueError):
        return ""


def _check(ok: bool | None) -> str:
    if ok is None:
        return _safe_text("—")
    return _safe_text("\u2713" if ok else "\u2717")


# ═════════════════════════════════════════════════════════════════════════════
#  Custom Flowables
# ═════════════════════════════════════════════════════════════════════════════

class _ColorRect(Flowable):
    """A simple filled rectangle — used for inline coloured bars."""

    def __init__(self, w, h, fill, stroke=None, radius=1):
        super().__init__()
        self.w = w
        self.h = h
        self.fill = fill
        self.stroke = stroke
        self.radius = radius

    def wrap(self, *_):
        return self.w, self.h

    def draw(self):
        c = self.canv
        c.setFillColor(self.fill)
        if self.stroke:
            c.setStrokeColor(self.stroke)
        c.roundRect(0, 0, self.w, self.h, self.radius,
                    stroke=1 if self.stroke else 0, fill=1)


class _KPIBlock(Flowable):
    """A single KPI tile: value + label rendered inline."""

    def __init__(self, label: str, value: str, sub: str = "",
                 accent: colors.Color = None, width: float = 42 * mm):
        super().__init__()
        self._label = label
        self._value = value
        self._sub = sub
        self._accent = accent or _ACCENT
        self.width = width
        self.height = 22 * mm

    def wrap(self, *_):
        return self.width, self.height

    def draw(self):
        c = self.canv
        w, h = self.width, self.height
        # Card background
        c.setFillColor(_BG_SOFT)
        c.roundRect(0, 0, w, h, 3, stroke=0, fill=1)
        # Top accent strip
        c.setFillColor(self._accent)
        c.rect(0, h - 2.5, w, 2.5, stroke=0, fill=1)
        # Value
        c.setFillColor(_TEXT_DARK)
        _register_fonts()
        fn = _FONT_BOLD if _FONTS_REGISTERED else "Helvetica-Bold"
        c.setFont(fn, 13)
        c.drawCentredString(w / 2, h - 13, self._value)
        # Sub
        if self._sub:
            fn2 = _FONT if _FONTS_REGISTERED else "Helvetica"
            c.setFont(fn2, 6.5)
            c.setFillColor(_TEXT_MUTE)
            c.drawCentredString(w / 2, h - 20, self._sub)
        # Label
        fn2 = _FONT if _FONTS_REGISTERED else "Helvetica"
        c.setFont(fn2, 7)
        c.setFillColor(_TEXT_MUTE)
        c.drawCentredString(w / 2, 4, self._label)


# ═════════════════════════════════════════════════════════════════════════════
#  Table builders
# ═════════════════════════════════════════════════════════════════════════════

_BASE_TS = TableStyle([
    ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",    (0, 0), (-1, -1), 7.5),
    ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
    ("BACKGROUND",  (0, 0), (-1, 0),  _BG_HEAD),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_WHITE, _BG_SOFT]),
    ("GRID",        (0, 0), (-1, -1), 0.25, _LINE),
    ("TOPPADDING",  (0, 0), (-1, -1), 2.5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
])


def _apply_ts(ts: TableStyle, extra: list) -> TableStyle:
    ts2 = TableStyle(list(ts._cmds))
    for cmd in extra:
        ts2.add(*cmd)
    return ts2


def _p(text: str, style: ParagraphStyle, trusted: bool = False) -> Paragraph:
    return Paragraph(str(text) if trusted else _xml_escape(text), style)


# ═════════════════════════════════════════════════════════════════════════════
#  Section builders
# ═════════════════════════════════════════════════════════════════════════════

def _section_header(title: str, st: dict) -> list:
    return [
        Paragraph(title.upper(), st["SectionTitle"]),
        HRFlowable(width="100%", thickness=0.5, color=_LINE, spaceAfter=4),
    ]


def _build_header_block(snap: dict, analyst: dict, note: str, st: dict) -> list:
    """Title block + run metadata."""
    run_id   = str(snap.get("runId") or "—")
    ts       = str(snap.get("ts") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    algo     = snap.get("algo", "—")
    bs_n     = snap.get("bsN", "—")
    miss     = snap.get("miss", "—")
    fname    = snap.get("fname", "—")
    a_type   = snap.get("analysisType", "SEM")

    now_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    title_data = [
        [
            Paragraph("NAVAL·SEM", st["ReportTitle"]),
            Paragraph(
                f"Run: <b>{_xml_escape(run_id[:12])}</b><br/>"
                f"Analysed: {_xml_escape(ts)}<br/>"
                f"Generated: {_xml_escape(now_str)}",
                st["RunMeta"]),
        ]
    ]
    title_tbl = Table(title_data, colWidths=[_CW * 0.6, _CW * 0.4])
    title_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
    ]))

    flowables: list = [
        title_tbl,
        Paragraph("Structural Equation Modelling · Results Report", st["ReportSubtitle"]),
        HRFlowable(width="100%", thickness=1, color=_ACCENT, spaceAfter=6),
    ]

    # ── Run metadata grid ────────────────────────────────────────────────────
    n_obs    = snap.get("n_obs")
    n_params = snap.get("n_params")
    cmb      = snap.get("cmb", "—")

    meta_pairs = [
        ("Analysis type", a_type),
        ("Algorithm",     algo),
        ("Bootstrap",     str(bs_n)),
        ("Missing",       miss),
        ("Dataset",       fname),
        ("CMB marker",    cmb or "—"),
    ]
    if n_obs is not None:
        meta_pairs.append(("Observations", str(n_obs)))
    if n_params is not None:
        meta_pairs.append(("Parameters", str(n_params)))

    # 4-column grid: label | value | label | value
    rows = []
    for i in range(0, len(meta_pairs), 2):
        left  = meta_pairs[i]
        right = meta_pairs[i + 1] if i + 1 < len(meta_pairs) else ("", "")
        rows.append([
            _p(left[0],  st["Muted"]),
            _p(left[1],  st["Small"]),
            _p(right[0], st["Muted"]),
            _p(right[1], st["Small"]),
        ])

    cw4 = [_CW * 0.18, _CW * 0.32, _CW * 0.18, _CW * 0.32]
    meta_tbl = Table(rows, colWidths=cw4)
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), _BG_SOFT),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_BG_SOFT, _WHITE]),
        ("GRID",        (0, 0), (-1, -1), 0.25, _LINE),
        ("TOPPADDING",  (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",(0, 0), (-1, -1), 4),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]))
    flowables.append(meta_tbl)

    # ── Analyst block ────────────────────────────────────────────────────────
    a_name  = (analyst or {}).get("name", "").strip()
    a_email = (analyst or {}).get("email", "").strip()
    a_org   = (analyst or {}).get("org", "").strip()
    if any([a_name, a_email, a_org]):
        parts = " · ".join(p for p in [a_name, a_email, a_org] if p)
        flowables.append(Spacer(1, 3))
        flowables.append(Paragraph(f"Prepared by: {_xml_escape(parts)}", st["Muted"]))

    # ── Note ─────────────────────────────────────────────────────────────────
    if note and note.strip():
        flowables.append(Spacer(1, 4))
        flowables.append(
            Paragraph(f"<i>Analyst note:</i> {_xml_escape(note.strip())}", st["Note"])
        )

    flowables.append(Spacer(1, 6))
    return flowables


def _build_syntax_block(snap: dict, st: dict) -> list:
    syntax = (snap.get("syntax") or "").strip()
    if not syntax:
        return []
    lines = syntax.split("\n")
    text  = "\n".join(lines[:40])   # cap at 40 lines
    if len(lines) > 40:
        text += f"\n… ({len(lines) - 40} more lines)"

    flowables = _section_header("Model Syntax", st)
    flowables.append(
        Paragraph(_xml_escape(text).replace("\n", "<br/>").replace(" ", "&nbsp;"), st["MonoBlock"])
    )
    flowables.append(Spacer(1, 4))
    return flowables


def _build_kpi_row(results: dict, st: dict) -> list:
    """4 KPI tiles: R², AVE, CR, α."""
    fit = (results or {}).get("fit") or {}
    params = (results or {}).get("parameters") or []
    latent = (results or {}).get("latent_variables") or {}

    # Average R²
    r2_vals = [p.get("r_squared") for p in params
                if p.get("r_squared") is not None]
    avg_r2 = (sum(r2_vals) / len(r2_vals)) if r2_vals else None

    # AVE / CR / alpha from fit
    ave_vals = [v for k, v in fit.items()
                if "ave" in k.lower() and isinstance(v, (int, float))]
    cr_vals  = [v for k, v in fit.items()
                if ("cr" in k.lower() or "composite" in k.lower())
                   and isinstance(v, (int, float))]
    al_vals  = [v for k, v in fit.items()
                if "alpha" in k.lower() and isinstance(v, (int, float))]

    # Also look in latent_variables dict for per-lv stats
    if isinstance(latent, dict):
        for lv_data in latent.values():
            if isinstance(lv_data, dict):
                if "ave" in lv_data and lv_data["ave"] is not None:
                    ave_vals.append(float(lv_data["ave"]))
                if "cr" in lv_data and lv_data["cr"] is not None:
                    cr_vals.append(float(lv_data["cr"]))
                if "alpha" in lv_data and lv_data["alpha"] is not None:
                    al_vals.append(float(lv_data["alpha"]))

    avg_ave = (sum(ave_vals) / len(ave_vals)) if ave_vals else None
    avg_cr  = (sum(cr_vals)  / len(cr_vals))  if cr_vals  else None
    avg_al  = (sum(al_vals)  / len(al_vals))  if al_vals  else None

    tiles = [
        ("Avg R²",        _fmt(avg_r2, 3), "Endogenous constructs", _ACCENT),
        ("Avg AVE",       _fmt(avg_ave, 3), "Convergent validity",   _GREEN),
        ("Avg CR",        _fmt(avg_cr,  3), "Composite reliability", _PURPLE),
        ("Avg \u03b1",    _fmt(avg_al,  3), "Cronbach\u2019s alpha", _AMBER),
    ]

    kpi_w = _CW / 4 - 3
    row_data = [[_KPIBlock(lbl, val, sub, color, kpi_w)
                 for lbl, val, sub, color in tiles]]
    tbl = Table(row_data, colWidths=[kpi_w + 3] * 4)
    tbl.setStyle(TableStyle([
        ("LEFTPADDING",  (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]))

    return [*_section_header("Key Metrics", st), tbl, Spacer(1, 6)]


def _build_fit_indices(results: dict, st: dict) -> list:
    fit = (results or {}).get("fit") or {}
    if not fit:
        return []

    # Well-known indices with thresholds
    INDEX_META = {
        "cfi":   ("CFI",                  _safe_text("≥ .95"), lambda v: v >= 0.95),
        "tli":   ("TLI",                  _safe_text("≥ .95"), lambda v: v >= 0.95),
        "nfi":   ("NFI",                  _safe_text("≥ .90"), lambda v: v >= 0.90),
        "ifi":   ("IFI",                  _safe_text("≥ .90"), lambda v: v >= 0.90),
        "rmsea": ("RMSEA",                _safe_text("≤ .06"), lambda v: v <= 0.06),
        "srmr":  ("SRMR",                 _safe_text("≤ .08"), lambda v: v <= 0.08),
        "gfi":   ("GFI",                  _safe_text("≥ .90"), lambda v: v >= 0.90),
        "chi2":  (_safe_text("χ²"),       "",                  None),
        "df":    ("df",                   "",                  None),
        "chi2_p":(_safe_text("p(χ²)"),   "",                  None),
        "cmin_df":("CMIN/df",             _safe_text("≤ 3.0"), lambda v: v <= 3.0),
    }

    rows = [[_p("Index", st["TH"]), _p("Value", st["TH"]),
             _p("Threshold", st["TH"]), _p("Verdict", st["TH"])]]
    extra_cmds = []

    for key, (label, thresh, check_fn) in INDEX_META.items():
        val = fit.get(key)
        if val is None:
            # try lowercase variants
            for k in fit:
                if k.lower().replace("_", "").replace("-", "") == \
                        key.lower().replace("_", "").replace("-", ""):
                    val = fit[k]
                    break
        if val is None:
            continue

        is_p = "p" in key.lower() or key.lower() in ("chi2_p",)
        val_str = _fmt_p(val) if is_p else _fmt(val, 3)
        if check_fn:
            try:
                ok = check_fn(float(val))
                verdict = "\u2713 Acceptable" if ok else "\u2717 Concern"
                color   = _GREEN if ok else _RED
            except (TypeError, ValueError):
                verdict = "—"
                color   = _TEXT_MUTE
        else:
            verdict = ""
            color   = _TEXT_MUTE

        row_i = len(rows)
        rows.append([
            _p(label,   st["TCL"]),
            _p(val_str, st["TCMono"]),
            _p(thresh,  st["TC"]),
            _p(verdict, st["TC"]),
        ])
        if verdict:
            extra_cmds.append(("TEXTCOLOR", (3, row_i), (3, row_i), color))

    if len(rows) <= 1:
        return []

    cw = [_CW * 0.20, _CW * 0.22, _CW * 0.30, _CW * 0.28]
    tbl = Table(rows, colWidths=cw, repeatRows=1)
    tbl.setStyle(_apply_ts(_BASE_TS, extra_cmds))

    return [*_section_header("Model Fit Indices", st), tbl, Spacer(1, 6)]


def _build_parameters(results: dict, st: dict) -> list:
    params = (results or {}).get("parameters") or []
    if not params:
        return []

    header = [_p(h, st["TH"]) for h in
              ["Relationship", "Std. Est.", "Std. Error", "t / z", "p-value", "Sig.", "95% CI"]]

    rows = [header]
    extra_cmds = []

    for p in params:
        op    = p.get("op", "")
        lhs   = p.get("lhs", "")
        rhs   = p.get("rhs", "")
        label = f"{rhs} {op} {lhs}" if op else _safe_text(f"{rhs} \u2192 {lhs}")
        est   = _fmt(_first_present(p, "std_estimate", "std.all", "est", "estimate"), 3)
        se    = _fmt(_first_present(p, "se", "std.error", "std_error"), 3)
        t_val = _fmt(_first_present(p, "t", "z", "stat", "z_value"), 3)
        pv    = _first_present(p, "pvalue", "p.value", "p", "p_value")
        p_str = _fmt_p(pv)
        stars = _sig_stars(pv)
        lo    = _first_present(p, "ci_lower", "ci.lower")
        hi    = _first_present(p, "ci_upper", "ci.upper")
        ci    = f"[{_fmt(lo,3)}, {_fmt(hi,3)}]" if lo is not None and hi is not None else "—"

        row_i = len(rows)
        rows.append([
            _p(label, st["TCL"]),
            _p(est,   st["TCMono"]),
            _p(se,    st["TCMono"]),
            _p(t_val, st["TCMono"]),
            _p(p_str, st["TCMono"]),
            _p(stars, st["TC"]),
            _p(ci,    st["TCMono"]),
        ])
        # Colour significance
        try:
            pf = float(pv)
            c  = _GREEN if pf < 0.05 else _TEXT_MUTE
            extra_cmds.append(("TEXTCOLOR", (5, row_i), (5, row_i), c))
        except (TypeError, ValueError):
            pass

    cw = [_CW*0.26, _CW*0.09, _CW*0.09, _CW*0.09, _CW*0.09, _CW*0.07, _CW*0.31]
    tbl = Table(rows, colWidths=cw, repeatRows=1)
    tbl.setStyle(_apply_ts(_BASE_TS, [
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        *extra_cmds,
    ]))

    return [*_section_header("Path Coefficients & Loadings", st), tbl, Spacer(1, 6)]


def _build_loadings(results: dict, st: dict) -> list:
    """Outer-model loadings table built from measurement parameters (op == '=~')."""
    params = (results or {}).get("parameters") or []
    meas_params = [p for p in params if p.get("op") == "=~"]
    if not meas_params:
        return []

    # Group by lhs (latent variable)
    from collections import defaultdict
    lv_groups: dict = defaultdict(list)
    for p in meas_params:
        lv_groups[p.get("lhs", "?")].append(p)

    all_rows = []
    extra_cmds: list = []

    for lv_name, indicators in lv_groups.items():
        row_i = len(all_rows) + 1  # +1 for header
        all_rows.append([
            _p(f"<b>{_xml_escape(lv_name)}</b>", st["TCL"], trusted=True),
            _p("", st["TC"]), _p("", st["TC"]),
            _p("", st["TC"]), _p("", st["TC"]),
        ])
        extra_cmds.append(("BACKGROUND", (0, row_i), (-1, row_i), _BG_ACCENT))

        for ind in indicators:
            name  = ind.get("rhs", "?")
            lod   = _fmt(_first_present(ind, "std_estimate", "estimate"), 3)
            se    = _fmt(_first_present(ind, "std_error", "se"), 3)
            t_val = _fmt(_first_present(ind, "z_value", "t"), 3)
            pv    = _first_present(ind, "p_value", "pvalue", "p")
            p_str = _fmt_p(pv)
            all_rows.append([
                _p(f"   {name}", st["TCL"]),
                _p(lod,   st["TCMono"]),
                _p(se,    st["TCMono"]),
                _p(t_val, st["TCMono"]),
                _p(p_str, st["TCMono"]),
            ])

    if not all_rows:
        return []

    header = [_p(h, st["TH"]) for h in
              ["Indicator", "Std. Loading", "Std. Error", "t-stat", "p-value"]]
    rows = [header] + all_rows

    cw = [_CW*0.35, _CW*0.16, _CW*0.16, _CW*0.16, _CW*0.17]
    tbl = Table(rows, colWidths=cw, repeatRows=1)
    tbl.setStyle(_apply_ts(_BASE_TS, extra_cmds))

    return [*_section_header("Outer Model Loadings", st), tbl, Spacer(1, 6)]


def _build_fornell_larcker(results: dict, st: dict) -> list:
    fl = (results or {}).get("fit", {}).get("fornell_larcker") or \
         (results or {}).get("fornell_larcker")
    if not fl or not isinstance(fl, dict):
        return []

    lv_names = sorted(fl.keys())
    if not lv_names:
        return []

    header = [_p("", st["TH"])] + [_p(n, st["TH"]) for n in lv_names]
    rows   = [header]
    extra_cmds: list = []

    for i, row_name in enumerate(lv_names):
        row_vals = fl.get(row_name) or {}
        cells    = [_p(row_name, st["TCL"])]
        for j, col_name in enumerate(lv_names):
            v = row_vals.get(col_name)
            if v is None:
                cells.append(_p("—", st["TC"]))
            else:
                cells.append(_p(_fmt(v, 3), st["TCMono"]))
                if i == j:  # diagonal (AVE sqrt) — highlight
                    extra_cmds.append(
                        ("BACKGROUND", (j+1, i+1), (j+1, i+1), _BG_ACCENT))
        rows.append(cells)

    n = len(lv_names)
    cw_lbl = _CW * 0.20
    cw_val = (_CW - cw_lbl) / n
    cw = [cw_lbl] + [cw_val] * n
    tbl = Table(rows, colWidths=cw, repeatRows=1)
    tbl.setStyle(_apply_ts(_BASE_TS, [
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        *extra_cmds,
    ]))

    return [*_section_header("Fornell-Larcker Criterion", st), tbl, Spacer(1, 6)]


def _build_htmt(htmt: dict, st: dict) -> list:
    if not htmt:
        return []
    matrix = htmt.get("matrix") or []
    if not matrix:
        return []

    flowables = _section_header("HTMT Ratios", st)

    # Overall verdict
    all_ok = htmt.get("all_acceptable")
    if all_ok is not None:
        lbl = _safe_text("\u2713 All HTMT ratios acceptable (< .90)" if all_ok
              else "\u2717 One or more HTMT ratios exceed threshold")
        color_hex = "#1B8A5A" if all_ok else "#C0392B"
        flowables.append(
            Paragraph(f'<font color="{color_hex}">{_xml_escape(lbl)}</font>', st["Small"]))
        flowables.append(Spacer(1, 3))

    header = [_p(h, st["TH"]) for h in
              ["Construct A", "Construct B", "HTMT", "Threshold", "Verdict"]]
    rows   = [header]
    extra_cmds: list = []

    for entry in matrix:
        ca = _first_present(entry, "lv1", "construct_a") or ""
        cb = _first_present(entry, "lv2", "construct_b") or ""
        v  = _first_present(entry, "htmt", "value")
        thresh = entry.get("threshold", 0.90)
        ok: bool | None = None
        try:
            ok = float(v) < float(thresh)
        except (TypeError, ValueError):
            pass

        verdict = ("\u2713 OK" if ok else "\u2717 Concern") if ok is not None else "—"
        row_i   = len(rows)
        rows.append([
            _p(ca,          st["TCL"]),
            _p(cb,          st["TCL"]),
            _p(_fmt(v, 3),  st["TCMono"]),
            _p(_fmt(thresh, 2), st["TC"]),
            _p(verdict,     st["TC"]),
        ])
        if ok is not None:
            extra_cmds.append(
                ("TEXTCOLOR", (4, row_i), (4, row_i), _GREEN if ok else _RED))

    cw = [_CW*0.24, _CW*0.24, _CW*0.16, _CW*0.16, _CW*0.20]
    tbl = Table(rows, colWidths=cw, repeatRows=1)
    tbl.setStyle(_apply_ts(_BASE_TS, [
        ("ALIGN", (0, 0), (1, -1), "LEFT"),
        *extra_cmds,
    ]))
    flowables += [tbl, Spacer(1, 6)]
    return flowables


def _build_mga(mga: dict, st: dict) -> list:
    if not mga:
        return []

    flowables = _section_header("Multi-Group Analysis", st)

    # ── MICOM / configural ──────────────────────────────────────────────────
    micom = mga.get("micom") or {}
    config = micom.get("configural") or {}
    partial = micom.get("partial_invariance") or {}
    full_inv = micom.get("full_invariance") or {}

    if config:
        flowables.append(Paragraph("Configural Model", st["ColHeader"]))
        kv = [(k, _fmt(v, 3) if isinstance(v, float) else str(v))
              for k, v in config.items() if k not in ("model",)]
        if kv:
            rows = [[_p(k, st["Muted"]), _p(v, st["Small"])] for k, v in kv]
            tbl = Table(rows, colWidths=[_CW*0.5, _CW*0.5])
            tbl.setStyle(TableStyle([
                ("GRID", (0,0),(-1,-1), 0.25, _LINE),
                ("TOPPADDING", (0,0),(-1,-1), 2),
                ("BOTTOMPADDING",(0,0),(-1,-1), 2),
                ("LEFTPADDING",(0,0),(-1,-1), 4),
            ]))
            flowables += [tbl, Spacer(1, 3)]

    # ── Path differences table ───────────────────────────────────────────────
    path_diffs = mga.get("path_differences") or mga.get("differences") or []
    if path_diffs and isinstance(path_diffs, list):
        flowables.append(Paragraph("Path Coefficient Differences", st["ColHeader"]))
        header = [_p(h, st["TH"]) for h in
                  ["Path", "Group 1", "Group 2", "Difference", "p-value", "Sig."]]
        rows = [header]
        for d in path_diffs:
            path   = d.get("path") or _safe_text(f"{d.get('rhs','?')} \u2192 {d.get('lhs','?')}")
            g1     = _fmt(d.get("g1") or d.get("group1"), 3)
            g2     = _fmt(d.get("g2") or d.get("group2"), 3)
            diff   = _fmt(d.get("diff") or d.get("difference"), 3)
            pv     = d.get("pvalue") or d.get("p")
            p_str  = _fmt_p(pv)
            stars  = _sig_stars(pv)
            rows.append([
                _p(path,  st["TCL"]),
                _p(g1,    st["TCMono"]),
                _p(g2,    st["TCMono"]),
                _p(diff,  st["TCMono"]),
                _p(p_str, st["TCMono"]),
                _p(stars, st["TC"]),
            ])
        cw = [_CW*0.28, _CW*0.14, _CW*0.14, _CW*0.14, _CW*0.16, _CW*0.14]
        tbl = Table(rows, colWidths=cw, repeatRows=1)
        tbl.setStyle(_apply_ts(_BASE_TS, [("ALIGN", (0,0),(0,-1), "LEFT")]))
        flowables += [tbl, Spacer(1, 6)]

    return flowables


def _build_predictive(pred: dict, st: dict) -> list:
    if not pred:
        return []

    flowables = _section_header("Predictive Power (PLSpredict / Q²)", st)

    q2   = pred.get("q2")        or []
    plsp = pred.get("plspredict") or []
    cvpat= pred.get("cvpat")      or []

    if q2 and isinstance(q2, list):
        flowables.append(Paragraph("Q² (Blindfolding)", st["ColHeader"]))
        header = [_p(h, st["TH"]) for h in ["Construct", "Q²", "Verdict"]]
        rows   = [header]
        for item in q2:
            lv = item.get("lv") or item.get("construct", "?")
            v  = item.get("q2") or item.get("value")
            ok_str = _safe_text("\u2713 Predictive relevance" if (
                v is not None and _safe_float(v, -1) > 0) else "\u2717 No predictive relevance")
            rows.append([_p(lv, st["TCL"]), _p(_fmt(v,3), st["TCMono"]),
                         _p(ok_str, st["TC"])])
        tbl = Table(rows, colWidths=[_CW*0.40, _CW*0.20, _CW*0.40], repeatRows=1)
        tbl.setStyle(_apply_ts(_BASE_TS, [("ALIGN", (0,0),(0,-1),"LEFT")]))
        flowables += [tbl, Spacer(1, 4)]

    if plsp and isinstance(plsp, list):
        flowables.append(Paragraph("PLSpredict", st["ColHeader"]))
        header = [_p(h, st["TH"]) for h in
                  ["Indicator","Q²_predict","RMSE_PLS","RMSE_LM","Verdict"]]
        rows   = [header]
        for item in plsp:
            ind  = item.get("indicator") or item.get("name", "?")
            q2p  = _fmt(item.get("q2_predict"), 3)
            rmse = _fmt(item.get("rmse_pls") or item.get("rmse"), 3)
            lm   = _fmt(item.get("rmse_lm"), 3)
            v_str= item.get("verdict") or ""
            rows.append([_p(ind, st["TCL"]),_p(q2p,st["TCMono"]),
                         _p(rmse,st["TCMono"]),_p(lm,st["TCMono"]),_p(v_str,st["TC"])])
        cw = [_CW*0.28,_CW*0.18,_CW*0.18,_CW*0.18,_CW*0.18]
        tbl = Table(rows, colWidths=cw, repeatRows=1)
        tbl.setStyle(_apply_ts(_BASE_TS, [("ALIGN",(0,0),(0,-1),"LEFT")]))
        flowables += [tbl, Spacer(1, 6)]

    return flowables


def _safe_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _build_diagram_section(diagram_png_b64: str, st: dict) -> list:
    if not diagram_png_b64:
        return []
    try:
        png_bytes = base64.b64decode(diagram_png_b64)
        img_buf   = io.BytesIO(png_bytes)
        # Fit within content width, max height 110mm
        max_w = _CW
        max_h = 110 * mm
        img = Image(img_buf, width=max_w, height=max_h, kind="bound")
        return [
            *_section_header("Path Diagram", st),
            img,
            Spacer(1, 6),
        ]
    except Exception:
        return []


# ═════════════════════════════════════════════════════════════════════════════
#  Footer / header on every page
# ═════════════════════════════════════════════════════════════════════════════

def _make_on_page(run_id: str, ts: str, total_pages_ref: list):
    """Return an onPage callback that draws header + footer on every page."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _register_fonts()
    fn = _FONT if _FONTS_REGISTERED else "Helvetica"

    def _on_page(canvas, doc):
        canvas.saveState()
        w, h = canvas._pagesize

        # ── Footer line ──────────────────────────────────────────────────
        canvas.setStrokeColor(_LINE)
        canvas.setLineWidth(0.3)
        canvas.line(_ML, _MB - 4, w - _MR, _MB - 4)

        canvas.setFont(fn, 6.5)
        canvas.setFillColor(_TEXT_MUTE)
        # Left: NAVAL-SEM + run id + ts
        canvas.drawString(_ML, _MB - 11,
            f"NAVAL-SEM \u00b7 Run {run_id[:12]} \u00b7 {ts}")
        # Right: page number
        page_str = f"Page {doc.page}"
        canvas.drawRightString(w - _MR, _MB - 11, page_str)
        # Centre: generated
        canvas.drawCentredString(w / 2, _MB - 11,
            f"Generated {now_str}")

        canvas.restoreState()

    return _on_page


# ═════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ═════════════════════════════════════════════════════════════════════════════

def generate_pdf(payload: dict) -> bytes:
    """
    Build and return PDF bytes from the /export/pdf payload dict.
    Runs synchronously — call in a thread-pool executor from async context.
    """
    _register_fonts()

    snap     = payload.get("snap")     or {}
    results  = payload.get("results")  or {}
    mga      = payload.get("mga")
    htmt     = payload.get("htmt")
    pred     = payload.get("predictive")
    diag_b64 = payload.get("diagram_png") or ""
    analyst  = payload.get("analyst")  or {}
    note     = payload.get("note")     or ""

    run_id   = snap.get("runId", "—")
    ts       = snap.get("ts", "—")

    st = _build_styles()

    # ── Story assembly ───────────────────────────────────────────────────────
    story: list = []
    story += _build_header_block(snap, analyst, note, st)
    story += _build_syntax_block(snap, st)
    story += _build_kpi_row(results, st)
    story += _build_fit_indices(results, st)
    story += _build_parameters(results, st)
    story += _build_loadings(results, st)
    story += _build_fornell_larcker(results, st)

    if htmt:
        story += _build_htmt(htmt, st)
    if mga:
        story += _build_mga(mga, st)
    if pred:
        story += _build_predictive(pred, st)

    # Diagram last (can be large)
    if diag_b64:
        story += _build_diagram_section(diag_b64, st)

    # ── Render to bytes ──────────────────────────────────────────────────────
    buf = io.BytesIO()
    total_pages_ref = [0]

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=_ML,
        rightMargin=_MR,
        topMargin=_MT,
        bottomMargin=_MB + 8,   # extra room for footer
        title=f"NAVAL-SEM Report — {run_id}",
        author=analyst.get("name") or "NAVAL-SEM",
        subject="Structural Equation Modelling Results",
        creator="NAVAL-SEM export_pdf.py",
    )

    on_page = _make_on_page(run_id, ts, total_pages_ref)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buf.getvalue()
