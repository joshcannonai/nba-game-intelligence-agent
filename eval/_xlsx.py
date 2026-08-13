"""Minimal .xlsx writer. Stdlib only, so the mass-eval workbook does not need openpyxl.

Supports strings, numbers, booleans, formulas, frozen header rows, column
widths, and Excel native row outline (collapse/expand). Styling is a header
row plus an optional totals row.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

COL_CACHE = [""]


def col_letter(idx: int) -> str:
    """1-based column index to Excel letters."""
    n = idx
    out = []
    while n:
        n, rem = divmod(n - 1, 26)
        out.append(chr(65 + rem))
    return "".join(reversed(out))


def cell_ref(row: int, col: int) -> str:
    return f"{col_letter(col)}{row}"


def _cell_xml(row: int, col: int, value, style: int | None = None) -> str:
    ref = cell_ref(row, col)
    style_attr = f' s="{style}"' if style else ""
    if value is None or value == "":
        return f'<c r="{ref}"{style_attr}/>' if style else ""
    if isinstance(value, bool):
        return f'<c r="{ref}"{style_attr} t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, str) and value.startswith("="):
        return f'<c r="{ref}"{style_attr}><f>{escape(value[1:])}</f></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    return f'<c r="{ref}"{style_attr} t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def _normalize_sheet(sheet) -> dict:
    if isinstance(sheet, dict):
        return sheet
    name, rows = sheet[0], sheet[1]
    opts = sheet[2] if len(sheet) > 2 else {}
    return {"name": name, "rows": rows, **opts}


def _sheet_xml(spec: dict) -> str:
    rows: list[list] = spec["rows"]
    freeze = spec.get("freeze", 1)
    group = spec.get("group")  # (first_detail, last_detail) 1-based inclusive
    collapsed = spec.get("collapsed", True)
    summary_below = spec.get("summary_below", False)
    widths = spec.get("widths")
    totals_row = spec.get("totals_row")  # 1-based row index to style as totals

    max_r = max(len(rows), 1)
    max_c = max((len(r) for r in rows), default=1)
    dim = f"A1:{cell_ref(max_r, max_c)}"
    out = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
    ]
    if group:
        below = "1" if summary_below else "0"
        out.append(
            f'<sheetPr><outlinePr summaryBelow="{below}" summaryRight="0"/></sheetPr>'
        )
    out.append(f'<dimension ref="{dim}"/>')
    out.append(
        '<sheetViews><sheetView workbookViewId="0" tabSelected="0" showOutlineSymbols="1">'
    )
    if freeze and max_r > freeze:
        out.append(
            f'<pane ySplit="{freeze}" topLeftCell="A{freeze + 1}" '
            'activePane="bottomLeft" state="frozen"/>'
        )
    out.append("</sheetView></sheetViews>")
    outline = ' outlineLevelRow="1"' if group else ""
    out.append(f'<sheetFormatPr defaultRowHeight="15"{outline}/>')
    if widths:
        out.append("<cols>")
        for i, width in enumerate(widths, start=1):
            out.append(f'<col min="{i}" max="{i}" width="{width}" customWidth="1"/>')
        out.append("</cols>")
    out.append("<sheetData>")
    group_lo, group_hi = group if group else (None, None)
    for r_i, row in enumerate(rows, start=1):
        style = None
        if r_i == 1:
            style = 1
        elif totals_row and r_i == totals_row:
            style = 2
        attrs = [f'r="{r_i}"']
        if group_lo is not None and group_lo <= r_i <= group_hi:
            attrs.append('outlineLevel="1"')
            if collapsed:
                attrs.append('hidden="1"')
        elif (
            collapsed
            and group_lo is not None
            and not summary_below
            and r_i == group_lo - 1
        ):
            attrs.append('collapsed="1"')
        cells = "".join(
            _cell_xml(r_i, c_i, v, style) for c_i, v in enumerate(row, start=1)
        )
        out.append(f"<row {' '.join(attrs)}>{cells}</row>")
    out.append("</sheetData>")
    out.append("</worksheet>")
    return "".join(out)


def write_xlsx(path: Path, sheets: list) -> Path:
    """sheets = [(name, rows)] or [(name, rows, options)] or dict specs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    specs = [_normalize_sheet(s) for s in sheets]
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _content_types(len(specs)))
        zf.writestr("_rels/.rels", _rels_root())
        zf.writestr("xl/workbook.xml", _workbook_xml(specs))
        zf.writestr("xl/_rels/workbook.xml.rels", _workbook_rels(len(specs)))
        zf.writestr("xl/styles.xml", _styles_xml())
        for i, spec in enumerate(specs, start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml(spec))
    path.write_bytes(buf.getvalue())
    return path


def _content_types(n: int) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for i in range(1, n + 1):
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(overrides)
        + "</Types>"
    )


def _rels_root() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def _workbook_rels(n: int) -> str:
    rels = [
        '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    ]
    for i in range(1, n + 1):
        rels.append(
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )


def _workbook_xml(specs: list[dict]) -> str:
    entries = []
    for i, spec in enumerate(specs, start=1):
        safe = escape(str(spec["name"])[:31])
        entries.append(f'<sheet name="{safe}" sheetId="{i}" r:id="rId{i}"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>" + "".join(entries) + "</sheets></workbook>"
    )


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="3">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1B3A4B"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFE8F1F5"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf/></cellStyleXfs>
  <cellXfs count="3">
    <xf xfId="0"/>
    <xf xfId="0" fontId="1" fillId="1" applyFont="1" applyFill="1"/>
    <xf xfId="0" fontId="2" fillId="2" applyFont="1" applyFill="1"/>
  </cellXfs>
</styleSheet>
"""
