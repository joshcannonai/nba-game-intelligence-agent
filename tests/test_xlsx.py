from pathlib import Path

from eval._xlsx import write_xlsx
from eval.build_mass_workbook import _model_rows, _summary_rows


def test_summary_uses_formulas_not_hardcoded_totals():
    rows = _summary_rows(10)
    joined = " ".join(str(c) for row in rows for c in row)
    assert "Model_A!A2" in joined
    assert "Model_D!M2" in joined
    assert "Games!O2" in joined
    assert "COUNTIF(Gemma_BC_10!A:A" in joined
    assert "COUNTIF(Gemma_BC!A:A" in joined
    assert "Vegas_Favorites" in joined
    assert "Always_Favorite" not in joined
    assert "0.659" not in joined
    assert "65.9" not in joined


def test_model_sheet_totals_row_sums_the_games_below():
    records = [
        {
            "game_id": "a",
            "game_date": "2025-10-21",
            "playoffs": 0,
            "away": "HOU",
            "home": "OKC",
            "cutoff": "2025-10-20",
            "actual_winner": "OKC",
            "A_p": 0.6,
            "A_pick": "OKC",
            "A_correct": 1,
            "A_american": -150,
            "A_decimal": 1.67,
            "A_pnl": 66.7,
        }
    ]
    rows = _model_rows("A", records)
    assert rows[1][0] == "=COUNTA(A3:A3)"
    assert rows[1][9] == "=SUM(J3:J3)"
    assert rows[1][12] == "=SUM(M3:M3)"


def test_stdlib_xlsx_round_trips_a_formula(tmp_path: Path):
    import zipfile

    path = tmp_path / "t.xlsx"
    write_xlsx(path, [("S", [["A", "B"], [1, "=A2+1"]])])
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("xl/worksheets/sheet1.xml").decode()
    assert "A2+1" in xml


def test_grouped_sheet_collapses_detail_rows(tmp_path: Path):
    import zipfile

    path = tmp_path / "g.xlsx"
    rows = [["H1", "H2"], ["=COUNTA(A3:A4)", "=SUM(B3:B4)"], ["g1", 1], ["g2", 2]]
    write_xlsx(
        path,
        [
            {
                "name": "Model_A",
                "rows": rows,
                "freeze": 2,
                "totals_row": 2,
                "group": (3, 4),
                "collapsed": True,
                "summary_below": False,
            }
        ],
    )
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("xl/worksheets/sheet1.xml").decode()
    assert 'summaryBelow="0"' in xml
    assert 'outlineLevel="1"' in xml
    assert 'hidden="1"' in xml
    assert 'collapsed="1"' in xml
    assert 'ySplit="2"' in xml
