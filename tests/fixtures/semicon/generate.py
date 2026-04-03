"""Generate semicon test fixture files with real-world data quality gotchas.

Usage:
    uv run python tests/fixtures/semicon/generate.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
SOURCES = HERE / "sources"
SOURCES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Base data (ground truth) — 10 lots
# ---------------------------------------------------------------------------

LOTS = [
    {
        "lot_id": "L001234",
        "wafer_count": 25,
        "operation": "ETCH_M1",
        "step_sequence": 450,
        "tool_id": "ETCH-07A",
        "track_in": "2025-03-15 08:12:33",
        "track_out": "2025-03-15 09:45:01",
        "recipe": "M1_ETCH_V3",
        "route": "LOGIC_28NM",
        "priority": 1,
        "hold": False,
        "hold_text": "NONE",
    },
    {
        "lot_id": "L001235",
        "wafer_count": 24,
        "operation": "LITHO_M2",
        "step_sequence": 520,
        "tool_id": "ASML-1980",
        "track_in": "2025-03-15 08:30:00",
        "track_out": "2025-03-15 10:00:00",
        "recipe": "M2_LITHO_EUV",
        "route": "LOGIC_28NM",
        "priority": 3,
        "hold": True,
        "hold_text": "QUALITY",
    },
    {
        "lot_id": "L001236",
        "wafer_count": 25,
        "operation": "DIFF_M1",
        "step_sequence": 380,
        "tool_id": "DIFF-03B",
        "track_in": "2025-03-15 09:00:00",
        "track_out": "2025-03-15 11:30:00",
        "recipe": "M1_DIFF_V2",
        "route": "LOGIC_28NM",
        "priority": 2,
        "hold": False,
        "hold_text": "NONE",
    },
    {
        "lot_id": "L001237",
        "wafer_count": 13,
        "operation": "CMP_M3",
        "step_sequence": 610,
        "tool_id": "CMP-12",
        "track_in": "2025-03-15 09:15:00",
        "track_out": "2025-03-15 12:00:00",
        "recipe": "M3_CMP_STD",
        "route": "LOGIC_28NM",
        "priority": 2,
        "hold": False,
        "hold_text": "NONE",
    },
    {
        "lot_id": "L001238",
        "wafer_count": 25,
        "operation": "CVD_ILD",
        "step_sequence": 440,
        "tool_id": "CVD-05",
        "track_in": "2025-03-15 09:30:00",
        "track_out": "2025-03-15 13:00:00",
        "recipe": "ILD_DEP_V1",
        "route": "LOGIC_28NM",
        "priority": 1,
        "hold": True,
        "hold_text": "ENGINEERING",
    },
    {
        "lot_id": "L005001",
        "wafer_count": 25,
        "operation": "IMPLANT_SD",
        "step_sequence": 250,
        "tool_id": "IMP-02A",
        "track_in": "2025-03-15 10:00:00",
        "track_out": "2025-03-15 11:15:00",
        "recipe": "SD_IMPLANT_HI",
        "route": "MEMORY_3D",
        "priority": 3,
        "hold": False,
        "hold_text": "NONE",
    },
    {
        "lot_id": "L005002",
        "wafer_count": 22,
        "operation": "ETCH_GATE",
        "step_sequence": 310,
        "tool_id": "ETCH-11C",
        "track_in": "2025-03-15 10:30:00",
        "track_out": "2025-03-15 12:45:00",
        "recipe": "GATE_ETCH_V2",
        "route": "MEMORY_3D",
        "priority": 2,
        "hold": False,
        "hold_text": "NONE",
    },
    {
        "lot_id": "L005003",
        "wafer_count": 25,
        "operation": "PVD_METAL",
        "step_sequence": 720,
        "tool_id": "PVD-08",
        "track_in": "2025-03-15 11:00:00",
        "track_out": "2025-03-15 14:00:00",
        "recipe": "M1_PVD_TI",
        "route": "MEMORY_3D",
        "priority": 3,
        "hold": False,
        "hold_text": "NONE",
    },
    {
        "lot_id": "L005004",
        "wafer_count": 1,
        "operation": "METROLOGY",
        "step_sequence": 455,
        "tool_id": "CD-SEM-3",
        "track_in": "2025-03-15 11:30:00",
        "track_out": "2025-03-15 11:45:00",
        "recipe": "CD_MEAS_V1",
        "route": "MEMORY_3D",
        "priority": 4,
        "hold": False,
        "hold_text": "NONE",
    },
    {
        "lot_id": "L005005",
        "wafer_count": 25,
        "operation": "CLEAN_PRE",
        "step_sequence": 100,
        "tool_id": "WET-01",
        "track_in": "2025-03-15 07:00:00",
        "track_out": None,
        "recipe": "PRE_CLEAN_SC1",
        "route": "LOGIC_28NM",
        "priority": 2,
        "hold": True,
        "hold_text": "QUALITY",
    },
]


def _ts_epoch(s: str | None) -> int:
    if s is None:
        return 0
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())


def _ts_epoch_ms(s: str | None) -> int:
    return _ts_epoch(s) * 1000


# ---------------------------------------------------------------------------
# File 1: Camstar MES — mostly clean, mixed naive/tz timestamps (gotcha 27)
# ---------------------------------------------------------------------------


def gen_camstar():
    carriers = [
        "FOUP-88",
        "FOUP-91",
        "FOUP-92",
        "FOUP-55",
        "FOUP-12",
        "FOUP-33",
        "FOUP-44",
        "FOUP-67",
        "FOUP-99",
        "FOUP-71",
    ]
    header = "LOT_ID,CARRIER_ID,QTY,CURRENT_OPER,OPER_SEQ,RESOURCE,TRACKIN_DT,TRACKOUT_DT,RECIPE_NAME,FLOW,LOT_PRIORITY,HOLD_STATUS"
    rows = []
    for i, lot in enumerate(LOTS):
        ti = lot["track_in"]
        to = lot["track_out"] or ""
        if i == 2:  # L001236 — UTC tz suffix
            ti = "2025-03-15T09:00:00+00:00"
            to = "2025-03-15T11:30:00+00:00"
        elif i == 5:  # L005001 — JST tz suffix (raw value, not UTC-converted)
            ti = "2025-03-15T10:00:00+09:00"
            to = "2025-03-15T11:15:00+09:00"
        rows.append(
            f"{lot['lot_id']},{carriers[i]},{lot['wafer_count']},{lot['operation']},"
            f"{lot['step_sequence']},{lot['tool_id']},{ti},{to},"
            f"{lot['recipe']},{lot['route']},{lot['priority']},{lot['hold_text']}"
        )
    (SOURCES / "camstar_mes.csv").write_text(header + "\n" + "\n".join(rows) + "\n")
    print("  camstar_mes.csv")


# ---------------------------------------------------------------------------
# File 2: FabX — unix epoch, 0/1 bools
# ---------------------------------------------------------------------------


def gen_fabx():
    header = "lot\twfrs\top_code\tseq\ttool\trecipe\tt_in\tt_out\troute\tprio\thold"
    rows = []
    for lot in LOTS:
        ti = _ts_epoch(lot["track_in"])
        to = _ts_epoch(lot["track_out"])
        hold = 1 if lot["hold"] else 0
        rows.append(
            f"{lot['lot_id']}\t{lot['wafer_count']}\t{lot['operation']}\t"
            f"{lot['step_sequence']}\t{lot['tool_id']}\t{lot['recipe']}\t"
            f"{ti}\t{to}\t{lot['route']}\t{lot['priority']}\t{hold}"
        )
    (SOURCES / "fabx.tsv").write_text(header + "\n" + "\n".join(rows) + "\n")
    print("  fabx.tsv")


# ---------------------------------------------------------------------------
# File 3: SAP ERP — semicolon delim, decimal comma, German names, units
# Gotchas: 4, 25, 7, 22, 16
# ---------------------------------------------------------------------------


def gen_sap():
    header = "PRIORITAET;REZEPT;WERKZEUG_ID;ANLAGE_ROUTE;SCHRITT_NR;LOS_NR;WAFER_ANZ;OPERATION;EINGANG_ZEIT;AUSGANG_ZEIT;HOLD_STATUS"
    rows = []
    for lot in LOTS:
        los_nr = lot["lot_id"].replace("L", "").zfill(6)
        wafer = f"{lot['wafer_count']},0 pcs"
        schritt = f"{lot['step_sequence']} steps"
        ti = datetime.strptime(lot["track_in"], "%Y-%m-%d %H:%M:%S")
        ti_str = ti.strftime("%d.%m.%Y %H:%M:%S")
        if lot["track_out"]:
            to = datetime.strptime(lot["track_out"], "%Y-%m-%d %H:%M:%S")
            to_str = to.strftime("%d.%m.%Y %H:%M:%S")
        else:
            to_str = ""
        hold = "Ja" if lot["hold"] else "Nein"
        rows.append(
            f"{lot['priority']};{lot['recipe']};{lot['tool_id']};{lot['route']};"
            f"{schritt};{los_nr};{wafer};{lot['operation']};{ti_str};{to_str};{hold}"
        )
    (SOURCES / "sap_erp.csv").write_text(header + "\n" + "\n".join(rows) + "\n")
    print("  sap_erp.csv")


# ---------------------------------------------------------------------------
# File 4: Promise legacy — UTF-8 BOM, French headers, type flip, mixed nulls
# Gotchas: 6, 5, 2, 8, 12
# ---------------------------------------------------------------------------


def gen_promise():
    header = (
        "Lot_Réf,Nb_Plaques,Opération,Séquence,Équipement,Heure_Entrée,Heure_Sortie,Recette,Parcours,Priorité,Blocage"
    )
    null_sentinels = ["NULL", "N/A", "-", "#N/A", ""]
    rows = []
    for i, lot in enumerate(LOTS):
        # Type flip: first 5 as bare int, last 5 with prefix
        if i < 5:
            lot_ref = lot["lot_id"].replace("L", "").lstrip("0") or "0"
        else:
            lot_ref = lot["lot_id"]

        ti = lot["track_in"]
        to = lot["track_out"] if lot["track_out"] else null_sentinels[i % len(null_sentinels)]

        # Mixed nulls in optional columns
        seq: str | int = lot["step_sequence"]
        route = lot["route"]
        if i == 3:
            seq = "N/A"
        if i == 6:
            route = "-"
        if i == 8:
            seq = "#N/A"

        hold = "Oui" if lot["hold"] else "Non"
        rows.append(
            f"{lot_ref},{lot['wafer_count']},{lot['operation']},{seq},"
            f"{lot['tool_id']},{ti},{to},{lot['recipe']},{route},"
            f"{lot['priority']},{hold}"
        )

    content = header + "\n" + "\n".join(rows) + "\n"
    bom = b"\xef\xbb\xbf"
    (SOURCES / "promise_legacy.csv").write_bytes(bom + content.encode("utf-8"))
    print("  promise_legacy.csv")


# ---------------------------------------------------------------------------
# File 5: InfinityQS SPC — XLSX junk rows, distractors, sci notation, sentinel dates
# Gotchas: 23, 14, 10, 24, 3
# ---------------------------------------------------------------------------


def gen_infinityqs():
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Lot Movement"

    # Garbage header rows
    ws.append(["InfinityQS SPC Export — Lot Movement Report"])
    ws.append(["Generated: 2025-03-15 14:00:00"])
    ws.append([])

    headers = [
        "Lot",
        "Wafers",
        "Op",
        "Step",
        "Tool",
        "Recipe",
        "Route",
        "Track_In",
        "Track_Out",
        "Priority",
        "Hold",
        "Cp",
        "Cpk",
        "Yield_Pct",
        "Defect_Count",
        "Operator",
        "Shift",
    ]
    ws.append(headers)

    operators = [
        "J.Kim",
        "S.Park",
        "M.Lee",
        "T.Chen",
        "R.Singh",
        "A.Tanaka",
        "P.Garcia",
        "K.Mueller",
        "D.Brown",
        "L.Wang",
    ]
    shifts = ["Day", "Night", "Day", "Day", "Night", "Day", "Night", "Day", "Day", "Night"]

    for i, lot in enumerate(LOTS):
        wfr = lot["wafer_count"]
        if i == 8:
            wfr = 1.0e0

        to = lot["track_out"] if lot["track_out"] else "9999-12-31 00:00:00"
        hold = "Y" if lot["hold"] else "N"

        ws.append(
            [
                lot["lot_id"],
                wfr,
                lot["operation"],
                lot["step_sequence"],
                lot["tool_id"],
                lot["recipe"],
                lot["route"],
                lot["track_in"],
                to,
                lot["priority"],
                hold,
                round(1.2 + i * 0.05, 2),
                round(1.1 + i * 0.04, 2),
                round(97.5 + i * 0.3, 1),
                max(0, 12 - i),
                operators[i],
                shifts[i],
            ]
        )

    # Trailing total row
    ws.append(
        [
            "TOTAL",
            210,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            1.33,
            1.21,
            98.5,
            47,
            "",
            "",
        ]
    )

    wb.save(SOURCES / "infinityqs_spc.xlsx")
    print("  infinityqs_spc.xlsx")


# ---------------------------------------------------------------------------
# File 6: FactoryTalk — JSON, split date/time, tz, text enums, mixed bools
# Gotchas: 15, 21, 17, 19
# ---------------------------------------------------------------------------


def gen_factorytalk():
    priority_map = {1: "Hot", 2: "Normal", 3: "Low", 4: "Monitoring"}
    hold_values = [False, True, "Yes", False, 1, "No", 0, False, "No", "Yes"]

    records = []
    for i, lot in enumerate(LOTS):
        dt_in = datetime.strptime(lot["track_in"], "%Y-%m-%d %H:%M:%S")
        in_date = dt_in.strftime("%Y-%m-%d")
        tz_suffix = "+09:00" if i in (2, 7) else "+00:00"
        in_time = dt_in.strftime("%H:%M:%S") + tz_suffix

        if lot["track_out"]:
            dt_out = datetime.strptime(lot["track_out"], "%Y-%m-%d %H:%M:%S")
            out_date = dt_out.strftime("%Y-%m-%d")
            out_time = dt_out.strftime("%H:%M:%S") + tz_suffix
        else:
            out_date = None
            out_time = None

        records.append(
            {
                "lot_number": lot["lot_id"],
                "wafer_qty": lot["wafer_count"],
                "process_step": lot["operation"],
                "step_num": lot["step_sequence"],
                "equipment": lot["tool_id"],
                "track_in_date": in_date,
                "track_in_time": in_time,
                "track_out_date": out_date,
                "track_out_time": out_time,
                "recipe_id": lot["recipe"],
                "production_route": lot["route"],
                "lot_priority": priority_map[lot["priority"]],
                "on_hold": hold_values[i],
            }
        )

    (SOURCES / "factorytalk.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    print("  factorytalk.json")


# ---------------------------------------------------------------------------
# File 7: AMHS Carrier — duplicate Date cols, terse names, whitespace, free-text hold
# Gotchas: 11, 13, 9, 18, 28
# ---------------------------------------------------------------------------


def gen_amhs():
    header = "id,Count (pcs),op,seq,eq,Date,Date,rcp,rte,pri,hold_reason"
    rows = []
    for i, lot in enumerate(LOTS):
        seq = str(lot["step_sequence"])
        rte = lot["route"]
        # Whitespace-only cells for optional values
        if i == 3:
            seq = "   "
        if i == 7:
            rte = "   "
        if i == 8:
            seq = "  "

        to = lot["track_out"] or ""

        # Free-text hold values
        hold_map = {
            0: "",
            1: "Quality Hold",
            2: "",
            3: "",
            4: "Engr",
            5: "",
            6: "",
            7: "",
            8: "",
            9: "Quality Hold",
        }
        hold = hold_map[i]

        rows.append(
            f"{lot['lot_id']},{lot['wafer_count']},{lot['operation']},{seq},"
            f"{lot['tool_id']},{lot['track_in']},{to},{lot['recipe']},{rte},"
            f"{lot['priority']},{hold}"
        )

    (SOURCES / "amhs_carrier.csv").write_text(header + "\n" + "\n".join(rows) + "\n")
    print("  amhs_carrier.csv")


# ---------------------------------------------------------------------------
# File 8: Partner Fab JP — Japanese headers, mixed dates, ms epoch, scrambled order
# Gotchas: 6(JP), 1, 20, 16
# ---------------------------------------------------------------------------


def gen_partner_jp():
    header = "工程\tロット番号\t装置\tウェーハ数\t開始時刻\tレシピ\t工程順序\t終了時刻\tルート\t優先度\t保留"

    date_formats_in = [
        "2025-03-15 08:12:33",
        "2025/03/15 08:30:00",
        "15-Mar-2025 09:00:00",
        "2025-03-15 09:15:00",
        "2025/03/15 09:30:00",
        "2025-03-15 10:00:00",
        "15-Mar-2025 10:30:00",
        "2025/03/15 11:00:00",
        "2025-03-15 11:30:00",
        "2025/03/15 07:00:00",
    ]

    hold_jp = {False: "なし", True: "品質"}

    rows = []
    for i, lot in enumerate(LOTS):
        ti = date_formats_in[i]

        if lot["track_out"] and i in (1, 4, 6):
            to = str(_ts_epoch_ms(lot["track_out"]))
        elif lot["track_out"]:
            to = lot["track_out"]
        else:
            to = ""

        hold = "技術" if lot["hold_text"] == "ENGINEERING" else hold_jp.get(lot["hold"], "なし")

        rows.append(
            f"{lot['operation']}\t{lot['lot_id']}\t{lot['tool_id']}\t{lot['wafer_count']}\t"
            f"{ti}\t{lot['recipe']}\t{lot['step_sequence']}\t{to}\t{lot['route']}\t"
            f"{lot['priority']}\t{hold}"
        )

    (SOURCES / "partner_fab_jp.tsv").write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print("  partner_fab_jp.tsv")


# ---------------------------------------------------------------------------
# File 9: Operator Manual Log — XLSX, maximum human messiness
# Gotchas: 1, 2, 8, 17, 26, 22
# ---------------------------------------------------------------------------


def gen_operator_log():
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Log"

    headers = [
        "Lot #",
        "Wafer Count",
        "Operation",
        "Step",
        "Tool",
        "Start Time",
        "End Time",
        "Recipe",
        "Route",
        "Priority",
        "On Hold?",
    ]
    ws.append(headers)

    date_formats = [
        "3/15/2025 8:12 AM",
        "2025-03-15 08:30",
        "15-Mar-25 9:00",
        "Mar 15 2025 09:15:00",
        "3/15/2025 9:30:00 AM",
        "2025-03-15T10:00:00",
        "15/03/2025 10:30",
        "2025.03.15 11:00:00",
        "Mar 15, 2025 11:30 AM",
        "3/15/25 7:00",
    ]

    lot_ids = [
        1234,
        1235,
        "L001236",
        1237,
        "L001238",
        "L005001",
        5002,
        "L005003",
        "L005004",
        "L005005",
    ]

    end_times = [
        "3/15/2025 9:45 AM",
        "2025-03-15 10:00",
        "15-Mar-25 11:30",
        "Mar 15 2025 12:00:00",
        "3/15/2025 1:00:00 PM",
        "2025-03-15T11:15:00",
        "15/03/2025 12:45",
        "2025.03.15 14:00:00",
        "Mar 15, 2025 11:45 AM",
        "N/A",
    ]

    priorities = [
        "HOT",
        "normal",
        "Med",
        "Med",
        "HOT!",
        "low",
        "Med",
        "low",
        "low",
        "Med",
    ]

    wafer_counts = [
        "25 wafers",
        24,
        25,
        "~13",
        "25 wafers",
        25,
        "22 wafers",
        25,
        1,
        "25 wafers",
    ]

    holds = ["N", "Y", "no", "N", "yes", "N", "N", "N", "N", "YES"]

    for i, lot in enumerate(LOTS):
        recipe = lot["recipe"]
        if i == 0:
            recipe = "M1_ETCH,V3"

        ws.append(
            [
                lot_ids[i],
                wafer_counts[i],
                lot["operation"],
                lot["step_sequence"],
                lot["tool_id"],
                date_formats[i],
                end_times[i],
                recipe,
                lot["route"],
                priorities[i],
                holds[i],
            ]
        )

    wb.save(SOURCES / "operator_log.xlsx")
    print("  operator_log.xlsx")


# ---------------------------------------------------------------------------
# File 10: Yield Pilot — Parquet with wrong dtypes, distractors
# Gotchas: 7, 3, 10, 14
# ---------------------------------------------------------------------------


def gen_yield_pilot():
    lot_ids = [int(lot["lot_id"].replace("L", "").lstrip("0") or "0") for lot in LOTS]
    track_outs = []
    for lot in LOTS:
        if lot["track_out"]:
            track_outs.append(datetime.strptime(lot["track_out"], "%Y-%m-%d %H:%M:%S"))
        else:
            track_outs.append(datetime(9999, 12, 31))

    df = pl.DataFrame(
        {
            "lot_id": pl.Series(lot_ids, dtype=pl.Int64),
            "wfr_count": pl.Series([float(lot["wafer_count"]) for lot in LOTS], dtype=pl.Float64),
            "oper": [lot["operation"] for lot in LOTS],
            "step_seq": pl.Series([lot["step_sequence"] for lot in LOTS], dtype=pl.Int32),
            "tool": [lot["tool_id"] for lot in LOTS],
            "track_in": [datetime.strptime(lot["track_in"], "%Y-%m-%d %H:%M:%S") for lot in LOTS],
            "track_out": track_outs,
            "recipe": [lot["recipe"] for lot in LOTS],
            "route": [lot["route"] for lot in LOTS],
            "priority": pl.Series([lot["priority"] for lot in LOTS], dtype=pl.Int8),
            "hold": pl.Series([1 if lot["hold"] else 0 for lot in LOTS], dtype=pl.Int8),
            "yield_pct": pl.Series([97.5 + i * 0.3 for i in range(10)], dtype=pl.Float64),
            "defect_count": pl.Series([max(0, 12 - i) for i in range(10)], dtype=pl.Int32),
            "review_status": [
                "Approved",
                "Pending",
                "Approved",
                "Approved",
                "Rejected",
                "Approved",
                "Pending",
                "Approved",
                "Approved",
                "Pending",
            ],
            "inspector": [
                "J.Kim",
                "S.Park",
                "M.Lee",
                "T.Chen",
                "R.Singh",
                "A.Tanaka",
                "P.Garcia",
                "K.Mueller",
                "D.Brown",
                "L.Wang",
            ],
        }
    )

    df.write_parquet(SOURCES / "yield_pilot.parquet")
    print("  yield_pilot.parquet")


# ---------------------------------------------------------------------------
# File 11: Broken Exporter — unquoted newlines, trailing junk
# Gotchas: 29, 24
# ---------------------------------------------------------------------------


def gen_broken():
    header = (
        "LOT_ID,QTY,CURRENT_OPER,OPER_SEQ,RESOURCE,TRACKIN_DT,TRACKOUT_DT,RECIPE_NAME,FLOW,LOT_PRIORITY,HOLD_STATUS"
    )
    lines = [header]

    for i, lot in enumerate(LOTS):
        to = lot["track_out"] or ""
        hold = lot["hold_text"]
        recipe = lot["recipe"]
        oper = lot["operation"]

        # Unquoted newlines
        if i == 2:
            recipe = "M1_DIFF\nV2"
        if i == 7:
            oper = "PVD\nMETAL"

        lines.append(
            f"{lot['lot_id']},{lot['wafer_count']},{oper},{lot['step_sequence']},"
            f"{lot['tool_id']},{lot['track_in']},{to},{recipe},{lot['route']},"
            f"{lot['priority']},{hold}"
        )

    # Trailing summary row
    lines.append("TOTAL,210,,,,,,,,,")

    (SOURCES / "broken_exporter.csv").write_text("\n".join(lines) + "\n")
    print("  broken_exporter.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"Generating semicon fixtures in {SOURCES}/\n")
    gen_camstar()
    gen_fabx()
    gen_sap()
    gen_promise()
    gen_infinityqs()
    gen_factorytalk()
    gen_amhs()
    gen_partner_jp()
    gen_operator_log()
    gen_yield_pilot()
    gen_broken()
    print(f"\nDone — {len(list(SOURCES.iterdir()))} files generated.")


if __name__ == "__main__":
    main()
