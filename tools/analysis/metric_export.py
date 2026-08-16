"""Formatted spreadsheet export of the manuscript metric table.

Extracted verbatim from Tensile_augen_gneiss.ipynb cell 55 by
``scripts/extract_analysis_sections.py``. The code is unchanged except that the
lithology-dependent numbers -- specimen ids, weak-plane spacing, phase-warp
amplitude and the output filename -- now come from the :class:`~tools.lithology.
Lithology` passed to :func:`main`, so both rocks run one implementation.

Scope
-----
This section covers both lithologies in one pass and is driven from Tensile_general_plots.ipynb, not from either lithology notebook.
"""
from __future__ import annotations


import pandas as pd
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter



# --- implementation ---------------------------------------------------


def format_sheet(ws):
    # Header formatting
    header_fill = PatternFill(
        fill_type="solid",
        fgColor="1F4E79"
    )

    header_font = Font(
        bold=True,
        color="FFFFFF"
    )

    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True
        )

    # Body formatting
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(
                vertical="center",
                wrap_text=True
            )

    # Freeze header
    ws.freeze_panes = "A2"

    # Auto-size columns, capped at 36
    for column_cells in ws.columns:
        max_length = 0

        for cell in column_cells:
            value = "" if cell.value is None else str(cell.value)
            max_length = max(max_length, len(value))

        column_letter = get_column_letter(column_cells[0].column)
        ws.column_dimensions[column_letter].width = min(
            max(max_length + 2, 10),
            36
        )



def main():
    """Run this section for both lithologies in one pass."""
    global angle_headers, angle_rows, df_angle, df_metrics, df_points, \
        df_summary, df_weak, metrics_headers, metrics_rows, out_xlsx, \
        point_headers, point_rows, summary_headers, summary_rows, \
        weak_headers, weak_rows, writer, ws
    summary_headers = [
        "Angle (deg)",
        "Augen Mean",
        "Augen SEM",
        "Psammitic Mean",
        "Psammitic SEM"
    ]
    summary_rows = [
        [0, 10.40857143, 0.17646568, 9.57, 0.38],
        [15, 9.84, 0.05507571, 8.43666667, 0.39959702],
        [30, 9.89833333, 0.24823936, 8.10666667, 0.24551533],
        [45, 8.945, 0.285, 6.86, 0.30550505],
        [60, 8.67666667, 0.40005555, 5.9, 0.21071308],
        [75, 7.774, 0.27742747, 4.8075, 0.2500125],
        [90, 6.115, 0.39050181, 3.21666667, 0.32986529],
    ]
    df_summary = pd.DataFrame(summary_rows, columns=summary_headers)
    metrics_headers = ["Rock Type", "Metric", "Value"]
    metrics_rows = [
        ["Augen gneiss", "Rows", 30],
        ["Augen gneiss", "sigma0 (MPa)", "10.5077 ± 0.1574"],
        ["Augen gneiss", "sigma90 (MPa)", "8.1105 ± 0.2178"],
        ["Augen gneiss", "eta", "0.2242 ± 0.0954"],
        ["Augen gneiss", "skew", "8.6944 ± 2.7504"],
        ["Augen gneiss", "activation maximum (deg)", 72.07],
        ["Augen gneiss", "R²", 0.8881],
        ["Augen gneiss", "RMSE (MPa)", 0.3717],
        ["Augen gneiss", "χ²red", 1.111],

        ["Psammitic schist", "Rows", 22],
        ["Psammitic schist", "sigma0 (MPa)", "9.5776 ± 0.3929"],
        ["Psammitic schist", "sigma90 (MPa)", "5.4120 ± 0.2396"],
        ["Psammitic schist", "eta", "3.1643 ± 3.4317"],
        ["Psammitic schist", "skew", "24.9832 ± 10.6975"],
        ["Psammitic schist", "activation maximum (deg)", 82.94],
        ["Psammitic schist", "R²", 0.9312],
        ["Psammitic schist", "RMSE (MPa)", 0.5498],
        ["Psammitic schist", "χ²red", 1.302],
    ]
    df_metrics = pd.DataFrame(metrics_rows, columns=metrics_headers)
    angle_headers = [
        "Rock Type",
        "Angle (deg)",
        "n",
        "Mean Residual",
        "Angle RMSE",
        "Angle SSE %"
    ]
    angle_rows = [
        ["Augen gneiss", 15, 5, -0.2474, 0.4726, "26.94%"],
        ["Augen gneiss", 45, 4, 0.2772, 0.4478, "19.35%"],
        ["Augen gneiss", 75, 4, 0.1756, 0.4454, "19.14%"],
        ["Augen gneiss", 90, 4, -0.1105, 0.3422, "11.30%"],
        ["Augen gneiss", 0, 5, -0.0377, 0.3408, "14.01%"],
        ["Augen gneiss", 30, 5, 0.1187, 0.2579, "8.02%"],
        ["Augen gneiss", 60, 3, -0.0577, 0.1311, "1.24%"],

        ["Psammitic schist", 15, 3, -0.5225, 0.7697, "26.73%"],
        ["Psammitic schist", 75, 4, -0.0596, 0.5963, "21.39%"],
        ["Psammitic schist", 0, 3, -0.0076, 0.5375, "13.03%"],
        ["Psammitic schist", 90, 3, -0.4087, 0.5135, "11.90%"],
        ["Psammitic schist", 30, 3, 0.3712, 0.5082, "11.65%"],
        ["Psammitic schist", 45, 3, 0.2012, 0.4766, "10.25%"],
        ["Psammitic schist", 60, 3, 0.1523, 0.3346, "5.05%"],
    ]
    df_angle = pd.DataFrame(angle_rows, columns=angle_headers)
    point_headers = [
        "Rock Type",
        "Angle (deg)",
        "Tensile Strength (MPa)",
        "Predicted",
        "Residual",
        "Squared Error",
        "SSE %"
    ]
    point_rows = [
        ["Augen gneiss", 45, 9.64, 8.8703, 0.7697, 0.5924, "14.29%"],
        ["Augen gneiss", 75, 8.18, 7.4544, 0.7256, 0.5265, "12.70%"],
        ["Augen gneiss", 15, 9.59, 10.2734, -0.6834, 0.4670, "11.26%"],
        ["Augen gneiss", 15, 9.71, 10.2734, -0.5634, 0.3174, "7.66%"],
        ["Augen gneiss", 0, 9.99, 10.5077, -0.5177, 0.2680, "6.46%"],
        ["Augen gneiss", 90, 7.61, 8.1105, -0.5005, 0.2505, "6.04%"],
        ["Augen gneiss", 15, 10.73, 10.2734, 0.4566, 0.2085, "5.03%"],
        ["Augen gneiss", 30, 10.11, 9.6753, 0.4347, 0.1890, "4.56%"],
        ["Augen gneiss", 0, 10.91, 10.5077, 0.4023, 0.1619, "3.90%"],
        ["Augen gneiss", 90, 8.51, 8.1105, 0.3995, 0.1596, "3.85%"],

        ["Psammitic schist", 15, 7.85, 8.9592, -1.1092, 1.2304, "18.50%"],
        ["Psammitic schist", 30, 8.59, 7.7355, 0.8545, 0.7302, "10.98%"],
        ["Psammitic schist", 90, 4.61, 5.4120, -0.8020, 0.6432, "9.67%"],
        ["Psammitic schist", 75, 2.74, 3.5271, -0.7871, 0.6196, "9.32%"],
        ["Psammitic schist", 0, 10.31, 9.5776, 0.7324, 0.5364, "8.07%"],
        ["Psammitic schist", 15, 8.26, 8.9592, -0.6992, 0.4889, "7.35%"],
        ["Psammitic schist", 75, 4.22, 3.5271, 0.6929, 0.4800, "7.22%"],
        ["Psammitic schist", 45, 7.26, 6.6588, 0.6012, 0.3615, "5.44%"],
        ["Psammitic schist", 60, 6.32, 5.7477, 0.5723, 0.3275, "4.93%"],
        ["Psammitic schist", 0, 9.05, 9.5776, -0.5276, 0.2783, "4.19%"],
    ]
    df_points = pd.DataFrame(point_rows, columns=point_headers)
    weak_headers = [
        "Rock",
        "eta",
        "kappa",
        "weakening_peak_angle_deg",
        "peak_angle_CI_low_deg",
        "peak_angle_CI_high_deg",
        "minimum_weakening_factor",
        "minimum_factor_CI_low",
        "minimum_factor_CI_high",
        "maximum_TI_reduction_pct",
        "reduction_CI_low_pct",
        "reduction_CI_high_pct"
    ]
    weak_rows = [
        [
            "Augen gneiss",
            0.224150,
            8.694422,
            72.072772,
            64.242534,
            82.941554,
            0.903821,
            0.757595,
            0.969215,
            9.61785,
            3.078538,
            24.240535
        ],
        [
            "Psammitic schist",
            3.164282,
            24.983232,
            82.941554,
            78.486336,
            82.941554,
            0.490038,
            0.365273,
            0.717385,
            50.99618,
            28.261456,
            63.472677
        ],
    ]
    df_weak = pd.DataFrame(weak_rows, columns=weak_headers)
    out_xlsx = "rock_strength_metrics.xlsx"
    with pd.ExcelWriter(
        out_xlsx,
        engine="openpyxl"
    ) as writer:

        df_summary.to_excel(
            writer,
            index=False,
            sheet_name="Summary Table"
        )

        df_metrics.to_excel(
            writer,
            index=False,
            sheet_name="Equation Fit"
        )

        df_angle.to_excel(
            writer,
            index=False,
            sheet_name="Angle RMSE"
        )

        df_points.to_excel(
            writer,
            index=False,
            sheet_name="Top RMSE Points"
        )

        df_weak.to_excel(
            writer,
            index=False,
            sheet_name="Weakening Factor"
        )

        # Apply formatting to every sheet
        for ws in writer.book.worksheets:
            format_sheet(ws)
    print(f"Excel file saved to: {out_xlsx}")
