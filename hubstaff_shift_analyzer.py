"""
Hubstaff Timesheet — Shift Hours & Efficiency Analyzer
-------------------------------------------------------

Run with:
    pip install streamlit pandas openpyxl altair
    streamlit run hubstaff_shift_analyzer.py

Upload a Hubstaff "timesheet report" export, .xlsx or .csv (the format
that has columns: Member, Organization, Time Zone, Projects, Task
Summary, Start, Start Time, Stop, Stop Time, Duration, Activity, Idle,
Manual, Notes, Reasons, Type, Payment type). The CSV path assumes the
same column layout as the xlsx export — if Hubstaff's CSV export uses
different column names, the column-check error below will say exactly
which ones are missing.

WHAT COUNTS AS A "SHIFT" / "WORK DAY"
--------------------------------------
Hubstaff logs a new row every time tracking starts/stops, and it also
splits a row at midnight (so a session that runs from 23:00 to 02:00
appears as two rows on two different calendar dates). Grouping by the
"Start" date column alone would therefore chop real, continuous night
shifts in half and misreport whether the 8-hour target was hit.

Instead, this tool groups consecutive entries (per employee, sorted by
start time) into a single "shift": as soon as the gap between when one
entry stops and the next one starts exceeds the configurable threshold
(default 12 hours), everything after that gap is treated as the next
shift/work day. This matches the rule: "a gap of more than 12 hours
between entries marks the start of the next day's shift."

WHAT "EFFICIENCY" MEANS HERE
------------------------------
Hubstaff already records an "Activity" percentage per tracked entry
(mouse/keyboard activity ratio while the timer ran). This tool reports
each shift's efficiency as the DURATION-WEIGHTED AVERAGE of that
Activity column across all entries in the shift — not a simple mean of
percentages (which would overweight tiny few-second entries) and not a
measure of "hours worked / 8-hour target" (that's a separate column,
"Meets Target"). If a different definition of efficiency is intended,
only the `weighted_efficiency` computation below needs to change.
"""

import datetime as dt
import io

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hubstaff Shift & Efficiency Analyzer", layout="wide")

REQUIRED_COLUMNS = {"Member", "Start", "Stop", "Activity"}


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------

def parse_local_dt(date_val, time_str) -> dt.datetime | None:
    """Combine a Hubstaff date cell with its paired 'HH:MM:SS+ZZ:ZZ' time
    string into a naive local datetime. The timezone offset is dropped
    because it is constant for a given export (shown in the 'Time Zone'
    column) and duration is computed from local wall-clock time, so
    keeping it naive avoids arithmetic surprises across the offset."""
    if pd.isna(date_val) or pd.isna(time_str) or str(time_str).strip() == "":
        return None
    time_str = str(time_str).strip()
    core = time_str[:8]  # "HH:MM:SS"
    try:
        h, m, s = (int(x) for x in core.split(":"))
    except ValueError:
        return None
    base_date = pd.to_datetime(date_val).date()
    return dt.datetime.combine(base_date, dt.time(h, m, s))


def _to_naive_datetime_series(series: pd.Series) -> pd.Series:
    """Parse a column that already holds a full date+time value in one
    cell (as opposed to a bare date paired with a separate time column).
    Handles both timezone-aware strings (e.g. '2026-09-16 17:37:22+05:30')
    and plain ones, and strips any timezone so hour arithmetic stays in
    local wall-clock time, consistent with the split-column path above."""
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    try:
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return parsed


def extract_start_stop(df: pd.DataFrame) -> tuple[list, list]:
    """Two Hubstaff export shapes have been seen in practice:
      1. xlsx-style: a bare date in 'Start'/'Stop' plus the time-of-day
         (with timezone offset) in separate 'Start Time'/'Stop Time' columns.
      2. csv-style: a single 'Start'/'Stop' column already holding the
         full timestamp.
    Detect which shape this file is and parse accordingly, so both work
    without the user needing to tell us which export type it is."""
    if {"Start Time", "Stop Time"}.issubset(df.columns):
        start_dt = [parse_local_dt(d, t) for d, t in zip(df["Start"], df["Start Time"])]
        stop_dt = [parse_local_dt(d, t) for d, t in zip(df["Stop"], df["Stop Time"])]
    else:
        start_dt = list(_to_naive_datetime_series(df["Start"]))
        stop_dt = list(_to_naive_datetime_series(df["Stop"]))
    return start_dt, stop_dt


def _read_table(file_bytes: bytes, file_ext: str) -> pd.DataFrame:
    """Read either a Hubstaff .xlsx or .csv export into a DataFrame.
    CSV encoding isn't guaranteed, so fall back through a couple of common
    ones rather than failing outright on the first mismatch."""
    if file_ext == "xlsx":
        return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

    if file_ext == "csv":
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "latin1"):
            try:
                return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
        raise ValueError(f"Could not decode this CSV file (tried utf-8/latin1): {last_error}")

    raise ValueError(f"Unsupported file type: .{file_ext}")


@st.cache_data(show_spinner=False)
def load_and_process(
    file_bytes: bytes, file_ext: str, daily_target_hours: float, shift_gap_hours: float
):
    df = _read_table(file_bytes, file_ext)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"This doesn't look like a Hubstaff timesheet export — missing column(s): "
            f"{', '.join(sorted(missing))}"
        )

    start_dt, stop_dt = extract_start_stop(df)
    df["start_dt"] = start_dt
    df["stop_dt"] = stop_dt
    df = df.dropna(subset=["start_dt", "stop_dt"]).copy()
    if df.empty:
        raise ValueError(
            "No rows with parseable Start/Stop times were found. This file's "
            "date/time format may differ from the ones this tool has been tested "
            "against — send a couple of sample rows so the parser can be adjusted."
        )

    df["duration_hours"] = (df["stop_dt"] - df["start_dt"]).dt.total_seconds() / 3600.0
    df["Activity"] = pd.to_numeric(df["Activity"], errors="coerce").fillna(0.0)

    summary_rows = []
    detail_frames = []

    for member, g in df.groupby("Member", sort=False):
        g = g.sort_values("start_dt").reset_index(drop=True)

        shift_id = 0
        shift_ids = [0]
        for i in range(1, len(g)):
            gap_hours = (g.loc[i, "start_dt"] - g.loc[i - 1, "stop_dt"]).total_seconds() / 3600.0
            if gap_hours > shift_gap_hours:
                shift_id += 1
            shift_ids.append(shift_id)
        g["shift_id"] = shift_ids
        detail_frames.append(g)

        for _, sg in g.groupby("shift_id"):
            total_hours = sg["duration_hours"].sum()
            weighted_efficiency = (
                (sg["duration_hours"] * sg["Activity"]).sum() / total_hours
                if total_hours > 0
                else 0.0
            )
            summary_rows.append(
                {
                    "Member": member,
                    "Shift Date": sg["start_dt"].iloc[0].date(),
                    "Shift Start": sg["start_dt"].iloc[0].strftime("%Y-%m-%d %H:%M"),
                    "Shift End": sg["stop_dt"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
                    "Total Hours": round(total_hours, 2),
                    "Target Hours": daily_target_hours,
                    "Shortfall (hrs)": round(max(0.0, daily_target_hours - total_hours), 2),
                    "Meets Target": "Yes" if total_hours >= daily_target_hours else "No",
                    "Avg Efficiency %": round(weighted_efficiency * 100, 1),
                    "Entries": len(sg),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["Member", "Shift Date"]).reset_index(drop=True)
    detail_df = (
        pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    )
    return summary_df, detail_df


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("Hubstaff Timesheet — Shift Hours & Efficiency Analyzer")
st.caption(
    "Upload a Hubstaff timesheet export. Entries are grouped into shifts using the "
    "gap rule below, then checked against the daily hour target."
)

with st.sidebar:
    st.header("Settings")
    daily_target_hours = st.number_input(
        "Daily target hours", min_value=1.0, max_value=24.0, value=8.0, step=0.5
    )
    shift_gap_hours = st.number_input(
        "Gap that starts a new shift (hours)",
        min_value=1.0,
        max_value=24.0,
        value=12.0,
        step=0.5,
        help="A gap between one entry's stop time and the next entry's start time "
        "larger than this counts as the start of the next day's shift.",
    )

uploaded = st.file_uploader(
    "Upload Hubstaff timesheet report (.xlsx or .csv)", type=["xlsx", "csv"]
)

if uploaded is None:
    st.info("Waiting for a .xlsx or .csv file.")
    st.stop()

file_ext = uploaded.name.rsplit(".", 1)[-1].lower()

try:
    summary_df, detail_df = load_and_process(
        uploaded.getvalue(), file_ext, daily_target_hours, shift_gap_hours
    )
except Exception as exc:  # noqa: BLE001 - surface any parsing problem to the user
    st.error(str(exc))
    st.stop()

if summary_df.empty:
    st.warning("No valid time entries were found in this file.")
    st.stop()

# ---- KPI row -------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Employees", summary_df["Member"].nunique())
col2.metric("Shifts analyzed", len(summary_df))
col3.metric(
    "Shifts meeting target",
    f"{(summary_df['Meets Target'] == 'Yes').sum()} / {len(summary_df)}",
)
col4.metric("Avg efficiency (all shifts) %", round(summary_df["Avg Efficiency %"].mean(), 1))

# ---- Summary table ---------------------------------------------------------
st.subheader("Shift Summary")


def highlight_row(row):
    color = "background-color: #ffe1e1" if row["Meets Target"] == "No" else "background-color: #e6ffe6"
    return [color] * len(row)


st.dataframe(
    summary_df.style.apply(highlight_row, axis=1),
    use_container_width=True,
    hide_index=True,
)

csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
st.download_button("Download summary as CSV", csv_bytes, "shift_summary.csv", "text/csv")

# ---- Charts -----------------------------------------------------------
st.subheader("Hours vs. Target and Efficiency Trend")

for member, mg in summary_df.groupby("Member"):
    st.markdown(f"**{member}**")
    mg = mg.copy()
    mg["Shift Date"] = mg["Shift Date"].astype(str)

    bar = (
        alt.Chart(mg)
        .mark_bar()
        .encode(
            x=alt.X("Shift Date:N", sort=None, title="Shift date"),
            y=alt.Y("Total Hours:Q", title="Hours"),
            color=alt.condition(
                alt.datum["Total Hours"] >= daily_target_hours,
                alt.value("#2ecc71"),
                alt.value("#e74c3c"),
            ),
            tooltip=["Shift Date", "Total Hours", "Meets Target", "Avg Efficiency %"],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [daily_target_hours]}))
        .mark_rule(strokeDash=[4, 4], color="black")
        .encode(y="y")
    )
    st.altair_chart(bar + rule, use_container_width=True)

    line = (
        alt.Chart(mg)
        .mark_line(point=True)
        .encode(
            x=alt.X("Shift Date:N", sort=None, title="Shift date"),
            y=alt.Y("Avg Efficiency %:Q", scale=alt.Scale(domain=[0, 100])),
            tooltip=["Shift Date", "Avg Efficiency %"],
        )
    )
    st.altair_chart(line, use_container_width=True)

with st.expander("Raw entry-level detail (with shift grouping)"):
    st.dataframe(detail_df, use_container_width=True)
