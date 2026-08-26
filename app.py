import io
import re
import json
import hashlib
import urllib.parse
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF
from supabase import create_client

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None


# =========================================================
# APP CONFIG
# =========================================================
st.set_page_config(
    page_title="AcadIntel 360",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

USAGE_COLUMNS = [
    "School", "Teacher", "Teacher Key", "Raw Module", "KPI Module",
    "Minutes", "DateTime", "Grade", "Subject", "Book", "Source File"
]

TEACHER_COLUMNS = [
    "School", "Teacher", "Teacher Key", "Rostered", "Status",
    "Lesson Delivery", "Lesson Target", "Lesson KPI %",
    "Library", "Library Target", "Library KPI %",
    "Other Modules", "Other Target", "Other KPI %",
    "Total Minutes", "Active Days", "Eligible Working Days",
    "Books Used", "Grades Covered", "Subjects Covered",
    "First Activity", "Last Activity", "Grade", "Subject", "Health Score"
]

SCHOOL_COLUMNS = [
    "School", "Teachers", "Active", "Inactive / Never Logged In",
    "Met All KPIs", "Overall Compliance %", "Health Score",
    "Lesson Delivery Minutes", "Library Minutes", "Other Modules Minutes",
    "Lesson Target / Day", "Library Target / Day", "Other Target / Day"
]

DEFAULT_KPI = {
    "lessonDelivery": ("Lesson Delivery", 10.0),
    "library": ("Library", 30.0),
    "otherModules": ("Other Modules", 15.0),
}

st.markdown(
    """
    <style>
    .block-container {padding-top:1rem; padding-bottom:3rem; max-width:1500px;}
    [data-testid="stMetric"] {background:white; border:1px solid #e8eaf2; padding:14px; border-radius:18px; box-shadow:0 8px 24px rgba(30,32,55,.055);}
    .hero {background:linear-gradient(120deg,#4338ca,#2563eb 55%,#0ea5e9); color:white; padding:26px; border-radius:24px; margin-bottom:18px; box-shadow:0 14px 36px rgba(55,48,163,.18);}
    .hero h1 {margin:0; font-size:38px;}
    .hero p {margin:.35rem 0 0; opacity:.92;}
    .card {background:white; border:1px solid #e8eaf2; border-radius:18px; padding:16px; margin:8px 0;}
    .ai-card {border-radius:18px; padding:16px 18px; margin:9px 0; border-left:5px solid #4f46e5; background:#f8fafc;}
    div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button, div[data-testid="stLinkButton"] a {border-radius:12px; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================
for key, default in {
    "raw": pd.DataFrame(columns=USAGE_COLUMNS),
    "import_errors": [],
    "ai_cache": {},
    "follow_school": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =========================================================
# BASIC HELPERS
# =========================================================
def norm_col(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def norm_name(value):
    text = str(value or "").strip()
    text = re.sub(r"^[\.\s]+", "", text)
    text = re.sub(r"\b(mrs|ms|mr|miss|dr)\.?\b", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def teacher_key(value):
    return re.sub(r"[^a-z0-9]", "", norm_name(value).lower())


def first_col(df, candidates):
    lookup = {norm_col(c): c for c in df.columns}
    for candidate in candidates:
        if norm_col(candidate) in lookup:
            return lookup[norm_col(candidate)]
    return None


def json_safe(value):
    return json.loads(json.dumps(value, default=str))


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def fmt_minutes(value):
    return f"{safe_float(value):,.1f} min"


# =========================================================
# SUPABASE
# =========================================================
@st.cache_resource
def get_supabase():
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_SECRET_KEY", "")
    if not url or not key:
        return None
    return create_client(url, key)


sb = get_supabase()


def db_select(table, filters=None, order=None):
    if sb is None:
        return []
    try:
        q = sb.table(table).select("*")
        for key, value in (filters or {}).items():
            q = q.eq(key, value)
        if order:
            q = q.order(order)
        return q.execute().data or []
    except Exception as exc:
        st.warning(f"Database read issue ({table}): {exc}")
        return []


def db_insert(table, payload):
    if sb is None:
        raise RuntimeError("Supabase is not connected.")
    return sb.table(table).insert(json_safe(payload)).execute().data


def db_update(table, payload, row_id):
    if sb is None:
        raise RuntimeError("Supabase is not connected.")
    return sb.table(table).update(json_safe(payload)).eq("id", row_id).execute().data


def db_delete(table, row_id):
    if sb is None:
        raise RuntimeError("Supabase is not connected.")
    return sb.table(table).delete().eq("id", row_id).execute().data


# =========================================================
# GEMINI - ROBUST DEPLOYMENT WITH MODEL DISCOVERY
# =========================================================
@st.cache_resource
def get_ai_client():
    key = st.secrets.get("GEMINI_API_KEY", "")
    if not key or genai is None:
        return None
    return genai.Client(api_key=key)


@st.cache_data(ttl=3600, show_spinner=False)
def discover_gemini_model():
    client = get_ai_client()
    if client is None:
        return None, "GEMINI_API_KEY missing or google-genai unavailable"

    configured = str(st.secrets.get("GEMINI_MODEL", "")).strip()
    preferred = [configured] if configured else []
    preferred += ["gemini-3.7-flash", "gemini-2.5-flash"]

    for name in preferred:
        if not name:
            continue
        try:
            client.models.get(model=name)
            return name, None
        except Exception:
            pass

    try:
        candidates = []
        for model in client.models.list():
            actions = list(getattr(model, "supported_actions", None) or [])
            name = str(getattr(model, "name", "") or "")
            if "generateContent" in actions and "gemini" in name.lower():
                clean = name.replace("models/", "")
                candidates.append(clean)
        if candidates:
            flash = [m for m in candidates if "flash" in m.lower()]
            return (flash[0] if flash else candidates[0]), None
    except Exception as exc:
        return None, str(exc)

    return None, "No Gemini model with generateContent access was found for this API key."


def ai_generate(prompt, force=False):
    client = get_ai_client()
    model, model_error = discover_gemini_model()
    if client is None or not model:
        raise RuntimeError(model_error or "Gemini is not connected.")

    cache_key = hashlib.sha256((model + "\n" + prompt).encode("utf-8")).hexdigest()
    if not force and cache_key in st.session_state.ai_cache:
        return st.session_state.ai_cache[cache_key], model

    config = None
    if genai_types is not None:
        config = genai_types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=5000,
        )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    st.session_state.ai_cache[cache_key] = text
    return text, model


# =========================================================
# RAW USERMETRICS IMPORT
# =========================================================
def detect_schema(df):
    return {
        "school": first_col(df, ["School", "School Name", "Institution", "Center", "Centre", "Institution Name"]),
        "teacher": first_col(df, ["Teacher", "Teacher Name", "User Name", "Username", "Name"]),
        "first": first_col(df, ["FirstName", "First Name"]),
        "last": first_col(df, ["LastName", "Last Name"]),
        "minutes": first_col(df, ["Duration (Minutes)", "Duration Minutes", "Minutes", "Minutes Logged", "Usage Minutes", "Duration"]),
        "module": first_col(df, ["Type", "Module", "Module Name", "Category"]),
        "date": first_col(df, ["StartTime", "Start Time", "Date", "Activity Date", "Log Date"]),
        "grade": first_col(df, ["Grade", "Class"]),
        "subject": first_col(df, ["Subject"]),
        "book": first_col(df, ["Book", "Book Name", "Content", "Content Name"]),
    }


def classify_module(value):
    key = norm_col(value)
    if key == "lessondelivery" or "lessondelivery" in key:
        return "Lesson Delivery"
    if key == "library" or "library" in key:
        return "Library"
    return "Other Modules"


def parse_minutes(series):
    # Supports numeric minutes or hh:mm:ss style duration strings.
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(1, int(len(series) * 0.8)):
        return numeric.fillna(0).clip(lower=0)
    td = pd.to_timedelta(series.astype(str), errors="coerce")
    return (td.dt.total_seconds() / 60).fillna(0).clip(lower=0)


@st.cache_data(show_spinner=False)
def parse_uploaded_file(file_bytes, filename):
    bio = io.BytesIO(file_bytes)
    df = pd.read_csv(bio) if filename.lower().endswith(".csv") else pd.read_excel(bio)
    schema = detect_schema(df)

    if not schema["school"]:
        raise ValueError(f"{filename}: School / Institution / Center column not found.")
    if not schema["minutes"]:
        raise ValueError(f"{filename}: Duration / Minutes column not found.")

    out = pd.DataFrame(index=df.index)
    out["School"] = df[schema["school"]].fillna("").astype(str).str.strip()

    if schema["teacher"]:
        out["Teacher"] = df[schema["teacher"]].fillna("").astype(str).map(norm_name)
    else:
        first = df[schema["first"]].fillna("").astype(str) if schema["first"] else pd.Series([""] * len(df), index=df.index)
        last = df[schema["last"]].fillna("").astype(str) if schema["last"] else pd.Series([""] * len(df), index=df.index)
        out["Teacher"] = (first + " " + last).str.strip().map(norm_name)

    out["Teacher"] = out["Teacher"].replace("", "Unattributed Activity")
    out["Teacher Key"] = out["Teacher"].map(teacher_key)
    out["Minutes"] = parse_minutes(df[schema["minutes"]])

    if schema["module"]:
        out["Raw Module"] = df[schema["module"]].fillna("").astype(str).str.strip()
    elif schema["book"]:
        out["Raw Module"] = "Book"
    else:
        out["Raw Module"] = "Other"

    out["KPI Module"] = out["Raw Module"].map(classify_module)
    out["DateTime"] = pd.to_datetime(df[schema["date"]], errors="coerce") if schema["date"] else pd.NaT
    out["Grade"] = df[schema["grade"]].fillna("").astype(str) if schema["grade"] else ""
    out["Subject"] = df[schema["subject"]].fillna("").astype(str) if schema["subject"] else ""
    out["Book"] = df[schema["book"]].fillna("").astype(str) if schema["book"] else ""
    out["Source File"] = filename

    out = out[out["School"].str.strip() != ""].copy()
    return out[USAGE_COLUMNS]


def combine_usage_files(files):
    frames, errors = [], []
    for file in files[:100]:
        try:
            frames.append(parse_uploaded_file(file.getvalue(), file.name))
        except Exception as exc:
            errors.append(str(exc))

    if not frames:
        return pd.DataFrame(columns=USAGE_COLUMNS), errors

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["School", "Teacher Key", "Raw Module", "Minutes", "DateTime", "Grade", "Subject", "Book"]
    ).reset_index(drop=True)
    return combined, errors


# =========================================================
# ROSTER + SHARED ACCOUNTS
# =========================================================
def roster_dataframe():
    rows = db_select("master_roster", {"active": True})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "school_name", "teacher_name", "teacher_key", "grade", "subject", "email", "phone"
    ])


def import_roster(files):
    incoming = []
    for file in files:
        bio = io.BytesIO(file.getvalue())
        df = pd.read_csv(bio) if file.name.lower().endswith(".csv") else pd.read_excel(bio)
        school_col = first_col(df, ["School", "School Name", "Institution", "Center"])
        teacher_col = first_col(df, ["Teacher", "Teacher Name", "Name"])
        fcol = first_col(df, ["FirstName", "First Name"])
        lcol = first_col(df, ["LastName", "Last Name"])
        grade_col = first_col(df, ["Grade", "Class"])
        subject_col = first_col(df, ["Subject"])
        email_col = first_col(df, ["Email", "Email ID"])
        phone_col = first_col(df, ["Phone", "Mobile", "Mobile Number"])

        if not school_col:
            raise ValueError(f"{file.name}: School column not found.")
        if not teacher_col and not (fcol or lcol):
            raise ValueError(f"{file.name}: Teacher name column not found.")

        if teacher_col:
            names = df[teacher_col].fillna("").astype(str).map(norm_name)
        else:
            first = df[fcol].fillna("").astype(str) if fcol else pd.Series([""] * len(df), index=df.index)
            last = df[lcol].fillna("").astype(str) if lcol else pd.Series([""] * len(df), index=df.index)
            names = (first + " " + last).str.strip().map(norm_name)

        for idx in df.index:
            school = str(df.loc[idx, school_col]).strip()
            teacher = names.loc[idx]
            if not school or not teacher:
                continue
            incoming.append({
                "school_name": school,
                "teacher_name": teacher,
                "teacher_key": teacher_key(teacher),
                "grade": str(df.loc[idx, grade_col]).strip() if grade_col else None,
                "subject": str(df.loc[idx, subject_col]).strip() if subject_col else None,
                "email": str(df.loc[idx, email_col]).strip() if email_col else None,
                "phone": str(df.loc[idx, phone_col]).strip() if phone_col else None,
                "active": True,
            })

    existing = {(r["school_name"], r["teacher_key"]): r for r in db_select("master_roster")}
    inserted = updated = 0
    for row in incoming:
        key = (row["school_name"], row["teacher_key"])
        if key in existing:
            db_update("master_roster", row, existing[key]["id"])
            updated += 1
        else:
            db_insert("master_roster", row)
            inserted += 1
    return inserted, updated


def shared_account_set():
    return {
        (r["school_name"], r["account_key"])
        for r in db_select("shared_accounts")
        if r.get("active", True)
    }


def is_shared_account(school, teacher, shared_accounts):
    tkey = teacher_key(teacher)
    skey = teacher_key(school)
    return (
        tkey in ("", "unattributedactivity")
        or (school, tkey) in shared_accounts
        or (tkey and tkey == skey)
    )


# =========================================================
# KPI SETTINGS
# =========================================================
def load_kpi_rows():
    return db_select("kpi_settings")


def effective_kpis(school=None):
    result = {k: v[1] for k, v in DEFAULT_KPI.items()}
    rows = load_kpi_rows()

    for module_key in result:
        matches = [r for r in rows if r.get("scope") == "GLOBAL" and r.get("module_key") == module_key and r.get("active", True)]
        if matches:
            matches.sort(key=lambda x: x.get("updated_at") or x.get("created_at") or "")
            result[module_key] = safe_float(matches[-1].get("target_minutes_per_day"), result[module_key])

    if school:
        for module_key in result:
            matches = [r for r in rows if r.get("scope") == "SCHOOL" and r.get("school_name") == school and r.get("module_key") == module_key and r.get("active", True)]
            if matches:
                matches.sort(key=lambda x: x.get("updated_at") or x.get("created_at") or "")
                result[module_key] = safe_float(matches[-1].get("target_minutes_per_day"), result[module_key])
    return result


def save_kpi(scope, module_key, value, school_name=None):
    rows = load_kpi_rows()
    existing = [r for r in rows if r.get("scope") == scope and r.get("module_key") == module_key and (r.get("school_name") or None) == (school_name or None)]
    payload = {
        "scope": scope,
        "school_name": school_name,
        "module_key": module_key,
        "module_name": DEFAULT_KPI[module_key][0],
        "target_minutes_per_day": float(value),
        "active": True,
        "updated_at": datetime.utcnow().isoformat(),
    }
    if existing:
        db_update("kpi_settings", payload, existing[-1]["id"])
    else:
        db_insert("kpi_settings", payload)


# =========================================================
# REVIEW PERIOD + ANALYTICS
# =========================================================
def working_days_between(start, end):
    if start > end:
        return 0
    dates = pd.date_range(start=start, end=end, freq="D")
    return sum(d.weekday() != 6 for d in dates)


def filter_period(df, start, end):
    if df.empty:
        return pd.DataFrame(columns=USAGE_COLUMNS)
    out = df.copy()
    if out["DateTime"].notna().any():
        d = out["DateTime"].dt.date
        out = out[(d >= start) & (d <= end)]
    return out


def build_teacher_record(school, teacher, tkey, activity, workdays, kpis, rostered, grade="", subject=""):
    activity = activity if isinstance(activity, pd.DataFrame) else pd.DataFrame(columns=USAGE_COLUMNS)

    def module_sum(name):
        if activity.empty:
            return 0.0
        return safe_float(activity.loc[activity["KPI Module"] == name, "Minutes"].sum())

    lesson = module_sum("Lesson Delivery")
    library = module_sum("Library")
    other = module_sum("Other Modules")
    total = safe_float(activity["Minutes"].sum()) if not activity.empty else 0.0
    active_days = int(activity["DateTime"].dropna().dt.date.nunique()) if not activity.empty else 0
    books = int(activity.loc[activity["Book"].astype(str).str.strip() != "", "Book"].nunique()) if not activity.empty else 0
    grades = int(activity.loc[activity["Grade"].astype(str).str.strip() != "", "Grade"].nunique()) if not activity.empty else 0
    subjects = int(activity.loc[activity["Subject"].astype(str).str.strip() != "", "Subject"].nunique()) if not activity.empty else 0
    first_activity = activity["DateTime"].min() if not activity.empty else pd.NaT
    last_activity = activity["DateTime"].max() if not activity.empty else pd.NaT

    lt = workdays * safe_float(kpis.get("lessonDelivery"))
    libt = workdays * safe_float(kpis.get("library"))
    ot = workdays * safe_float(kpis.get("otherModules"))

    lp = (lesson / lt * 100) if lt > 0 else 0
    libp = (library / libt * 100) if libt > 0 else 0
    op = (other / ot * 100) if ot > 0 else 0
    consistency = min((active_days / workdays * 100), 100) if workdays > 0 else 0
    health = round(min(lp, 100) * .40 + min(libp, 100) * .35 + min(op, 100) * .15 + consistency * .10)

    if total <= 0:
        status = "Never Logged In" if rostered else "0 Usage"
    elif lp >= 100 and libp >= 100 and op >= 100:
        status = "Meeting All KPIs"
    elif max(lp, libp, op) >= 100:
        status = "Partially Meeting"
    else:
        status = "Below KPI"

    return {
        "School": school,
        "Teacher": teacher,
        "Teacher Key": tkey,
        "Rostered": rostered,
        "Status": status,
        "Lesson Delivery": round(lesson, 1),
        "Lesson Target": round(lt, 1),
        "Lesson KPI %": round(lp, 1),
        "Library": round(library, 1),
        "Library Target": round(libt, 1),
        "Library KPI %": round(libp, 1),
        "Other Modules": round(other, 1),
        "Other Target": round(ot, 1),
        "Other KPI %": round(op, 1),
        "Total Minutes": round(total, 1),
        "Active Days": active_days,
        "Eligible Working Days": int(workdays),
        "Books Used": books,
        "Grades Covered": grades,
        "Subjects Covered": subjects,
        "First Activity": first_activity,
        "Last Activity": last_activity,
        "Grade": grade or "",
        "Subject": subject or "",
        "Health Score": int(max(0, min(100, health))),
    }


def build_analytics(raw, start_date, end_date, working_days_override=None):
    workdays = int(working_days_override) if working_days_override else working_days_between(start_date, end_date)
    roster = roster_dataframe()
    shared = shared_account_set()
    usage = raw.copy()

    if usage.empty:
        personal = pd.DataFrame(columns=USAGE_COLUMNS)
        shared_usage = pd.DataFrame(columns=USAGE_COLUMNS)
    else:
        usage["Is Shared"] = usage.apply(lambda r: is_shared_account(r["School"], r["Teacher"], shared), axis=1)
        personal = usage[~usage["Is Shared"]].copy()
        shared_usage = usage[usage["Is Shared"]].copy()

    usage_map = {}
    if not personal.empty:
        for (school, tkey), group in personal.groupby(["School", "Teacher Key"], dropna=False):
            usage_map[(str(school), str(tkey))] = group

    records, seen = [], set()

    if not roster.empty:
        for _, row in roster.iterrows():
            school = str(row.get("school_name", "")).strip()
            tkey = str(row.get("teacher_key", "")).strip()
            teacher = str(row.get("teacher_name", "")).strip()
            if not school or not tkey or not teacher:
                continue
            seen.add((school, tkey))
            activity = usage_map.get((school, tkey), pd.DataFrame(columns=USAGE_COLUMNS))
            records.append(build_teacher_record(
                school, teacher, tkey, activity, workdays, effective_kpis(school), True,
                row.get("grade", ""), row.get("subject", "")
            ))

    for (school, tkey), activity in usage_map.items():
        if (school, tkey) in seen:
            continue
        teacher = str(activity["Teacher"].iloc[0]).strip()
        records.append(build_teacher_record(
            school, teacher, tkey, activity, workdays, effective_kpis(school), False
        ))

    teacher_df = pd.DataFrame(records, columns=TEACHER_COLUMNS)
    if teacher_df.empty:
        return teacher_df, pd.DataFrame(columns=SCHOOL_COLUMNS), shared_usage

    school_records = []
    for school, group in teacher_df.groupby("School"):
        total = len(group)
        active = int((pd.to_numeric(group["Total Minutes"], errors="coerce").fillna(0) > 0).sum())
        inactive = int(group["Status"].isin(["Never Logged In", "0 Usage"]).sum())
        fully_met = int(((group["Lesson KPI %"] >= 100) & (group["Library KPI %"] >= 100) & (group["Other KPI %"] >= 100)).sum())
        health = int(round(pd.to_numeric(group["Health Score"], errors="coerce").fillna(0).mean()))
        kpis = effective_kpis(school)
        school_records.append({
            "School": school,
            "Teachers": total,
            "Active": active,
            "Inactive / Never Logged In": inactive,
            "Met All KPIs": fully_met,
            "Overall Compliance %": round(fully_met / total * 100, 1) if total else 0.0,
            "Health Score": health,
            "Lesson Delivery Minutes": round(group["Lesson Delivery"].sum(), 1),
            "Library Minutes": round(group["Library"].sum(), 1),
            "Other Modules Minutes": round(group["Other Modules"].sum(), 1),
            "Lesson Target / Day": kpis["lessonDelivery"],
            "Library Target / Day": kpis["library"],
            "Other Target / Day": kpis["otherModules"],
        })

    school_df = pd.DataFrame(school_records, columns=SCHOOL_COLUMNS)
    if not school_df.empty:
        school_df = school_df.sort_values(["Health Score", "Overall Compliance %"], ascending=False).reset_index(drop=True)
    return teacher_df, school_df, shared_usage


# =========================================================
# REPORT PROMPTS + DETERMINISTIC SUMMARY
# =========================================================
def school_verified_facts(school_row, teacher_data, raw_school, start_date, end_date, workdays):
    module_breakdown = {}
    if not raw_school.empty:
        module_breakdown = raw_school.groupby("Raw Module")["Minutes"].sum().sort_values(ascending=False).head(20).round(1).to_dict()

    priority = teacher_data.sort_values(["Health Score", "Total Minutes"]).head(8)
    top = teacher_data.sort_values(["Health Score", "Total Minutes"], ascending=False).head(5)

    return {
        "review_period": {"start": str(start_date), "end": str(end_date), "working_days": workdays},
        "school_summary": json_safe(school_row.to_dict()),
        "priority_teachers": json_safe(priority[["Teacher", "Status", "Health Score", "Lesson KPI %", "Library KPI %", "Other KPI %", "Total Minutes", "Active Days"]].to_dict("records")),
        "top_teachers": json_safe(top[["Teacher", "Status", "Health Score", "Lesson KPI %", "Library KPI %", "Other KPI %", "Total Minutes", "Active Days"]].to_dict("records")),
        "module_breakdown_minutes": json_safe(module_breakdown),
    }


def teacher_verified_facts(teacher_row, evidence, start_date, end_date):
    logs = []
    if not evidence.empty:
        logs = evidence[["DateTime", "Raw Module", "KPI Module", "Grade", "Subject", "Book", "Minutes"]].sort_values("DateTime", ascending=False).head(120).astype(str).to_dict("records")
    return {
        "review_period": {"start": str(start_date), "end": str(end_date)},
        "teacher_summary": json_safe(teacher_row.to_dict()),
        "activity_evidence": logs,
    }


def school_report_prompt(facts, action_days):
    return f"""
You are AcadIntel 360, an academic implementation intelligence analyst.
Use ONLY the VERIFIED FACTS below. KPI calculations are already completed by Python; never recalculate or invent them.
Never invent causes, commitments, infrastructure issues, dates, or teacher behaviour. If a cause is not established, say: "The data does not establish the cause."

Create a management-ready School 360 report with these exact headings:
EXECUTIVE SUMMARY
KPI SCORECARD INTERPRETATION
IMPLEMENTATION STRENGTHS
CRITICAL GAPS
TEACHERS REQUIRING ATTENTION
TOP PERFORMERS
MODULE ADOPTION PATTERN
{action_days}-DAY ACTION PLAN
NEXT REVIEW TARGETS
EVIDENCE

Tone: constructive, polite, motivational, specific and professional. No markdown asterisks.
VERIFIED FACTS:
{json.dumps(facts, default=str)}
"""


def teacher_report_prompt(facts, action_days):
    return f"""
You are AcadIntel 360. Use ONLY the VERIFIED FACTS below. KPI calculations are final and must not be altered.
Never invent reasons or behaviour.
Create a detailed Teacher 360 report using these headings:
EXECUTIVE DIAGNOSIS
KPI SCORECARD INTERPRETATION
ACTIVITY CONSISTENCY
CONTENT AND CURRICULUM ENGAGEMENT
STRENGTHS
IMPLEMENTATION GAPS
{action_days}-DAY DEVELOPMENT ACTION PLAN
NEXT REVIEW TARGET
MOTIVATIONAL CLOSING
EVIDENCE
No markdown asterisks. Tone must be respectful, developmental and measurable.
VERIFIED FACTS:
{json.dumps(facts, default=str)}
"""


def whatsapp_prompt(facts, contact_role="School Management"):
    return f"""
Draft a concise customized WhatsApp performance update addressed to {contact_role}.
Use ONLY verified facts. Do not invent causes. No markdown asterisks. Use tasteful Unicode icons.
Include review period, school health/compliance, important positive point, priority gap, next action, and a polite closing.
VERIFIED FACTS:
{json.dumps(facts, default=str)}
"""


def call_script_prompt(facts, previous_followups):
    return f"""
Prepare a natural KDM call script for an Academic Consultant. Use ONLY verified facts and the listed follow-ups.
Do not invent reasons or commitments.
Headings: OPENING, POSITIVE START, DATA-BACKED UPDATE, KEY QUESTIONS, PRIORITY GAPS, DESIRED COMMITMENT, CLOSING, POST-CALL NOTE.
No markdown asterisks.
VERIFIED FACTS:
{json.dumps(facts, default=str)}
PREVIOUS FOLLOW-UPS:
{json.dumps(previous_followups[:5], default=str)}
"""


def deterministic_school_summary(school_row, action_days):
    return (
        f"EXECUTIVE SUMMARY\n"
        f"{school_row['School']} has a Health Score of {school_row['Health Score']}/100 with overall full-KPI compliance of {school_row['Overall Compliance %']}%. "
        f"There are {school_row['Teachers']} teachers in the current analysis, of whom {school_row['Active']} are active and {school_row['Inactive / Never Logged In']} are inactive or never logged in.\n\n"
        f"KPI POSITION\n"
        f"Lesson Delivery: {school_row['Lesson Delivery Minutes']:.1f} minutes. Library: {school_row['Library Minutes']:.1f} minutes. Other Modules: {school_row['Other Modules Minutes']:.1f} minutes.\n\n"
        f"{action_days}-DAY ACTION PLAN\n"
        f"Prioritize inactive and below-KPI teachers, review usage evidence, obtain school commitments, and recheck the same KPI metrics at the next review.\n\n"
        f"EVIDENCE\n"
        f"All figures above are calculated directly from the loaded UserMetrics data and Master Roster for the selected review period."
    )


# =========================================================
# REPORT RENDERING + PDF
# =========================================================
def clean_ai_text(text):
    return (text or "").replace("**", "").replace("###", "").replace("##", "").strip()


def render_report(text):
    clean = clean_ai_text(text)
    if not clean:
        return
    headings = re.compile(r"(?m)^([A-Z][A-Z0-9 /&\-]{2,60})\s*$")
    matches = list(headings.finditer(clean))
    if not matches:
        st.markdown(f'<div class="ai-card">{clean.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        return
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(clean)
        body = clean[start:end].strip()
        if body:
            st.markdown(
                f'<div class="ai-card"><b>{title.title()}</b><br><br>{body.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )


class ReportPDF(FPDF):
    def header(self):
        self.set_fill_color(67, 56, 202)
        self.rect(0, 0, 210, 24, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 16)
        self.set_xy(12, 7)
        self.cell(0, 8, "AcadIntel 360", 0, 1)
        self.set_font("Helvetica", "", 8)
        self.set_x(12)
        self.cell(0, 4, "Academic Intelligence - Evidence - Action", 0, 1)
        self.set_text_color(20, 20, 20)

    def footer(self):
        self.set_y(-12)
        self.set_text_color(100, 100, 100)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 6, f"Generated {datetime.now().strftime('%d %b %Y %H:%M')} | Page {self.page_no()}", 0, 0, "C")


def pdf_safe(value):
    return str(value).encode("latin-1", "replace").decode("latin-1")


def make_text_pdf(title, subtitle, report_text, evidence_df=None):
    pdf = ReportPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.ln(18)
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 8, pdf_safe(title))
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, pdf_safe(subtitle))
    pdf.ln(3)

    for block in clean_ai_text(report_text).split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if len(lines) > 1 and lines[0].strip().isupper():
            pdf.set_fill_color(242, 244, 255)
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(0, 7, pdf_safe(lines[0].strip()), fill=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5.5, pdf_safe("\n".join(lines[1:])))
        else:
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5.5, pdf_safe(block))
        pdf.ln(2)

    if evidence_df is not None and not evidence_df.empty:
        pdf.add_page()
        pdf.ln(18)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 7, "Evidence Audit Trail", 0, 1)
        cols = ["DateTime", "Raw Module", "Grade", "Subject", "Book", "Minutes"]
        widths = [27, 32, 20, 32, 65, 14]
        pdf.set_font("Helvetica", "B", 7)
        for c, w in zip(cols, widths):
            pdf.cell(w, 6, c.replace("DateTime", "Date"), 1, 0, "C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 6.3)
        for _, row in evidence_df.sort_values("DateTime", ascending=False).head(150).iterrows():
            vals = [str(row.get(c, "")) for c in cols]
            vals[0] = vals[0][:16]
            vals[1] = vals[1][:18]
            vals[2] = vals[2][:12]
            vals[3] = vals[3][:18]
            vals[4] = vals[4][:36]
            vals[5] = f"{safe_float(row.get('Minutes')):.1f}"
            for value, w in zip(vals, widths):
                pdf.cell(w, 5.5, pdf_safe(value), 1, 0, "L")
            pdf.ln()

    return bytes(pdf.output())


# =========================================================
# FOLLOW-UP DIALOG
# =========================================================
@st.dialog("Add Follow-Up")
def followup_dialog(school_name):
    with st.form("followup_form", clear_on_submit=True):
        followup_date = st.date_input("Follow-up date", value=date.today() + timedelta(days=7))
        issue = st.text_area("Specific issue / gap")
        commitment = st.text_area("Last commitment")
        status = st.selectbox("Status", ["Open", "In Progress", "Resolved"])
        remarks = st.text_area("Remarks")
        if st.form_submit_button("Save Follow-Up", use_container_width=True):
            if not issue.strip():
                st.error("Please enter the issue/gap.")
                return
            db_insert("followups", {
                "school_name": school_name,
                "followup_date": str(followup_date),
                "issue": issue.strip(),
                "last_commitment": commitment.strip(),
                "status": status,
                "remarks": remarks.strip(),
            })
            st.session_state.follow_school = None
            st.rerun()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("🎓 AcadIntel 360")
    st.caption("Academic Intelligence • Evidence • Action")

    if sb is None:
        st.error("Supabase not connected")
    else:
        st.success("Supabase connected")

    model_name, model_error = discover_gemini_model()
    if model_name:
        st.success(f"Gemini ready: {model_name}")
    else:
        st.error(f"Gemini not ready: {model_error}")

    uploaded_files = st.file_uploader(
        "Upload raw UserMetrics files",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        help="Upload up to 100 raw company exports together.",
    )
    if uploaded_files and st.button("⚡ Process Raw Data", use_container_width=True):
        with st.spinner("Processing and validating files..."):
            data, errors = combine_usage_files(uploaded_files)
            st.session_state.raw = data
            st.session_state.import_errors = errors
        if not data.empty:
            st.success(f"{len(data):,} unique activity rows loaded.")
        for error in errors[:8]:
            st.warning(error)

    raw = st.session_state.raw
    if not raw.empty and raw["DateTime"].notna().any():
        min_date = raw["DateTime"].min().date()
        max_date = raw["DateTime"].max().date()
    else:
        min_date = date.today() - timedelta(days=7)
        max_date = date.today()

    st.divider()
    start_date = st.date_input("From", value=min_date)
    end_date = st.date_input("To", value=max_date, min_value=start_date)
    default_workdays = working_days_between(start_date, end_date)

    with st.expander("Advanced review settings"):
        override_enabled = st.checkbox("Manually set working days", value=False)
        custom_workdays = st.number_input("Working days", min_value=1, max_value=365, value=max(1, default_workdays), step=1)

    workdays = int(custom_workdays) if override_enabled else default_workdays
    st.caption(f"KPI denominator: {workdays} working day(s)")

    page = st.radio("Navigate", [
        "Command Center", "School 360", "Teacher 360", "Follow-Ups", "KPI & Roster", "Ask AcadIntel"
    ])


period_raw = filter_period(st.session_state.raw, start_date, end_date)
teachers, schools, shared_usage = build_analytics(period_raw, start_date, end_date, workdays)

if st.session_state.raw.empty and roster_dataframe().empty:
    st.markdown('<div class="hero"><h1>AcadIntel 360</h1><p>Upload UserMetrics files to begin evidence-backed school and teacher intelligence.</p></div>', unsafe_allow_html=True)
    st.info("Start by uploading your raw UserMetrics files in the left sidebar.")
    st.stop()


# =========================================================
# COMMAND CENTER
# =========================================================
if page == "Command Center":
    st.markdown('<div class="hero"><h1>Academic Command Center</h1><p>Portfolio health, priorities and follow-up intelligence at a glance.</p></div>', unsafe_allow_html=True)

    if teachers.empty:
        st.warning("No teacher analytics are available for the selected review period.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Schools", len(schools))
        c2.metric("Teachers", len(teachers))
        c3.metric("Never Logged In", int((teachers["Status"] == "Never Logged In").sum()))
        c4.metric("Meeting All KPIs", int((teachers["Status"] == "Meeting All KPIs").sum()))
        c5.metric("Average Health", f"{teachers['Health Score'].mean():.0f}/100")

        st.subheader("📊 School Health")
        fig = px.bar(
            schools.sort_values("Health Score"), x="Health Score", y="School", orientation="h",
            color="Health Score", color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
            range_color=[0, 100], text="Health Score"
        )
        fig.update_layout(height=max(420, len(schools) * 36), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("🚨 Priority Teachers")
            priority = teachers.sort_values(["Health Score", "Total Minutes"]).head(15)
            st.dataframe(priority[["School", "Teacher", "Status", "Health Score", "Lesson KPI %", "Library KPI %", "Other KPI %"]], use_container_width=True, hide_index=True)
        with right:
            st.subheader("🧩 Module Mix")
            if not period_raw.empty:
                mix = period_raw.groupby("KPI Module")["Minutes"].sum().reset_index()
                pie = px.pie(mix, names="KPI Module", values="Minutes", hole=.55)
                st.plotly_chart(pie, use_container_width=True)

        st.subheader("📅 Follow-Up Pulse")
        fu_rows = db_select("followups")
        if fu_rows:
            fu = pd.DataFrame(fu_rows)
            fu["followup_date"] = pd.to_datetime(fu["followup_date"]).dt.date
            today = date.today()
            due = fu[(fu["followup_date"] == today) & (fu["status"] != "Resolved")]
            overdue = fu[(fu["followup_date"] < today) & (fu["status"] != "Resolved")]
            upcoming = fu[(fu["followup_date"] > today) & (fu["status"] != "Resolved")]
            a, b, c = st.columns(3)
            a.metric("Due Today", len(due)); b.metric("Overdue", len(overdue)); c.metric("Upcoming", len(upcoming))
        else:
            st.info("No follow-ups saved yet.")


# =========================================================
# SCHOOL 360
# =========================================================
elif page == "School 360":
    if schools.empty:
        st.warning("No school data available.")
        st.stop()

    school = st.selectbox("Select School", schools["School"].tolist())
    school_row = schools[schools["School"] == school].iloc[0]
    school_teachers = teachers[teachers["School"] == school].copy()
    school_raw = period_raw[period_raw["School"] == school].copy()
    facts = school_verified_facts(school_row, school_teachers, school_raw, start_date, end_date, workdays)

    st.title(f"🏫 {school}")
    st.caption(f"Review period: {start_date} to {end_date} • {workdays} working day(s)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Health", f"{school_row['Health Score']}/100")
    c2.metric("Compliance", f"{school_row['Overall Compliance %']}%")
    c3.metric("Teachers", school_row["Teachers"])
    c4.metric("Active", school_row["Active"])
    c5.metric("Inactive", school_row["Inactive / Never Logged In"])

    contact_rows = db_select("schools", {"school_name": school})
    contact = contact_rows[0] if contact_rows else {}

    with st.expander("📞 School Contact & Communication", expanded=True):
        col1, col2 = st.columns(2)
        contact_name = col1.text_input("KDM Name", value=contact.get("contact_name") or "")
        contact_role = col2.text_input("KDM Role", value=contact.get("contact_role") or "Principal")
        phone = col1.text_input("Mobile number with country code", value=contact.get("contact_phone") or "")
        group_url = col2.text_input("WhatsApp Group link", value=contact.get("whatsapp_group_url") or "")
        if st.button("Save Contact"):
            payload = {
                "school_name": school, "contact_name": contact_name, "contact_role": contact_role,
                "contact_phone": phone, "whatsapp_group_url": group_url, "updated_at": datetime.utcnow().isoformat()
            }
            if contact:
                db_update("schools", payload, contact["id"])
            else:
                db_insert("schools", payload)
            st.success("Contact saved.")
            st.rerun()

        clean_phone = re.sub(r"\D", "", phone)
        b1, b2, b3, b4 = st.columns(4)
        if clean_phone:
            b1.link_button("📞 Call KDM", f"tel:+{clean_phone}", use_container_width=True)
            b2.link_button("💬 WhatsApp Personal", f"https://wa.me/{clean_phone}", use_container_width=True)
        else:
            b1.button("📞 Call KDM", disabled=True, use_container_width=True)
            b2.button("💬 WhatsApp Personal", disabled=True, use_container_width=True)
        if group_url:
            b3.link_button("👥 WhatsApp Group", group_url, use_container_width=True)
        else:
            b3.button("👥 WhatsApp Group", disabled=True, use_container_width=True)
        if b4.button("📅 Follow Up", use_container_width=True):
            st.session_state.follow_school = school

    if st.session_state.follow_school == school:
        followup_dialog(school)

    st.subheader("📈 KPI Overview")
    module_df = pd.DataFrame({
        "Module": ["Lesson Delivery", "Library", "Other Modules"],
        "Actual": [school_row["Lesson Delivery Minutes"], school_row["Library Minutes"], school_row["Other Modules Minutes"]],
        "Expected if one teacher": [school_row["Lesson Target / Day"] * workdays, school_row["Library Target / Day"] * workdays, school_row["Other Target / Day"] * workdays],
    })
    fig = px.bar(module_df.melt(id_vars="Module", var_name="Measure", value_name="Minutes"), x="Module", y="Minutes", color="Measure", barmode="group", text_auto=".1f")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Teacher-level KPI percentages below are the authoritative compliance measure. The school chart shows total module volume and a one-teacher benchmark only, so aggregate volume does not distort individual compliance.")

    st.subheader("👩‍🏫 Teacher Performance")
    st.dataframe(
        school_teachers[["Teacher", "Status", "Health Score", "Lesson Delivery", "Lesson KPI %", "Library", "Library KPI %", "Other Modules", "Other KPI %", "Active Days", "Books Used"]].sort_values(["Health Score", "Total Minutes"]),
        use_container_width=True, hide_index=True
    )

    st.subheader("🧠 School Report")
    action_days = st.radio("Action plan interval", [7, 15], horizontal=True, key="school_action_days")
    auto_ai = st.checkbox("Auto-generate AI report for this school", value=True, key=f"auto_school_{school}")
    report_key = f"school_report::{school}::{start_date}::{end_date}::{workdays}::{action_days}"

    if report_key not in st.session_state:
        st.session_state[report_key] = deterministic_school_summary(school_row, action_days)

    should_generate = auto_ai and not st.session_state.get(report_key + "::ai_done", False)
    generate_clicked = st.button("✨ Generate / Refresh AI Report", use_container_width=True)

    if should_generate or generate_clicked:
        try:
            with st.spinner("Gemini is preparing the evidence-backed report..."):
                text, used_model = ai_generate(school_report_prompt(facts, action_days), force=generate_clicked)
            st.session_state[report_key] = text
            st.session_state[report_key + "::ai_done"] = True
            st.session_state[report_key + "::model"] = used_model
            db_insert("report_history", {
                "report_level": "School", "school_name": school, "action_plan_days": action_days,
                "report_text": text, "verified_facts": facts
            })
        except Exception as exc:
            st.error(f"Gemini report generation failed: {exc}")
            st.info("The deterministic report remains available below, so reporting is not blocked.")

    report_text = st.session_state[report_key]
    render_report(report_text)
    if st.session_state.get(report_key + "::model"):
        st.caption(f"AI narrative generated with {st.session_state[report_key + '::model']}. KPI values are calculated by Python, not Gemini.")

    pdf_bytes = make_text_pdf(
        f"School 360 Intelligence Report - {school}",
        f"Review Period: {start_date} to {end_date} | Working Days: {workdays}",
        report_text,
    )

    msg_key = report_key + "::whatsapp"
    if msg_key not in st.session_state:
        st.session_state[msg_key] = (
            f"Dear Sir/Ma'am,\n\nPlease find the AcadIntel 360 performance report for {school} for {start_date} to {end_date}. "
            f"Current Health Score: {school_row['Health Score']}/100 | Full KPI Compliance: {school_row['Overall Compliance %']}%. "
            f"The report includes teacher-level evidence, implementation gaps and the {action_days}-day action plan.\n\nRegards,\nDilip Kumar Vishwakarma"
        )

    g1, g2, g3 = st.columns(3)
    g1.download_button("⬇ Download PDF", data=pdf_bytes, file_name=re.sub(r"[^A-Za-z0-9]+", "_", school) + "_School_360.pdf", mime="application/pdf", use_container_width=True)
    share_url = "https://wa.me/?text=" + urllib.parse.quote(st.session_state[msg_key])
    g2.link_button("📤 Share Message on WhatsApp", share_url, use_container_width=True)
    if group_url:
        g3.link_button("👥 Open School WhatsApp Group", group_url, use_container_width=True)
    else:
        g3.button("👥 Open School WhatsApp Group", disabled=True, use_container_width=True)

    st.text_area("Customized share message", st.session_state[msg_key], height=160, key=f"msg_edit_{school}")
    if st.button("✨ AI-Customize Share Message", use_container_width=True):
        try:
            text, _ = ai_generate(whatsapp_prompt(facts, contact_role), force=True)
            st.session_state[msg_key] = clean_ai_text(text)
            st.rerun()
        except Exception as exc:
            st.error(f"Could not customize message: {exc}")

    st.caption("WhatsApp Web cannot automatically attach a PDF to a group from a normal browser link. Download the PDF with one click, then use the pre-written share message / group button.")

    st.subheader("☎️ KDM Calling Script")
    script_key = report_key + "::script"
    if st.button("Generate KDM Call Script", use_container_width=True):
        try:
            previous = db_select("followups", {"school_name": school}, order="followup_date")
            script, _ = ai_generate(call_script_prompt(facts, previous), force=True)
            st.session_state[script_key] = script
            db_insert("call_scripts", {"school_name": school, "target_role": contact_role, "script_text": script, "verified_facts": facts})
        except Exception as exc:
            st.error(f"Could not generate call script: {exc}")
    if st.session_state.get(script_key):
        render_report(st.session_state[script_key])


# =========================================================
# TEACHER 360
# =========================================================
elif page == "Teacher 360":
    if teachers.empty:
        st.warning("No teacher data available.")
        st.stop()

    school_filter = st.selectbox("School", sorted(teachers["School"].unique()), key="teacher_school")
    teacher_options = teachers[teachers["School"] == school_filter]["Teacher"].sort_values().tolist()
    teacher_name = st.selectbox("Teacher", teacher_options)
    teacher_row = teachers[(teachers["School"] == school_filter) & (teachers["Teacher"] == teacher_name)].iloc[0]
    evidence = period_raw[(period_raw["School"] == school_filter) & (period_raw["Teacher Key"] == teacher_row["Teacher Key"])].copy()
    facts = teacher_verified_facts(teacher_row, evidence, start_date, end_date)

    st.title(f"👩‍🏫 {teacher_name}")
    st.caption(f"{school_filter} • {start_date} to {end_date}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Health", f"{teacher_row['Health Score']}/100")
    c2.metric("Lesson KPI", f"{teacher_row['Lesson KPI %']}%")
    c3.metric("Library KPI", f"{teacher_row['Library KPI %']}%")
    c4.metric("Other KPI", f"{teacher_row['Other KPI %']}%")

    module_data = pd.DataFrame({
        "Module": ["Lesson Delivery", "Library", "Other Modules"],
        "Actual": [teacher_row["Lesson Delivery"], teacher_row["Library"], teacher_row["Other Modules"]],
        "Target": [teacher_row["Lesson Target"], teacher_row["Library Target"], teacher_row["Other Target"]],
    })
    fig = px.bar(module_data.melt(id_vars="Module", var_name="Measure", value_name="Minutes"), x="Module", y="Minutes", color="Measure", barmode="group", text_auto=".1f")
    st.plotly_chart(fig, use_container_width=True)

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Active Days", f"{teacher_row['Active Days']}/{teacher_row['Eligible Working Days']}")
    e2.metric("Books Used", teacher_row["Books Used"])
    e3.metric("Grades Covered", teacher_row["Grades Covered"])
    e4.metric("Subjects Covered", teacher_row["Subjects Covered"])

    st.subheader("📚 Granular Evidence")
    if evidence.empty:
        st.info("No activity was recorded for this teacher in the selected period.")
    else:
        st.dataframe(evidence[["DateTime", "Raw Module", "KPI Module", "Grade", "Subject", "Book", "Minutes"]].sort_values("DateTime", ascending=False), use_container_width=True, hide_index=True)
        daily = evidence.dropna(subset=["DateTime"]).assign(ActivityDate=lambda x: x["DateTime"].dt.date).groupby("ActivityDate")["Minutes"].sum().reset_index()
        if not daily.empty:
            st.plotly_chart(px.line(daily, x="ActivityDate", y="Minutes", markers=True, title="Daily Activity Trend"), use_container_width=True)

    action_days = st.radio("Development plan interval", [7, 15], horizontal=True, key="teacher_action_days")
    key = f"teacher_report::{school_filter}::{teacher_name}::{start_date}::{end_date}::{workdays}::{action_days}"
    if key not in st.session_state:
        st.session_state[key] = (
            f"EXECUTIVE DIAGNOSIS\n{teacher_name} has a Health Score of {teacher_row['Health Score']}/100 and status '{teacher_row['Status']}'.\n\n"
            f"KPI SCORECARD\nLesson Delivery {teacher_row['Lesson KPI %']}%, Library {teacher_row['Library KPI %']}%, Other Modules {teacher_row['Other KPI %']}%.\n\n"
            f"{action_days}-DAY DEVELOPMENT ACTION PLAN\nPrioritize the lowest KPI areas and review verified activity evidence at the next checkpoint."
        )

    auto_teacher = st.checkbox("Auto-generate detailed AI report", value=True, key=f"auto_teacher_{teacher_row['Teacher Key']}")
    should_generate = auto_teacher and not st.session_state.get(key + "::ai_done", False)
    clicked = st.button("✨ Generate / Refresh Teacher 360 Report", use_container_width=True)
    if should_generate or clicked:
        try:
            with st.spinner("Gemini is preparing the Teacher 360 report..."):
                text, used_model = ai_generate(teacher_report_prompt(facts, action_days), force=clicked)
            st.session_state[key] = text
            st.session_state[key + "::ai_done"] = True
            st.session_state[key + "::model"] = used_model
            db_insert("report_history", {
                "report_level": "Teacher", "school_name": school_filter, "teacher_name": teacher_name,
                "action_plan_days": action_days, "report_text": text, "verified_facts": facts
            })
        except Exception as exc:
            st.error(f"Gemini report generation failed: {exc}")

    render_report(st.session_state[key])
    pdf_bytes = make_text_pdf(
        f"Teacher 360 Intelligence Report - {teacher_name}",
        f"School: {school_filter} | Review Period: {start_date} to {end_date}",
        st.session_state[key], evidence
    )
    st.download_button("⬇ Download Teacher 360 PDF", data=pdf_bytes, file_name=re.sub(r"[^A-Za-z0-9]+", "_", teacher_name) + "_360_Audit_Report.pdf", mime="application/pdf", use_container_width=True)


# =========================================================
# FOLLOW-UPS
# =========================================================
elif page == "Follow-Ups":
    st.title("📅 Implementation Follow-Up Center")
    rows = db_select("followups", order="followup_date")
    if not rows:
        st.info("No follow-ups saved yet. Add them from School 360.")
    else:
        fu = pd.DataFrame(rows)
        fu["followup_date"] = pd.to_datetime(fu["followup_date"]).dt.date
        today = date.today()
        tabs = st.tabs(["🔴 Overdue", "🟠 Due Today", "🔵 Upcoming", "✅ Resolved"])
        subsets = [
            fu[(fu["followup_date"] < today) & (fu["status"] != "Resolved")],
            fu[(fu["followup_date"] == today) & (fu["status"] != "Resolved")],
            fu[(fu["followup_date"] > today) & (fu["status"] != "Resolved")],
            fu[fu["status"] == "Resolved"],
        ]
        for tab, subset in zip(tabs, subsets):
            with tab:
                if subset.empty:
                    st.caption("Nothing here.")
                for _, row in subset.sort_values("followup_date").iterrows():
                    with st.container(border=True):
                        a, b, c = st.columns([2.5, 1, 1])
                        a.subheader(row["school_name"])
                        a.write(row["issue"])
                        a.caption("Last commitment: " + (row.get("last_commitment") or "—"))
                        b.write(str(row["followup_date"])); b.write(row["status"])
                        if row["status"] != "Resolved" and c.button("Mark Resolved", key=f"resolve_{row['id']}"):
                            db_update("followups", {"status": "Resolved", "updated_at": datetime.utcnow().isoformat()}, row["id"])
                            st.rerun()
                        if c.button("Delete", key=f"delete_{row['id']}"):
                            db_delete("followups", row["id"])
                            st.rerun()


# =========================================================
# KPI & ROSTER
# =========================================================
elif page == "KPI & Roster":
    st.title("⚙️ KPI, Master Roster & Shared Accounts")
    t1, t2, t3, t4 = st.tabs(["🎯 KPI Settings", "👩‍🏫 Master Roster", "🏫 Shared Accounts", "🤖 Gemini Diagnostics"])

    with t1:
        current = effective_kpis()
        st.subheader("Global Default KPIs")
        with st.form("global_kpis"):
            l = st.number_input("Lesson Delivery minutes/day", min_value=0.0, value=float(current["lessonDelivery"]), step=1.0)
            lib = st.number_input("Library minutes/day", min_value=0.0, value=float(current["library"]), step=1.0)
            oth = st.number_input("Other Modules combined minutes/day", min_value=0.0, value=float(current["otherModules"]), step=1.0)
            if st.form_submit_button("Save Global KPI", use_container_width=True):
                save_kpi("GLOBAL", "lessonDelivery", l)
                save_kpi("GLOBAL", "library", lib)
                save_kpi("GLOBAL", "otherModules", oth)
                st.success("Global KPIs saved.")
                st.rerun()

        possible = set(schools["School"].tolist() if not schools.empty else [])
        possible.update(r["school_name"] for r in db_select("schools"))
        if possible:
            st.subheader("School-Specific KPI Override")
            s = st.selectbox("School", sorted(possible), key="kpi_school")
            eff = effective_kpis(s)
            with st.form("school_kpis"):
                sl = st.number_input("Lesson Delivery", min_value=0.0, value=float(eff["lessonDelivery"]), step=1.0)
                sLib = st.number_input("Library", min_value=0.0, value=float(eff["library"]), step=1.0)
                so = st.number_input("Other Modules", min_value=0.0, value=float(eff["otherModules"]), step=1.0)
                if st.form_submit_button("Save School Override", use_container_width=True):
                    save_kpi("SCHOOL", "lessonDelivery", sl, s)
                    save_kpi("SCHOOL", "library", sLib, s)
                    save_kpi("SCHOOL", "otherModules", so, s)
                    st.success("School-specific KPIs saved.")
                    st.rerun()

    with t2:
        roster_files = st.file_uploader("Upload Master Roster CSV/XLSX", type=["csv", "xlsx"], accept_multiple_files=True, key="roster")
        if roster_files and st.button("Import Master Roster", use_container_width=True):
            try:
                inserted, updated = import_roster(roster_files)
                st.success(f"Roster imported: {inserted} new, {updated} updated.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        roster = roster_dataframe()
        if not roster.empty:
            st.metric("Active roster teachers", len(roster))
            st.dataframe(roster[["school_name", "teacher_name", "grade", "subject", "email", "phone"]], use_container_width=True, hide_index=True)

    with t3:
        st.write("Shared school-level accounts are excluded from individual teacher rankings and inactivity calculations.")
        ss = st.text_input("School name", key="shared_school")
        sa = st.text_input("Shared account name", key="shared_account")
        if st.button("Add Shared Account", use_container_width=True):
            if ss.strip() and sa.strip():
                key = teacher_key(sa)
                existing = [r for r in db_select("shared_accounts") if r["school_name"] == ss.strip() and r["account_key"] == key]
                if not existing:
                    db_insert("shared_accounts", {"school_name": ss.strip(), "account_name": sa.strip(), "account_key": key, "active": True})
                st.success("Shared account saved.")
                st.rerun()
            else:
                st.error("Enter both school and account name.")
        for row in db_select("shared_accounts"):
            a, b = st.columns([4, 1])
            a.write(f"{row['school_name']} — {row['account_name']}")
            if b.button("Remove", key=f"rm_shared_{row['id']}"):
                db_delete("shared_accounts", row["id"])
                st.rerun()

    with t4:
        st.write("Gemini integration uses the official google-genai SDK and automatically discovers a model that supports generateContent.")
        model, err = discover_gemini_model()
        if model:
            st.success(f"Gemini model ready: {model}")
            if st.button("Run Gemini Test", use_container_width=True):
                try:
                    answer, used = ai_generate("Reply with exactly: AcadIntel Gemini connection successful.", force=True)
                    st.success(f"{used}: {answer}")
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.error(err or "Gemini unavailable")


# =========================================================
# ASK ACADINTEL
# =========================================================
elif page == "Ask AcadIntel":
    st.title("✨ Ask AcadIntel")
    st.caption("Gemini is allowed to interpret only verified calculated facts from the current review period.")
    question = st.text_area("Ask a portfolio question", placeholder="Which schools need my attention most and why?")
    if st.button("Ask AcadIntel", use_container_width=True):
        facts = {
            "period": {"start": str(start_date), "end": str(end_date), "working_days": workdays},
            "schools": json_safe(schools.to_dict("records") if not schools.empty else []),
            "priority_teachers": json_safe(teachers.sort_values(["Health Score", "Total Minutes"]).head(30).to_dict("records") if not teachers.empty else []),
        }
        prompt = f"""
You are AcadIntel 360. Answer ONLY from VERIFIED FACTS. Never invent metrics, causes, commitments, names or dates.
QUESTION: {question}
Use headings: FACTS, INTERPRETATION, RECOMMENDED ACTIONS, EVIDENCE. No markdown asterisks.
VERIFIED FACTS: {json.dumps(facts, default=str)}
"""
        try:
            with st.spinner("Analysing verified facts..."):
                answer, model = ai_generate(prompt, force=True)
            render_report(answer)
            st.caption(f"Generated with {model}. KPI calculations were produced by Python before AI interpretation.")
        except Exception as exc:
            st.error(f"Gemini error: {exc}")
