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
    section[data-testid="stSidebar"] {min-width:300px !important; max-width:300px !important;}
    .stSelectbox > div > div {border-radius:14px;}
    div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:18px;}
    .ai-card {border-radius:18px; padding:16px 18px; margin:9px 0; border-left:5px solid #4f46e5; background:#f8fafc;}
    div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button, div[data-testid="stLinkButton"] a {border-radius:12px; font-weight:700;}
    
    html, body, [class*="css"] {font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
    .insight-box,.success-box,.warning-box{padding:18px 20px;border-radius:18px;min-height:118px;line-height:1.55;box-shadow:0 8px 24px rgba(15,23,42,.05);}
    .insight-box{background:linear-gradient(135deg,#eef2ff,#f8fafc);border:1px solid #c7d2fe;}
    .success-box{background:linear-gradient(135deg,#ecfdf5,#f8fafc);border:1px solid #a7f3d0;}
    .warning-box{background:linear-gradient(135deg,#fff7ed,#fffbeb);border:1px solid #fed7aa;}
    div[data-testid="stTabs"] button{font-weight:750;}
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
    "share_follow_school": None,
    "analytics_cache": {},
    "db_cache": {},
    "raw_version": 0,
    "db_version": 0,
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


def _invalidate_db_cache(table=None):
    cache = st.session_state.setdefault("db_cache", {})
    if table is None:
        cache.clear()
    else:
        cache.pop(table, None)
    st.session_state.db_version = int(st.session_state.get("db_version", 0)) + 1
    st.session_state.analytics_cache = {}


def _db_all(table, refresh=False):
    """Load each Supabase table once per Streamlit session.
    This avoids repeated network round-trips on every widget rerun.
    """
    if sb is None:
        return []
    cache = st.session_state.setdefault("db_cache", {})
    if refresh or table not in cache:
        try:
            cache[table] = sb.table(table).select("*").execute().data or []
        except Exception as exc:
            st.warning(f"Database read issue ({table}): {exc}")
            cache[table] = []
    return list(cache.get(table, []))


def db_select(table, filters=None, order=None, refresh=False):
    rows = _db_all(table, refresh=refresh)
    for key, value in (filters or {}).items():
        rows = [r for r in rows if r.get(key) == value]
    if order:
        rows = sorted(rows, key=lambda r: (r.get(order) is None, r.get(order)))
    return rows


def db_insert(table, payload):
    if sb is None:
        raise RuntimeError("Supabase is not connected.")
    data = sb.table(table).insert(json_safe(payload)).execute().data
    _invalidate_db_cache(table)
    return data


def db_update(table, payload, row_id):
    if sb is None:
        raise RuntimeError("Supabase is not connected.")
    data = sb.table(table).update(json_safe(payload)).eq("id", row_id).execute().data
    _invalidate_db_cache(table)
    return data


def db_delete(table, row_id):
    if sb is None:
        raise RuntimeError("Supabase is not connected.")
    data = sb.table(table).delete().eq("id", row_id).execute().data
    _invalidate_db_cache(table)
    return data


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
    preferred += ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash"]

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
    """Fast path: call the configured/default Flash model directly.
    Model discovery is only attempted if that call fails.
    """
    client = get_ai_client()
    if client is None:
        raise RuntimeError("Gemini is not connected. Check GEMINI_API_KEY in Streamlit Secrets.")

    configured = str(st.secrets.get("GEMINI_MODEL", "")).strip()
    model = configured or "gemini-2.5-flash"
    cache_key = hashlib.sha256((model + "\n" + prompt).encode("utf-8")).hexdigest()
    if not force and cache_key in st.session_state.ai_cache:
        return st.session_state.ai_cache[cache_key], model

    try:
        response = client.models.generate_content(model=model, contents=prompt)
        text = clean_ai_text(response.text or "")
        st.session_state.ai_cache[cache_key] = text
        return text, model
    except Exception as first_error:
        fallback_model, discovery_error = discover_gemini_model()
        if not fallback_model:
            raise RuntimeError(discovery_error or str(first_error))
        fallback_key = hashlib.sha256((fallback_model + "\n" + prompt).encode("utf-8")).hexdigest()
        if not force and fallback_key in st.session_state.ai_cache:
            return st.session_state.ai_cache[fallback_key], fallback_model
        response = client.models.generate_content(model=fallback_model, contents=prompt)
        text = clean_ai_text(response.text or "")
        st.session_state.ai_cache[fallback_key] = text
        return text, fallback_model


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


def kpi_scope_details(school=None):
    """Return global values, active local overrides, effective values and source labels."""
    rows = load_kpi_rows()
    global_values = {k: v[1] for k, v in DEFAULT_KPI.items()}

    for module_key in global_values:
        matches = [
            r for r in rows
            if r.get("scope") == "GLOBAL"
            and r.get("module_key") == module_key
            and r.get("active", True)
        ]
        if matches:
            matches.sort(key=lambda x: x.get("updated_at") or x.get("created_at") or "")
            global_values[module_key] = safe_float(
                matches[-1].get("target_minutes_per_day"),
                global_values[module_key],
            )

    local_values = {}
    if school:
        for module_key in global_values:
            matches = [
                r for r in rows
                if r.get("scope") == "SCHOOL"
                and r.get("school_name") == school
                and r.get("module_key") == module_key
                and r.get("active", True)
            ]
            if matches:
                matches.sort(key=lambda x: x.get("updated_at") or x.get("created_at") or "")
                local_values[module_key] = safe_float(
                    matches[-1].get("target_minutes_per_day"),
                    global_values[module_key],
                )

    effective = dict(global_values)
    effective.update(local_values)

    source = {
        module_key: (
            f"Local override — {school}"
            if module_key in local_values
            else "Global default"
        )
        for module_key in global_values
    }

    return {
        "global": global_values,
        "local": local_values,
        "effective": effective,
        "source": source,
        "has_local_override": bool(local_values),
    }


def reset_school_kpis_to_global(school_name):
    """Deactivate all active school-specific KPI rows so the school inherits global defaults."""
    rows = load_kpi_rows()
    changed = 0
    for row in rows:
        if (
            row.get("scope") == "SCHOOL"
            and row.get("school_name") == school_name
            and row.get("active", True)
        ):
            db_update(
                "kpi_settings",
                {
                    "active": False,
                    "updated_at": datetime.utcnow().isoformat(),
                },
                row["id"],
            )
            changed += 1
    return changed


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
# PREMIUM REPORT UI + GRAPHICAL PDF ENGINE
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
    palette = ["#EEF2FF", "#ECFEFF", "#F0FDF4", "#FFF7ED", "#FDF4FF", "#F8FAFC"]
    borders = ["#4F46E5", "#0891B2", "#16A34A", "#EA580C", "#A21CAF", "#475569"]
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        s = match.end()
        e = matches[i + 1].start() if i + 1 < len(matches) else len(clean)
        body = clean[s:e].strip()
        if body:
            bg = palette[i % len(palette)]
            border = borders[i % len(borders)]
            html = (
                f'<div style="background:{bg};border-left:5px solid {border};padding:18px 20px;'
                f'border-radius:16px;margin:10px 0 14px 0;box-shadow:0 6px 20px rgba(15,23,42,.04)">'
                f'<div style="font-weight:800;font-size:15px;color:{border};margin-bottom:8px">{title.title()}</div>'
                f'<div style="line-height:1.62;color:#243045">{body.replace(chr(10), "<br>")}</div></div>'
            )
            st.markdown(html, unsafe_allow_html=True)


def teacher_auto_insights(row, action_days=7):
    kpis = {
        "Lesson Delivery": safe_float(row.get("Lesson KPI %")),
        "Library": safe_float(row.get("Library KPI %")),
        "Other Modules": safe_float(row.get("Other KPI %")),
    }
    strongest = max(kpis, key=kpis.get)
    weakest = min(kpis, key=kpis.get)
    total = safe_float(row.get("Total Minutes"))
    active_days = int(safe_float(row.get("Active Days")))
    eligible = max(1, int(safe_float(row.get("Eligible Working Days"))))
    consistency = round(active_days / eligible * 100, 1)
    if total <= 0:
        diagnosis = "No measurable platform activity was recorded in the selected review period."
        strength = "The roster provides a clear baseline for a structured activation plan."
        focus = "First login, guided navigation and the first measurable usage checkpoint."
    else:
        diagnosis = f"Recorded {total:.1f} minutes across {active_days} active day(s), giving {consistency:.1f}% activity-day consistency."
        strength = f"Relative strength: {strongest} at {kpis[strongest]:.1f}% of the configured KPI."
        focus = f"Primary focus: {weakest}, currently at {kpis[weakest]:.1f}% of the configured KPI."
    plan = (
        f"For the next {action_days} days: establish a daily usage rhythm, prioritize {weakest}, "
        "use the relevant classroom/content workflow, and verify improvement at the next review using the same KPI denominator."
    )
    return diagnosis, strength, focus, plan


def render_graphical_school_report(school_row, school_teachers, school_raw, workdays, action_days):
    teacher_count = max(1, int(safe_float(school_row.get("Teachers"))))
    kpi_df = pd.DataFrame({
        "KPI": ["Lesson Delivery", "Library", "Other Modules"],
        "Actual": [
            safe_float(school_row.get("Lesson Delivery Minutes")),
            safe_float(school_row.get("Library Minutes")),
            safe_float(school_row.get("Other Modules Minutes")),
        ],
        "Target": [
            safe_float(school_row.get("Lesson Target / Day")) * workdays * teacher_count,
            safe_float(school_row.get("Library Target / Day")) * workdays * teacher_count,
            safe_float(school_row.get("Other Target / Day")) * workdays * teacher_count,
        ],
    })
    kpi_df["Achievement %"] = (kpi_df["Actual"] / kpi_df["Target"].replace(0, pd.NA) * 100).fillna(0).round(1)

    st.markdown("### Performance cockpit")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Institution Health", f"{safe_float(school_row.get('Health Score')):.0f}/100")
    k2.metric("Full KPI Compliance", f"{safe_float(school_row.get('Overall Compliance %')):.1f}%")
    k3.metric("Active Teachers", f"{int(safe_float(school_row.get('Active')))}/{teacher_count}")
    k4.metric("Met All KPIs", int(safe_float(school_row.get("Met All KPIs"))))

    left, right = st.columns([1.15, 1])
    with left:
        fig = px.bar(
            kpi_df.melt(id_vars=["KPI", "Achievement %"], value_vars=["Actual", "Target"], var_name="Measure", value_name="Minutes"),
            x="KPI", y="Minutes", color="Measure", barmode="group", text_auto=".0f",
            color_discrete_map={"Actual": "#4F46E5", "Target": "#CBD5E1"},
            title="KPI delivery: actual vs review-period target",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if not school_teachers.empty:
            rank = school_teachers.sort_values("Health Score", ascending=True).tail(12)
            fig2 = px.bar(
                rank, x="Health Score", y="Teacher", orientation="h", color="Health Score",
                color_continuous_scale=["#FCA5A5", "#FBBF24", "#34D399"], range_color=[0, 100],
                text="Health Score", title="Teacher health ranking",
            )
            fig2.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        if not school_raw.empty:
            module = school_raw.groupby("Raw Module", as_index=False)["Minutes"].sum().sort_values("Minutes", ascending=False).head(10)
            fig3 = px.bar(module.sort_values("Minutes"), x="Minutes", y="Raw Module", orientation="h", text_auto=".1f", title="Module adoption mix")
            fig3.update_layout(margin=dict(l=10, r=10, t=55, b=10), showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
    with right:
        status = school_teachers["Status"].value_counts().rename_axis("Status").reset_index(name="Teachers") if not school_teachers.empty else pd.DataFrame()
        if not status.empty:
            fig4 = px.pie(status, names="Status", values="Teachers", hole=.58, title="Teacher implementation status")
            fig4.update_layout(margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Teacher scorecard")
    if not school_teachers.empty:
        view = school_teachers[[
            "Teacher", "Status", "Health Score", "Lesson KPI %", "Library KPI %", "Other KPI %",
            "Total Minutes", "Active Days", "Books Used", "Grades Covered", "Subjects Covered"
        ]].sort_values(["Health Score", "Total Minutes"], ascending=[True, True])
        st.dataframe(view, use_container_width=True, hide_index=True, height=min(520, 90 + len(view) * 36))

    st.markdown("### Action organizer")
    low = school_teachers.sort_values(["Health Score", "Total Minutes"]).head(5) if not school_teachers.empty else pd.DataFrame()
    high = school_teachers.sort_values(["Health Score", "Total Minutes"], ascending=False).head(3) if not school_teachers.empty else pd.DataFrame()
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="insight-box"><b>Protect</b><br>' + (", ".join(high["Teacher"].astype(str).tolist()) if not high.empty else "No benchmark teachers yet") + '<br><span>Recognise relatively stronger implementation and capture repeatable practices.</span></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="warning-box"><b>Prioritise</b><br>' + (", ".join(low["Teacher"].astype(str).tolist()) if not low.empty else "No priority teachers") + '<br><span>Review the lowest KPI, activity-day consistency and evidence before the next follow-up.</span></div>', unsafe_allow_html=True)
    with c:
        st.markdown(f'<div class="success-box"><b>{action_days}-day checkpoint</b><br>Re-run the same review period logic after the intervention.<br><span>Compare KPI %, active days and module breadth teacher by teacher.</span></div>', unsafe_allow_html=True)


class ReportPDF(FPDF):
    """Executive A4 report canvas with predictable margins and restrained branding."""

    NAVY = (15, 23, 42)
    INDIGO = (79, 70, 229)
    CYAN = (14, 165, 233)
    GREEN = (16, 185, 129)
    AMBER = (245, 158, 11)
    RED = (239, 68, 68)
    SLATE = (100, 116, 139)
    LIGHT = (248, 250, 252)
    BORDER = (226, 232, 240)

    def header(self):
        # Clean executive band; no full-page rainbow effect.
        self.set_fill_color(*self.NAVY)
        self.rect(0, 0, 210, 20, "F")
        self.set_fill_color(*self.INDIGO)
        self.rect(0, 20, 210, 2.2, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 15)
        self.set_xy(12, 5.5)
        self.cell(0, 6, "AcadIntel 360", 0, 1)
        self.set_font("Helvetica", "", 7.5)
        self.set_x(12)
        self.cell(0, 3.8, "Academic Intelligence | Evidence | Action", 0, 1)
        self.set_text_color(*self.NAVY)

    def footer(self):
        self.set_y(-10)
        self.set_draw_color(*self.BORDER)
        self.line(12, self.get_y(), 198, self.get_y())
        self.set_y(-8.5)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*self.SLATE)
        self.cell(0, 4, f"Confidential academic implementation report | Page {self.page_no()}", 0, 0, "C")


def pdf_safe(value):
    return str(value).encode("latin-1", "replace").decode("latin-1")


def _wrap_pdf_text(pdf, text, width, font="Helvetica", style="", size=7.4):
    """Return wrapped lines based on actual PDF font metrics."""
    text = pdf_safe(text or "")
    pdf.set_font(font, style, size)
    lines = []
    for paragraph in text.split("\n"):
        if paragraph == "":
            lines.append("")
            continue
        words = paragraph.split()
        current = ""
        for word in words:
            candidate = word if not current else current + " " + word
            if pdf.get_string_width(candidate) <= width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                # Handle an exceptionally long token safely.
                if pdf.get_string_width(word) <= width:
                    current = word
                else:
                    chunk = ""
                    for ch in word:
                        cand = chunk + ch
                        if pdf.get_string_width(cand) <= width:
                            chunk = cand
                        else:
                            if chunk:
                                lines.append(chunk)
                            chunk = ch
                    current = chunk
        if current:
            lines.append(current)
    return lines or [""]


def _ensure_space(pdf, required_h, top_y=28):
    if pdf.get_y() + required_h > 278:
        pdf.add_page()
        pdf.set_y(top_y)


def pdf_title_block(pdf, title, subtitle=None, eyebrow=None):
    if eyebrow:
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(*ReportPDF.INDIGO)
        pdf.cell(0, 4, pdf_safe(eyebrow.upper()), 0, 1)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*ReportPDF.NAVY)
    pdf.cell(0, 8, pdf_safe(title), 0, 1)
    if subtitle:
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*ReportPDF.SLATE)
        pdf.multi_cell(pdf.epw, 4.2, pdf_safe(subtitle))
        pdf.set_x(pdf.l_margin)
    pdf.ln(2)


def pdf_card(pdf, x, y, w, h, label, value, accent=None, subtext=None):
    accent = accent or ReportPDF.INDIGO
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(*ReportPDF.BORDER)
    pdf.rect(x, y, w, h, style="DF", round_corners=True, corner_radius=2.5)
    pdf.set_fill_color(*accent)
    pdf.rect(x, y, 2.2, h, style="F", round_corners=True, corner_radius=1)
    pdf.set_xy(x + 5, y + 4)
    pdf.set_font("Helvetica", "", 7.2)
    pdf.set_text_color(*ReportPDF.SLATE)
    pdf.cell(w - 8, 3.5, pdf_safe(label), 0, 1)
    pdf.set_xy(x + 5, y + 8.8)
    pdf.set_font("Helvetica", "B", 12.5)
    pdf.set_text_color(*ReportPDF.NAVY)
    pdf.cell(w - 8, 6, pdf_safe(str(value)[:24]), 0, 1)
    if subtext:
        pdf.set_xy(x + 5, y + h - 5.5)
        pdf.set_font("Helvetica", "", 6.5)
        pdf.set_text_color(*ReportPDF.SLATE)
        pdf.cell(w - 8, 3.5, pdf_safe(str(subtext)[:38]), 0, 0)


def pdf_section_title(pdf, title, subtitle=None):
    _ensure_space(pdf, 14 if subtitle else 10)
    x = pdf.l_margin
    y = pdf.get_y()
    pdf.set_fill_color(241, 245, 249)
    pdf.rect(x, y, pdf.epw, 7.5, style="F", round_corners=True, corner_radius=1.8)
    pdf.set_xy(x + 4, y + 1.8)
    pdf.set_font("Helvetica", "B", 9.2)
    pdf.set_text_color(*ReportPDF.NAVY)
    pdf.cell(pdf.epw - 8, 4, pdf_safe(title.upper()), 0, 1)
    pdf.set_y(y + 9)
    if subtitle:
        pdf.set_font("Helvetica", "", 7.1)
        pdf.set_text_color(*ReportPDF.SLATE)
        pdf.multi_cell(pdf.epw, 3.8, pdf_safe(subtitle))
        pdf.set_x(pdf.l_margin)
        pdf.ln(1)


def pdf_progress(pdf, label, pct, x=None, y=None, w=84, show_value=True, detail=None):
    x = pdf.get_x() if x is None else x
    y = pdf.get_y() if y is None else y
    pct = max(0.0, safe_float(pct))
    capped = min(pct, 100.0)
    pdf.set_xy(x, y)
    pdf.set_font("Helvetica", "B", 7.3)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(w - 22, 4, pdf_safe(label), 0, 0)
    if show_value:
        pdf.set_xy(x + w - 22, y)
        pdf.cell(22, 4, f"{pct:.1f}%", 0, 0, "R")
    bar_y = y + 5
    pdf.set_fill_color(226, 232, 240)
    pdf.rect(x, bar_y, w, 3.6, style="F", round_corners=True, corner_radius=1.8)
    if capped < 40:
        rgb = ReportPDF.RED
    elif capped < 75:
        rgb = ReportPDF.AMBER
    else:
        rgb = ReportPDF.GREEN
    pdf.set_fill_color(*rgb)
    if capped > 0:
        pdf.rect(x, bar_y, max(1, w * capped / 100), 3.6, style="F", round_corners=True, corner_radius=1.8)
    next_y = bar_y + 5.5
    if detail:
        pdf.set_xy(x, next_y)
        pdf.set_font("Helvetica", "", 6.6)
        pdf.set_text_color(*ReportPDF.SLATE)
        pdf.cell(w, 3.5, pdf_safe(detail), 0, 0)
        next_y += 3.5
    return next_y


def pdf_hbar_chart(pdf, title, rows, max_value=None, height_each=7, max_rows=10, value_suffix=""):
    rows = list(rows or [])[:max_rows]
    needed = 12 + max(1, len(rows)) * height_each
    _ensure_space(pdf, needed)
    pdf_section_title(pdf, title)
    if not rows:
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*ReportPDF.SLATE)
        pdf.cell(0, 6, "No measurable data available.", 0, 1)
        return
    max_value = max_value or max(safe_float(v) for _, v in rows) or 1
    max_value = max(max_value, 1)
    label_w = 54
    value_w = 20
    bar_w = pdf.epw - label_w - value_w - 5
    for label, value in rows:
        y = pdf.get_y()
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(51, 65, 85)
        pdf.set_xy(pdf.l_margin, y)
        pdf.cell(label_w, 4.5, pdf_safe(str(label)[:34]), 0, 0)
        x = pdf.l_margin + label_w
        pdf.set_fill_color(241, 245, 249)
        pdf.rect(x, y + 0.7, bar_w, 3.6, style="F", round_corners=True, corner_radius=1.4)
        fill = max(0, min(bar_w, bar_w * safe_float(value) / max_value))
        pdf.set_fill_color(*ReportPDF.INDIGO)
        if fill > 0:
            pdf.rect(x, y + 0.7, max(1, fill), 3.6, style="F", round_corners=True, corner_radius=1.4)
        pdf.set_xy(x + bar_w + 2, y)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(value_w - 2, 4.5, f"{safe_float(value):.1f}{value_suffix}", 0, 0, "R")
        pdf.set_y(y + height_each)


def pdf_note_box(pdf, title, body, kind="info", max_lines=None):
    """Measured note box: never clips or overlaps the next section."""
    palette = {
        "info": ((239, 246, 255), (37, 99, 235)),
        "success": ((240, 253, 244), (22, 163, 74)),
        "warning": ((255, 247, 237), (234, 88, 12)),
        "purple": ((245, 243, 255), (124, 58, 237)),
        "neutral": ((248, 250, 252), (71, 85, 105)),
    }
    bg, accent = palette.get(kind, palette["info"])
    inner_w = pdf.epw - 12
    lines = _wrap_pdf_text(pdf, body, inner_w, size=7.2)
    if max_lines:
        lines = lines[:max_lines]
    line_h = 3.65
    h = 11 + len(lines) * line_h
    _ensure_space(pdf, h + 3)
    x = pdf.l_margin
    y = pdf.get_y()
    pdf.set_fill_color(*bg)
    pdf.set_draw_color(*accent)
    pdf.rect(x, y, pdf.epw, h, style="DF", round_corners=True, corner_radius=2.2)
    pdf.set_fill_color(*accent)
    pdf.rect(x, y, 2.1, h, style="F", round_corners=True, corner_radius=1)
    pdf.set_xy(x + 5, y + 3)
    pdf.set_text_color(*accent)
    pdf.set_font("Helvetica", "B", 8.2)
    pdf.cell(pdf.epw - 10, 4, pdf_safe(title), 0, 1)
    pdf.set_text_color(51, 65, 85)
    pdf.set_font("Helvetica", "", 7.2)
    ty = y + 8.5
    for line in lines:
        pdf.set_xy(x + 5, ty)
        pdf.cell(inner_w, line_h, pdf_safe(line), 0, 0)
        ty += line_h
    pdf.set_y(y + h + 3)


def pdf_table(pdf, headers, rows, widths, aligns=None, header_fill=(241,245,249), font_size=7.0, row_h=5.2):
    """Simple executive table with predictable pagination."""
    aligns = aligns or ["L"] * len(headers)
    _ensure_space(pdf, row_h * 3 + 5)
    x0 = pdf.l_margin
    pdf.set_fill_color(*header_fill)
    pdf.set_draw_color(*ReportPDF.BORDER)
    pdf.set_font("Helvetica", "B", font_size)
    pdf.set_text_color(*ReportPDF.NAVY)
    for i, (h, w) in enumerate(zip(headers, widths)):
        pdf.cell(w, row_h, pdf_safe(h), 1, 0, aligns[i], True)
    pdf.ln(row_h)
    pdf.set_font("Helvetica", "", font_size)
    pdf.set_text_color(51, 65, 85)
    for row in rows:
        if pdf.get_y() + row_h > 276:
            pdf.add_page(); pdf.set_y(28)
            pdf.set_x(x0)
            pdf.set_fill_color(*header_fill)
            pdf.set_font("Helvetica", "B", font_size)
            pdf.set_text_color(*ReportPDF.NAVY)
            for i, (h, w) in enumerate(zip(headers, widths)):
                pdf.cell(w, row_h, pdf_safe(h), 1, 0, aligns[i], True)
            pdf.ln(row_h)
            pdf.set_font("Helvetica", "", font_size)
            pdf.set_text_color(51, 65, 85)
        pdf.set_x(x0)
        for i, (val, w) in enumerate(zip(row, widths)):
            txt = pdf_safe(val)
            if pdf.get_string_width(txt) > w - 2:
                while len(txt) > 3 and pdf.get_string_width(txt + "...") > w - 2:
                    txt = txt[:-1]
                txt += "..."
            pdf.cell(w, row_h, txt, 1, 0, aligns[i])
        pdf.ln(row_h)


def add_signature(pdf, signature_name="Dilip Kumar Vishwakarma"):
    _ensure_space(pdf, 22)
    pdf.ln(3)
    y = pdf.get_y()
    pdf.set_draw_color(203, 213, 225)
    pdf.line(pdf.l_margin, y, pdf.l_margin + 62, y)
    pdf.set_y(y + 2)
    pdf.set_font("Helvetica", "B", 8.8)
    pdf.set_text_color(*ReportPDF.NAVY)
    pdf.cell(0, 4.6, pdf_safe(signature_name), 0, 1)
    pdf.set_font("Helvetica", "", 7.2)
    pdf.set_text_color(*ReportPDF.SLATE)
    pdf.cell(0, 4, "Academic Consultant | OneLern Academic Team", 0, 1)


def teacher_math_components(row):
    workdays = max(0, int(safe_float(row.get("Eligible Working Days"))))
    active_days = max(0, int(safe_float(row.get("Active Days"))))
    consistency = min((active_days / workdays * 100), 100) if workdays else 0
    items = [
        ("Lesson Delivery", safe_float(row.get("Lesson KPI %")), 40.0),
        ("Library", safe_float(row.get("Library KPI %")), 35.0),
        ("Other Modules", safe_float(row.get("Other KPI %")), 15.0),
        ("Consistency", consistency, 10.0),
    ]
    out = []
    total = 0
    for label, achieved, weight in items:
        contribution = min(max(achieved, 0), 100) * weight / 100
        total += contribution
        out.append((label, achieved, weight, contribution))
    return out, total, consistency


def teacher_math_explanation(row):
    components, total, consistency = teacher_math_components(row)
    workdays = int(safe_float(row.get("Eligible Working Days")))
    active = int(safe_float(row.get("Active Days")))
    return (
        "Teacher Health combines KPI achievement and activity consistency. "
        "Weights: Lesson Delivery 40%, Library 35%, Other Modules 15%, Consistency 10%. "
        "Each component is capped at 100% before weighting. "
        f"Consistency = {active}/{workdays} x 100 = {consistency:.1f}%. "
        f"Weighted total = {total:.2f}, rounded to {safe_float(row.get('Health Score')):.0f}/100."
    )


def school_math_explanation(school_row, teacher_data, workdays):
    total = max(0, int(safe_float(school_row.get("Teachers"))))
    active = max(0, int(safe_float(school_row.get("Active"))))
    met_all = max(0, int(safe_float(school_row.get("Met All KPIs"))))
    compliance = safe_float(school_row.get("Overall Compliance %"))
    health = safe_float(school_row.get("Health Score"))
    if teacher_data is not None and not teacher_data.empty:
        scores = [safe_float(v) for v in teacher_data["Health Score"].tolist()]
        score_sum = sum(scores)
        average_text = f"sum of Teacher Health Scores {score_sum:.0f} / {len(scores)} teachers = {score_sum / len(scores):.2f}, rounded to {health:.0f}/100"
    else:
        average_text = "no teacher-level Health Scores were available"
    return (
        f"Institution Health = {average_text}. "
        f"Full KPI Compliance = {met_all}/{total} x 100 = {compliance:.1f}%. "
        f"Active Teachers = {active}/{total} with recorded usage above 0 minutes. "
        "Met All KPIs counts teachers whose Lesson Delivery, Library and Other Modules are all at least 100%. "
        f"Eligible working days used in this report: {workdays}."
    )


def kpi_target_math_explanation(school_row, workdays, teacher_count):
    school_name = str(school_row.get("School") or "").strip()
    scope = kpi_scope_details(school_name if school_name else None)
    ld = safe_float(school_row.get("Lesson Target / Day"))
    lib = safe_float(school_row.get("Library Target / Day"))
    oth = safe_float(school_row.get("Other Target / Day"))
    return (
        f"Effective daily benchmarks - Lesson Delivery {ld:.1f} min/day ({scope['source']['lessonDelivery']}), "
        f"Library {lib:.1f} min/day ({scope['source']['library']}), Other Modules {oth:.1f} min/day ({scope['source']['otherModules']}). "
        "Per-teacher review target = daily benchmark x eligible working days. "
        "Institution target = per-teacher review target x included teachers. "
        f"For {workdays} days and {teacher_count} teachers: Lesson {ld*workdays*teacher_count:.1f} min, "
        f"Library {lib*workdays*teacher_count:.1f} min, Other Modules {oth*workdays*teacher_count:.1f} min."
    )


def teacher_auto_insights(row, action_days=7):
    # preserve deterministic, auditable insights
    lesson = safe_float(row.get("Lesson KPI %"))
    library = safe_float(row.get("Library KPI %"))
    other = safe_float(row.get("Other KPI %"))
    total = safe_float(row.get("Total Minutes"))
    active = int(safe_float(row.get("Active Days")))
    workdays = max(1, int(safe_float(row.get("Eligible Working Days"))))
    consistency = active / workdays * 100
    metrics = {"Lesson Delivery": lesson, "Library": library, "Other Modules": other}
    strongest = max(metrics, key=metrics.get)
    weakest = min(metrics, key=metrics.get)
    diagnosis = f"{total:.1f} minutes recorded across {active}/{workdays} active days ({consistency:.1f}% consistency)."
    strength = f"Strongest relative adoption: {strongest} at {metrics[strongest]:.1f}% of configured KPI."
    focus = f"Primary implementation gap: {weakest} at {metrics[weakest]:.1f}% of configured KPI."
    plan = (
        f"Next {action_days} days: establish a daily usage rhythm, prioritise {weakest}, "
        "use the relevant classroom/content workflow, and review the same KPI denominator at the next checkpoint."
    )
    return diagnosis, strength, focus, plan


def _teacher_status_counts(teacher_data):
    if teacher_data is None or teacher_data.empty:
        return []
    counts = teacher_data["Status"].value_counts()
    order = ["Meeting All KPIs", "Partially Meeting", "Below KPI", "Never Logged In", "0 Usage"]
    return [(name, int(counts.get(name, 0))) for name in order if int(counts.get(name, 0)) > 0]


def _school_priority_text(teacher_data):
    if teacher_data is None or teacher_data.empty:
        return "No teacher-level evidence available for prioritisation."
    priority = teacher_data.sort_values(["Health Score", "Total Minutes"]).head(4)
    parts = []
    for _, r in priority.iterrows():
        kpis = {
            "Lesson Delivery": safe_float(r.get("Lesson KPI %")),
            "Library": safe_float(r.get("Library KPI %")),
            "Other Modules": safe_float(r.get("Other KPI %")),
        }
        weak = min(kpis, key=kpis.get)
        parts.append(f"{r.get('Teacher')}: {weak} {kpis[weak]:.1f}%, health {safe_float(r.get('Health Score')):.0f}/100")
    return "; ".join(parts) + "."


def add_teacher_report_page(pdf, row, evidence, start_date, end_date, action_days, signature_name="Dilip Kumar Vishwakarma"):
    """One executive page per teacher. Dense mathematics is moved into a compact contribution table."""
    pdf.add_page()
    pdf.set_y(28)
    pdf_title_block(
        pdf,
        str(row.get("Teacher", "Teacher 360")),
        f"Teacher 360 | {row.get('School','')} | {start_date} to {end_date}",
        "Teacher implementation profile",
    )

    y = pdf.get_y(); gap = 3; w = (pdf.epw - gap * 3) / 4
    cards = [
        ("Health", f"{safe_float(row.get('Health Score')):.0f}/100", ReportPDF.INDIGO, "Weighted implementation score"),
        ("Active Days", f"{int(safe_float(row.get('Active Days')))}/{int(safe_float(row.get('Eligible Working Days')))}", ReportPDF.CYAN, "Usage-day consistency"),
        ("Total Usage", f"{safe_float(row.get('Total Minutes')):.1f} min", ReportPDF.GREEN, "Selected review period"),
        ("Status", str(row.get('Status','')), ReportPDF.AMBER, "Against configured KPIs"),
    ]
    for i, (lab, val, acc, sub) in enumerate(cards):
        pdf_card(pdf, pdf.l_margin + i * (w + gap), y, w, 21, lab, val, acc, sub)
    pdf.set_y(y + 25)

    pdf_section_title(pdf, "KPI scorecard", "Actual achievement against this teacher's configured review-period targets")
    y = pdf.get_y()
    y = pdf_progress(pdf, "Lesson Delivery", row.get("Lesson KPI %"), pdf.l_margin, y, pdf.epw,
                     detail=f"{safe_float(row.get('Lesson Delivery')):.1f} / {safe_float(row.get('Lesson Target')):.1f} min")
    y = pdf_progress(pdf, "Library", row.get("Library KPI %"), pdf.l_margin, y + 1, pdf.epw,
                     detail=f"{safe_float(row.get('Library')):.1f} / {safe_float(row.get('Library Target')):.1f} min")
    y = pdf_progress(pdf, "Other Modules", row.get("Other KPI %"), pdf.l_margin, y + 1, pdf.epw,
                     detail=f"{safe_float(row.get('Other Modules')):.1f} / {safe_float(row.get('Other Target')):.1f} min")
    pdf.set_y(y + 2)

    components, total_calc, consistency = teacher_math_components(row)
    pdf_section_title(pdf, "Health score mathematics", "Transparent weighted calculation - capped at 100% per component")
    rows = [(label, f"{ach:.1f}%", f"{weight:.0f}%", f"{contrib:.2f}") for label, ach, weight, contrib in components]
    rows.append(("TOTAL", "", "", f"{total_calc:.2f} -> {safe_float(row.get('Health Score')):.0f}/100"))
    pdf_table(pdf, ["Component", "Achievement", "Weight", "Contribution"], rows, [62, 38, 28, 58], ["L","R","R","R"], font_size=6.9, row_h=4.9)
    pdf.ln(2)

    diagnosis, strength, focus, plan = teacher_auto_insights(row, action_days)
    # Two-column executive insight organizer
    _ensure_space(pdf, 31)
    x = pdf.l_margin; y0 = pdf.get_y(); gap = 3; col = (pdf.epw - gap) / 2
    pdf.set_xy(x, y0)
    _mini_box(pdf, x, y0, col, 27, "WHAT THE DATA SAYS", diagnosis + " " + strength, "info")
    _mini_box(pdf, x + col + gap, y0, col, 27, "PRIORITY ACTION", focus + " " + plan, "warning")
    pdf.set_y(y0 + 31)

    if evidence is not None and not evidence.empty:
        # Compact top modules only; raw evidence remains available in app.
        modules = evidence.groupby("Raw Module")["Minutes"].sum().sort_values(ascending=False).head(5)
        pdf_hbar_chart(pdf, "Top module utilisation", [(str(k), safe_float(v)) for k, v in modules.items()], max_rows=5, height_each=6.2)

    pdf_note_box(
        pdf,
        "Closing note",
        "At the next review, compare the same KPI percentages, active-day consistency and content breadth. Recognise verified improvement and agree a measurable commitment for any persistent gap.",
        "success",
        max_lines=4,
    )

    # Keep each Teacher 360 profile self-contained on one page.
    # A compact sign-off is anchored above the footer rather than triggering a new page.
    sig_y = 263
    pdf.set_draw_color(203, 213, 225)
    pdf.line(pdf.l_margin, sig_y, pdf.l_margin + 52, sig_y)
    pdf.set_xy(pdf.l_margin, sig_y + 1.8)
    pdf.set_font("Helvetica", "B", 7.8)
    pdf.set_text_color(*ReportPDF.NAVY)
    pdf.cell(70, 4, pdf_safe(signature_name), 0, 1)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 6.5)
    pdf.set_text_color(*ReportPDF.SLATE)
    pdf.cell(90, 3.5, "Academic Consultant | OneLern Academic Team", 0, 1)


def _mini_box(pdf, x, y, w, h, title, body, kind="info"):
    palette = {
        "info": ((239,246,255),(37,99,235)),
        "success": ((240,253,244),(22,163,74)),
        "warning": ((255,247,237),(234,88,12)),
        "purple": ((245,243,255),(124,58,237)),
    }
    bg, accent = palette.get(kind, palette["info"])
    pdf.set_fill_color(*bg); pdf.set_draw_color(*ReportPDF.BORDER)
    pdf.rect(x, y, w, h, style="DF", round_corners=True, corner_radius=2.2)
    pdf.set_fill_color(*accent); pdf.rect(x, y, 2, h, "F")
    pdf.set_xy(x+5, y+3); pdf.set_font("Helvetica","B",7.6); pdf.set_text_color(*accent)
    pdf.cell(w-9,4,pdf_safe(title),0,1)
    lines = _wrap_pdf_text(pdf, body, w-10, size=6.8)[:5]
    ty=y+8.3; pdf.set_font("Helvetica","",6.8); pdf.set_text_color(51,65,85)
    for line in lines:
        pdf.set_xy(x+5,ty); pdf.cell(w-10,3.3,pdf_safe(line),0,0); ty += 3.3


def make_premium_school_pack_pdf(school_row, teacher_data, raw_school, start_date, end_date, workdays, action_days=7, ai_text="", signature_name="Dilip Kumar Vishwakarma"):
    """Executive pack: 2 school pages + one self-contained page per teacher."""
    pdf = ReportPDF()
    pdf.set_auto_page_break(True, 14)
    pdf.set_margins(12, 12, 12)

    # ---------------- PAGE 1: EXECUTIVE COCKPIT ----------------
    pdf.add_page(); pdf.set_y(28)
    school = str(school_row.get("School", "School"))
    pdf_title_block(
        pdf,
        "School 360 Intelligence Report",
        f"{school} | Review period {start_date} to {end_date} | {workdays} eligible working days | {len(teacher_data)} teacher profiles",
        "Executive implementation brief",
    )

    y = pdf.get_y(); gap = 3; w = (pdf.epw - gap * 3) / 4
    school_cards = [
        ("Institution Health", f"{safe_float(school_row.get('Health Score')):.0f}/100", ReportPDF.INDIGO, "Average teacher health"),
        ("Full KPI Compliance", f"{safe_float(school_row.get('Overall Compliance %')):.1f}%", ReportPDF.CYAN, "Met all 3 KPIs"),
        ("Active Teachers", f"{int(safe_float(school_row.get('Active')))}/{int(safe_float(school_row.get('Teachers')))}", ReportPDF.GREEN, "Usage above 0 min"),
        ("Met All KPIs", str(int(safe_float(school_row.get('Met All KPIs')))), ReportPDF.AMBER, "Teacher count"),
    ]
    for i, (lab, val, acc, sub) in enumerate(school_cards):
        pdf_card(pdf, pdf.l_margin + i * (w + gap), y, w, 21, lab, val, acc, sub)
    pdf.set_y(y + 25)

    teacher_count = max(1, int(safe_float(school_row.get("Teachers"))))
    targets = {
        "Lesson Delivery": safe_float(school_row.get("Lesson Target / Day")) * workdays * teacher_count,
        "Library": safe_float(school_row.get("Library Target / Day")) * workdays * teacher_count,
        "Other Modules": safe_float(school_row.get("Other Target / Day")) * workdays * teacher_count,
    }
    actuals = {
        "Lesson Delivery": safe_float(school_row.get("Lesson Delivery Minutes")),
        "Library": safe_float(school_row.get("Library Minutes")),
        "Other Modules": safe_float(school_row.get("Other Modules Minutes")),
    }
    pdf_section_title(pdf, "Institutional KPI scorecard", "Actual usage against cumulative configured school targets")
    y = pdf.get_y()
    for label in ["Lesson Delivery", "Library", "Other Modules"]:
        pct = actuals[label] / targets[label] * 100 if targets[label] else 0
        y = pdf_progress(pdf, label, pct, pdf.l_margin, y, pdf.epw,
                         detail=f"Actual {actuals[label]:.1f} min | Target {targets[label]:.1f} min")
        y += 1
    pdf.set_y(y + 2)

    # Teacher status distribution and priority in two columns.
    _ensure_space(pdf, 42)
    status_rows = _teacher_status_counts(teacher_data)
    x = pdf.l_margin; y0 = pdf.get_y(); gap = 4; col = (pdf.epw-gap)/2
    _mini_box(pdf, x, y0, col, 36, "IMPLEMENTATION SIGNAL", school_math_explanation(school_row, teacher_data, workdays), "info")
    _mini_box(pdf, x+col+gap, y0, col, 36, "PRIORITY TEACHERS", _school_priority_text(teacher_data), "warning")
    pdf.set_y(y0+40)

    if not teacher_data.empty:
        health = teacher_data.sort_values("Health Score", ascending=False).head(8)
        pdf_hbar_chart(pdf, "Teacher health ranking", [(str(r["Teacher"]), safe_float(r["Health Score"])) for _, r in health.iterrows()], max_value=100, max_rows=8, height_each=6.2)

    # ---------------- PAGE 2: METHODOLOGY + ACTION ----------------
    pdf.add_page(); pdf.set_y(28)
    pdf_title_block(pdf, "Executive Interpretation & Methodology", f"{school} | Transparent calculations, adoption pattern and next actions", "Decision support")

    # Compact target table replaces oversized narrative boxes.
    scope = kpi_scope_details(school)
    ld = safe_float(school_row.get("Lesson Target / Day")); lib = safe_float(school_row.get("Library Target / Day")); oth = safe_float(school_row.get("Other Target / Day"))
    target_rows = [
        ("Lesson Delivery", f"{ld:.1f}", scope['source']['lessonDelivery'], f"{ld*workdays:.1f}", f"{ld*workdays*teacher_count:.1f}"),
        ("Library", f"{lib:.1f}", scope['source']['library'], f"{lib*workdays:.1f}", f"{lib*workdays*teacher_count:.1f}"),
        ("Other Modules", f"{oth:.1f}", scope['source']['otherModules'], f"{oth*workdays:.1f}", f"{oth*workdays*teacher_count:.1f}"),
    ]
    pdf_section_title(pdf, "KPI target mathematics", "Review target = daily benchmark x working days; school target = teacher review target x included teachers")
    pdf_table(pdf, ["Module", "Min/day", "Source", "Per teacher", "School target"], target_rows, [42,24,49,34,37], ["L","R","L","R","R"], font_size=6.6, row_h=5.2)
    pdf.ln(2)

    # Institution Health contribution table.
    pdf_section_title(pdf, "Institution Health mathematics", "Institution Health is the arithmetic mean of individual Teacher Health Scores")
    health_rows = []
    if teacher_data is not None and not teacher_data.empty:
        for _, r in teacher_data.sort_values("Teacher").iterrows():
            health_rows.append((str(r.get("Teacher")), f"{safe_float(r.get('Health Score')):.0f}/100"))
        score_sum = sum(safe_float(r.get("Health Score")) for _, r in teacher_data.iterrows())
        health_rows.append(("Average", f"{score_sum:.0f}/{len(teacher_data)} = {score_sum/len(teacher_data):.2f} -> {safe_float(school_row.get('Health Score')):.0f}/100"))
    pdf_table(pdf, ["Teacher / calculation", "Health"], health_rows, [126,60], ["L","R"], font_size=6.8, row_h=4.8)
    pdf.ln(2)

    if raw_school is not None and not raw_school.empty:
        modules = raw_school.groupby("Raw Module")["Minutes"].sum().sort_values(ascending=False).head(8)
        pdf_hbar_chart(pdf, "Module adoption", [(str(k), safe_float(v)) for k, v in modules.items()], max_rows=8, height_each=6.0)

    if not teacher_data.empty:
        top = teacher_data.sort_values(["Health Score","Total Minutes"], ascending=False).head(3)
        strengths = "Reference points: " + ", ".join(top["Teacher"].astype(str).tolist()) + ". Use their stronger behaviours as examples while validating the underlying activity evidence."
    else:
        strengths = "No teacher-level evidence available."
    pdf_note_box(pdf, "Implementation strengths", strengths, "success", max_lines=4)
    pdf_note_box(pdf, "Management priority", _school_priority_text(teacher_data), "warning", max_lines=5)
    pdf_note_box(pdf, f"{action_days}-day action organiser",
                 f"Days 1-2: validate the lowest KPI and obtain teacher-specific commitments. Days 3-5: reinforce the weakest workflow through guided practice. Days 6-{action_days}: track active days and KPI movement. At the checkpoint, compare the same metrics teacher by teacher and record the next commitment.",
                 "purple", max_lines=5)

    if ai_text:
        excerpt = clean_ai_text(ai_text)
        pdf_note_box(pdf, "AI interpretation - verified facts only", excerpt, "info", max_lines=8)

    pdf_note_box(pdf, "Closing note",
                 "This report converts verified platform evidence into focused academic implementation action. The next review should recognise measurable improvement, isolate persistent gaps and agree specific commitments with school leadership.",
                 "success", max_lines=4)
    add_signature(pdf, signature_name)

    # ---------------- TEACHER SECTION ----------------
    if not teacher_data.empty:
        for _, row in teacher_data.sort_values("Teacher").iterrows():
            evidence = raw_school[raw_school["Teacher Key"] == row.get("Teacher Key")].copy() if raw_school is not None and not raw_school.empty else pd.DataFrame(columns=USAGE_COLUMNS)
            add_teacher_report_page(pdf, row, evidence, start_date, end_date, action_days, signature_name)

    return bytes(pdf.output())


def make_premium_teacher_pdf(teacher_row, evidence, start_date, end_date, action_days=7):
    pdf = ReportPDF(); pdf.set_auto_page_break(True, 14); pdf.set_margins(12,12,12)
    add_teacher_report_page(pdf, teacher_row, evidence, start_date, end_date, action_days)
    return bytes(pdf.output())


@st.cache_data(show_spinner=False, max_entries=60)
def cached_text_pdf(title, subtitle, report_text):
    pdf=ReportPDF(); pdf.set_auto_page_break(True,14); pdf.set_margins(12,12,12); pdf.add_page(); pdf.ln(17)
    pdf.set_font("Helvetica","B",16); pdf.set_text_color(15,23,42); pdf.set_x(pdf.l_margin); pdf.multi_cell(pdf.epw,8,pdf_safe(title)); pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica","",8); pdf.set_text_color(100,116,139); pdf.multi_cell(pdf.epw,5,pdf_safe(subtitle)); pdf.set_x(pdf.l_margin); pdf.ln(3)
    pdf_note_box(pdf,"Report",clean_ai_text(report_text),"info")
    add_signature(pdf)
    return bytes(pdf.output())


def make_text_pdf(title, subtitle, report_text, evidence_df=None):
    return cached_text_pdf(title, subtitle, report_text)


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
# SMART WHATSAPP -> FOLLOW-UP -> CALENDAR WORKFLOW
# =========================================================
def _next_school_day(d):
    # Sundays are excluded from the implementation review cycle.
    while d.weekday() == 6:
        d += timedelta(days=1)
    return d


def suggest_followup(school_row):
    health = float(school_row.get("Health Score", 0) or 0)
    compliance = float(school_row.get("Overall Compliance %", 0) or 0)
    inactive = int(school_row.get("Inactive / Never Logged In", 0) or 0)

    if inactive > 0 or health < 25 or compliance <= 10:
        gap_days = 3
        urgency = "High priority"
    elif health < 50 or compliance < 40:
        gap_days = 5
        urgency = "Priority"
    elif health < 75 or compliance < 70:
        gap_days = 7
        urgency = "Standard review"
    else:
        gap_days = 10
        urgency = "Sustainability review"

    suggested_date = _next_school_day(date.today() + timedelta(days=gap_days))

    if inactive > 0:
        issue = (
            f"{inactive} teacher(s) are inactive / never logged in. Review activation, "
            f"KPI movement and implementation barriers after the shared report."
        )
    elif compliance < 100:
        issue = (
            f"Review progress against the shared School 360 report. Current full KPI "
            f"compliance is {compliance:.1f}% and Health Score is {health:.0f}/100."
        )
    else:
        issue = (
            "Review sustained implementation after the shared School 360 report and "
            "confirm that current KPI performance is being maintained."
        )

    return suggested_date, urgency, issue


def _calendar_payload(school, follow_date, follow_time, issue, commitment):
    start_dt = datetime.combine(follow_date, follow_time)
    end_dt = start_dt + timedelta(minutes=30)
    title = f"AcadIntel Follow-Up - {school}"
    details = (
        f"School: {school}\n"
        f"Report shared via WhatsApp: {date.today().strftime('%d %b %Y')}\n"
        f"Follow-up focus: {issue}\n"
        f"Last commitment: {commitment or 'To be confirmed during follow-up'}\n\n"
        "Prepared through AcadIntel 360.\n"
        "Dilip Kumar Vishwakarma | Academic Consultant"
    )
    return title, details, start_dt, end_dt


def google_calendar_url(school, follow_date, follow_time, issue, commitment):
    title, details, start_dt, end_dt = _calendar_payload(
        school, follow_date, follow_time, issue, commitment
    )
    params = {
        "action": "TEMPLATE",
        "text": title,
        "dates": f"{start_dt.strftime('%Y%m%dT%H%M%S')}/{end_dt.strftime('%Y%m%dT%H%M%S')}",
        "details": details,
        "ctz": "Asia/Kolkata",
    }
    return "https://calendar.google.com/calendar/render?" + urllib.parse.urlencode(params)


def outlook_calendar_url(school, follow_date, follow_time, issue, commitment):
    title, details, start_dt, end_dt = _calendar_payload(
        school, follow_date, follow_time, issue, commitment
    )
    params = {
        "path": "/calendar/action/compose",
        "rru": "addevent",
        "subject": title,
        "startdt": start_dt.strftime('%Y-%m-%dT%H:%M:%S'),
        "enddt": end_dt.strftime('%Y-%m-%dT%H:%M:%S'),
        "body": details,
    }
    return "https://outlook.office.com/calendar/0/deeplink/compose?" + urllib.parse.urlencode(params)


def make_ics_event(school, follow_date, follow_time, issue, commitment):
    title, details, start_dt, end_dt = _calendar_payload(
        school, follow_date, follow_time, issue, commitment
    )

    def esc(v):
        return str(v).replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")

    uid = hashlib.sha256(
        f"{school}|{start_dt.isoformat()}|{issue}".encode("utf-8")
    ).hexdigest()[:24] + "@acadintel360"

    ics = (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "PRODID:-//AcadIntel 360//Academic Follow-Up//EN\r\n"
        "CALSCALE:GREGORIAN\r\n"
        "BEGIN:VEVENT\r\n"
        f"UID:{uid}\r\n"
        f"DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}\r\n"
        f"DTSTART;TZID=Asia/Kolkata:{start_dt.strftime('%Y%m%dT%H%M%S')}\r\n"
        f"DTEND;TZID=Asia/Kolkata:{end_dt.strftime('%Y%m%dT%H%M%S')}\r\n"
        f"SUMMARY:{esc(title)}\r\n"
        f"DESCRIPTION:{esc(details)}\r\n"
        "BEGIN:VALARM\r\n"
        "TRIGGER:-PT30M\r\n"
        "ACTION:DISPLAY\r\n"
        f"DESCRIPTION:{esc('Follow up with ' + school)}\r\n"
        "END:VALARM\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n"
    )
    return ics.encode("utf-8")


@st.dialog("💬 Report Shared — Schedule the Next Follow-Up", width="large")
def share_followup_dialog(school, message, school_row, group_url=""):
    suggested_date, urgency, suggested_issue = suggest_followup(school_row)

    st.markdown(
        f"**AcadIntel suggestion:** {urgency}. Based on the current implementation "
        f"position, review **{school}** on **{suggested_date.strftime('%d %b %Y')}**."
    )
    st.caption(
        "This pop-up is triggered when you choose to share the report. WhatsApp does not "
        "send a confirmation back to Streamlit, so AcadIntel schedules from your share action."
    )

    c1, c2 = st.columns(2)
    follow_date = c1.date_input(
        "Suggested follow-up date", value=suggested_date, key=f"share_fu_date_{school}"
    )
    follow_time = c2.time_input(
        "Reminder time", value=datetime.strptime("10:00", "%H:%M").time(), key=f"share_fu_time_{school}"
    )

    issue = st.text_area(
        "Suggested follow-up focus", value=suggested_issue, height=110, key=f"share_fu_issue_{school}"
    )
    commitment = st.text_area(
        "Last commitment / expected commitment",
        value="Review the School 360 report and confirm measurable improvement before the next review.",
        height=90,
        key=f"share_fu_commitment_{school}",
    )

    google_url = google_calendar_url(school, follow_date, follow_time, issue, commitment)
    outlook_url = outlook_calendar_url(school, follow_date, follow_time, issue, commitment)
    ics_bytes = make_ics_event(school, follow_date, follow_time, issue, commitment)
    whatsapp_url = "https://wa.me/?text=" + urllib.parse.quote(message)

    st.markdown("#### 1. Save the follow-up")
    s1, s2 = st.columns([1.35, 1])
    if s1.button("✅ Save Follow-Up in AcadIntel", type="primary", use_container_width=True, key=f"save_share_fu_{school}"):
        db_insert("followups", {
            "school_name": school,
            "followup_date": str(follow_date),
            "issue": issue.strip(),
            "last_commitment": commitment.strip(),
            "status": "Open",
            "remarks": "Created from WhatsApp report-sharing workflow.",
        })
        st.session_state[f"share_fu_saved::{school}::{follow_date}"] = True
        st.success("Follow-up saved in AcadIntel 360.")

    s2.download_button(
        "⏰ Calendar Alarm (.ics)",
        data=ics_bytes,
        file_name=re.sub(r"[^A-Za-z0-9]+", "_", school) + "_Follow_Up.ics",
        mime="text/calendar",
        use_container_width=True,
        help="Works with most calendar apps and includes a 30-minute reminder alarm.",
    )

    st.markdown("#### 2. Add it to your calendar")
    g1, g2 = st.columns(2)
    g1.link_button("📅 Add to Google Calendar", google_url, use_container_width=True)
    g2.link_button("🗓️ Add to Outlook Calendar", outlook_url, use_container_width=True)

    st.markdown("#### 3. Share the report")
    w1, w2 = st.columns(2)
    w1.link_button("💬 Open WhatsApp with Message", whatsapp_url, use_container_width=True)
    if group_url:
        w2.link_button("👥 Open Saved School Group", group_url, use_container_width=True)
    else:
        w2.button("👥 School group not saved", disabled=True, use_container_width=True)

    if st.button("Done", use_container_width=True, key=f"close_share_fu_{school}"):
        st.session_state.share_follow_school = None
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

    if get_ai_client() is not None:
        st.caption("⚡ Gemini: ready on demand")
    else:
        st.error("Gemini not connected")

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
            st.session_state.raw_version = int(st.session_state.get("raw_version", 0)) + 1
            st.session_state.analytics_cache = {}
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
        "⚡ Quick Desk", "Command Center", "School 360", "Teacher 360", "Follow-Ups", "KPI & Roster", "Ask AcadIntel"
    ])


analytics_key = (
    int(st.session_state.get("raw_version", 0)),
    int(st.session_state.get("db_version", 0)),
    str(start_date),
    str(end_date),
    int(workdays),
)

_cached = st.session_state.setdefault("analytics_cache", {}).get(analytics_key)
if _cached is None:
    period_raw = filter_period(st.session_state.raw, start_date, end_date)
    teachers, schools, shared_usage = build_analytics(period_raw, start_date, end_date, workdays)
    st.session_state.analytics_cache = {analytics_key: (period_raw, teachers, schools, shared_usage)}
else:
    period_raw, teachers, schools, shared_usage = _cached

if st.session_state.raw.empty and roster_dataframe().empty:
    st.markdown('<div class="hero"><h1>AcadIntel 360</h1><p>Upload UserMetrics files to begin evidence-backed school and teacher intelligence.</p></div>', unsafe_allow_html=True)
    st.info("Start by uploading your raw UserMetrics files in the left sidebar.")
    st.stop()


# =========================================================
# QUICK DESK - PREMIUM DAILY CONTROL DESK
# =========================================================
if page == "⚡ Quick Desk":
    st.markdown(
        '<div class="hero"><h1>⚡ AcadIntel Control Desk</h1><p>School insight, graphical report, every Teacher 360, PDF, WhatsApp, call and follow-up — from one screen.</p></div>',
        unsafe_allow_html=True,
    )
    if schools.empty:
        st.warning("No school data is available for the selected review period. Upload and process UserMetrics first.")
        st.stop()

    top1, top2 = st.columns([2.2,1])
    with top1:
        school = st.selectbox("🏫 Select school", schools["School"].tolist(), key="quick_school")
    with top2:
        action_days = st.selectbox("🎯 Action plan", [7,15], format_func=lambda x:f"{x}-day plan", key="quick_action_days")
    action_days=int(action_days)

    school_row=schools[schools["School"]==school].iloc[0]
    school_teachers=teachers[teachers["School"]==school].copy()
    school_raw=period_raw[period_raw["School"]==school].copy()
    facts=school_verified_facts(school_row,school_teachers,school_raw,start_date,end_date,workdays)

    contact_rows=db_select("schools",{"school_name":school})
    contact=contact_rows[0] if contact_rows else {}
    phone=str(contact.get("contact_phone") or "")
    group_url=str(contact.get("whatsapp_group_url") or "")
    clean_phone=re.sub(r"\D","",phone)

    priority=school_teachers.sort_values(["Health Score","Total Minutes"]).head(4) if not school_teachers.empty else pd.DataFrame()
    priority_names=", ".join(priority["Teacher"].astype(str).tolist()) if not priority.empty else "No priority teachers identified"
    message=(
        f"Dear Sir/Ma'am,\n\n"
        f"Please find the implementation performance update for {school} for {start_date} to {end_date} ({workdays} working days).\n\n"
        f"📊 Health Score: {school_row['Health Score']}/100\n"
        f"🎯 Full KPI Compliance: {school_row['Overall Compliance %']}%\n"
        f"👩‍🏫 Active Teachers: {school_row['Active']}/{school_row['Teachers']}\n"
        f"⚠️ Priority Review: {priority_names}\n\n"
        f"The attached School 360 pack contains the graphical school dashboard and a separate Teacher 360 report for every teacher, along with evidence-backed action priorities.\n\n"
        f"Kindly review the same so that we can align the next implementation actions.\n\n"
        f"Regards,\nDilip Kumar Vishwakarma"
    )

    ai_key=f"quick_report::{school}::{start_date}::{end_date}::{workdays}::{action_days}"
    ai_text=st.session_state.get(ai_key,"")
    pdf_key="premium_pack::"+hashlib.sha256((school+str(start_date)+str(end_date)+str(workdays)+str(action_days)+ai_text+str(st.session_state.get('raw_version',0))).encode()).hexdigest()
    if pdf_key not in st.session_state:
        try:
            st.session_state[pdf_key]=make_premium_school_pack_pdf(school_row,school_teachers,school_raw,start_date,end_date,workdays,action_days,ai_text)
            st.session_state[pdf_key+"::error"]=""
        except Exception as exc:
            st.session_state[pdf_key]=None
            st.session_state[pdf_key+"::error"]=str(exc)
    pdf_bytes=st.session_state[pdf_key]

    st.markdown("### 🚀 One-tap actions")
    a1,a2,a3,a4,a5=st.columns(5)
    if pdf_bytes:
        a1.download_button("⬇ Full PDF Pack",pdf_bytes,file_name=re.sub(r"[^A-Za-z0-9]+","_",school)+"_360_Full_Report.pdf",mime="application/pdf",use_container_width=True,help="School overview + a separate Teacher 360 report for every teacher in one PDF.")
    else:
        a1.button("⬇ PDF unavailable",disabled=True,use_container_width=True)
    if a2.button("💬 Share + Follow-Up", use_container_width=True, type="primary"):
        st.session_state.share_follow_school = school
    if group_url: a3.link_button("👥 School Group",group_url,use_container_width=True)
    else: a3.button("👥 Add Group",disabled=True,use_container_width=True)
    if clean_phone: a4.link_button("📞 Call KDM",f"tel:+{clean_phone}",use_container_width=True)
    else: a4.button("📞 Add Number",disabled=True,use_container_width=True)
    if a5.button("📅 Follow Up",use_container_width=True): st.session_state.follow_school=school
    if st.session_state.follow_school==school: followup_dialog(school)
    if st.session_state.get("share_follow_school") == school:
        share_followup_dialog(school, message, school_row, group_url)
    if st.session_state.get(pdf_key+"::error"): st.error("PDF could not be prepared: "+st.session_state[pdf_key+"::error"])

    tabs=st.tabs(["📊 Report Dashboard","👩‍🏫 Teacher Reports","✨ AI Interpretation","💬 Communication"])

    with tabs[0]:
        render_graphical_school_report(school_row,school_teachers,school_raw,workdays,action_days)

    with tabs[1]:
        st.markdown("### Every teacher at your fingertips")
        st.caption("The Full PDF Pack above already contains a separate report for every teacher. You can also inspect or download one teacher individually here.")
        if school_teachers.empty:
            st.info("No teacher records are available.")
        else:
            teacher_name=st.selectbox("Select teacher",school_teachers["Teacher"].sort_values().tolist(),key="quick_teacher")
            tr=school_teachers[school_teachers["Teacher"]==teacher_name].iloc[0]
            ev=school_raw[school_raw["Teacher Key"]==tr["Teacher Key"]].copy()
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Health",f"{tr['Health Score']}/100")
            c2.metric("Lesson KPI",f"{tr['Lesson KPI %']}%")
            c3.metric("Library KPI",f"{tr['Library KPI %']}%")
            c4.metric("Other KPI",f"{tr['Other KPI %']}%")
            tdf=pd.DataFrame({"KPI":["Lesson Delivery","Library","Other Modules"],"Achievement":[tr["Lesson KPI %"],tr["Library KPI %"],tr["Other KPI %"]]})
            tf=px.bar(tdf,x="KPI",y="Achievement",text_auto=".1f",range_y=[0,max(110,float(tdf["Achievement"].max())*1.1)],title=f"{teacher_name} — KPI achievement %")
            st.plotly_chart(tf,use_container_width=True)
            diagnosis,strength,focus,plan=teacher_auto_insights(tr,action_days)
            x1,x2=st.columns(2)
            x1.markdown(f'<div class="success-box"><b>Relative strength</b><br>{strength}</div>',unsafe_allow_html=True)
            x2.markdown(f'<div class="warning-box"><b>Priority focus</b><br>{focus}</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="insight-box"><b>{action_days}-day plan</b><br>{plan}</div>',unsafe_allow_html=True)
            teacher_pdf=make_premium_teacher_pdf(tr,ev,start_date,end_date,action_days)
            st.download_button("⬇ Download this Teacher 360 PDF",teacher_pdf,file_name=re.sub(r"[^A-Za-z0-9]+","_",teacher_name)+"_360_Report.pdf",mime="application/pdf",use_container_width=True)
            if not ev.empty:
                st.dataframe(ev[["DateTime","Raw Module","Grade","Subject","Book","Minutes"]].sort_values("DateTime",ascending=False),use_container_width=True,hide_index=True,height=320)

    with tabs[2]:
        st.markdown("### Gemini — optional deeper interpretation")
        st.caption("The graphical PDF and Teacher 360 reports work without Gemini. AI only enriches the interpretation; it never recalculates KPIs.")
        if st.button("✨ Generate evidence-backed management interpretation",type="primary",use_container_width=True,key=f"premium_ai_{school}"):
            try:
                with st.spinner("Gemini is interpreting verified facts..."):
                    text_ai,used_model=ai_generate(school_report_prompt(facts,action_days),force=True)
                st.session_state[ai_key]=text_ai
                st.session_state[ai_key+"::model"]=used_model
                st.rerun()
            except Exception as exc:
                st.error(f"Gemini could not generate the interpretation: {exc}")
        if st.session_state.get(ai_key):
            render_report(st.session_state[ai_key])
            if st.session_state.get(ai_key+"::model"): st.caption("Gemini model: "+st.session_state[ai_key+"::model"])
        else:
            st.info("No AI wait is required for the report. The graphical deterministic report is already available in the first tab and PDF pack.")

    with tabs[3]:
        st.markdown("### Ready-to-send school communication")
        st.text_area("Customized WhatsApp message",message,height=240,key=f"premium_message_{school}")
        c1,c2=st.columns(2)
        if c1.button("💬 Share + Schedule Follow-Up", use_container_width=True, type="primary"):
            st.session_state.share_follow_school = school
            st.rerun()
        if group_url: c2.link_button("👥 Open saved school group",group_url,use_container_width=True)
        else: c2.button("👥 Save group link below",disabled=True,use_container_width=True)
        with st.expander("School contact settings",expanded=not(bool(phone) and bool(group_url))):
            p1,p2=st.columns(2)
            quick_phone=p1.text_input("KDM mobile number",value=phone,key=f"quick_phone_{school}")
            quick_group=p2.text_input("WhatsApp group link",value=group_url,key=f"quick_group_{school}")
            if st.button("Save school contact",key=f"save_quick_contact_{school}"):
                payload={"school_name":school,"contact_phone":quick_phone.strip(),"whatsapp_group_url":quick_group.strip(),"updated_at":datetime.utcnow().isoformat()}
                if contact: db_update("schools",payload,contact["id"])
                else: db_insert("schools",payload)
                st.rerun()


# =========================================================
# COMMAND CENTER
# =========================================================
elif page == "Command Center":
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
# REPORT & WHATSAPP HUB
# =========================================================
elif page == "__legacy_report_hub__":
    if schools.empty:
        st.warning("No school data available. Upload and process UserMetrics first.")
        st.stop()

    st.markdown(
        """
        <div class="hero">
            <h1>Report & WhatsApp Hub</h1>
            <p>Generate a complete school report, download it, and share a customized school message from one screen.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    school = st.selectbox("Select School", schools["School"].tolist(), key="hub_school")
    school_row = schools[schools["School"] == school].iloc[0]
    school_teachers = teachers[teachers["School"] == school].copy()
    school_raw = period_raw[period_raw["School"] == school].copy()
    facts = school_verified_facts(school_row, school_teachers, school_raw, start_date, end_date, workdays)

    contact_rows = db_select("schools", {"school_name": school})
    contact = contact_rows[0] if contact_rows else {}
    phone = str(contact.get("contact_phone") or "")
    group_url = str(contact.get("whatsapp_group_url") or "")
    contact_role = str(contact.get("contact_role") or "School Management")
    clean_phone = re.sub(r"\D", "", phone)

    a, b, c, d = st.columns(4)
    a.metric("Health", f"{school_row['Health Score']}/100")
    b.metric("Compliance", f"{school_row['Overall Compliance %']}%")
    c.metric("Active Teachers", school_row["Active"])
    d.metric("Inactive", school_row["Inactive / Never Logged In"])

    action_days = st.radio("Action plan interval", [7, 15], horizontal=True, key="hub_action_days")
    auto_once = st.toggle("Auto-generate once when I open this school", value=False, help="Keep this OFF for maximum speed. Turn it ON only when you want automatic AI generation.")
    report_key = f"hub_report::{school}::{start_date}::{end_date}::{workdays}::{action_days}"

    if report_key not in st.session_state:
        st.session_state[report_key] = deterministic_school_summary(school_row, action_days)

    clicked = st.button("⚡ Generate Complete Report + Share Message", type="primary", use_container_width=True)
    should_auto = auto_once and not st.session_state.get(report_key + "::ai_done", False)

    if clicked or should_auto:
        try:
            with st.spinner("Generating the evidence-backed report with Gemini..."):
                text, used_model = ai_generate(school_report_prompt(facts, action_days), force=clicked)
            st.session_state[report_key] = text
            st.session_state[report_key + "::ai_done"] = True
            st.session_state[report_key + "::model"] = used_model
            db_insert("report_history", {
                "report_level": "School",
                "school_name": school,
                "action_plan_days": action_days,
                "report_text": text,
                "verified_facts": facts,
            })
        except Exception as exc:
            st.error(f"Gemini report generation failed: {exc}")
            st.info("A deterministic report is still available immediately below.")

    report_text = st.session_state[report_key]

    priority = school_teachers.sort_values(["Health Score", "Total Minutes"]).head(4) if not school_teachers.empty else pd.DataFrame()
    priority_names = ", ".join(priority["Teacher"].astype(str).tolist()) if not priority.empty else "No priority teachers identified"
    message = (
        f"Dear Sir/Ma'am,\n\n"
        f"Please find the implementation performance update for {school} for {start_date} to {end_date} ({workdays} working days).\n\n"
        f"📊 Health Score: {school_row['Health Score']}/100\n"
        f"🎯 Full KPI Compliance: {school_row['Overall Compliance %']}%\n"
        f"👩‍🏫 Active Teachers: {school_row['Active']}/{school_row['Teachers']}\n"
        f"⚠️ Priority Review: {priority_names}\n\n"
        f"The attached/report copy includes module-wise evidence, teacher-level gaps and the {action_days}-day action plan. "
        f"Kindly review the same so that we can align the next implementation actions.\n\n"
        f"Regards,\nDilip Kumar Vishwakarma"
    )

    pdf_key = "pdf::" + hashlib.sha256((school + str(start_date) + str(end_date) + report_text).encode("utf-8")).hexdigest()
    if pdf_key not in st.session_state:
        st.session_state[pdf_key] = cached_text_pdf(
            f"School 360 Intelligence Report - {school}",
            f"Review Period: {start_date} to {end_date} | Working Days: {workdays}",
            report_text,
        )
    pdf_bytes = st.session_state[pdf_key]

    st.subheader("One-Click Actions")
    q1, q2, q3, q4 = st.columns(4)
    q1.download_button(
        "⬇ Download Report PDF",
        data=pdf_bytes,
        file_name=re.sub(r"[^A-Za-z0-9]+", "_", school) + "_School_360.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
    q2.link_button(
        "💬 Share Customized WhatsApp",
        "https://wa.me/?text=" + urllib.parse.quote(message),
        use_container_width=True,
    )
    if group_url:
        q3.link_button("👥 Open School WhatsApp Group", group_url, use_container_width=True)
    else:
        q3.button("👥 Add Group Link in School 360", disabled=True, use_container_width=True)
    if clean_phone:
        q4.link_button("📞 Call KDM", f"tel:+{clean_phone}", use_container_width=True)
    else:
        q4.button("📞 Add KDM Number in School 360", disabled=True, use_container_width=True)

    st.text_area("Customized WhatsApp message", message, height=220, key=f"hub_message_{school}")
    st.subheader("School 360 Report")
    render_report(report_text)
    if st.session_state.get(report_key + "::model"):
        st.caption(f"Gemini model: {st.session_state[report_key + '::model']}. KPI calculations remain deterministic in Python.")


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
    auto_ai = st.checkbox("Auto-generate AI report for this school", value=False, key=f"auto_school_{school}", help="Keep OFF for fastest browsing; use the Generate button when needed.")
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

    pdf_key = "pdf::" + hashlib.sha256((school + str(start_date) + str(end_date) + report_text).encode("utf-8")).hexdigest()
    if pdf_key not in st.session_state:
        st.session_state[pdf_key] = make_text_pdf(
            f"School 360 Intelligence Report - {school}",
            f"Review Period: {start_date} to {end_date} | Working Days: {workdays}",
            report_text,
        )
    pdf_bytes = st.session_state[pdf_key]

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

    auto_teacher = st.checkbox("Auto-generate detailed AI report", value=False, key=f"auto_teacher_{teacher_row['Teacher Key']}", help="Keep OFF for fastest browsing; use the Generate button when needed.")
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
    pdf_bytes = make_premium_teacher_pdf(teacher_row, evidence, start_date, end_date, action_days)
    st.download_button("⬇ Download Graphical Teacher 360 PDF", data=pdf_bytes, file_name=re.sub(r"[^A-Za-z0-9]+", "_", teacher_name) + "_360_Audit_Report.pdf", mime="application/pdf", use_container_width=True)


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
        st.subheader("🌐 Global KPI Benchmarks")
        st.caption(
            "These values become the default for every school that does not have its own local override."
        )

        global_scope = kpi_scope_details()
        global_values = global_scope["global"]

        with st.form("global_kpis"):
            l = st.number_input(
                "Lesson Delivery — minutes/day",
                min_value=0.0,
                value=float(global_values["lessonDelivery"]),
                step=1.0,
            )
            lib = st.number_input(
                "Library — minutes/day",
                min_value=0.0,
                value=float(global_values["library"]),
                step=1.0,
            )
            oth = st.number_input(
                "Other Modules — combined minutes/day",
                min_value=0.0,
                value=float(global_values["otherModules"]),
                step=1.0,
            )

            if st.form_submit_button("Save Global Benchmarks", use_container_width=True):
                save_kpi("GLOBAL", "lessonDelivery", l)
                save_kpi("GLOBAL", "library", lib)
                save_kpi("GLOBAL", "otherModules", oth)
                st.success("Global KPI benchmarks saved.")
                st.rerun()

        st.info(
            "Global changes affect all schools that are inheriting the global benchmarks. "
            "Schools with a Local Override remain unchanged."
        )

        possible = set(schools["School"].tolist() if not schools.empty else [])
        possible.update(r["school_name"] for r in db_select("schools"))

        if possible:
            st.divider()
            st.subheader("🏫 Local School KPI Override")
            st.caption(
                "Use this only when a particular school's subscription package, implementation phase, "
                "infrastructure or agreed benchmark requires different targets."
            )

            s = st.selectbox("Select School", sorted(possible), key="kpi_school")
            detail = kpi_scope_details(s)
            eff = detail["effective"]

            source_rows = pd.DataFrame(
                [
                    {
                        "Module": "Lesson Delivery",
                        "Global": detail["global"]["lessonDelivery"],
                        "Local": detail["local"].get("lessonDelivery", "—"),
                        "Effective": eff["lessonDelivery"],
                        "Source": detail["source"]["lessonDelivery"],
                    },
                    {
                        "Module": "Library",
                        "Global": detail["global"]["library"],
                        "Local": detail["local"].get("library", "—"),
                        "Effective": eff["library"],
                        "Source": detail["source"]["library"],
                    },
                    {
                        "Module": "Other Modules",
                        "Global": detail["global"]["otherModules"],
                        "Local": detail["local"].get("otherModules", "—"),
                        "Effective": eff["otherModules"],
                        "Source": detail["source"]["otherModules"],
                    },
                ]
            )
            st.dataframe(source_rows, use_container_width=True, hide_index=True)

            if detail["has_local_override"]:
                st.success(f"{s} is currently using a Local KPI Override.")
            else:
                st.info(f"{s} is currently inheriting the Global KPI Benchmarks.")

            with st.form("school_kpis"):
                sl = st.number_input(
                    "Local Lesson Delivery — minutes/day",
                    min_value=0.0,
                    value=float(eff["lessonDelivery"]),
                    step=1.0,
                )
                sLib = st.number_input(
                    "Local Library — minutes/day",
                    min_value=0.0,
                    value=float(eff["library"]),
                    step=1.0,
                )
                so = st.number_input(
                    "Local Other Modules — combined minutes/day",
                    min_value=0.0,
                    value=float(eff["otherModules"]),
                    step=1.0,
                )

                save_local = st.form_submit_button(
                    "Save Local Override for This School",
                    use_container_width=True,
                )

                if save_local:
                    save_kpi("SCHOOL", "lessonDelivery", sl, s)
                    save_kpi("SCHOOL", "library", sLib, s)
                    save_kpi("SCHOOL", "otherModules", so, s)
                    st.success(f"Local KPI override saved for {s}.")
                    st.rerun()

            if detail["has_local_override"]:
                if st.button(
                    "↩ Reset This School to Global Benchmarks",
                    use_container_width=True,
                    key="reset_school_kpi",
                ):
                    reset_school_kpis_to_global(s)
                    st.success(f"{s} will now inherit the Global KPI Benchmarks.")
                    st.rerun()

            st.caption(
                "School cumulative targets are calculated automatically as: "
                "daily benchmark × eligible working days × number of teachers. "
                "Eligible working days can already be manually overridden from the Review Period controls."
            )

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
