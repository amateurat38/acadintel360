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
except Exception:
    genai = None


# =========================================================
# APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AcadIntel 360",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

USAGE_COLUMNS = [
    "School",
    "Teacher",
    "Teacher Key",
    "Raw Module",
    "KPI Module",
    "Minutes",
    "DateTime",
    "Grade",
    "Subject",
    "Book",
    "Source File",
]

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e8eaf2;
        padding: 16px;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(30,32,55,.055);
    }

    .hero {
        background: linear-gradient(
            120deg,
            #4438ca,
            #2563eb 55%,
            #0ea5e9
        );
        color: white;
        padding: 28px;
        border-radius: 24px;
        margin-bottom: 18px;
        box-shadow: 0 14px 36px rgba(55,48,163,.18);
    }

    .hero h1 {
        margin: 0;
        font-size: 40px;
    }

    .hero p {
        margin: .35rem 0 0;
        opacity: .9;
    }

    .ai-card {
        border-radius: 18px;
        padding: 16px 18px;
        margin: 9px 0;
        border: 1px solid rgba(0,0,0,.04);
    }

    .ai-title {
        font-weight: 800;
        font-size: 15px;
        margin-bottom: 7px;
    }

    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stLinkButton"] a {
        border-radius: 12px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "raw" not in st.session_state:
    st.session_state.raw = pd.DataFrame(columns=USAGE_COLUMNS)

if "import_errors" not in st.session_state:
    st.session_state.import_errors = []

if "ai_cache" not in st.session_state:
    st.session_state.ai_cache = {}

if "follow_school" not in st.session_state:
    st.session_state.follow_school = None


# =========================================================
# GENERAL HELPERS
# =========================================================

def json_safe(value):
    return json.loads(
        json.dumps(
            value,
            default=lambda x: (
                x.item()
                if hasattr(x, "item")
                else str(x)
            ),
        )
    )


def norm_col(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower(),
    )


def norm_name(value):
    text = str(value or "").strip()

    text = re.sub(
        r"^[\.\s]+",
        "",
        text,
    )

    text = re.sub(
        r"\b(mrs|ms|mr|miss|dr)\.?\b",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def teacher_key(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        norm_name(value).lower(),
    )


def first_col(df, candidates):

    lookup = {
        norm_col(c): c
        for c in df.columns
    }

    for candidate in candidates:

        key = norm_col(candidate)

        if key in lookup:
            return lookup[key]

    return None


# =========================================================
# SUPABASE
# =========================================================

@st.cache_resource
def get_supabase():

    url = st.secrets.get(
        "SUPABASE_URL",
        "",
    )

    key = st.secrets.get(
        "SUPABASE_SECRET_KEY",
        "",
    )

    if not url or not key:
        return None

    return create_client(
        url,
        key,
    )


sb = get_supabase()


def db_select(
    table,
    filters=None,
    order=None,
):

    if sb is None:
        return []

    query = (
        sb.table(table)
        .select("*")
    )

    for key, value in (
        filters or {}
    ).items():

        query = query.eq(
            key,
            value,
        )

    if order:
        query = query.order(order)

    response = query.execute()

    return response.data or []


def db_insert(
    table,
    payload,
):

    if sb is None:
        raise RuntimeError(
            "Supabase is not connected."
        )

    return (
        sb.table(table)
        .insert(json_safe(payload))
        .execute()
        .data
    )


def db_update(
    table,
    payload,
    row_id,
):

    if sb is None:
        raise RuntimeError(
            "Supabase is not connected."
        )

    return (
        sb.table(table)
        .update(json_safe(payload))
        .eq("id", row_id)
        .execute()
        .data
    )


def db_delete(
    table,
    row_id,
):

    if sb is None:
        raise RuntimeError(
            "Supabase is not connected."
        )

    return (
        sb.table(table)
        .delete()
        .eq("id", row_id)
        .execute()
        .data
    )


# =========================================================
# COMPANY USERMETRICS IMPORT
# =========================================================

def detect_schema(df):

    return {

        "school": first_col(
            df,
            [
                "School",
                "School Name",
                "Institution",
                "Center",
                "Centre",
                "Institution Name",
            ],
        ),

        "teacher": first_col(
            df,
            [
                "Teacher",
                "Teacher Name",
                "User Name",
                "Username",
                "Name",
            ],
        ),

        "first": first_col(
            df,
            [
                "FirstName",
                "First Name",
            ],
        ),

        "last": first_col(
            df,
            [
                "LastName",
                "Last Name",
            ],
        ),

        "minutes": first_col(
            df,
            [
                "Duration (Minutes)",
                "Duration Minutes",
                "Minutes",
                "Minutes Logged",
                "Usage Minutes",
                "Duration",
            ],
        ),

        "module": first_col(
            df,
            [
                "Type",
                "Module",
                "Module Name",
                "Category",
            ],
        ),

        "date": first_col(
            df,
            [
                "StartTime",
                "Start Time",
                "Date",
                "Activity Date",
                "Log Date",
            ],
        ),

        "grade": first_col(
            df,
            [
                "Grade",
                "Class",
            ],
        ),

        "subject": first_col(
            df,
            [
                "Subject",
            ],
        ),

        "book": first_col(
            df,
            [
                "Book",
                "Book Name",
                "Content",
                "Content Name",
            ],
        ),
    }


@st.cache_data(
    show_spinner=False
)
def parse_uploaded_file(
    file_bytes,
    filename,
):

    bio = io.BytesIO(
        file_bytes
    )

    if filename.lower().endswith(
        ".csv"
    ):

        df = pd.read_csv(bio)

    else:

        df = pd.read_excel(bio)

    schema = detect_schema(df)

    if not schema["school"]:

        raise ValueError(
            f"{filename}: "
            "School / Institution / Center "
            "column not found."
        )

    if not schema["minutes"]:

        raise ValueError(
            f"{filename}: "
            "Duration / Minutes column "
            "not found."
        )

    out = pd.DataFrame()

    out["School"] = (
        df[schema["school"]]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if schema["teacher"]:

        out["Teacher"] = (
            df[schema["teacher"]]
            .fillna("")
            .astype(str)
            .map(norm_name)
        )

    else:

        if schema["first"]:

            first = (
                df[schema["first"]]
                .fillna("")
                .astype(str)
            )

        else:

            first = pd.Series(
                [""] * len(df),
                index=df.index,
            )

        if schema["last"]:

            last = (
                df[schema["last"]]
                .fillna("")
                .astype(str)
            )

        else:

            last = pd.Series(
                [""] * len(df),
                index=df.index,
            )

        out["Teacher"] = (
            (first + " " + last)
            .str.strip()
            .map(norm_name)
        )

    out["Teacher"] = (
        out["Teacher"]
        .replace(
            "",
            "Unattributed Activity",
        )
    )

    out["Teacher Key"] = (
        out["Teacher"]
        .map(teacher_key)
    )

    out["Minutes"] = (
        pd.to_numeric(
            df[schema["minutes"]],
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
    )

    if schema["module"]:

        out["Raw Module"] = (
            df[schema["module"]]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    elif schema["book"]:

        out["Raw Module"] = "Book"

    else:

        out["Raw Module"] = "Other"

    if schema["date"]:

        out["DateTime"] = (
            pd.to_datetime(
                df[schema["date"]],
                errors="coerce",
            )
        )

    else:

        out["DateTime"] = pd.NaT

    if schema["grade"]:

        out["Grade"] = (
            df[schema["grade"]]
            .fillna("")
            .astype(str)
        )

    else:

        out["Grade"] = ""

    if schema["subject"]:

        out["Subject"] = (
            df[schema["subject"]]
            .fillna("")
            .astype(str)
        )

    else:

        out["Subject"] = ""

    if schema["book"]:

        out["Book"] = (
            df[schema["book"]]
            .fillna("")
            .astype(str)
        )

    else:

        out["Book"] = ""

    out["Source File"] = filename

    def classify_module(value):

        module = str(value).lower()

        if (
            "lessondelivery" in module
            or "lesson delivery" in module
            or "lesson" in module
        ):

            return "Lesson Delivery"

        if "library" in module:

            return "Library"

        return "Other Modules"

    out["KPI Module"] = (
        out["Raw Module"]
        .map(classify_module)
    )

    out = out[
        out["School"].str.strip() != ""
    ].copy()

    return out[USAGE_COLUMNS]


def combine_usage_files(files):

    frames = []
    errors = []

    for file in files[:100]:

        try:

            frames.append(
                parse_uploaded_file(
                    file.getvalue(),
                    file.name,
                )
            )

        except Exception as error:

            errors.append(
                str(error)
            )

    if not frames:

        return (
            pd.DataFrame(
                columns=USAGE_COLUMNS
            ),
            errors,
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    fingerprint_columns = [

        "School",
        "Teacher Key",
        "Raw Module",
        "Minutes",
        "DateTime",
        "Grade",
        "Subject",
        "Book",
    ]

    combined = (
        combined.drop_duplicates(
            subset=fingerprint_columns
        )
        .reset_index(drop=True)
    )

    return combined, errors


# =========================================================
# MASTER ROSTER
# =========================================================

def roster_dataframe():

    rows = db_select(
        "master_roster",
        {
            "active": True,
        },
    )

    if not rows:

        return pd.DataFrame(
            columns=[
                "school_name",
                "teacher_name",
                "teacher_key",
                "grade",
                "subject",
                "email",
                "phone",
            ]
        )

    return pd.DataFrame(rows)


def import_roster(files):

    incoming = []

    for file in files:

        bio = io.BytesIO(
            file.getvalue()
        )

        if file.name.lower().endswith(
            ".csv"
        ):

            df = pd.read_csv(bio)

        else:

            df = pd.read_excel(bio)

        school_col = first_col(
            df,
            [
                "School",
                "School Name",
                "Institution",
                "Center",
            ],
        )

        teacher_col = first_col(
            df,
            [
                "Teacher",
                "Teacher Name",
                "Name",
            ],
        )

        first_col_name = first_col(
            df,
            [
                "FirstName",
                "First Name",
            ],
        )

        last_col_name = first_col(
            df,
            [
                "LastName",
                "Last Name",
            ],
        )

        grade_col = first_col(
            df,
            [
                "Grade",
                "Class",
            ],
        )

        subject_col = first_col(
            df,
            ["Subject"],
        )

        email_col = first_col(
            df,
            [
                "Email",
                "Email ID",
            ],
        )

        phone_col = first_col(
            df,
            [
                "Phone",
                "Mobile",
                "Mobile Number",
            ],
        )

        if not school_col:

            raise ValueError(
                f"{file.name}: "
                "School column not found."
            )

        if (
            not teacher_col
            and not (
                first_col_name
                or last_col_name
            )
        ):

            raise ValueError(
                f"{file.name}: "
                "Teacher name column not found."
            )

        if teacher_col:

            names = (
                df[teacher_col]
                .fillna("")
                .astype(str)
                .map(norm_name)
            )

        else:

            if first_col_name:

                first = (
                    df[first_col_name]
                    .fillna("")
                    .astype(str)
                )

            else:

                first = pd.Series(
                    [""] * len(df),
                    index=df.index,
                )

            if last_col_name:

                last = (
                    df[last_col_name]
                    .fillna("")
                    .astype(str)
                )

            else:

                last = pd.Series(
                    [""] * len(df),
                    index=df.index,
                )

            names = (
                (first + " " + last)
                .str.strip()
                .map(norm_name)
            )

        for index in df.index:

            school = str(
                df.loc[
                    index,
                    school_col,
                ]
            ).strip()

            teacher = names.loc[index]

            if not school or not teacher:
                continue

            incoming.append(
                {
                    "school_name": school,
                    "teacher_name": teacher,
                    "teacher_key": teacher_key(
                        teacher
                    ),
                    "grade": (
                        str(
                            df.loc[
                                index,
                                grade_col,
                            ]
                        ).strip()
                        if grade_col
                        else None
                    ),
                    "subject": (
                        str(
                            df.loc[
                                index,
                                subject_col,
                            ]
                        ).strip()
                        if subject_col
                        else None
                    ),
                    "email": (
                        str(
                            df.loc[
                                index,
                                email_col,
                            ]
                        ).strip()
                        if email_col
                        else None
                    ),
                    "phone": (
                        str(
                            df.loc[
                                index,
                                phone_col,
                            ]
                        ).strip()
                        if phone_col
                        else None
                    ),
                    "active": True,
                }
            )

    existing = {

        (
            row["school_name"],
            row["teacher_key"],
        ): row

        for row in db_select(
            "master_roster"
        )
    }

    inserted = 0
    updated = 0

    for row in incoming:

        key = (
            row["school_name"],
            row["teacher_key"],
        )

        if key in existing:

            db_update(
                "master_roster",
                row,
                existing[key]["id"],
            )

            updated += 1

        else:

            db_insert(
                "master_roster",
                row,
            )

            inserted += 1

    return inserted, updated


# =========================================================
# SHARED ACCOUNTS
# =========================================================

def shared_account_set():

    rows = db_select(
        "shared_accounts"
    )

    return {

        (
            row["school_name"],
            row["account_key"],
        )

        for row in rows

        if row.get(
            "active",
            True,
        )
    }


def is_shared_account(
    school,
    teacher,
    shared_accounts,
):

    tkey = teacher_key(
        teacher
    )

    skey = teacher_key(
        school
    )

    if tkey in (
        "",
        "unattributedactivity",
    ):

        return True

    return (
        (
            school,
            tkey,
        ) in shared_accounts
        or (
            tkey
            and tkey == skey
        )
    )


# =========================================================
# KPI ENGINE
# =========================================================

DEFAULT_KPI = {

    "lessonDelivery": (
        "Lesson Delivery",
        10.0,
    ),

    "library": (
        "Library",
        30.0,
    ),

    "otherModules": (
        "Other Modules",
        15.0,
    ),
}


def load_kpi_rows():

    return db_select(
        "kpi_settings"
    )


def effective_kpis(
    school=None,
):

    rows = load_kpi_rows()

    result = {

        key: value[1]

        for key, value
        in DEFAULT_KPI.items()
    }

    globals_only = [

        row

        for row in rows

        if (
            row.get("scope")
            == "GLOBAL"
            and row.get(
                "active",
                True,
            )
        )
    ]

    for module_key in result:

        matches = [

            row

            for row in globals_only

            if (
                row.get(
                    "module_key"
                )
                == module_key
            )
        ]

        if matches:

            matches.sort(
                key=lambda x:
                (
                    x.get("updated_at")
                    or x.get("created_at")
                    or ""
                )
            )

            result[module_key] = float(
                matches[-1][
                    "target_minutes_per_day"
                ]
            )

    if school:

        school_rows = [

            row

            for row in rows

            if (
                row.get("scope")
                == "SCHOOL"
                and row.get(
                    "school_name"
                )
                == school
                and row.get(
                    "active",
                    True,
                )
            )
        ]

        for module_key in result:

            matches = [

                row

                for row in school_rows

                if (
                    row.get(
                        "module_key"
                    )
                    == module_key
                )
            ]

            if matches:

                matches.sort(
                    key=lambda x:
                    (
                        x.get("updated_at")
                        or x.get("created_at")
                        or ""
                    )
                )

                result[module_key] = float(
                    matches[-1][
                        "target_minutes_per_day"
                    ]
                )

    return result


def save_kpi(
    scope,
    module_key,
    value,
    school_name=None,
):

    rows = load_kpi_rows()

    existing = [

        row

        for row in rows

        if (
            row.get("scope")
            == scope
            and row.get(
                "module_key"
            )
            == module_key
            and (
                row.get(
                    "school_name"
                )
                or None
            )
            == (
                school_name
                or None
            )
        )
    ]

    payload = {

        "scope": scope,

        "school_name":
            school_name,

        "module_key":
            module_key,

        "module_name":
            DEFAULT_KPI[
                module_key
            ][0],

        "target_minutes_per_day":
            float(value),

        "active":
            True,

        "updated_at":
            datetime.utcnow()
            .isoformat(),
    }

    if existing:

        db_update(
            "kpi_settings",
            payload,
            existing[-1]["id"],
        )

    else:

        db_insert(
            "kpi_settings",
            payload,
        )


# =========================================================
# REVIEW PERIOD
# =========================================================

def working_days_between(
    start,
    end,
):

    dates = pd.date_range(
        start=start,
        end=end,
        freq="D",
    )

    return max(
        1,
        sum(
            day.weekday() != 6
            for day in dates
        ),
    )


def filter_period(
    df,
    start,
    end,
):

    if df.empty:

        return pd.DataFrame(
            columns=USAGE_COLUMNS
        )

    out = df.copy()

    if (
        "DateTime" in out
        and out["DateTime"]
        .notna()
        .any()
    ):

        activity_date = (
            out["DateTime"]
            .dt.date
        )

        out = out[
            (
                activity_date
                >= start
            )
            & (
                activity_date
                <= end
            )
        ]

    return out


# =========================================================
# ANALYTICS ENGINE
# =========================================================

def build_teacher_record(
    school,
    teacher,
    tkey,
    activity,
    workdays,
    kpis,
    rostered,
    grade="",
    subject="",
):

    if (
        activity is None
        or activity.empty
    ):

        lesson = 0
        library = 0
        other = 0
        total = 0
        active_days = 0
        books = 0
        grades = 0
        subjects = 0
        first_activity = None
        last_activity = None

    else:

        lesson = (
            activity.loc[
                activity[
                    "KPI Module"
                ]
                == "Lesson Delivery",
                "Minutes",
            ]
            .sum()
        )

        library = (
            activity.loc[
                activity[
                    "KPI Module"
                ]
                == "Library",
                "Minutes",
            ]
            .sum()
        )

        other = (
            activity.loc[
                activity[
                    "KPI Module"
                ]
                == "Other Modules",
                "Minutes",
            ]
            .sum()
        )

        total = (
            activity[
                "Minutes"
            ]
            .sum()
        )

        active_days = (
            activity[
                "DateTime"
            ]
            .dropna()
            .dt.date
            .nunique()
        )

        books = (
            activity.loc[
                activity["Book"]
                .str.strip()
                != "",
                "Book",
            ]
            .nunique()
        )

        grades = (
            activity.loc[
                activity["Grade"]
                .str.strip()
                != "",
                "Grade",
            ]
            .nunique()
        )

        subjects = (
            activity.loc[
                activity["Subject"]
                .str.strip()
                != "",
                "Subject",
            ]
            .nunique()
        )

        first_activity = (
            activity[
                "DateTime"
            ]
            .min()
        )

        last_activity = (
            activity[
                "DateTime"
            ]
            .max()
        )

    lesson_target = (
        workdays
        * kpis[
            "lessonDelivery"
        ]
    )

    library_target = (
        workdays
        * kpis[
            "library"
        ]
    )

    other_target = (
        workdays
        * kpis[
            "otherModules"
        ]
    )

    lesson_pct = (
        lesson
        / lesson_target
        * 100
        if lesson_target
        else 0
    )

    library_pct = (
        library
        / library_target
        * 100
        if library_target
        else 0
    )

    other_pct = (
        other
        / other_target
        * 100
        if other_target
        else 0
    )

    consistency = min(
        (
            active_days
            / max(
                workdays,
                1,
            )
        )
        * 100,
        100,
    )

    health = round(

        min(
            lesson_pct,
            100,
        )
        * .40

        + min(
            library_pct,
            100,
        )
        * .35

        + min(
            other_pct,
            100,
        )
        * .15

        + consistency
        * .10
    )

    if total == 0:

        status = (
            "Never Logged In"
            if rostered
            else "0 Usage"
        )

    elif (
        lesson_pct >= 100
        and library_pct >= 100
        and other_pct >= 100
    ):

        status = (
            "Meeting All KPIs"
        )

    elif (
        max(
            lesson_pct,
            library_pct,
            other_pct,
        )
        >= 100
    ):

        status = (
            "Partially Meeting"
        )

    else:

        status = "Below KPI"

    return {

        "School": school,
        "Teacher": teacher,
        "Teacher Key": tkey,
        "Rostered": rostered,
        "Status": status,

        "Lesson Delivery":
            round(
                float(lesson),
                1,
            ),

        "Lesson Target":
            round(
                float(
                    lesson_target
                ),
                1,
            ),

        "Lesson KPI %":
            round(
                float(
                    lesson_pct
                ),
                1,
            ),

        "Library":
            round(
                float(
                    library
                ),
                1,
            ),

        "Library Target":
            round(
                float(
                    library_target
                ),
                1,
            ),

        "Library KPI %":
            round(
                float(
                    library_pct
                ),
                1,
            ),

        "Other Modules":
            round(
                float(
                    other
                ),
                1,
            ),

        "Other Target":
            round(
                float(
                    other_target
                ),
                1,
            ),

        "Other KPI %":
            round(
                float(
                    other_pct
                ),
                1,
            ),

        "Total Minutes":
            round(
                float(
                    total
                ),
                1,
            ),

        "Active Days":
            int(
                active_days
            ),

        "Eligible Working Days":
            int(
                workdays
            ),

        "Books Used":
            int(
                books
            ),

        "Grades Covered":
            int(
                grades
            ),

        "Subjects Covered":
            int(
                subjects
            ),

        "First Activity":
            first_activity,

        "Last Activity":
            last_activity,

        "Grade":
            grade or "",

        "Subject":
            subject or "",

        "Health Score":
            int(
                max(
                    0,
                    min(
                        100,
                        health,
                    ),
                )
            ),
    }


def build_analytics(
    raw,
    start_date,
    end_date,
):

    workdays = (
        working_days_between(
            start_date,
            end_date,
        )
    )

    roster = (
        roster_dataframe()
    )

    shared = (
        shared_account_set()
    )

    usage = raw.copy()

    if usage.empty:

        personal = pd.DataFrame(
            columns=USAGE_COLUMNS
        )

        shared_usage = (
            pd.DataFrame(
                columns=USAGE_COLUMNS
            )
        )

    else:

        usage["Is Shared"] = (
            usage.apply(
                lambda row:
                is_shared_account(
                    row["School"],
                    row["Teacher"],
                    shared,
                ),
                axis=1,
            )
        )

        personal = (
            usage[
                ~usage["Is Shared"]
            ]
            .copy()
        )

        shared_usage = (
            usage[
                usage["Is Shared"]
            ]
            .copy()
        )

    usage_map = {}

    if not personal.empty:

        for (
            school,
            tkey,
        ), group in personal.groupby(
            [
                "School",
                "Teacher Key",
            ],
            dropna=False,
        ):

            usage_map[
                (
                    school,
                    tkey,
                )
            ] = group

    records = []
    seen = set()

    if not roster.empty:

        for _, row in (
            roster.iterrows()
        ):

            school = (
                row["school_name"]
            )

            tkey = (
                row["teacher_key"]
            )

            teacher = (
                row["teacher_name"]
            )

            seen.add(
                (
                    school,
                    tkey,
                )
            )

            activity = (
                usage_map.get(
                    (
                        school,
                        tkey,
                    ),
                    pd.DataFrame(
                        columns=USAGE_COLUMNS
                    ),
                )
            )

            records.append(
                build_teacher_record(
                    school,
                    teacher,
                    tkey,
                    activity,
                    workdays,
                    effective_kpis(
                        school
                    ),
                    True,
                    row.get(
                        "grade"
                    ),
                    row.get(
                        "subject"
                    ),
                )
            )

    for (
        school,
        tkey,
    ), activity in (
        usage_map.items()
    ):

        if (
            school,
            tkey,
        ) in seen:

            continue

        teacher = (
            activity[
                "Teacher"
            ]
            .iloc[0]
        )

        records.append(
            build_teacher_record(
                school,
                teacher,
                tkey,
                activity,
                workdays,
                effective_kpis(
                    school
                ),
                False,
            )
        )

    teacher_df = pd.DataFrame(
        records
    )

    if teacher_df.empty:

    teacher_df = pd.DataFrame(
        columns=[
            "School",
            "Teacher",
            "Teacher Key",
            "Rostered",
            "Status",
            "Lesson Delivery",
            "Lesson Target",
            "Lesson KPI %",
            "Library",
            "Library Target",
            "Library KPI %",
            "Other Modules",
            "Other Target",
            "Other KPI %",
            "Total Minutes",
            "Active Days",
            "Eligible Working Days",
            "Books Used",
            "Grades Covered",
            "Subjects Covered",
            "First Activity",
            "Last Activity",
            "Grade",
            "Subject",
            "Health Score",
        ]
    )

    school_df = pd.DataFrame(
        columns=[
            "School",
            "Teachers",
            "Active",
            "Inactive / Never Logged In",
            "Met All KPIs",
            "Overall Compliance %",
            "Health Score",
            "Lesson Delivery Minutes",
            "Library Minutes",
            "Other Modules Minutes",
            "Lesson Target / Day",
            "Library Target / Day",
            "Other Target / Day",
        ]
    )

    return (
        teacher_df,
        school_df,
        shared_usage,
    )

    schools = []

    for school, group in (
        teacher_df.groupby(
            "School"
        )
    ):

        total = len(group)

        active = int(
            (
                group[
                    "Total Minutes"
                ]
                > 0
            ).sum()
        )

        inactive = int(
            (
                group["Status"]
                .isin(
                    [
                        "Never Logged In",
                        "0 Usage",
                    ]
                )
            ).sum()
        )

        fully_met = int(
            (
                (
                    group[
                        "Lesson KPI %"
                    ]
                    >= 100
                )
                & (
                    group[
                        "Library KPI %"
                    ]
                    >= 100
                )
                & (
                    group[
                        "Other KPI %"
                    ]
                    >= 100
                )
            ).sum()
        )

        health = (
            int(
                round(
                    group[
                        "Health Score"
                    ]
                    .mean()
                )
            )
            if total
            else 0
        )

        kpis = effective_kpis(
            school
        )

        schools.append(
            {

                "School":
                    school,

                "Teachers":
                    total,

                "Active":
                    active,

                "Inactive / Never Logged In":
                    inactive,

                "Met All KPIs":
                    fully_met,

                "Overall Compliance %":
                    round(
                        (
                            fully_met
                            / total
                            * 100
                        )
                        if total
                        else 0,
                        1,
                    ),

                "Health Score":
                    health,

                "Lesson Delivery Minutes":
                    round(
                        group[
                            "Lesson Delivery"
                        ]
                        .sum(),
                        1,
                    ),

                "Library Minutes":
                    round(
                        group[
                            "Library"
                        ]
                        .sum(),
                        1,
                    ),

                "Other Modules Minutes":
                    round(
                        group[
                            "Other Modules"
                        ]
                        .sum(),
                        1,
                    ),

                "Lesson Target / Day":
                    kpis[
                        "lessonDelivery"
                    ],

                "Library Target / Day":
                    kpis[
                        "library"
                    ],

                "Other Target / Day":
                    kpis[
                        "otherModules"
                    ],
            }
        )

    school_df = (
        pd.DataFrame(
            schools
        )
        .sort_values(
            [
                "Health Score",
                "Overall Compliance %",
            ],
            ascending=False,
        )
    )

    return (
        teacher_df,
        school_df,
        shared_usage,
    )


# =========================================================
# GEMINI AI
# =========================================================

@st.cache_resource
def get_ai_client():

    key = st.secrets.get(
        "GEMINI_API_KEY",
        "",
    )

    if (
        not key
        or genai is None
    ):

        return None

    return genai.Client(
        api_key=key
    )


def ai_generate(
    prompt,
):

    cache_key = (
        hashlib.sha256(
            prompt.encode()
        )
        .hexdigest()
    )

    if (
        cache_key
        in st.session_state.ai_cache
    ):

        return (
            st.session_state.ai_cache[
                cache_key
            ]
        )

    client = get_ai_client()

    if client is None:

        return (
            "AI is not configured. "
            "Check GEMINI_API_KEY "
            "in Streamlit Secrets."
        )

    model = st.secrets.get(
        "GEMINI_MODEL",
        "gemini-2.5-flash",
    )

    response = (
        client.models.generate_content(
            model=model,
            contents=prompt,
        )
    )

    text = (
        response.text
        or ""
    )

    text = (
        text.replace(
            "**",
            "",
        )
        .replace(
            "###",
            "",
        )
        .replace(
            "##",
            "",
        )
        .strip()
    )

    st.session_state.ai_cache[
        cache_key
    ] = text

    return text


# =========================================================
# AI PROMPTS
# =========================================================

def school_ai_prompt(
    school_row,
    teacher_data,
    raw_school,
    action_days,
):

    lowest = (
        teacher_data
        .sort_values(
            [
                "Health Score",
                "Total Minutes",
            ]
        )
        .head(6)
    )

    highest = (
        teacher_data
        .sort_values(
            [
                "Health Score",
                "Total Minutes",
            ],
            ascending=False,
        )
        .head(5)
    )

    module_breakdown = {}

    if not raw_school.empty:

        module_breakdown = (
            raw_school
            .groupby(
                "Raw Module"
            )["Minutes"]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(15)
            .round(1)
            .to_dict()
        )

    facts = {

        "school":
            json_safe(
                school_row.to_dict()
            ),

        "lowest_teachers":
            json_safe(
                lowest[
                    [
                        "Teacher",
                        "Status",
                        "Health Score",
                        "Lesson KPI %",
                        "Library KPI %",
                        "Other KPI %",
                        "Total Minutes",
                        "Active Days",
                    ]
                ]
                .to_dict(
                    "records"
                )
            ),

        "top_teachers":
            json_safe(
                highest[
                    [
                        "Teacher",
                        "Status",
                        "Health Score",
                        "Lesson KPI %",
                        "Library KPI %",
                        "Other KPI %",
                        "Total Minutes",
                        "Active Days",
                    ]
                ]
                .to_dict(
                    "records"
                )
            ),

        "module_breakdown_minutes":
            json_safe(
                module_breakdown
            ),

        "action_plan_days":
            action_days,
    }

    return f"""
You are AcadIntel 360, an academic implementation intelligence analyst.

Use ONLY the VERIFIED FACTS below.

Never invent:
numbers,
causes,
commitments,
dates,
teacher behaviour,
infrastructure problems,
or explanations not established by the data.

Whenever the data does not establish a cause, explicitly state:
"The data does not establish the cause."

Create a polished school-management intelligence report.

Use these exact plain-text headings:

EXECUTIVE SUMMARY
PERFORMANCE DIAGNOSIS
STRENGTHS
CRITICAL GAPS
TEACHERS REQUIRING ATTENTION
TOP PERFORMERS
MODULE ADOPTION PATTERN
{action_days}-DAY ACTION PLAN
NEXT REVIEW TARGETS
EVIDENCE

Tone:
professional,
constructive,
respectful,
management-ready,
concise but insightful.

Do not use markdown asterisks or hash symbols.

VERIFIED FACTS:

{json.dumps(facts, default=str)}
"""


def teacher_ai_prompt(
    teacher_row,
    evidence,
    action_days,
):

    if evidence.empty:

        recent_evidence = []

    else:

        recent_evidence = (
            evidence[
                [
                    "DateTime",
                    "Raw Module",
                    "KPI Module",
                    "Grade",
                    "Subject",
                    "Book",
                    "Minutes",
                ]
            ]
            .sort_values(
                "DateTime",
                ascending=False,
            )
            .head(100)
            .astype(str)
            .to_dict(
                "records"
            )
        )

    facts = {

        "teacher":
            json_safe(
                teacher_row.to_dict()
            ),

        "recent_activity_evidence":
            recent_evidence,

        "action_plan_days":
            action_days,
    }

    return f"""
You are AcadIntel 360.

Create a detailed evidence-backed Teacher 360 implementation report.

Use ONLY the verified facts below.

Never invent causes or behaviour that the data does not establish.

Use these exact headings:

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

Tone:
polite,
developmental,
motivational,
specific,
professional.

The {action_days}-day action plan must contain measurable steps.

If usage is zero, state it clearly but respectfully.

Do not use markdown asterisks or hash symbols.

VERIFIED FACTS:

{json.dumps(facts, default=str)}
"""


def call_script_prompt(
    school_row,
    teacher_data,
    previous_followups,
):

    priorities = (
        teacher_data
        .sort_values(
            [
                "Health Score",
                "Total Minutes",
            ]
        )
        .head(6)
    )

    facts = {

        "school":
            json_safe(
                school_row.to_dict()
            ),

        "priority_teachers":
            json_safe(
                priorities[
                    [
                        "Teacher",
                        "Status",
                        "Lesson KPI %",
                        "Library KPI %",
                        "Other KPI %",
                        "Health Score",
                        "Total Minutes",
                    ]
                ]
                .to_dict(
                    "records"
                )
            ),

        "previous_followups":
            json_safe(
                previous_followups[
                    :5
                ]
            ),
    }

    return f"""
You are preparing an Academic Consultant to call a school KDM such as Principal, Director or Coordinator.

Use only the verified facts below.

Do not invent reasons or commitments.

Create a natural speaking script with these headings:

OPENING
POSITIVE START
DATA-BACKED PERFORMANCE UPDATE
KEY QUESTIONS TO ASK
PRIORITY TEACHERS / GAPS
DESIRED COMMITMENT
CLOSING
POST-CALL FOLLOW-UP NOTE

The language should sound natural when spoken aloud.

It must be respectful, professional and firm where required.

Do not use markdown asterisks.

VERIFIED FACTS:

{json.dumps(facts, default=str)}
"""


def whatsapp_prompt(
    school_row,
    teacher_data,
    contact_name,
):

    priorities = (
        teacher_data
        .sort_values(
            [
                "Health Score",
                "Total Minutes",
            ]
        )
        .head(5)
    )

    facts = {

        "school":
            json_safe(
                school_row.to_dict()
            ),

        "teachers_requiring_attention":
            json_safe(
                priorities[
                    [
                        "Teacher",
                        "Status",
                        "Lesson KPI %",
                        "Library KPI %",
                        "Other KPI %",
                        "Total Minutes",
                    ]
                ]
                .to_dict(
                    "records"
                )
            ),

        "contact_name":
            contact_name,
    }

    return f"""
Draft a concise WhatsApp performance update to school management.

Use only the verified facts below.

Do not use markdown asterisks.

Use tasteful Unicode icons.

Tone:
professional,
constructive,
respectful.

Include:
review summary,
KPI position,
teachers requiring attention,
specific next action,
polite closing.

VERIFIED FACTS:

{json.dumps(facts, default=str)}
"""


# =========================================================
# BEAUTIFUL AI RESPONSE CARDS
# =========================================================

AI_STYLES = {

    "EXECUTIVE":
        (
            "#eef2ff",
            "#4338ca",
            "🧠",
        ),

    "PERFORMANCE":
        (
            "#eff6ff",
            "#1d4ed8",
            "📊",
        ),

    "STRENGTH":
        (
            "#ecfdf5",
            "#047857",
            "✅",
        ),

    "CRITICAL":
        (
            "#fff7ed",
            "#c2410c",
            "⚠️",
        ),

    "GAP":
        (
            "#fff7ed",
            "#c2410c",
            "⚠️",
        ),

    "ACTION":
        (
            "#f5f3ff",
            "#7c3aed",
            "🎯",
        ),

    "TARGET":
        (
            "#ecfeff",
            "#0e7490",
            "📌",
        ),

    "EVIDENCE":
        (
            "#f8fafc",
            "#334155",
            "🔎",
        ),

    "MOTIVATIONAL":
        (
            "#f0fdf4",
            "#15803d",
            "🌱",
        ),

    "CONTENT":
        (
            "#fdf4ff",
            "#a21caf",
            "📚",
        ),

    "ACTIVITY":
        (
            "#f0f9ff",
            "#0369a1",
            "📅",
        ),

    "TOP":
        (
            "#fffbeb",
            "#b45309",
            "🌟",
        ),

    "MODULE":
        (
            "#f5f3ff",
            "#6d28d9",
            "🧩",
        ),
}


def render_ai(text):

    clean = (
        (text or "")
        .replace(
            "**",
            "",
        )
        .replace(
            "#",
            "",
        )
        .strip()
    )

    if not clean:
        return

    headings = [

        "EXECUTIVE SUMMARY",
        "EXECUTIVE DIAGNOSIS",
        "PERFORMANCE DIAGNOSIS",
        "KPI SCORECARD INTERPRETATION",
        "ACTIVITY CONSISTENCY",
        "CONTENT AND CURRICULUM ENGAGEMENT",
        "STRENGTHS",
        "CRITICAL GAPS",
        "IMPLEMENTATION GAPS",
        "TEACHERS REQUIRING ATTENTION",
        "TOP PERFORMERS",
        "MODULE ADOPTION PATTERN",
        "7-DAY ACTION PLAN",
        "15-DAY ACTION PLAN",
        "7-DAY DEVELOPMENT ACTION PLAN",
        "15-DAY DEVELOPMENT ACTION PLAN",
        "NEXT REVIEW TARGETS",
        "NEXT REVIEW TARGET",
        "MOTIVATIONAL CLOSING",
        "EVIDENCE",
        "OPENING",
        "POSITIVE START",
        "DATA-BACKED PERFORMANCE UPDATE",
        "KEY QUESTIONS TO ASK",
        "PRIORITY TEACHERS / GAPS",
        "DESIRED COMMITMENT",
        "CLOSING",
        "POST-CALL FOLLOW-UP NOTE",
        "FACTS",
        "INTERPRETATION",
        "RECOMMENDED ACTIONS",
    ]

    pattern = re.compile(

        r"(?im)^("

        + "|".join(
            re.escape(h)
            for h in headings
        )

        + r")\s*:?\s*$"
    )

    matches = list(
        pattern.finditer(
            clean
        )
    )

    if not matches:

        st.markdown(
            f"""
            <div class="ai-card"
            style="background:#f8fafc;">
            {clean.replace(chr(10), "<br>")}
            </div>
            """,
            unsafe_allow_html=True,
        )

        return

    for index, match in (
        enumerate(matches)
    ):

        title = (
            match.group(1)
            .strip()
            .upper()
        )

        start = match.end()

        end = (
            matches[
                index + 1
            ].start()
            if index + 1
            < len(matches)
            else len(clean)
        )

        body = (
            clean[
                start:end
            ]
            .strip()
        )

        if not body:
            continue

        style_key = next(
            (
                key
                for key
                in AI_STYLES
                if key in title
            ),
            "EXECUTIVE",
        )

        bg, colour, icon = (
            AI_STYLES[
                style_key
            ]
        )

        body_html = (
            body.replace(
                "\n",
                "<br>",
            )
        )

        st.markdown(
            f"""
            <div class="ai-card"
            style="
            background:{bg};
            border-left:5px solid {colour};
            ">
                <div
                class="ai-title"
                style="color:{colour}">
                {icon} {title.title()}
                </div>

                <div>
                {body_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# PDF REPORTS
# =========================================================

class ReportPDF(FPDF):

    def header(self):

        self.set_fill_color(
            67,
            56,
            202,
        )

        self.rect(
            0,
            0,
            210,
            25,
            "F",
        )

        self.set_text_color(
            255,
            255,
            255,
        )

        self.set_font(
            "Helvetica",
            "B",
            16,
        )

        self.set_xy(
            12,
            8,
        )

        self.cell(
            0,
            8,
            "AcadIntel 360",
            0,
            1,
        )

        self.set_font(
            "Helvetica",
            "",
            8,
        )

        self.set_x(12)

        self.cell(
            0,
            4,
            "Academic Intelligence • Evidence • Action",
            0,
            1,
        )

        self.set_text_color(
            20,
            20,
            20,
        )

    def footer(self):

        self.set_y(-12)

        self.set_text_color(
            110,
            110,
            110,
        )

        self.set_font(
            "Helvetica",
            "",
            8,
        )

        self.cell(
            0,
            6,
            (
                f"Generated "
                f"{datetime.now().strftime('%d %b %Y %H:%M')}"
                f" • Page {self.page_no()}"
            ),
            0,
            0,
            "C",
        )


def safe_pdf_text(value):

    return (
        str(value)
        .encode(
            "latin-1",
            "replace",
        )
        .decode(
            "latin-1"
        )
    )


def split_ai_sections(text):

    clean = (
        (text or "")
        .replace(
            "**",
            "",
        )
        .replace(
            "#",
            "",
        )
        .strip()
    )

    lines = clean.splitlines()

    result = []

    current = None
    buffer = []

    for line in lines:

        stripped = (
            line.strip()
        )

        is_heading = (
            stripped.isupper()
            and 2
            <= len(stripped)
            <= 70
        )

        if is_heading:

            if (
                current
                and buffer
            ):

                result.append(
                    (
                        current,
                        "\n".join(
                            buffer
                        ).strip(),
                    )
                )

            current = stripped
            buffer = []

        else:

            buffer.append(
                line
            )

    if (
        current
        and buffer
    ):

        result.append(
            (
                current,
                "\n".join(
                    buffer
                ).strip(),
            )
        )

    if (
        not result
        and clean
    ):

        result = [
            (
                "ACADEMIC INTELLIGENCE",
                clean,
            )
        ]

    return result


def pdf_section(
    pdf,
    title,
    body,
):

    pdf.set_fill_color(
        242,
        244,
        255,
    )

    pdf.set_font(
        "Helvetica",
        "B",
        11,
    )

    pdf.multi_cell(
        0,
        7,
        safe_pdf_text(
            title
        ),
        fill=True,
    )

    pdf.set_font(
        "Helvetica",
        "",
        9,
    )

    for paragraph in (
        str(body)
        .split("\n")
    ):

        if paragraph.strip():

            pdf.multi_cell(
                0,
                5.5,
                safe_pdf_text(
                    paragraph.strip()
                ),
            )

    pdf.ln(2)


def make_school_pdf(
    school_row,
    teacher_data,
    ai_text,
    start_date,
    end_date,
):

    pdf = ReportPDF()

    pdf.set_auto_page_break(
        True,
        15,
    )

    pdf.add_page()

    pdf.ln(18)

    pdf.set_font(
        "Helvetica",
        "B",
        17,
    )

    pdf.multi_cell(
        0,
        8,
        safe_pdf_text(
            "School 360 Intelligence Report\n"
            + school_row["School"]
        ),
    )

    pdf.set_font(
        "Helvetica",
        "",
        9,
    )

    pdf.cell(
        0,
        6,
        (
            f"Review Period: "
            f"{start_date} to {end_date}"
        ),
        0,
        1,
    )

    pdf.ln(3)

    metrics = [

        (
            "Health Score",
            f"{school_row['Health Score']}/100",
        ),

        (
            "Overall Compliance",
            f"{school_row['Overall Compliance %']}%",
        ),

        (
            "Teachers",
            school_row["Teachers"],
        ),

        (
            "Active Teachers",
            school_row["Active"],
        ),

        (
            "Inactive / Never Logged In",
            school_row[
                "Inactive / Never Logged In"
            ],
        ),

        (
            "Met All KPIs",
            school_row[
                "Met All KPIs"
            ],
        ),
    ]

    for key, value in metrics:

        pdf.set_fill_color(
            239,
            242,
            255,
        )

        pdf.set_font(
            "Helvetica",
            "B",
            9,
        )

        pdf.cell(
            70,
            7,
            safe_pdf_text(key),
            1,
            0,
            "L",
            True,
        )

        pdf.set_font(
            "Helvetica",
            "",
            9,
        )

        pdf.cell(
            0,
            7,
            safe_pdf_text(value),
            1,
            1,
        )

    pdf.ln(4)

    pdf.set_font(
        "Helvetica",
        "B",
        11,
    )

    pdf.cell(
        0,
        7,
        "Teacher Performance Snapshot",
        0,
        1,
    )

    headers = [

        "Teacher",
        "Status",
        "Health",
        "Lesson %",
        "Library %",
        "Other %",
    ]

    widths = [
        55,
        38,
        18,
        22,
        22,
        22,
    ]

    pdf.set_font(
        "Helvetica",
        "B",
        7.5,
    )

    for header, width in zip(
        headers,
        widths,
    ):

        pdf.cell(
            width,
            6,
            header,
            1,
            0,
            "C",
        )

    pdf.ln()

    pdf.set_font(
        "Helvetica",
        "",
        7,
    )

    snapshot = (
        teacher_data
        .sort_values(
            [
                "Health Score",
                "Total Minutes",
            ]
        )
        .head(30)
    )

    for _, row in (
        snapshot.iterrows()
    ):

        values = [

            row["Teacher"],
            row["Status"],
            row["Health Score"],
            row["Lesson KPI %"],
            row["Library KPI %"],
            row["Other KPI %"],
        ]

        for value, width in zip(
            values,
            widths,
        ):

            pdf.cell(
                width,
                6,
                safe_pdf_text(
                    value
                )[:28],
                1,
                0,
                "L",
            )

        pdf.ln()

    pdf.ln(4)

    for title, body in (
        split_ai_sections(
            ai_text
        )
    ):

        pdf_section(
            pdf,
            title,
            body,
        )

    return bytes(
        pdf.output()
    )


def make_teacher_pdf(
    teacher_row,
    evidence,
    ai_text,
    start_date,
    end_date,
):

    pdf = ReportPDF()

    pdf.set_auto_page_break(
        True,
        15,
    )

    pdf.add_page()

    pdf.ln(18)

    pdf.set_font(
        "Helvetica",
        "B",
        17,
    )

    pdf.multi_cell(
        0,
        8,
        safe_pdf_text(
            "Teacher 360 Intelligence Report\n"
            + teacher_row["Teacher"]
        ),
    )

    pdf.set_font(
        "Helvetica",
        "",
        9,
    )

    pdf.cell(
        0,
        6,
        safe_pdf_text(
            "School: "
            + teacher_row["School"]
        ),
        0,
        1,
    )

    pdf.cell(
        0,
        6,
        (
            f"Review Period: "
            f"{start_date} to {end_date}"
        ),
        0,
        1,
    )

    pdf.ln(3)

    metrics = [

        (
            "Health Score",
            f"{teacher_row['Health Score']}/100",
        ),

        (
            "Status",
            teacher_row["Status"],
        ),

        (
            "Lesson Delivery",
            (
                f"{teacher_row['Lesson Delivery']} / "
                f"{teacher_row['Lesson Target']} min "
                f"({teacher_row['Lesson KPI %']}%)"
            ),
        ),

        (
            "Library",
            (
                f"{teacher_row['Library']} / "
                f"{teacher_row['Library Target']} min "
                f"({teacher_row['Library KPI %']}%)"
            ),
        ),

        (
            "Other Modules",
            (
                f"{teacher_row['Other Modules']} / "
                f"{teacher_row['Other Target']} min "
                f"({teacher_row['Other KPI %']}%)"
            ),
        ),

        (
            "Active Days",
            (
                f"{teacher_row['Active Days']} / "
                f"{teacher_row['Eligible Working Days']}"
            ),
        ),

        (
            "Books Used",
            teacher_row["Books Used"],
        ),

        (
            "Grades Covered",
            teacher_row["Grades Covered"],
        ),

        (
            "Subjects Covered",
            teacher_row["Subjects Covered"],
        ),
    ]

    for key, value in metrics:

        pdf.set_fill_color(
            239,
            242,
            255,
        )

        pdf.set_font(
            "Helvetica",
            "B",
            9,
        )

        pdf.cell(
            60,
            7,
            safe_pdf_text(key),
            1,
            0,
            "L",
            True,
        )

        pdf.set_font(
            "Helvetica",
            "",
            9,
        )

        pdf.cell(
            0,
            7,
            safe_pdf_text(value),
            1,
            1,
        )

    pdf.ln(4)

    for title, body in (
        split_ai_sections(
            ai_text
        )
    ):

        pdf_section(
            pdf,
            title,
            body,
        )

    pdf.add_page()

    pdf.ln(18)

    pdf.set_font(
        "Helvetica",
        "B",
        12,
    )

    pdf.cell(
        0,
        7,
        "Evidence Audit Trail",
        0,
        1,
    )

    headers = [
        "Date",
        "Module",
        "Grade",
        "Subject",
        "Book",
        "Min",
    ]

    widths = [
        25,
        32,
        20,
        32,
        67,
        14,
    ]

    pdf.set_font(
        "Helvetica",
        "B",
        7,
    )

    for header, width in zip(
        headers,
        widths,
    ):

        pdf.cell(
            width,
            6,
            header,
            1,
            0,
            "C",
        )

    pdf.ln()

    pdf.set_font(
        "Helvetica",
        "",
        6.5,
    )

    if evidence.empty:

        pdf.cell(
            0,
            7,
            "No activity evidence available.",
            1,
            1,
        )

    else:

        evidence = (
            evidence
            .sort_values(
                "DateTime",
                ascending=False,
            )
            .head(150)
        )

        for _, row in (
            evidence.iterrows()
        ):

            values = [

                str(
                    row["DateTime"]
                )[:16],

                str(
                    row["Raw Module"]
                )[:18],

                str(
                    row["Grade"]
                )[:12],

                str(
                    row["Subject"]
                )[:18],

                str(
                    row["Book"]
                )[:38],

                f"{float(row['Minutes']):.1f}",
            ]

            for value, width in zip(
                values,
                widths,
            ):

                pdf.cell(
                    width,
                    5.5,
                    safe_pdf_text(
                        value
                    ),
                    1,
                    0,
                    "L",
                )

            pdf.ln()

    return bytes(
        pdf.output()
    )


# =========================================================
# FOLLOW-UP MODAL
# =========================================================

@st.dialog(
    "Add / Update Follow-Up"
)
def followup_dialog(
    school_name,
):

    with st.form(
        "followup_form",
        clear_on_submit=True,
    ):

        followup_date = (
            st.date_input(
                "Follow-up date",
                value=(
                    date.today()
                    + timedelta(
                        days=7
                    )
                ),
            )
        )

        issue = st.text_area(
            "Specific issue / gap identified"
        )

        commitment = (
            st.text_area(
                "Last commitment made"
            )
        )

        status = st.selectbox(
            "Status",
            [
                "Open",
                "In Progress",
                "Resolved",
            ],
        )

        remarks = st.text_area(
            "Remarks (optional)"
        )

        submitted = (
            st.form_submit_button(
                "Save Follow-Up",
                use_container_width=True,
            )
        )

        if submitted:

            if not issue.strip():

                st.error(
                    "Please enter the issue/gap."
                )

                return

            db_insert(
                "followups",
                {
                    "school_name":
                        school_name,

                    "followup_date":
                        str(
                            followup_date
                        ),

                    "issue":
                        issue.strip(),

                    "last_commitment":
                        commitment.strip(),

                    "status":
                        status,

                    "remarks":
                        remarks.strip(),
                },
            )

            st.session_state.follow_school = None

            st.success(
                "Follow-up saved."
            )

            st.rerun()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title(
        "🎓 AcadIntel 360"
    )

    st.caption(
        "Academic Intelligence • Evidence • Action"
    )

    uploaded_files = (
        st.file_uploader(
            "Upload raw UserMetrics files",
            type=[
                "csv",
                "xlsx",
            ],
            accept_multiple_files=True,
            help=(
                "Upload up to "
                "100 CSV/XLSX files together."
            ),
        )
    )

    if uploaded_files:

        if len(
            uploaded_files
        ) > 100:

            st.error(
                "Maximum 100 files at a time."
            )

        elif st.button(
            "⚡ Process Raw Data",
            use_container_width=True,
        ):

            with st.spinner(
                "Combining, validating and de-duplicating..."
            ):

                data, errors = (
                    combine_usage_files(
                        uploaded_files
                    )
                )

                st.session_state.raw = data

                st.session_state.import_errors = errors

            if not data.empty:

                st.success(
                    f"{len(data):,} unique activity rows loaded "
                    f"from {len(uploaded_files)} file(s)."
                )

            for error in errors[:10]:

                st.warning(error)

    raw = st.session_state.raw

    if (
        not raw.empty
        and raw[
            "DateTime"
        ]
        .notna()
        .any()
    ):

        minimum_date = (
            raw[
                "DateTime"
            ]
            .min()
            .date()
        )

        maximum_date = (
            raw[
                "DateTime"
            ]
            .max()
            .date()
        )

    else:

        minimum_date = (
            date.today()
            - timedelta(
                days=7
            )
        )

        maximum_date = (
            date.today()
        )

    st.divider()

    st.caption(
        "Review Period"
    )

    start_date = (
        st.date_input(
            "From",
            value=minimum_date,
        )
    )

    end_date = (
        st.date_input(
            "To",
            value=maximum_date,
            min_value=start_date,
        )
    )

    workdays = (
        working_days_between(
            start_date,
            end_date,
        )
    )

    st.caption(
        f"{workdays} working day(s), "
        "Sundays excluded"
    )

    page = st.radio(
        "Navigate",
        [
            "Command Center",
            "School 360",
            "Teacher 360",
            "Follow-Ups",
            "KPI & Roster",
            "Ask AcadIntel",
        ],
    )


# =========================================================
# BUILD CURRENT ANALYTICS
# =========================================================

period_raw = filter_period(
    st.session_state.raw,
    start_date,
    end_date,
)

teachers, schools, shared_usage = (
    build_analytics(
        period_raw,
        start_date,
        end_date,
    )
)


# =========================================================
# EMPTY STATE
# =========================================================

if (
    st.session_state.raw.empty
    and roster_dataframe().empty
):

    st.markdown(
        """
        <div class="hero">
            <h1>AcadIntel 360</h1>
            <p>
            Upload UserMetrics files and an optional Master Roster
            to generate evidence-backed School 360 and Teacher 360 intelligence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Start by uploading your raw UserMetrics files "
        "from the left sidebar."
    )

    st.stop()


# =========================================================
# COMMAND CENTER
# =========================================================

if page == "Command Center":

    st.markdown(
        """
        <div class="hero">
            <h1>Academic Command Center</h1>
            <p>
            What requires attention, why it matters,
            and what should happen next.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if teachers.empty:

        st.warning(
            "No teacher analytics are available "
            "for this review period."
        )

    else:

        c1, c2, c3, c4, c5 = (
            st.columns(5)
        )

        c1.metric(
            "Schools",
            len(schools),
        )

        c2.metric(
            "Teachers",
            len(teachers),
        )

        c3.metric(
            "Never Logged In",
            int(
                (
                    teachers["Status"]
                    == "Never Logged In"
                ).sum()
            ),
        )

        c4.metric(
            "Meeting All KPIs",
            int(
                (
                    teachers["Status"]
                    == "Meeting All KPIs"
                ).sum()
            ),
        )

        c5.metric(
            "Average Health",
            f"{teachers['Health Score'].mean():.0f}/100",
        )

        st.subheader(
            "📊 School Health"
        )

        school_chart = (
            px.bar(
                schools.sort_values(
                    "Health Score"
                ),
                x="Health Score",
                y="School",
                orientation="h",
                color="Health Score",
                color_continuous_scale=[
                    "#ef4444",
                    "#f59e0b",
                    "#10b981",
                ],
                range_color=[
                    0,
                    100,
                ],
                text="Health Score",
            )
        )

        school_chart.update_layout(
            height=max(
                420,
                len(schools)
                * 36,
            ),
            coloraxis_showscale=False,
        )

        st.plotly_chart(
            school_chart,
            use_container_width=True,
        )

        left, right = (
            st.columns(
                [
                    1.2,
                    1,
                ]
            )
        )

        with left:

            st.subheader(
                "🚨 Priority Teachers"
            )

            priority = (
                teachers
                .sort_values(
                    [
                        "Health Score",
                        "Total Minutes",
                    ]
                )
                .head(15)
            )

            st.dataframe(
                priority[
                    [
                        "School",
                        "Teacher",
                        "Status",
                        "Health Score",
                        "Lesson KPI %",
                        "Library KPI %",
                        "Other KPI %",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        with right:

            st.subheader(
                "🧩 Module Mix"
            )

            if not period_raw.empty:

                mix = (
                    period_raw
                    .groupby(
                        "KPI Module"
                    )["Minutes"]
                    .sum()
                    .reset_index()
                )

                mix_chart = (
                    px.pie(
                        mix,
                        names="KPI Module",
                        values="Minutes",
                        hole=.55,
                        color="KPI Module",
                        color_discrete_map={
                            "Lesson Delivery":
                                "#4f46e5",
                            "Library":
                                "#0ea5e9",
                            "Other Modules":
                                "#10b981",
                        },
                    )
                )

                st.plotly_chart(
                    mix_chart,
                    use_container_width=True,
                )

        st.subheader(
            "📅 Follow-Up Pulse"
        )

        followup_rows = (
            db_select(
                "followups"
            )
        )

        if followup_rows:

            followups = pd.DataFrame(
                followup_rows
            )

            followups[
                "followup_date"
            ] = (
                pd.to_datetime(
                    followups[
                        "followup_date"
                    ]
                )
                .dt.date
            )

            today = date.today()

            due = followups[
                (
                    followups[
                        "followup_date"
                    ]
                    == today
                )
                & (
                    followups["status"]
                    != "Resolved"
                )
            ]

            overdue = followups[
                (
                    followups[
                        "followup_date"
                    ]
                    < today
                )
                & (
                    followups["status"]
                    != "Resolved"
                )
            ]

            upcoming = followups[
                (
                    followups[
                        "followup_date"
                    ]
                    > today
                )
                & (
                    followups["status"]
                    != "Resolved"
                )
            ]

            a, b, c = st.columns(3)

            a.metric(
                "Due Today",
                len(due),
            )

            b.metric(
                "Overdue",
                len(overdue),
            )

            c.metric(
                "Upcoming",
                len(upcoming),
            )

        else:

            st.info(
                "No follow-ups saved yet."
            )


# =========================================================
# SCHOOL 360
# =========================================================

elif page == "School 360":

    if schools.empty:

        st.warning(
            "No school data available."
        )

        st.stop()

    school = st.selectbox(
        "Select School",
        schools[
            "School"
        ].tolist(),
    )

    school_row = (
        schools[
            schools["School"]
            == school
        ]
        .iloc[0]
    )

    school_teachers = (
        teachers[
            teachers["School"]
            == school
        ]
        .copy()
    )

    school_raw = (
        period_raw[
            period_raw["School"]
            == school
        ]
        .copy()
    )

    st.title(
        f"🏫 {school}"
    )

    st.caption(
        f"Review period: "
        f"{start_date} to {end_date} "
        f"• {workdays} working day(s)"
    )

    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )

    c1.metric(
        "Health",
        f"{school_row['Health Score']}/100",
    )

    c2.metric(
        "Compliance",
        f"{school_row['Overall Compliance %']}%",
    )

    c3.metric(
        "Teachers",
        school_row["Teachers"],
    )

    c4.metric(
        "Active",
        school_row["Active"],
    )

    c5.metric(
        "Never Logged In",
        school_row[
            "Inactive / Never Logged In"
        ],
    )

    # -----------------------------------------------------
    # SCHOOL CONTACT / COMMUNICATION
    # -----------------------------------------------------

    contact_rows = (
        db_select(
            "schools",
            {
                "school_name":
                    school,
            },
        )
    )

    contact = (
        contact_rows[0]
        if contact_rows
        else {}
    )

    with st.expander(
        "📞 School Contact & One-Click Communication",
        expanded=True,
    ):

        left, right = (
            st.columns(2)
        )

        contact_name = (
            left.text_input(
                "KDM Name",
                value=(
                    contact.get(
                        "contact_name"
                    )
                    or ""
                ),
            )
        )

        contact_role = (
            right.text_input(
                "KDM Role",
                value=(
                    contact.get(
                        "contact_role"
                    )
                    or "Principal"
                ),
            )
        )

        phone = (
            left.text_input(
                "Mobile number with country code",
                value=(
                    contact.get(
                        "contact_phone"
                    )
                    or ""
                ),
            )
        )

        group_url = (
            right.text_input(
                "WhatsApp Group link",
                value=(
                    contact.get(
                        "whatsapp_group_url"
                    )
                    or ""
                ),
            )
        )

        if st.button(
            "Save Contact"
        ):

            payload = {

                "school_name":
                    school,

                "contact_name":
                    contact_name,

                "contact_role":
                    contact_role,

                "contact_phone":
                    phone,

                "whatsapp_group_url":
                    group_url,

                "updated_at":
                    datetime.utcnow()
                    .isoformat(),
            }

            if contact:

                db_update(
                    "schools",
                    payload,
                    contact["id"],
                )

            else:

                db_insert(
                    "schools",
                    payload,
                )

            st.success(
                "Contact saved."
            )

            st.rerun()

        clean_phone = re.sub(
            r"\D",
            "",
            phone,
        )

        b1, b2, b3, b4 = (
            st.columns(4)
        )

        if clean_phone:

            b1.link_button(
                "📞 Call KDM",
                f"tel:+{clean_phone}",
                use_container_width=True,
            )

            b2.link_button(
                "💬 WhatsApp Personal",
                f"https://wa.me/{clean_phone}",
                use_container_width=True,
            )

        else:

            b1.button(
                "📞 Call KDM",
                disabled=True,
                use_container_width=True,
            )

            b2.button(
                "💬 WhatsApp Personal",
                disabled=True,
                use_container_width=True,
            )

        if group_url:

            b3.link_button(
                "👥 WhatsApp Group",
                group_url,
                use_container_width=True,
            )

        else:

            b3.button(
                "👥 WhatsApp Group",
                disabled=True,
                use_container_width=True,
            )

        if b4.button(
            "📅 Follow Up",
            use_container_width=True,
        ):

            st.session_state.follow_school = school

    if (
        st.session_state.follow_school
        == school
    ):

        followup_dialog(
            school
        )

    # -----------------------------------------------------
    # KPI GRAPH
    # -----------------------------------------------------

    st.subheader(
        "📈 Digital KPI Overview"
    )

    kpis = effective_kpis(
        school
    )

    module_summary = (
        pd.DataFrame(
            {
                "Module": [
                    "Lesson Delivery",
                    "Library",
                    "Other Modules",
                ],

                "Actual": [
                    school_row[
                        "Lesson Delivery Minutes"
                    ],
                    school_row[
                        "Library Minutes"
                    ],
                    school_row[
                        "Other Modules Minutes"
                    ],
                ],

                "Daily Target": [
                    kpis[
                        "lessonDelivery"
                    ],
                    kpis[
                        "library"
                    ],
                    kpis[
                        "otherModules"
                    ],
                ],
            }
        )
    )

    module_summary[
        "Review Target"
    ] = (
        module_summary[
            "Daily Target"
        ]
        * workdays
    )

    kpi_chart_data = (
        module_summary.melt(
            id_vars="Module",
            value_vars=[
                "Actual",
                "Review Target",
            ],
            var_name="Measure",
            value_name="Minutes",
        )
    )

    kpi_chart = px.bar(
        kpi_chart_data,
        x="Module",
        y="Minutes",
        color="Measure",
        barmode="group",
        color_discrete_map={
            "Actual":
                "#4f46e5",
            "Review Target":
                "#cbd5e1",
        },
        text_auto=".1f",
    )

    st.plotly_chart(
        kpi_chart,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # TEACHER DISTRIBUTION
    # -----------------------------------------------------

    st.subheader(
        "👩‍🏫 Teacher Performance Distribution"
    )

    status_counts = (
        school_teachers[
            "Status"
        ]
        .value_counts()
        .reset_index()
    )

    status_counts.columns = [
        "Status",
        "Teachers",
    ]

    distribution_chart = (
        px.bar(
            status_counts,
            x="Status",
            y="Teachers",
            color="Status",
            text_auto=True,
            color_discrete_map={
                "Meeting All KPIs":
                    "#10b981",
                "Partially Meeting":
                    "#3b82f6",
                "Below KPI":
                    "#f59e0b",
                "Never Logged In":
                    "#ef4444",
                "0 Usage":
                    "#ef4444",
            },
        )
    )

    st.plotly_chart(
        distribution_chart,
        use_container_width=True,
    )

    st.dataframe(
        school_teachers[
            [
                "Teacher",
                "Status",
                "Health Score",
                "Lesson Delivery",
                "Lesson KPI %",
                "Library",
                "Library KPI %",
                "Other Modules",
                "Other KPI %",
                "Active Days",
                "Books Used",
            ]
        ]
        .sort_values(
            [
                "Health Score",
                "Total Minutes",
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------
    # SCHOOL AI REPORT
    # -----------------------------------------------------

    st.subheader(
        "🧠 School Intelligence"
    )

    action_days = st.radio(
        "Action plan interval",
        [
            7,
            15,
        ],
        horizontal=True,
        key="school_action_days",
    )

    if st.button(
        "✨ Generate School 360 AI Report",
        use_container_width=True,
    ):

        with st.spinner(
            "Generating evidence-backed school intelligence..."
        ):

            ai_text = ai_generate(
                school_ai_prompt(
                    school_row,
                    school_teachers,
                    school_raw,
                    action_days,
                )
            )

        st.session_state[
            f"school_ai::{school}::{start_date}::{end_date}"
        ] = ai_text

        db_insert(
            "report_history",
            {
                "report_level":
                    "School",

                "school_name":
                    school,

                "action_plan_days":
                    action_days,

                "report_text":
                    ai_text,

                "verified_facts":
                    json_safe(
                        school_row.to_dict()
                    ),
            },
        )

    school_ai_key = (
        f"school_ai::{school}::"
        f"{start_date}::{end_date}"
    )

    school_ai = (
        st.session_state.get(
            school_ai_key,
            "",
        )
    )

    if school_ai:

        render_ai(
            school_ai
        )

        school_pdf = (
            make_school_pdf(
                school_row,
                school_teachers,
                school_ai,
                start_date,
                end_date,
            )
        )

        st.download_button(
            "⬇ Download School 360 PDF",
            data=school_pdf,
            file_name=(
                re.sub(
                    r"[^A-Za-z0-9]+",
                    "_",
                    school,
                )
                + "_School_360_Report.pdf"
            ),
            mime="application/pdf",
            use_container_width=True,
        )

    # -----------------------------------------------------
    # CALL SCRIPT
    # -----------------------------------------------------

    st.subheader(
        "☎️ AI KDM Calling Script"
    )

    previous_followups = (
        db_select(
            "followups",
            {
                "school_name":
                    school,
            },
            order="followup_date",
        )
    )

    if st.button(
        "Generate Data-Backed Call Script",
        use_container_width=True,
    ):

        with st.spinner(
            "Preparing call brief..."
        ):

            script = ai_generate(
                call_script_prompt(
                    school_row,
                    school_teachers,
                    previous_followups,
                )
            )

        st.session_state[
            f"script::{school}"
        ] = script

        db_insert(
            "call_scripts",
            {
                "school_name":
                    school,

                "target_role":
                    contact_role,

                "script_text":
                    script,

                "verified_facts":
                    json_safe(
                        school_row.to_dict()
                    ),
            },
        )

    script = (
        st.session_state.get(
            f"script::{school}",
            "",
        )
    )

    if script:

        render_ai(
            script
        )

    # -----------------------------------------------------
    # WHATSAPP
    # -----------------------------------------------------

    st.subheader(
        "💬 AI WhatsApp Message"
    )

    if st.button(
        "Generate WhatsApp Performance Message",
        use_container_width=True,
    ):

        whatsapp_message = (
            ai_generate(
                whatsapp_prompt(
                    school_row,
                    school_teachers,
                    contact_name,
                )
            )
        )

        st.session_state[
            f"wa::{school}"
        ] = whatsapp_message

    whatsapp_message = (
        st.session_state.get(
            f"wa::{school}",
            "",
        )
    )

    if whatsapp_message:

        whatsapp_message = (
            whatsapp_message
            .replace(
                "**",
                "",
            )
        )

        st.text_area(
            "Copy-ready WhatsApp message",
            whatsapp_message,
            height=220,
        )

        if clean_phone:

            encoded_message = (
                urllib.parse.quote(
                    whatsapp_message
                )
            )

            st.link_button(
                "Open WhatsApp with this message",
                (
                    f"https://wa.me/"
                    f"{clean_phone}"
                    f"?text={encoded_message}"
                ),
                use_container_width=True,
            )


# =========================================================
# TEACHER 360
# =========================================================

elif page == "Teacher 360":

    if teachers.empty:

        st.warning(
            "No teacher data available."
        )

        st.stop()

    school_filter = (
        st.selectbox(
            "School",
            sorted(
                teachers[
                    "School"
                ].unique()
            ),
            key="teacher_school",
        )
    )

    teacher_options = (
        teachers[
            teachers["School"]
            == school_filter
        ]["Teacher"]
        .sort_values()
        .tolist()
    )

    teacher_name = (
        st.selectbox(
            "Teacher",
            teacher_options,
        )
    )

    teacher_row = (
        teachers[
            (
                teachers["School"]
                == school_filter
            )
            & (
                teachers["Teacher"]
                == teacher_name
            )
        ]
        .iloc[0]
    )

    teacher_evidence = (
        period_raw[
            (
                period_raw["School"]
                == school_filter
            )
            & (
                period_raw[
                    "Teacher Key"
                ]
                == teacher_row[
                    "Teacher Key"
                ]
            )
        ]
        .copy()
    )

    st.title(
        f"👩‍🏫 {teacher_name}"
    )

    st.caption(
        f"{school_filter} "
        f"• Review period "
        f"{start_date} to {end_date}"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Health",
        f"{teacher_row['Health Score']}/100",
    )

    c2.metric(
        "Lesson KPI",
        f"{teacher_row['Lesson KPI %']}%",
    )

    c3.metric(
        "Library KPI",
        f"{teacher_row['Library KPI %']}%",
    )

    c4.metric(
        "Other KPI",
        f"{teacher_row['Other KPI %']}%",
    )

    module_data = pd.DataFrame(
        {
            "Module": [
                "Lesson Delivery",
                "Library",
                "Other Modules",
            ],

            "Actual": [
                teacher_row[
                    "Lesson Delivery"
                ],
                teacher_row[
                    "Library"
                ],
                teacher_row[
                    "Other Modules"
                ],
            ],

            "Target": [
                teacher_row[
                    "Lesson Target"
                ],
                teacher_row[
                    "Library Target"
                ],
                teacher_row[
                    "Other Target"
                ],
            ],
        }
    )

    teacher_kpi_chart = (
        px.bar(
            module_data.melt(
                id_vars="Module",
                value_vars=[
                    "Actual",
                    "Target",
                ],
                var_name="Measure",
                value_name="Minutes",
            ),
            x="Module",
            y="Minutes",
            color="Measure",
            barmode="group",
            color_discrete_map={
                "Actual":
                    "#4f46e5",
                "Target":
                    "#cbd5e1",
            },
            text_auto=".1f",
        )
    )

    st.plotly_chart(
        teacher_kpi_chart,
        use_container_width=True,
    )

    e1, e2, e3, e4 = (
        st.columns(4)
    )

    e1.metric(
        "Active Days",
        (
            f"{teacher_row['Active Days']}/"
            f"{teacher_row['Eligible Working Days']}"
        ),
    )

    e2.metric(
        "Books Used",
        teacher_row[
            "Books Used"
        ],
    )

    e3.metric(
        "Grades Covered",
        teacher_row[
            "Grades Covered"
        ],
    )

    e4.metric(
        "Subjects Covered",
        teacher_row[
            "Subjects Covered"
        ],
    )

    st.subheader(
        "📚 Granular Activity Evidence"
    )

    if teacher_evidence.empty:

        st.info(
            "No activity was recorded "
            "for this teacher during "
            "the selected period."
        )

    else:

        evidence_columns = [

            "DateTime",
            "Raw Module",
            "KPI Module",
            "Grade",
            "Subject",
            "Book",
            "Minutes",
        ]

        st.dataframe(
            teacher_evidence[
                evidence_columns
            ]
            .sort_values(
                "DateTime",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

        daily_activity = (
            teacher_evidence
            .dropna(
                subset=[
                    "DateTime"
                ]
            )
            .assign(
                ActivityDate=lambda x:
                x[
                    "DateTime"
                ].dt.date
            )
            .groupby(
                "ActivityDate"
            )["Minutes"]
            .sum()
            .reset_index()
        )

        if not daily_activity.empty:

            trend_chart = (
                px.line(
                    daily_activity,
                    x="ActivityDate",
                    y="Minutes",
                    markers=True,
                    title="Daily Activity Trend",
                )
            )

            st.plotly_chart(
                trend_chart,
                use_container_width=True,
            )

        granular_modules = (
            teacher_evidence
            .groupby(
                "Raw Module"
            )["Minutes"]
            .sum()
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        granular_chart = (
            px.bar(
                granular_modules,
                x="Raw Module",
                y="Minutes",
                color="Raw Module",
                title="Granular Module Usage",
            )
        )

        st.plotly_chart(
            granular_chart,
            use_container_width=True,
        )

    action_days = st.radio(
        "Development plan interval",
        [
            7,
            15,
        ],
        horizontal=True,
        key="teacher_action_days",
    )

    if st.button(
        "✨ Generate Detailed Teacher 360 Report",
        use_container_width=True,
    ):

        with st.spinner(
            "Generating constructive evidence-backed report..."
        ):

            teacher_ai = ai_generate(
                teacher_ai_prompt(
                    teacher_row,
                    teacher_evidence,
                    action_days,
                )
            )

        teacher_ai_key = (
            f"teacher_ai::"
            f"{school_filter}::"
            f"{teacher_name}::"
            f"{start_date}::"
            f"{end_date}"
        )

        st.session_state[
            teacher_ai_key
        ] = teacher_ai

        db_insert(
            "report_history",
            {
                "report_level":
                    "Teacher",

                "school_name":
                    school_filter,

                "teacher_name":
                    teacher_name,

                "action_plan_days":
                    action_days,

                "report_text":
                    teacher_ai,

                "verified_facts":
                    json_safe(
                        teacher_row.to_dict()
                    ),
            },
        )

    teacher_ai_key = (
        f"teacher_ai::"
        f"{school_filter}::"
        f"{teacher_name}::"
        f"{start_date}::"
        f"{end_date}"
    )

    teacher_ai = (
        st.session_state.get(
            teacher_ai_key,
            "",
        )
    )

    if teacher_ai:

        render_ai(
            teacher_ai
        )

        teacher_pdf = (
            make_teacher_pdf(
                teacher_row,
                teacher_evidence,
                teacher_ai,
                start_date,
                end_date,
            )
        )

        st.download_button(
            "⬇ Download Detailed Teacher 360 PDF",
            data=teacher_pdf,
            file_name=(
                re.sub(
                    r"[^A-Za-z0-9]+",
                    "_",
                    teacher_name,
                )
                + "_360_Audit_Report.pdf"
            ),
            mime="application/pdf",
            use_container_width=True,
        )


# =========================================================
# FOLLOW-UP CENTER
# =========================================================

elif page == "Follow-Ups":

    st.title(
        "📅 Implementation Follow-Up Center"
    )

    followup_rows = (
        db_select(
            "followups",
            order="followup_date",
        )
    )

    if not followup_rows:

        st.info(
            "No follow-ups saved yet. "
            "Use the Follow Up button "
            "inside School 360."
        )

    else:

        followups = pd.DataFrame(
            followup_rows
        )

        followups[
            "followup_date"
        ] = (
            pd.to_datetime(
                followups[
                    "followup_date"
                ]
            )
            .dt.date
        )

        today = date.today()

        tabs = st.tabs(
            [
                "🔴 Overdue",
                "🟠 Due Today",
                "🔵 Upcoming",
                "✅ Resolved",
            ]
        )

        subsets = [

            followups[
                (
                    followups[
                        "followup_date"
                    ]
                    < today
                )
                & (
                    followups["status"]
                    != "Resolved"
                )
            ],

            followups[
                (
                    followups[
                        "followup_date"
                    ]
                    == today
                )
                & (
                    followups["status"]
                    != "Resolved"
                )
            ],

            followups[
                (
                    followups[
                        "followup_date"
                    ]
                    > today
                )
                & (
                    followups["status"]
                    != "Resolved"
                )
            ],

            followups[
                followups["status"]
                == "Resolved"
            ],
        ]

        for tab, subset in zip(
            tabs,
            subsets,
        ):

            with tab:

                if subset.empty:

                    st.caption(
                        "Nothing here."
                    )

                else:

                    for _, row in (
                        subset.sort_values(
                            "followup_date"
                        )
                        .iterrows()
                    ):

                        with st.container(
                            border=True
                        ):

                            c1, c2, c3 = (
                                st.columns(
                                    [
                                        2,
                                        1,
                                        1,
                                    ]
                                )
                            )

                            c1.markdown(
                                f"**{row['school_name']}**"
                            )

                            c1.write(
                                row["issue"]
                            )

                            c1.caption(
                                "Last commitment: "
                                + (
                                    row.get(
                                        "last_commitment"
                                    )
                                    or "—"
                                )
                            )

                            c2.write(
                                str(
                                    row[
                                        "followup_date"
                                    ]
                                )
                            )

                            c2.write(
                                row["status"]
                            )

                            if (
                                row["status"]
                                != "Resolved"
                            ):

                                if c3.button(
                                    "Mark Resolved",
                                    key=(
                                        f"resolve_"
                                        f"{row['id']}"
                                    ),
                                ):

                                    db_update(
                                        "followups",
                                        {
                                            "status":
                                                "Resolved",

                                            "updated_at":
                                                datetime.utcnow()
                                                .isoformat(),
                                        },
                                        row["id"],
                                    )

                                    st.rerun()

                            if c3.button(
                                "Delete",
                                key=(
                                    f"delete_fu_"
                                    f"{row['id']}"
                                ),
                            ):

                                db_delete(
                                    "followups",
                                    row["id"],
                                )

                                st.rerun()


# =========================================================
# KPI / ROSTER / SHARED ACCOUNTS
# =========================================================

elif page == "KPI & Roster":

    st.title(
        "⚙️ KPI, Master Roster & Shared Accounts"
    )

    tab1, tab2, tab3 = (
        st.tabs(
            [
                "🎯 KPI Settings",
                "👩‍🏫 Master Roster",
                "🏫 Shared Accounts",
            ]
        )
    )

    # -----------------------------------------------------
    # KPI SETTINGS
    # -----------------------------------------------------

    with tab1:

        st.subheader(
            "Global Default KPIs"
        )

        current = effective_kpis()

        with st.form(
            "global_kpi"
        ):

            lesson = (
                st.number_input(
                    "Lesson Delivery minutes/day",
                    min_value=0.0,
                    value=float(
                        current[
                            "lessonDelivery"
                        ]
                    ),
                    step=1.0,
                )
            )

            library = (
                st.number_input(
                    "Library minutes/day",
                    min_value=0.0,
                    value=float(
                        current[
                            "library"
                        ]
                    ),
                    step=1.0,
                )
            )

            other = (
                st.number_input(
                    "Other Modules combined minutes/day",
                    min_value=0.0,
                    value=float(
                        current[
                            "otherModules"
                        ]
                    ),
                    step=1.0,
                )
            )

            if st.form_submit_button(
                "Save Global KPI",
                use_container_width=True,
            ):

                save_kpi(
                    "GLOBAL",
                    "lessonDelivery",
                    lesson,
                )

                save_kpi(
                    "GLOBAL",
                    "library",
                    library,
                )

                save_kpi(
                    "GLOBAL",
                    "otherModules",
                    other,
                )

                st.success(
                    "Global KPIs saved."
                )

                st.rerun()

        possible_schools = set()

        if not schools.empty:

            possible_schools.update(
                schools[
                    "School"
                ].tolist()
            )

        possible_schools.update(
            [
                row["school_name"]
                for row
                in db_select(
                    "schools"
                )
            ]
        )

        possible_schools = sorted(
            possible_schools
        )

        if possible_schools:

            st.subheader(
                "School-Specific KPI Override"
            )

            selected_school = (
                st.selectbox(
                    "School",
                    possible_schools,
                    key="kpi_school",
                )
            )

            effective = (
                effective_kpis(
                    selected_school
                )
            )

            with st.form(
                "school_kpi"
            ):

                s_lesson = (
                    st.number_input(
                        "Lesson Delivery",
                        min_value=0.0,
                        value=float(
                            effective[
                                "lessonDelivery"
                            ]
                        ),
                        step=1.0,
                    )
                )

                s_library = (
                    st.number_input(
                        "Library",
                        min_value=0.0,
                        value=float(
                            effective[
                                "library"
                            ]
                        ),
                        step=1.0,
                    )
                )

                s_other = (
                    st.number_input(
                        "Other Modules",
                        min_value=0.0,
                        value=float(
                            effective[
                                "otherModules"
                            ]
                        ),
                        step=1.0,
                    )
                )

                if st.form_submit_button(
                    "Save School Override",
                    use_container_width=True,
                ):

                    save_kpi(
                        "SCHOOL",
                        "lessonDelivery",
                        s_lesson,
                        selected_school,
                    )

                    save_kpi(
                        "SCHOOL",
                        "library",
                        s_library,
                        selected_school,
                    )

                    save_kpi(
                        "SCHOOL",
                        "otherModules",
                        s_other,
                        selected_school,
                    )

                    st.success(
                        "School KPI override saved."
                    )

                    st.rerun()

    # -----------------------------------------------------
    # MASTER ROSTER
    # -----------------------------------------------------

    with tab2:

        st.write(
            "Upload the Master Roster so teachers "
            "with no usage can be identified as "
            "Never Logged In."
        )

        roster_files = (
            st.file_uploader(
                "Upload Master Roster CSV/XLSX",
                type=[
                    "csv",
                    "xlsx",
                ],
                accept_multiple_files=True,
                key="roster_upload",
            )
        )

        if (
            roster_files
            and st.button(
                "Import Master Roster",
                use_container_width=True,
            )
        ):

            try:

                inserted, updated = (
                    import_roster(
                        roster_files
                    )
                )

                st.success(
                    f"Roster imported: "
                    f"{inserted} new, "
                    f"{updated} updated."
                )

                st.rerun()

            except Exception as error:

                st.error(
                    str(error)
                )

        roster = roster_dataframe()

        if not roster.empty:

            st.metric(
                "Active roster teachers",
                len(roster),
            )

            st.dataframe(
                roster[
                    [
                        "school_name",
                        "teacher_name",
                        "grade",
                        "subject",
                        "email",
                        "phone",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    # -----------------------------------------------------
    # SHARED SCHOOL ACCOUNTS
    # -----------------------------------------------------

    with tab3:

        st.write(
            "Shared school accounts are excluded "
            "from individual teacher rankings "
            "and inactive-teacher analytics."
        )

        shared_school = (
            st.text_input(
                "School name",
                key="shared_school",
            )
        )

        shared_name = (
            st.text_input(
                "Shared account name",
                key="shared_name",
            )
        )

        if st.button(
            "Add Shared Account",
            use_container_width=True,
        ):

            if (
                shared_school.strip()
                and shared_name.strip()
            ):

                key = teacher_key(
                    shared_name
                )

                existing = [

                    row

                    for row
                    in db_select(
                        "shared_accounts"
                    )

                    if (
                        row["school_name"]
                        == shared_school.strip()
                        and row["account_key"]
                        == key
                    )
                ]

                if not existing:

                    db_insert(
                        "shared_accounts",
                        {
                            "school_name":
                                shared_school.strip(),

                            "account_name":
                                shared_name.strip(),

                            "account_key":
                                key,

                            "active":
                                True,
                        },
                    )

                st.success(
                    "Shared account saved."
                )

                st.rerun()

            else:

                st.error(
                    "Enter both school "
                    "and account name."
                )

        shared_rows = (
            db_select(
                "shared_accounts"
            )
        )

        if shared_rows:

            for row in shared_rows:

                left, right = (
                    st.columns(
                        [
                            4,
                            1,
                        ]
                    )
                )

                left.write(
                    f"{row['school_name']} "
                    f"— {row['account_name']}"
                )

                if right.button(
                    "Remove",
                    key=(
                        f"shared_del_"
                        f"{row['id']}"
                    ),
                ):

                    db_delete(
                        "shared_accounts",
                        row["id"],
                    )

                    st.rerun()


# =========================================================
# ASK ACADINTEL
# =========================================================

elif page == "Ask AcadIntel":

    st.title(
        "✨ Ask AcadIntel"
    )

    st.caption(
        "Responses are constrained to "
        "the verified calculated facts "
        "currently loaded."
    )

    question = (
        st.text_area(
            "Ask a portfolio question",
            placeholder=(
                "Which schools need my "
                "attention most and why?"
            ),
        )
    )

    if st.button(
        "Ask AcadIntel",
        use_container_width=True,
    ):

        facts = {

            "period": {
                "start":
                    str(
                        start_date
                    ),

                "end":
                    str(
                        end_date
                    ),

                "working_days":
                    workdays,
            },

            "schools":
                json_safe(
                    schools.to_dict(
                        "records"
                    )
                    if not schools.empty
                    else []
                ),

            "priority_teachers":
                json_safe(
                    teachers
                    .sort_values(
                        [
                            "Health Score",
                            "Total Minutes",
                        ]
                    )
                    .head(30)
                    .to_dict(
                        "records"
                    )
                    if not teachers.empty
                    else []
                ),
        }

        prompt = f"""
You are AcadIntel 360.

Answer ONLY from the verified facts below.

Never invent:
metrics,
causes,
commitments,
names,
or dates.

If the available data does not answer the question, state that clearly.

QUESTION:

{question}

Use these plain headings:

FACTS
INTERPRETATION
RECOMMENDED ACTIONS
EVIDENCE

VERIFIED FACTS:

{json.dumps(facts, default=str)}
"""

        with st.spinner(
            "Analysing verified facts..."
        ):

            answer = (
                ai_generate(
                    prompt
                )
            )

        render_ai(
            answer
        )
