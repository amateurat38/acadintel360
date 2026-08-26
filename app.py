import io
import re
import json
import urllib.parse
import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF

try:
    from google import genai
except Exception:
    genai = None


# ---------------------------------------------------------
# APP CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="AcadIntel 360",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e8eaf2;
        padding: 16px;
        border-radius: 18px;
        box-shadow: 0 5px 22px rgba(30,32,55,.05);
    }

    .hero {
        background: linear-gradient(120deg,#4937d8,#377fee);
        color:white;
        padding:28px;
        border-radius:24px;
        margin-bottom:20px;
    }

    .hero h1 {
        margin:0;
        font-size:42px;
    }

    .hero p {
        opacity:.88;
        margin:5px 0 0 0;
    }

    .intel-card {
        background:white;
        border:1px solid #e9ebf3;
        border-radius:18px;
        padding:18px;
        margin:8px 0;
        box-shadow:0 6px 24px rgba(20,30,60,.04);
    }

    .small-muted {
        color:#717487;
        font-size:13px;
    }

    div[data-testid="stDownloadButton"] button,
    div[data-testid="stButton"] button {
        border-radius:12px;
        font-weight:700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

defaults = {
    "raw": pd.DataFrame(),
    "contacts": {},
    "ai_cache": {},
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def normalize_col(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def first_column(df, names):
    lookup = {normalize_col(c): c for c in df.columns}

    for name in names:
        key = normalize_col(name)
        if key in lookup:
            return lookup[key]

    return None


def normalize_teacher(name):
    s = str(name or "").strip()

    s = re.sub(r"^[\.\s]+", "", s)

    s = re.sub(
        r"\b(mrs|ms|mr|miss|dr)\.?\b",
        "",
        s,
        flags=re.I,
    )

    s = re.sub(r"\s+", " ", s).strip()

    return s


def identify_schema(df):
    return {
        "school": first_column(
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
        "first": first_column(
            df,
            [
                "FirstName",
                "First Name",
            ],
        ),
        "last": first_column(
            df,
            [
                "LastName",
                "Last Name",
            ],
        ),
        "teacher": first_column(
            df,
            [
                "Teacher",
                "Teacher Name",
                "User Name",
                "Username",
                "Name",
            ],
        ),
        "minutes": first_column(
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
        "module": first_column(
            df,
            [
                "Type",
                "Module",
                "Module Name",
                "Category",
            ],
        ),
        "date": first_column(
            df,
            [
                "StartTime",
                "Start Time",
                "Date",
                "Activity Date",
                "Log Date",
            ],
        ),
        "grade": first_column(
            df,
            [
                "Grade",
                "Class",
            ],
        ),
        "subject": first_column(
            df,
            [
                "Subject",
            ],
        ),
        "book": first_column(
            df,
            [
                "Book",
                "Book Name",
                "Content",
            ],
        ),
    }


def transform_file(df, filename):
    schema = identify_schema(df)

    if not schema["school"]:
        raise ValueError(
            f"{filename}: School/Institution/Center column not found."
        )

    if not schema["minutes"]:
        raise ValueError(
            f"{filename}: Duration/Minutes column not found."
        )

    out = pd.DataFrame()

    out["School"] = df[schema["school"]].astype(str).str.strip()

    if schema["teacher"]:
        out["Teacher"] = (
            df[schema["teacher"]]
            .fillna("")
            .astype(str)
            .map(normalize_teacher)
        )

    elif schema["first"] or schema["last"]:
        if schema["first"]:
            first = df[schema["first"]].fillna("").astype(str)
        else:
            first = pd.Series([""] * len(df), index=df.index)

        if schema["last"]:
            last = df[schema["last"]].fillna("").astype(str)
        else:
            last = pd.Series([""] * len(df), index=df.index)

        out["Teacher"] = (
            (first + " " + last)
            .str.strip()
            .map(normalize_teacher)
        )

    else:
        out["Teacher"] = "Unattributed Activity"

    out["Minutes"] = (
        pd.to_numeric(
            df[schema["minutes"]],
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
    )

    if schema["module"]:
        out["Module"] = df[schema["module"]].fillna("").astype(str)
    else:
        out["Module"] = "Platform"

    if schema["date"]:
        out["Date"] = pd.to_datetime(
            df[schema["date"]],
            errors="coerce",
        )
    else:
        out["Date"] = pd.NaT

    if schema["grade"]:
        out["Grade"] = df[schema["grade"]].fillna("").astype(str)
    else:
        out["Grade"] = ""

    if schema["subject"]:
        out["Subject"] = df[schema["subject"]].fillna("").astype(str)
    else:
        out["Subject"] = ""

    if schema["book"]:
        out["Book"] = df[schema["book"]].fillna("").astype(str)
    else:
        out["Book"] = ""

    out["Source File"] = filename

    out = out[
        out["School"].notna()
        & (out["School"].str.strip() != "")
    ]

    return out


def load_files(files):
    frames = []
    errors = []

    for file in files[:100]:
        try:
            name = file.name

            if name.lower().endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            frames.append(transform_file(df, name))

        except Exception as e:
            errors.append(str(e))

    if not frames:
        return pd.DataFrame(), errors

    result = pd.concat(frames, ignore_index=True)

    return result, errors


def module_group(value):
    x = str(value).lower()

    if "lesson" in x or "prep" in x or "delivery" in x:
        return "Lesson Preparation"

    if "library" in x:
        return "Library"

    if "book" in x or "content" in x:
        return "Content / Books"

    return "Other Platform"


def working_days(df):
    if df.empty or df["Date"].dropna().empty:
        return 1

    dates = (
        df["Date"]
        .dropna()
        .dt.normalize()
        .drop_duplicates()
    )

    return max(
        1,
        sum(d.weekday() != 6 for d in dates),
    )


def teacher_summary(df):
    days = working_days(df)

    result = []

    for (school, teacher), g in df.groupby(
        ["School", "Teacher"],
        dropna=False,
    ):
        lesson = g.loc[
            g["Module Group"] == "Lesson Preparation",
            "Minutes",
        ].sum()

        library = g.loc[
            g["Module Group"] == "Library",
            "Minutes",
        ].sum()

        content = g.loc[
            g["Module Group"] == "Content / Books",
            "Minutes",
        ].sum()

        total = g["Minutes"].sum()

        lesson_target = days * 10
        library_target = days * 30

        lesson_pct = (
            lesson / lesson_target * 100
            if lesson_target
            else 0
        )

        library_pct = (
            library / library_target * 100
            if library_target
            else 0
        )

        active = total > 0

        score = round(
            min(100, lesson_pct) * 0.45
            + min(100, library_pct) * 0.35
            + (100 if active else 0) * 0.20
        )

        result.append(
            {
                "School": school,
                "Teacher": teacher,
                "Lesson Prep": round(lesson, 1),
                "Library": round(library, 1),
                "Content": round(content, 1),
                "Total": round(total, 1),
                "Lesson KPI %": round(lesson_pct, 1),
                "Library KPI %": round(library_pct, 1),
                "Health": int(score),
                "Active": active,
            }
        )

    return pd.DataFrame(result)


def school_summary(df):
    teachers = teacher_summary(df)

    result = []

    for school, g in teachers.groupby("School"):
        total = len(g)
        active = int(g["Active"].sum())
        inactive = total - active

        met = int(
            (
                (g["Lesson KPI %"] >= 100)
                & (g["Library KPI %"] >= 100)
            ).sum()
        )

        health = round(
            g["Health"].mean()
            if total
            else 0
        )

        compliance = round(
            met / total * 100
            if total
            else 0,
            1,
        )

        result.append(
            {
                "School": school,
                "Teachers": total,
                "Active": active,
                "Inactive": inactive,
                "Met Both KPIs": met,
                "Compliance": compliance,
                "Health": health,
            }
        )

    return pd.DataFrame(result)


def get_ai_client():
    key = st.secrets.get("GEMINI_API_KEY", "")

    if not key or genai is None:
        return None

    return genai.Client(api_key=key)


def ai_generate(prompt):
    if prompt in st.session_state.ai_cache:
        return st.session_state.ai_cache[prompt]

    client = get_ai_client()

    if client is None:
        return (
            "AI is not configured. Add GEMINI_API_KEY "
            "to Streamlit Secrets."
        )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    text = response.text or ""
    text = text.replace("**", "")

    st.session_state.ai_cache[prompt] = text

    return text


def ai_school_prompt(school, school_df, teacher_df):
    facts = teacher_df[
        teacher_df["School"] == school
    ].copy()

    lowest = (
        facts
        .sort_values("Health")
        .head(5)[
            [
                "Teacher",
                "Health",
                "Lesson KPI %",
                "Library KPI %",
                "Total",
            ]
        ]
        .to_dict("records")
    )

    highest = (
        facts
        .sort_values(
            "Health",
            ascending=False,
        )
        .head(5)[
            [
                "Teacher",
                "Health",
                "Lesson KPI %",
                "Library KPI %",
                "Total",
            ]
        ]
        .to_dict("records")
    )

    summary = school_summary(school_df)
    row = summary.iloc[0].to_dict()

    return f"""
You are AcadIntel 360, an academic implementation intelligence analyst.

Use ONLY the verified facts supplied below.
Never fabricate numbers, reasons, commitments, dates, or teacher behaviour.

Create a management-ready school insight.

Use these plain headings without Markdown symbols:

EXECUTIVE SUMMARY
STRENGTHS
CRITICAL RISKS
TEACHERS REQUIRING ATTENTION
TOP PERFORMERS
RECOMMENDED MANAGEMENT ACTIONS
NEXT REVIEW TARGETS
EVIDENCE

Verified school facts:
{json.dumps(row, default=str)}

Lowest teacher indicators:
{json.dumps(lowest, default=str)}

Highest teacher indicators:
{json.dumps(highest, default=str)}

Distinguish observed facts from interpretation.
Do not blame teachers.
Do not invent a reason for low performance.
"""


def ai_teacher_prompt(row):
    return f"""
You are AcadIntel 360.

Create an evidence-backed Teacher 360 academic implementation review.

Do not invent facts.

Use plain headings without Markdown symbols:

EXECUTIVE DIAGNOSIS
STRENGTHS
DEVELOPMENT GAPS
BEHAVIOURAL PATTERN
RECOMMENDED INTERVENTION
NEXT REVIEW TARGET
EVIDENCE

Verified facts:

Teacher: {row['Teacher']}
School: {row['School']}
Lesson Preparation minutes: {row['Lesson Prep']}
Lesson KPI achievement: {row['Lesson KPI %']}%
Library minutes: {row['Library']}
Library KPI achievement: {row['Library KPI %']}%
Content/Book minutes: {row['Content']}
Total usage: {row['Total']}
Health score: {row['Health']}/100
Active: {row['Active']}

Interpret the pattern, but explicitly state when the data
does not establish a cause.
"""


def pdf_bytes(title, lines):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font(
        "Helvetica",
        "B",
        18,
    )

    pdf.multi_cell(
        0,
        10,
        title,
    )

    pdf.ln(3)

    pdf.set_font(
        "Helvetica",
        "",
        10,
    )

    for line in lines:
        safe = str(line).encode(
            "latin-1",
            "replace",
        ).decode(
            "latin-1"
        )

        pdf.multi_cell(
            0,
            6,
            safe,
        )

        pdf.ln(1)

    return bytes(pdf.output())


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.title("🎓 AcadIntel 360")

    st.caption(
        "Academic Intelligence • Evidence • Action"
    )

    uploaded = st.file_uploader(
        "Upload UserMetrics files",
        type=[
            "csv",
            "xlsx",
        ],
        accept_multiple_files=True,
        help="Select up to 100 files.",
    )

    if uploaded:

        if len(uploaded) > 100:
            st.error(
                "Maximum 100 files at a time."
            )

        else:

            if st.button(
                "⚡ Process Files",
                use_container_width=True,
            ):

                with st.spinner(
                    "Reading and calculating..."
                ):

                    df, errors = load_files(uploaded)

                    if not df.empty:

                        df["Module Group"] = (
                            df["Module"]
                            .map(module_group)
                        )

                        st.session_state.raw = df

                        st.success(
                            f"{len(df):,} rows loaded from "
                            f"{len(uploaded)} file(s)."
                        )

                    for error in errors:
                        st.warning(error)

    page = st.radio(
        "Navigate",
        [
            "Command Center",
            "Schools",
            "Teachers",
            "Ask AcadIntel",
            "Contacts & Communication",
            "Reports",
        ],
    )

    st.divider()

    st.caption(
        "All performance figures are calculated "
        "before AI interpretation."
    )


# ---------------------------------------------------------
# DATA CHECK
# ---------------------------------------------------------

raw = st.session_state.raw

if raw.empty:

    st.markdown(
        """
        <div class="hero">
        <h1>AcadIntel 360</h1>
        <p>Upload your company's UserMetrics CSV/XLSX files
        to begin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Upload one or multiple company UserMetrics files "
        "from the sidebar."
    )

    st.stop()


teachers = teacher_summary(raw)
schools = school_summary(raw)


# ---------------------------------------------------------
# COMMAND CENTER
# ---------------------------------------------------------

if page == "Command Center":

    st.markdown(
        """
        <div class="hero">
        <h1>Academic Command Center</h1>
        <p>What requires attention, why it matters,
        and what should happen next.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Schools",
        len(schools),
    )

    c2.metric(
        "Teachers",
        len(teachers),
    )

    c3.metric(
        "Inactive Teachers",
        int((~teachers["Active"]).sum()),
    )

    c4.metric(
        "Avg. Health",
        f"{teachers['Health'].mean():.0f}/100",
    )

    st.subheader("📊 Portfolio Health")

    fig = px.bar(
        schools.sort_values("Health"),
        x="Health",
        y="School",
        orientation="h",
        color="Health",
        color_continuous_scale=[
            "#ef4444",
            "#f59e0b",
            "#10b981",
        ],
        range_color=[0, 100],
    )

    fig.update_layout(
        height=max(400, len(schools) * 34),
        coloraxis_showscale=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("🚨 Immediate Attention")

    critical = teachers.sort_values(
        [
            "Health",
            "Total",
        ]
    ).head(10)

    st.dataframe(
        critical,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# SCHOOLS
# ---------------------------------------------------------

elif page == "Schools":

    school = st.selectbox(
        "Select School",
        schools["School"].tolist(),
    )

    s = schools[
        schools["School"] == school
    ].iloc[0]

    school_raw = raw[
        raw["School"] == school
    ]

    school_teachers = teachers[
        teachers["School"] == school
    ]

    st.title(f"🏫 {school}")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Health",
        f"{s['Health']}/100",
    )

    c2.metric(
        "Compliance",
        f"{s['Compliance']}%",
    )

    c3.metric(
        "Active",
        f"{s['Active']}/{s['Teachers']}",
    )

    c4.metric(
        "Inactive",
        s["Inactive"],
    )

    st.subheader(
        "📈 Teacher Performance Distribution"
    )

    chart = school_teachers.copy()

    chart["Tier"] = pd.cut(
        chart["Health"],
        bins=[
            -1,
            0,
            49,
            74,
            99,
            100,
        ],
        labels=[
            "Inactive",
            "Critical",
            "Developing",
            "Meeting",
            "Excellent",
        ],
    )

    distribution = (
        chart["Tier"]
        .value_counts()
        .reset_index()
    )

    distribution.columns = [
        "Tier",
        "Teachers",
    ]

    fig = px.bar(
        distribution,
        x="Tier",
        y="Teachers",
        color="Tier",
        color_discrete_map={
            "Inactive": "#ef4444",
            "Critical": "#f97316",
            "Developing": "#f59e0b",
            "Meeting": "#3b82f6",
            "Excellent": "#10b981",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader(
        "👩‍🏫 Teacher KPI Matrix"
    )

    st.dataframe(
        school_teachers,
        use_container_width=True,
        hide_index=True,
    )

    if st.button(
        "✨ Generate School Intelligence",
        use_container_width=True,
    ):

        with st.spinner(
            "Generating evidence-backed insight..."
        ):

            insight = ai_generate(
                ai_school_prompt(
                    school,
                    school_raw,
                    teachers,
                )
            )

            st.session_state[
                f"school_ai_{school}"
            ] = insight

    insight = st.session_state.get(
        f"school_ai_{school}",
        "",
    )

    if insight:

        st.markdown(
            '<div class="intel-card">',
            unsafe_allow_html=True,
        )

        st.subheader(
            "🧠 Academic Intelligence"
        )

        st.write(insight)

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# TEACHERS
# ---------------------------------------------------------

elif page == "Teachers":

    teacher = st.selectbox(
        "Select Teacher",
        teachers[
            "Teacher"
        ].sort_values().tolist(),
    )

    t = teachers[
        teachers["Teacher"] == teacher
    ].iloc[0]

    st.title(
        f"👩‍🏫 {teacher}"
    )

    st.caption(
        t["School"]
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Health",
        f"{t['Health']}/100",
    )

    c2.metric(
        "Lesson KPI",
        f"{t['Lesson KPI %']}%",
    )

    c3.metric(
        "Library KPI",
        f"{t['Library KPI %']}%",
    )

    module_df = pd.DataFrame(
        {
            "Module": [
                "Lesson Preparation",
                "Library",
                "Content / Books",
            ],
            "Minutes": [
                t["Lesson Prep"],
                t["Library"],
                t["Content"],
            ],
        }
    )

    fig = px.bar(
        module_df,
        x="Module",
        y="Minutes",
        color="Module",
        color_discrete_sequence=[
            "#6255e8",
            "#35a7ff",
            "#0fac78",
        ],
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    if st.button(
        "✨ Generate Teacher 360 Intelligence",
        use_container_width=True,
    ):

        with st.spinner(
            "Analysing verified teacher evidence..."
        ):

            insight = ai_generate(
                ai_teacher_prompt(t)
            )

            st.session_state[
                f"teacher_ai_{teacher}"
            ] = insight

    insight = st.session_state.get(
        f"teacher_ai_{teacher}",
        "",
    )

    if insight:

        st.markdown(
            '<div class="intel-card">',
            unsafe_allow_html=True,
        )

        st.subheader(
            "🧠 Teacher 360 Intelligence"
        )

        st.write(insight)

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    teacher_evidence = raw[
        (raw["School"] == t["School"])
        & (
            raw["Teacher"]
            .map(normalize_teacher)
            == normalize_teacher(teacher)
        )
    ][
        [
            "Date",
            "Module",
            "Grade",
            "Subject",
            "Book",
            "Minutes",
        ]
    ]

    st.subheader(
        "🔎 Evidence Audit Trail"
    )

    st.dataframe(
        teacher_evidence.sort_values(
            "Date",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# ASK ACADINTEL
# ---------------------------------------------------------

elif page == "Ask AcadIntel":

    st.title(
        "✨ Ask AcadIntel"
    )

    question = st.text_area(
        "Ask anything about the uploaded portfolio",
        placeholder=(
            "Which schools require immediate attention?"
        ),
    )

    if st.button(
        "Ask AcadIntel",
        use_container_width=True,
    ):

        facts = {
            "schools": schools.to_dict(
                "records"
            ),
            "lowest_teachers": (
                teachers.sort_values(
                    "Health"
                )
                .head(20)
                .to_dict(
                    "records"
                )
            ),
        }

        prompt = f"""
Use only the verified facts below.

Do not fabricate values.

QUESTION:
{question}

VERIFIED FACTS:
{json.dumps(facts, default=str)}

Answer using plain headings:

FACTS
INTERPRETATION
RECOMMENDED ACTIONS
EVIDENCE
"""

        with st.spinner("Analysing..."):
            response = ai_generate(prompt)

        st.markdown(
            '<div class="intel-card">',
            unsafe_allow_html=True,
        )

        st.write(response)

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# CONTACTS & COMMUNICATION
# ---------------------------------------------------------

elif page == "Contacts & Communication":

    st.title(
        "📞 School Contacts & Communication"
    )

    school = st.selectbox(
        "School",
        schools["School"],
    )

    existing = (
        st.session_state.contacts
        .get(
            school,
            {},
        )
    )

    contact_name = st.text_input(
        "Principal / Coordinator Name",
        existing.get(
            "name",
            "",
        ),
    )

    phone = st.text_input(
        "Mobile with country code",
        existing.get(
            "phone",
            "",
        ),
    )

    group = st.text_input(
        "WhatsApp Group Invite/Share URL",
        existing.get(
            "group",
            "",
        ),
    )

    if st.button("Save Contact"):
        st.session_state.contacts[
            school
        ] = {
            "name": contact_name,
            "phone": phone,
            "group": group,
        }

        st.success("Saved.")

    school_row = schools[
        schools["School"] == school
    ].iloc[0]

    school_teacher_rows = teachers[
        teachers["School"] == school
    ]

    if st.button(
        "✨ Generate WhatsApp Message"
    ):

        prompt = f"""
Create a concise management WhatsApp performance message.

No markdown stars.

Use professional, constructive language.

Verified school facts:
{school_row.to_dict()}

Teachers requiring attention:
{
school_teacher_rows
.sort_values("Health")
.head(5)
.to_dict("records")
}
"""

        st.session_state[
            "wa_message"
        ] = ai_generate(prompt)

    message = st.text_area(
        "WhatsApp Draft",
        st.session_state.get(
            "wa_message",
            "",
        ),
        height=240,
    )

    c1, c2, c3 = st.columns(3)

    clean_phone = re.sub(
        r"\D",
        "",
        phone,
    )

    encoded = urllib.parse.quote(message)

    if clean_phone:

        c1.link_button(
            "📱 WhatsApp Personal",
            f"https://wa.me/{clean_phone}?text={encoded}",
            use_container_width=True,
        )

        c2.link_button(
            "📞 Call",
            f"tel:+{clean_phone}",
            use_container_width=True,
        )

    if group:

        c3.link_button(
            "👥 WhatsApp Group",
            group,
            use_container_width=True,
        )


# ---------------------------------------------------------
# REPORTS
# ---------------------------------------------------------

elif page == "Reports":

    st.title("📄 Report Studio")

    report_type = st.radio(
        "Report Level",
        [
            "School",
            "Teacher",
        ],
        horizontal=True,
    )

    if report_type == "School":

        school = st.selectbox(
            "School",
            schools["School"],
        )

        s = schools[
            schools["School"] == school
        ].iloc[0]

        insight = st.session_state.get(
            f"school_ai_{school}",
            "",
        )

        lines = [
            f"School: {school}",
            f"Health Score: {s['Health']}/100",
            f"Compliance: {s['Compliance']}%",
            f"Teachers: {s['Teachers']}",
            f"Active: {s['Active']}",
            f"Inactive: {s['Inactive']}",
            "",
            "Academic Intelligence:",
            insight
            or "Generate School Intelligence first for AI interpretation.",
        ]

        pdf = pdf_bytes(
            f"AcadIntel 360 - {school}",
            lines,
        )

        st.download_button(
            "⬇ Download School Report",
            pdf,
            file_name=(
                f"{school.replace(' ','_')}_"
                "School_360_Report.pdf"
            ),
            mime="application/pdf",
            use_container_width=True,
        )

    else:

        teacher = st.selectbox(
            "Teacher",
            teachers["Teacher"],
        )

        t = teachers[
            teachers["Teacher"] == teacher
        ].iloc[0]

        insight = st.session_state.get(
            f"teacher_ai_{teacher}",
            "",
        )

        lines = [
            f"Teacher: {teacher}",
            f"School: {t['School']}",
            f"Health: {t['Health']}/100",
            f"Lesson Prep: {t['Lesson Prep']} min",
            f"Lesson KPI: {t['Lesson KPI %']}%",
            f"Library: {t['Library']} min",
            f"Library KPI: {t['Library KPI %']}%",
            f"Content / Books: {t['Content']} min",
            "",
            "Teacher 360 Intelligence:",
            insight
            or "Generate Teacher Intelligence first for AI interpretation.",
        ]

        pdf = pdf_bytes(
            f"Teacher 360 Audit - {teacher}",
            lines,
        )

        st.download_button(
            "⬇ Download Teacher 360 Report",
            pdf,
            file_name=(
                f"{teacher.replace(' ','_')}_"
                "360_Audit_Report.pdf"
            ),
            mime="application/pdf",
            use_container_width=True,
        )
