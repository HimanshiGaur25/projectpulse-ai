# Optional ML dependency: the app can still run if Windows blocks sklearn.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
    SKLEARN_ERROR = ""
except (ImportError, OSError) as exc:
    SKLEARN_AVAILABLE = False
    SKLEARN_ERROR = str(exc)
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, timedelta
from io import StringIO


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ProjectPulse AI",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------
# Demo data
# --------------------------------------------------

def demo_projects():
    today = date.today()

    return pd.DataFrame([
        {
            "project_name": "CRM Migration",
            "client": "Acme Demo",
            "owner": "Aarav",
            "deadline": today + timedelta(days=5),
            "completion_pct": 62,
            "overdue_days": 0,
            "blockers": 2,
            "last_update_days": 3,
        },
        {
            "project_name": "Marketing Automation",
            "client": "Northstar Demo",
            "owner": "Priya",
            "deadline": today + timedelta(days=18),
            "completion_pct": 78,
            "overdue_days": 0,
            "blockers": 0,
            "last_update_days": 1,
        },
        {
            "project_name": "Sales Dashboard",
            "client": "Vertex Demo",
            "owner": "Kabir",
            "deadline": today - timedelta(days=2),
            "completion_pct": 85,
            "overdue_days": 2,
            "blockers": 1,
            "last_update_days": 5,
        },
        {
            "project_name": "HubSpot Implementation",
            "client": "BrightPath Demo",
            "owner": "Meera",
            "deadline": today + timedelta(days=10),
            "completion_pct": 35,
            "overdue_days": 0,
            "blockers": 3,
            "last_update_days": 4,
        },
        {
            "project_name": "Website Revamp",
            "client": "Orbit Demo",
            "owner": "Rohan",
            "deadline": today + timedelta(days=30),
            "completion_pct": 55,
            "overdue_days": 0,
            "blockers": 0,
            "last_update_days": 2,
        },
    ])


# --------------------------------------------------
# Health scoring rules
# --------------------------------------------------

def calculate_health(row):
    """Transparent rule-based score. Higher is healthier."""
    score = 100

    if row["overdue_days"] > 0:
        score -= min(35, row["overdue_days"] * 10)

    score -= min(30, row["blockers"] * 10)

    if row["last_update_days"] >= 4:
        score -= 15
    elif row["last_update_days"] >= 2:
        score -= 5

    days_left = (row["deadline"] - date.today()).days

    if days_left <= 7 and row["completion_pct"] < 70:
        score -= 20

    return max(0, min(100, score))


def health_label(score):
    if score >= 75:
        return "Healthy"
    if score >= 45:
        return "At Risk"
    return "Critical"


def risk_reason(row):
    reasons = []

    if row["overdue_days"] > 0:
        reasons.append(
            f"{row['overdue_days']} day(s) overdue"
        )

    if row["blockers"] > 0:
        reasons.append(
            f"{row['blockers']} open blocker(s)"
        )

    if row["last_update_days"] >= 4:
        reasons.append("No recent project update")

    days_left = (row["deadline"] - date.today()).days

    if days_left <= 7 and row["completion_pct"] < 70:
        reasons.append(
            "Deadline close, progress below 70%"
        )

    if not reasons:
        return "No rule-based warning detected"

    return "; ".join(reasons)



# --------------------------------------------------
# App header
# --------------------------------------------------

# --------------------------------------------------
# Slack message classification
# --------------------------------------------------

def classify_message(message):
    text = str(message).lower()

    categories = [
        (
            "Blocker",
            [
                "blocked", "blocker", "blocking",
                "can't proceed", "cannot proceed",
                "access issue", "waiting on",
            ],
        ),
        (
            "Delivery concern",
            [
                "behind schedule", "delay", "delayed",
                "missed deadline", "late delivery",
                "at risk", "timeline",
            ],
        ),
        (
            "Scope change",
            [
                "out of scope", "scope change",
                "add another", "additional feature",
                "new requirement", "extra work",
            ],
        ),
        (
            "Urgency",
            [
                "urgent", "asap", "immediately",
                "critical", "as soon as possible",
            ],
        ),
        (
            "Negative feedback",
            [
                "disappointed", "unhappy", "frustrated",
                "not acceptable", "poor quality",
                "not satisfied",
            ],
        ),
        (
            "Positive feedback",
            [
                "great job", "looks great", "thank you",
                "thanks for", "excellent", "well done",
            ],
        ),
    ]

    for category, keywords in categories:
        if any(keyword in text for keyword in keywords):
            return category

    return "Neutral"

def train_slack_model(training_df):
    required_columns = {"message", "category"}

    if not required_columns.issubset(training_df.columns):
        raise ValueError(
            "Training CSV must contain message and category columns."
        )

    data = training_df[["message", "category"]].copy()
    data = data.dropna()

    data["message"] = data["message"].astype(str).str.strip()
    data["category"] = data["category"].astype(str).str.strip()

    data = data[
        (data["message"] != "") &
        (data["category"] != "")
    ]

    if data["category"].nunique() < 2:
        raise ValueError(
            "Training data must contain at least 2 categories."
        )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                stop_words="english",
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ])

    model.fit(data["message"], data["category"])

    return model, data

st.title("ProjectPulse AI")

st.caption(
    "Project health and client intelligence for delivery teams"
)

st.info(
    "DEMO MODE — This dashboard uses synthetic sample data. "
    "Health scores are rule-based indicators, not ML predictions."
)


# --------------------------------------------------
# Data input and sample CSV download
# --------------------------------------------------

today = date.today()

sample_projects = demo_projects()

sample_csv = sample_projects.to_csv(index=False)


with st.sidebar:
    st.header("Project data")

    st.download_button(
        label="Download project sample CSV",
        data=sample_csv,
        file_name="projectpulse_sample.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload project CSV",
        type=["csv"],
        key="project_csv",
    )
    
    st.divider()
    st.subheader("ML model training")

    uploaded_training_file = st.file_uploader(
        "Upload labelled Slack messages",
        type=["csv"],
        key="training_csv",
    )

    st.caption(
        "CSV must contain message and category columns."
    )
    st.caption(
        "Project columns: project_name, client, owner, "
        "deadline, completion_pct, overdue_days, "
        "blockers, last_update_days"
    )

    st.divider()
    st.header("Slack data")

    sample_slack_csv = """timestamp,client,project_name,sender,message
2026-09-24 09:15,Vertex Demo,Sales Dashboard,Client PM,The dashboard delivery is already behind schedule.
2026-09-24 10:30,Vertex Demo,Sales Dashboard,Client Lead,We are blocked until the data access issue is resolved.
2026-09-24 11:00,Acme Demo,CRM Migration,Client PM,Can we add another workflow to the original scope?
2026-09-24 12:00,Northstar Demo,Marketing Automation,Client Lead,Thanks for the quick turnaround. The campaign looks great.
2026-09-24 13:00,BrightPath Demo,HubSpot Implementation,Client PM,Please share an update on the remaining implementation tasks.
"""

    st.download_button(
        label="Download Slack sample CSV",
        data=sample_slack_csv,
        file_name="projectpulse_slack_sample.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded_slack_file = st.file_uploader(
        "Upload Slack messages CSV",
        type=["csv"],
        key="slack_csv",
    )

    st.caption(
        "Required columns: timestamp, client, project_name, "
        "sender, message"
    )

    st.divider()
    st.caption("ProjectPulse AI · MVP")

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload project CSV",
        type=["csv"],
    )

    st.caption(
        "Required columns: project_name, client, owner, "
        "deadline, completion_pct, overdue_days, "
        "blockers, last_update_days"
    )

    st.divider()

    st.caption("ProjectPulse AI · MVP")


# --------------------------------------------------
# Load and validate project data
# --------------------------------------------------

if uploaded_file is not None:
    try:
        projects = pd.read_csv(uploaded_file)

        required_columns = [
            "project_name",
            "client",
            "owner",
            "deadline",
            "completion_pct",
            "overdue_days",
            "blockers",
            "last_update_days",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in projects.columns
        ]

        if missing_columns:
            st.error(
                "CSV is missing these columns: "
                + ", ".join(missing_columns)
            )
            st.stop()

        if projects.empty:
            st.error("The uploaded CSV contains no project rows.")
            st.stop()

        # Remove rows without essential project information.
        essential_columns = [
            "project_name",
            "client",
            "owner",
            "deadline",
        ]

        if projects[essential_columns].isna().any().any():
            st.error(
                "Project name, client, owner, and deadline "
                "cannot be blank."
            )
            st.stop()

        # Parse deadlines.
        projects["deadline"] = pd.to_datetime(
            projects["deadline"],
            errors="coerce",
        ).dt.date

        if projects["deadline"].isna().any():
            st.error(
                "Some deadlines are invalid. Use YYYY-MM-DD format."
            )
            st.stop()

        # Convert numeric fields.
        numeric_columns = [
            "completion_pct",
            "overdue_days",
            "blockers",
            "last_update_days",
        ]

        for column in numeric_columns:
            projects[column] = pd.to_numeric(
                projects[column],
                errors="coerce",
            )

        if projects[numeric_columns].isna().any().any():
            st.error(
                "Numeric fields contain blank or invalid values. "
                "Check completion_pct, overdue_days, blockers, "
                "and last_update_days."
            )
            st.stop()

        if not projects["completion_pct"].between(0, 100).all():
            st.error(
                "completion_pct must be between 0 and 100."
            )
            st.stop()

        if (projects[numeric_columns[1:]] < 0).any().any():
            st.error(
                "Overdue days, blockers, and last update days "
                "cannot be negative."
            )
            st.stop()

        projects["project_name"] = (
            projects["project_name"].astype(str).str.strip()
        )
        projects["client"] = (
            projects["client"].astype(str).str.strip()
        )
        projects["owner"] = (
            projects["owner"].astype(str).str.strip()
        )

        if (
            projects[["project_name", "client", "owner"]]
            .eq("")
            .any()
            .any()
        ):
            st.error(
                "Project name, client, and owner cannot be blank."
            )
            st.stop()

        data_source = "Uploaded CSV"

    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()

else:
    projects = demo_projects()
    data_source = "Synthetic demo data"


# --------------------------------------------------
# Calculate health and risk explanations
# --------------------------------------------------

projects["health_score"] = projects.apply(
    calculate_health,
    axis=1,
)

projects["health"] = projects["health_score"].apply(
    health_label
)

projects["risk_reason"] = projects.apply(
    risk_reason,
    axis=1,
)

st.caption(f"Current source: {data_source}")


# --------------------------------------------------
# Portfolio filters
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    clients = ["All clients"] + sorted(
        projects["client"].dropna().unique().tolist()
    )

    selected_client = st.selectbox(
        "Client",
        clients,
    )

with col2:
    statuses = [
        "All statuses",
        "Healthy",
        "At Risk",
        "Critical",
    ]

    selected_status = st.selectbox(
        "Health status",
        statuses,
    )


filtered = projects.copy()

if selected_client != "All clients":
    filtered = filtered[
        filtered["client"] == selected_client
    ]

if selected_status != "All statuses":
    filtered = filtered[
        filtered["health"] == selected_status
    ]

if filtered.empty:
    st.warning("No projects match these filters.")
    st.stop()


# --------------------------------------------------
# Portfolio metrics
# --------------------------------------------------

total = len(filtered)

healthy = (filtered["health"] == "Healthy").sum()
at_risk = (filtered["health"] == "At Risk").sum()
critical = (filtered["health"] == "Critical").sum()

avg_health = filtered["health_score"].mean()

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Projects", total)
m2.metric("Healthy", int(healthy))
m3.metric("At Risk", int(at_risk))
m4.metric("Critical", int(critical))
m5.metric("Avg. Health", f"{avg_health:.0f}/100")

st.divider()


# --------------------------------------------------
# Charts
# --------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("Project health")

    chart = px.bar(
        filtered.sort_values("health_score"),
        x="health_score",
        y="project_name",
        color="health",
        orientation="h",
        range_x=[0, 100],
        labels={
            "health_score": "Health score",
            "project_name": "Project",
            "health": "Status",
        },
    )

    st.plotly_chart(
        chart,
        use_container_width=True,
    )

with right:
    st.subheader("Project completion")

    completion_chart = px.bar(
        filtered,
        x="project_name",
        y="completion_pct",
        range_y=[0, 100],
        labels={
            "project_name": "Project",
            "completion_pct": "Complete (%)",
        },
    )

    st.plotly_chart(
        completion_chart,
        use_container_width=True,
    )


# --------------------------------------------------
# Project portfolio table
# --------------------------------------------------

st.subheader("Project portfolio")

display = filtered[
    [
        "project_name",
        "client",
        "owner",
        "deadline",
        "completion_pct",
        "health_score",
        "health",
        "blockers",
    ]
].copy()

display = display.rename(columns={
    "project_name": "Project",
    "client": "Client",
    "owner": "Owner",
    "deadline": "Deadline",
    "completion_pct": "Complete (%)",
    "health_score": "Health score",
    "health": "Status",
    "blockers": "Open blockers",
})

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# Project detail and evidence panel
# --------------------------------------------------

st.divider()
st.subheader("Project detail & evidence")

project_names = filtered["project_name"].tolist()

selected_project = st.selectbox(
    "Select a project to investigate",
    project_names,
    key="detail_project",
)

row = filtered[
    filtered["project_name"] == selected_project
].iloc[0]

days_left = (row["deadline"] - date.today()).days


# Project summary
st.markdown(f"### {row['project_name']}")

st.caption(
    f"Client: {row['client']} · "
    f"Owner: {row['owner']} · "
    f"Deadline: {row['deadline']}"
)

d1, d2, d3, d4 = st.columns(4)

d1.metric(
    "Health score",
    f"{row['health_score']}/100",
)

d2.metric(
    "Status",
    row["health"],
)

d3.metric(
    "Completion",
    f"{row['completion_pct']}%",
)

if days_left < 0:
    deadline_label = f"{abs(days_left)} days overdue"
else:
    deadline_label = f"{days_left} days left"

d4.metric(
    "Deadline",
    deadline_label,
)


# --------------------------------------------------
# Evidence behind the health status
# --------------------------------------------------

evidence = []

if row["overdue_days"] > 0:
    evidence.append({
        "Signal": "Overdue work",
        "Observed evidence": (
            f"{row['overdue_days']} day(s) overdue"
        ),
        "Why it matters": (
            "Delivery is already past its recorded schedule."
        ),
    })

if row["blockers"] > 0:
    evidence.append({
        "Signal": "Open blockers",
        "Observed evidence": (
            f"{row['blockers']} blocker(s)"
        ),
        "Why it matters": (
            "Unresolved blockers may prevent planned work."
        ),
    })

if row["last_update_days"] >= 4:
    evidence.append({
        "Signal": "Stale project update",
        "Observed evidence": (
            f"{row['last_update_days']} days since last update"
        ),
        "Why it matters": (
            "The current project state may need confirmation."
        ),
    })

if days_left <= 7 and row["completion_pct"] < 70:
    evidence.append({
        "Signal": "Deadline proximity",
        "Observed evidence": (
            f"{days_left} day(s) until deadline; "
            f"{row['completion_pct']}% complete"
        ),
        "Why it matters": (
            "The deadline is close relative to recorded progress."
        ),
    })

if not evidence:
    evidence.append({
        "Signal": "No warning triggered",
        "Observed evidence": (
            "No configured risk rule was triggered"
        ),
        "Why it matters": (
            "This does not guarantee delivery; keep monitoring."
        ),
    })

st.markdown("#### Evidence behind this status")

st.dataframe(
    pd.DataFrame(evidence),
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# Suggested PM follow-up
# --------------------------------------------------

st.markdown("#### Suggested PM follow-up")

actions = []

if row["overdue_days"] > 0:
    actions.append(
        "Confirm the revised delivery date and recovery plan "
        "with the project owner."
    )

if row["blockers"] > 0:
    actions.append(
        "Review open blockers, assign an owner, and agree "
        "on a resolution date."
    )

if row["last_update_days"] >= 4:
    actions.append(
        "Request a current status update from the project owner."
    )

if days_left <= 7 and row["completion_pct"] < 70:
    actions.append(
        "Review remaining scope and confirm whether the "
        "deadline is achievable."
    )

if not actions:
    actions.append(
        "Continue routine monitoring and keep the next "
        "project update current."
    )

for action in actions:
    st.markdown(f"- {action}")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.caption(
    "Evidence is derived from the project fields provided. "
    "Suggested actions are prompts for human review. "
    "Health scores are rule-based indicators, not trained "
    "ML predictions. They do not establish client satisfaction "
    "or guarantee delivery outcomes."
)

# --------------------------------------------------
# Slack message intelligence
# --------------------------------------------------

st.divider()
st.subheader("Client communication intelligence")

st.caption(
    "Messages are classified using transparent keyword rules. "
    "This is a demo baseline, not an ML model or a measure "
    "of client satisfaction."
)

if uploaded_slack_file is None:
    st.info(
        "Upload a Slack messages CSV from the sidebar to "
        "explore communication signals."
    )

else:
    try:
        messages = pd.read_csv(uploaded_slack_file)

        required_message_columns = [
            "timestamp",
            "client",
            "project_name",
            "sender",
            "message",
        ]

        missing_message_columns = [
            column
            for column in required_message_columns
            if column not in messages.columns
        ]

        if missing_message_columns:
            st.error(
                "Slack CSV is missing these columns: "
                + ", ".join(missing_message_columns)
            )
            st.stop()

        if messages.empty:
            st.warning("The Slack CSV contains no messages.")
            st.stop()

        messages = messages.dropna(
            subset=["client", "project_name", "message"]
        ).copy()

        messages["client"] = (
            messages["client"].astype(str).str.strip()
        )
        messages["project_name"] = (
            messages["project_name"].astype(str).str.strip()
        )
        messages["message"] = (
            messages["message"].astype(str).str.strip()
        )

        messages["timestamp"] = pd.to_datetime(
            messages["timestamp"],
            errors="coerce",
        )

        messages = messages[
            messages["message"] != ""
        ].copy()

        if messages.empty:
            st.warning(
                "No usable messages found. Check the CSV values."
            )
            st.stop()

        messages["category"] = messages["message"].apply(
            classify_message
        )

        # Match communications to the project currently selected.
        project_messages = messages[
            (
                messages["project_name"]
                == str(row["project_name"])
            )
            &
            (
                messages["client"]
                == str(row["client"])
            )
        ].copy()

        st.markdown(
            f"#### Messages for {row['project_name']}"
        )

        st.caption(
            f"{len(project_messages)} matching message(s) "
            f"from the uploaded file."
        )

        if project_messages.empty:
            st.info(
                "No messages matched this project and client. "
                "Check spelling in the project CSV and Slack CSV."
            )

        else:
            # Summary counts
            category_counts = (
                project_messages["category"]
                .value_counts()
                .rename_axis("Category")
                .reset_index(name="Messages")
            )

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Message categories**")
                st.dataframe(
                    category_counts,
                    use_container_width=True,
                    hide_index=True,
                )

            with c2:
                st.markdown("**Signals to review**")

                signal_categories = [
                    "Blocker",
                    "Delivery concern",
                    "Scope change",
                    "Urgency",
                    "Negative feedback",
                ]

                signal_count = project_messages[
                    project_messages["category"].isin(
                        signal_categories
                    )
                ].shape[0]

                st.metric(
                    "Messages with potential risk signals",
                    signal_count,
                )

                st.caption(
                    "These are keyword matches for human review, "
                    "not confirmed project risks."
                )

            st.markdown("**Message evidence**")

            message_display = project_messages[
                [
                    "timestamp",
                    "sender",
                    "category",
                    "message",
                ]
            ].copy()

            message_display = message_display.rename(
                columns={
                    "timestamp": "Timestamp",
                    "sender": "Sender",
                    "category": "Detected category",
                    "message": "Message",
                }
            )

            st.dataframe(
                message_display,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**Suggested PM follow-up**")

            category_actions = {
                "Blocker": (
                    "Confirm the blocker, owner, and expected "
                    "resolution date."
                ),
                "Delivery concern": (
                    "Check the delivery plan and confirm the "
                    "latest timeline with the project team."
                ),
                "Scope change": (
                    "Review the request against the agreed scope "
                    "before committing to additional work."
                ),
                "Urgency": (
                    "Confirm the deadline and clarify what needs "
                    "immediate attention."
                ),
                "Negative feedback": (
                    "Review the specific concern with the account "
                    "or project owner."
                ),
            }

            detected_categories = set(
                project_messages["category"]
            )

            actions_to_review = [
                category_actions[category]
                for category in signal_categories
                if category in detected_categories
            ]

            if actions_to_review:
                for action in actions_to_review:
                    st.markdown(f"- {action}")
            else:
                st.write(
                    "No configured risk keywords were detected. "
                    "Continue normal project monitoring."
                )

    except Exception as exc:
        st.error(
            f"Could not process the Slack CSV: {exc}"
        )
st.divider()
st.header("Slack ML classifier")

if not SKLEARN_AVAILABLE:
    st.warning(
        "The ML model is unavailable in this environment because "
        "scikit-learn could not be loaded."
    )
    st.caption(
        "Windows Application Control may have blocked a compiled "
        "scikit-learn component. Your project dashboard and the "
        "keyword-based Slack classifier can still run."
    )
else:
    if uploaded_training_file is None:
        st.info(
            "Upload slack_training.csv in the sidebar "
            "to train the message classifier."
        )
    elif uploaded_slack_file is None:
        st.info(
            "Upload a Slack messages CSV to compare "
            "ML predictions with keyword classifications."
        )
    else:
        try:
            training_df = pd.read_csv(uploaded_training_file)
            slack_df = pd.read_csv(uploaded_slack_file)
            model, training_data = train_slack_model(training_df)

            if "message" not in slack_df.columns:
                st.error("Slack CSV must contain a message column.")
            else:
                messages_for_model = (
                    slack_df["message"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )
                valid = messages_for_model != ""
                comparison = slack_df.loc[valid].copy()
                messages_for_model = messages_for_model.loc[valid]

                if comparison.empty:
                    st.warning("No usable Slack messages found.")
                else:
                    comparison["Keyword category"] = (
                        messages_for_model.apply(classify_message)
                    )
                    comparison["ML category"] = model.predict(
                        messages_for_model
                    )

                    st.success(
                        f"Model trained on {len(training_data)} labelled "
                        f"messages across "
                        f"{training_data['category'].nunique()} categories."
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Messages analysed", len(comparison))
                    with col2:
                        agreement = (
                            comparison["Keyword category"]
                            == comparison["ML category"]
                        ).mean()
                        st.metric(
                            "Keyword/ML agreement", f"{agreement:.0%}"
                        )

                    st.caption(
                        "Agreement is not model accuracy. Ground-truth "
                        "labels are needed to measure accuracy."
                    )
                    st.subheader("Compare predictions")
                    st.dataframe(
                        comparison[
                            ["message", "Keyword category", "ML category"]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.subheader("ML category distribution")
                    st.bar_chart(comparison["ML category"].value_counts())
        except Exception as exc:
            st.error(f"Could not train the model: {exc}")
