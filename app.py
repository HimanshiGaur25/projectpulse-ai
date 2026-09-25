
from pathlib import Path

import pandas as pd
import streamlit as st

from logic import (
    SKLEARN_AVAILABLE,
    SKLEARN_ERROR,
    calculate_health,
    health_label,
    risk_reason,
    train_slack_model,
)

BASE_DIR = Path(__file__).resolve().parent
TRAINING_FILE = BASE_DIR / "slack_training.csv"

st.set_page_config(
    page_title="ProjectPulse AI",
    page_icon="📊",
    layout="wide",
)

st.title("ProjectPulse AI")
st.caption("Project health monitoring and Slack message analysis")

project_tab, slack_tab = st.tabs(
    ["Project Health", "Slack Analysis"]
)

# ---------------- PROJECT HEALTH ----------------
with project_tab:
    st.subheader("Project Health Analysis")
    st.write("Upload a CSV containing your project details.")

    project_file = st.file_uploader(
        "Upload project CSV",
        type=["csv"],
        key="project_csv",
    )

    if project_file is not None:
        try:
            projects = pd.read_csv(project_file)

            required = {
                "project_name",
                "client",
                "owner",
                "deadline",
                "completion_pct",
                "overdue_days",
                "blockers",
                "last_update_days",
            }

            missing = required - set(projects.columns)

            if missing:
                st.error(
                    "Missing columns: " + ", ".join(sorted(missing))
                )
            else:
                projects["deadline"] = pd.to_datetime(
                    projects["deadline"], errors="coerce"
                ).dt.date

                numeric_columns = [
                    "completion_pct",
                    "overdue_days",
                    "blockers",
                    "last_update_days",
                ]

                for column in numeric_columns:
                    projects[column] = pd.to_numeric(
                        projects[column], errors="coerce"
                    )

                projects = projects.dropna(
                    subset=["deadline"] + numeric_columns
                )

                if projects.empty:
                    st.warning("No valid project rows found.")
                else:
                    projects["health_score"] = projects.apply(
                        calculate_health, axis=1
                    )
                    projects["status"] = projects[
                        "health_score"
                    ].apply(health_label)
                    projects["risk_reason"] = projects.apply(
                        risk_reason, axis=1
                    )

                    total = len(projects)
                    healthy = (projects["status"] == "Healthy").sum()
                    at_risk = (projects["status"] == "At Risk").sum()
                    critical = (projects["status"] == "Critical").sum()

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Projects", total)
                    c2.metric("Healthy", int(healthy))
                    c3.metric("At Risk", int(at_risk))
                    c4.metric("Critical", int(critical))

                    st.dataframe(
                        projects,
                        use_container_width=True,
                        hide_index=True,
                    )

                    csv = projects.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Project Results",
                        data=csv,
                        file_name="project_health_results.csv",
                        mime="text/csv",
                    )

        except Exception as exc:
            st.error(f"Could not analyse this CSV: {exc}")


# ---------------- SLACK ANALYSIS ----------------
with slack_tab:
    st.subheader("Slack Message Analysis")
    st.write("Upload a CSV containing a `message` column.")

    if not SKLEARN_AVAILABLE:
        st.error(f"Scikit-learn is unavailable: {SKLEARN_ERROR}")
    elif not TRAINING_FILE.exists():
        st.error(
            "slack_training.csv was not found. "
            "Add it to the root of your GitHub repository."
        )
    else:
        try:
            @st.cache_resource
            def load_slack_model():
                training_data = pd.read_csv(TRAINING_FILE)
                return train_slack_model(training_data)

            model, training_data = load_slack_model()

            st.caption(
                f"Model loaded with {len(training_data)} training examples."
            )

            slack_file = st.file_uploader(
                "Upload Slack messages CSV",
                type=["csv"],
                key="slack_csv",
            )

            if slack_file is not None:
                messages = pd.read_csv(slack_file)

                if "message" not in messages.columns:
                    st.error("Your CSV must contain a 'message' column.")
                else:
                    messages = messages.copy()
                    messages["message"] = (
                        messages["message"].fillna("").astype(str).str.strip()
                    )
                    messages = messages[
                        messages["message"] != ""
                    ].copy()

                    if messages.empty:
                        st.warning("No messages found in this CSV.")
                    else:
                        predictions = model.predict(messages["message"])
                        probabilities = model.predict_proba(
                            messages["message"]
                        ).max(axis=1)

                        messages["predicted_category"] = predictions
                        messages["prediction_probability"] = (
                            probabilities * 100
                        ).round(1)

                        counts = messages[
                            "predicted_category"
                        ].value_counts()

                        c1, c2 = st.columns(2)
                        c1.metric("Messages Analysed", len(messages))
                        c2.metric(
                            "Categories Found",
                            messages["predicted_category"].nunique(),
                        )

                        st.subheader("Category Breakdown")
                        st.bar_chart(counts)

                        selected_categories = st.multiselect(
                            "Filter by category",
                            options=sorted(
                                messages["predicted_category"].unique()
                            ),
                            default=sorted(
                                messages["predicted_category"].unique()
                            ),
                        )

                        filtered = messages[
                            messages["predicted_category"].isin(
                                selected_categories
                            )
                        ]

                        st.dataframe(
                            filtered,
                            use_container_width=True,
                            hide_index=True,
                        )

                        csv = filtered.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "Download Slack Results",
                            data=csv,
                            file_name="slack_analysis_results.csv",
                            mime="text/csv",
                        )

        except Exception as exc:
            st.error(f"Could not analyse Slack messages: {exc}")