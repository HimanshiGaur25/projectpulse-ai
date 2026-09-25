
from pathlib import Path
from io import BytesIO

import pandas as pd

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from logic import (
    calculate_health,
    health_label,
    risk_reason,
    train_slack_model,
)


# ============================================
# PATHS AND APP CONFIGURATION
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
TRAINING_FILE = BASE_DIR / "slack_training.csv"

app = FastAPI(
    title="ProjectPulse AI",
    description="Project health and communication analysis",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# PROJECT HEALTH CONFIGURATION
# ============================================

REQUIRED_PROJECT_COLUMNS = {
    "project_name",
    "deadline",
    "completion_pct",
    "overdue_days",
    "blockers",
    "last_update_days",
}


# ============================================
# SLACK MODEL
# ============================================

slack_model = None
slack_training_rows = 0
slack_model_error = None


def load_slack_model():
    """
    Train the Slack classifier using slack_training.csv.
    """

    global slack_model
    global slack_training_rows
    global slack_model_error

    if not TRAINING_FILE.exists():
        slack_model_error = (
            f"Training file not found: {TRAINING_FILE}"
        )
        print(slack_model_error)
        return

    try:
        training_df = pd.read_csv(TRAINING_FILE)

        model, cleaned_data = train_slack_model(training_df)

        slack_model = model
        slack_training_rows = len(cleaned_data)
        slack_model_error = None

        print("Slack ML model trained successfully.")
        print(f"Training examples: {slack_training_rows}")
        print(
            "Categories:",
            sorted(cleaned_data["category"].unique().tolist()),
        )

    except Exception as exc:
        slack_model = None
        slack_model_error = str(exc)

        print("Slack model training failed:", slack_model_error)


# Train once when the backend starts
load_slack_model()


# ============================================
# HEALTH CHECK
# ============================================

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "application": "ProjectPulse AI",
        "slack_model_ready": slack_model is not None,
    }


# ============================================
# PROJECT HEALTH ANALYSIS
# ============================================

@app.post("/api/projects/analyze")
async def analyze_projects(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        df.columns = df.columns.str.strip().str.lower()

        missing = REQUIRED_PROJECT_COLUMNS - set(df.columns)

        if missing:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing required columns: "
                    + ", ".join(sorted(missing))
                ),
            )

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV contains no project rows.",
            )

        # Optional descriptive fields
        if "client" not in df.columns:
            df["client"] = "—"

        if "owner" not in df.columns:
            df["owner"] = "—"

        # Parse deadlines
        df["deadline"] = pd.to_datetime(
            df["deadline"],
            errors="coerce",
        ).dt.date

        if df["deadline"].isna().any():
            raise HTTPException(
                status_code=400,
                detail="Some deadline values are missing or invalid.",
            )

        # Validate numeric fields
        numeric_columns = [
            "completion_pct",
            "overdue_days",
            "blockers",
            "last_update_days",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        if df[numeric_columns].isna().any().any():
            raise HTTPException(
                status_code=400,
                detail=(
                    "Some project metric values are missing or invalid."
                ),
            )

        results = []

        for _, row in df.iterrows():
            project = row.to_dict()

            score = calculate_health(project)
            label = health_label(score)
            reason = risk_reason(project)

            results.append({
                "project_name": str(project["project_name"]),
                "client": str(project["client"]),
                "owner": str(project["owner"]),
                "deadline": project["deadline"].isoformat(),
                "completion_pct": float(project["completion_pct"]),
                "overdue_days": int(project["overdue_days"]),
                "blockers": int(project["blockers"]),
                "last_update_days": int(
                    project["last_update_days"]
                ),
                "health_score": score,
                "health_label": label,
                "risk_reason": reason,
            })

        counts = {
            "total": len(results),
            "healthy": sum(
                p["health_label"] == "Healthy"
                for p in results
            ),
            "at_risk": sum(
                p["health_label"] == "At Risk"
                for p in results
            ),
            "critical": sum(
                p["health_label"] == "Critical"
                for p in results
            ),
        }

        return {
            "status": "success",
            "filename": file.filename,
            "summary": counts,
            "projects": results,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not analyze the project CSV: {exc}",
        )


# ============================================
# SLACK MESSAGE ANALYSIS
# ============================================

@app.post("/api/slack/analyze")
async def analyze_slack(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a Slack CSV file.",
        )

    if slack_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The Slack model is not available. "
                "Check slack_training.csv and the backend terminal."
            ),
        )

    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        if "message" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The Slack CSV must contain a 'message' column."
                ),
            )

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded Slack CSV contains no rows.",
            )

        # Remove blank messages, retaining original row information
        df["message"] = df["message"].fillna("").astype(str).str.strip()

        df = df[df["message"] != ""].copy()

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="No non-empty messages were found in the CSV.",
            )

        # Predict categories using the trained ML model
        messages = df["message"].tolist()

        predictions = slack_model.predict(messages)

        # Model confidence, where available
        if hasattr(slack_model, "predict_proba"):
            probabilities = slack_model.predict_proba(messages)
            confidences = probabilities.max(axis=1)
        else:
            confidences = [None] * len(messages)

        results = []

        for index, (_, row) in enumerate(df.iterrows()):
            result = {
                "message": row["message"],
                "category": str(predictions[index]),
                "confidence": (
                    round(float(confidences[index]) * 100, 1)
                    if confidences[index] is not None
                    else None
                ),
            }

            # Include useful Slack metadata if provided
            for column in [
                "user",
                "username",
                "channel",
                "date",
                "timestamp",
            ]:
                if column in df.columns:
                    value = row[column]

                    result[column] = (
                        None if pd.isna(value) else str(value)
                    )

            results.append(result)

        # Count messages in each predicted category
        category_counts = (
            pd.Series(predictions)
            .value_counts()
            .to_dict()
        )

        counts = {
            str(category): int(count)
            for category, count in category_counts.items()
        }

        return {
            "status": "success",
            "filename": file.filename,
            "summary": {
                "total_messages": len(results),
                "categories": len(counts),
                "category_counts": counts,
            },
            "messages": results,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not analyze the Slack CSV: {exc}",
        )


# ============================================
# FRONTEND
# Keep this AFTER all API routes
# ============================================

app.mount(
    "/",
    StaticFiles(
        directory=str(FRONTEND_DIR),
        html=True,
    ),
    name="frontend",
)