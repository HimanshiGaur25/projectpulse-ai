from datetime import date, timedelta
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
    SKLEARN_ERROR = ""
except (ImportError, OSError) as exc:
    SKLEARN_AVAILABLE = False
    SKLEARN_ERROR = str(exc)

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

