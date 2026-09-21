
"""
AI-Powered Customer Churn & Revenue Analytics
==============================================

End-to-end data analytics + AI/ML portfolio project.

Features:
- Generates a realistic synthetic customer dataset.
- Cleans and validates data.
- Performs descriptive analytics and KPI analysis.
- Trains a Logistic Regression churn prediction model.
- Evaluates the model using accuracy, precision, recall, F1 and ROC-AUC.
- Identifies the most influential churn factors.
- Scores customers by churn risk.
- Creates revenue/churn charts and CSV outputs.
- Produces an AI-style business action report from model results.

Run:
    pip install -r requirements.txt
    python ai_customer_churn_analytics.py
"""

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR / "output"
OUTPUT.mkdir(exist_ok=True)


def generate_customer_data(n=6000):
    """Generate synthetic customer records with a realistic churn signal."""
    rng = np.random.default_rng(SEED)

    tenure = rng.integers(1, 73, n)
    monthly_charges = np.round(rng.normal(3200, 1100, n).clip(700, 8500), 2)
    support_tickets = rng.poisson(2.0, n).clip(0, 10)
    late_payments = rng.poisson(1.1, n).clip(0, 7)
    usage_hours = np.round(rng.normal(42, 16, n).clip(2, 100), 1)

    contract = rng.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        n, p=[0.58, 0.27, 0.15]
    )
    payment = rng.choice(
        ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash"],
        n, p=[0.30, 0.24, 0.18, 0.20, 0.08]
    )
    region = rng.choice(
        ["North", "South", "East", "West"],
        n, p=[0.25, 0.30, 0.20, 0.25]
    )
    plan = rng.choice(
        ["Basic", "Standard", "Premium"],
        n, p=[0.42, 0.38, 0.20]
    )
    support_plan = rng.choice(["Yes", "No"], n, p=[0.35, 0.65])

    # Construct a latent churn probability so the ML task has learnable structure.
    logit = (
        -2.2
        + 0.035 * support_tickets
        + 0.28 * late_payments
        - 0.022 * tenure
        + 0.012 * (monthly_charges - 3000)
        - 0.018 * usage_hours
        + np.where(contract == "Month-to-Month", 1.15, 0)
        + np.where(contract == "One Year", 0.30, 0)
        + np.where(plan == "Basic", 0.30, 0)
        + np.where(support_plan == "No", 0.15, 0)
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    df = pd.DataFrame({
        "customer_id": [f"CUST{100000+i}" for i in range(n)],
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "support_tickets": support_tickets,
        "late_payments": late_payments,
        "usage_hours": usage_hours,
        "contract": contract,
        "payment_method": payment,
        "region": region,
        "plan": plan,
        "support_plan": support_plan,
        "churn": churn
    })

    # Add a few missing values to demonstrate cleaning.
    for col in ["monthly_charges", "usage_hours", "payment_method"]:
        idx = rng.choice(n, size=20, replace=False)
        df.loc[idx, col] = np.nan

    return df


def clean_data(df):
    df = df.copy()
    df = df.drop_duplicates(subset=["customer_id"])
    numeric_cols = [
        "tenure_months", "monthly_charges",
        "support_tickets", "late_payments", "usage_hours"
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    categorical_cols = ["contract", "payment_method", "region", "plan", "support_plan"]
    for c in categorical_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    df["annual_value"] = (df["monthly_charges"] * 12).round(2)
    return df


def train_model(df):
    features = [
        "tenure_months", "monthly_charges", "support_tickets",
        "late_payments", "usage_hours", "contract",
        "payment_method", "region", "plan", "support_plan"
    ]
    target = "churn"

    X = df[features]
    y = df[target]

    numeric = [
        "tenure_months", "monthly_charges",
        "support_tickets", "late_payments", "usage_hours"
    ]
    categorical = ["contract", "payment_method", "region", "plan", "support_plan"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=SEED))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    }

    return model, X_test, y_test, probabilities, predictions, metrics


def feature_importance(model):
    prep = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]

    names = prep.get_feature_names_out()
    coefficients = clf.coef_[0]

    result = pd.DataFrame({
        "feature": names,
        "coefficient": coefficients,
        "absolute_impact": np.abs(coefficients)
    }).sort_values("absolute_impact", ascending=False)

    return result


def create_customer_risk_table(df, model):
    features = [
        "tenure_months", "monthly_charges", "support_tickets",
        "late_payments", "usage_hours", "contract",
        "payment_method", "region", "plan", "support_plan"
    ]
    scored = df.copy()
    scored["churn_probability"] = model.predict_proba(scored[features])[:, 1]

    scored["risk_level"] = pd.cut(
        scored["churn_probability"],
        bins=[-0.01, 0.30, 0.60, 1.00],
        labels=["Low", "Medium", "High"]
    )

    scored["recommended_action"] = np.select(
        [
            scored["risk_level"].eq("High"),
            scored["risk_level"].eq("Medium")
        ],
        [
            "Priority retention call + targeted offer",
            "Send personalized engagement offer"
        ],
        default="Standard engagement"
    )

    return scored.sort_values("churn_probability", ascending=False)


def create_kpis(df, metrics):
    total = len(df)
    churned = int(df["churn"].sum())
    annual_revenue_at_risk = df.loc[df["churn"] == 1, "annual_value"].sum()

    rows = [
        ["Total Customers", total],
        ["Churned Customers", churned],
        ["Churn Rate %", round(churned / total * 100, 2)],
        ["Average Monthly Charges", round(df["monthly_charges"].mean(), 2)],
        ["Annual Revenue at Risk", round(annual_revenue_at_risk, 2)],
        ["Model Accuracy", round(metrics["Accuracy"], 4)],
        ["Model Precision", round(metrics["Precision"], 4)],
        ["Model Recall", round(metrics["Recall"], 4)],
        ["Model F1 Score", round(metrics["F1 Score"], 4)],
        ["Model ROC-AUC", round(metrics["ROC-AUC"], 4)],
    ]
    return pd.DataFrame(rows, columns=["KPI", "Value"])


def create_sql_analysis(df):
    db_path = OUTPUT / "customer_analytics.db"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    df.to_sql("customers", conn, index=False, if_exists="replace")

    query = """
        SELECT
            contract,
            COUNT(*) AS customers,
            ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
            ROUND(AVG(churn) * 100, 2) AS churn_rate_pct
        FROM customers
        GROUP BY contract
        ORDER BY churn_rate_pct DESC;
    """
    result = pd.read_sql_query(query, conn)
    conn.close()
    return result


def create_charts(df, risk, importance):
    # 1. Churn by contract
    contract_stats = df.groupby("contract", as_index=False)["churn"].mean()
    contract_stats["churn_pct"] = contract_stats["churn"] * 100

    plt.figure(figsize=(8, 5))
    plt.bar(contract_stats["contract"], contract_stats["churn_pct"])
    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract")
    plt.ylabel("Churn Rate (%)")
    plt.tight_layout()
    plt.savefig(OUTPUT / "churn_by_contract.png", dpi=160)
    plt.close()

    # 2. Feature importance
    top = importance.head(10).sort_values("coefficient")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["coefficient"])
    plt.title("Top AI Model Drivers of Churn")
    plt.xlabel("Logistic Regression Coefficient")
    plt.tight_layout()
    plt.savefig(OUTPUT / "ai_feature_importance.png", dpi=160)
    plt.close()

    # 3. Risk distribution
    risk_counts = risk["risk_level"].value_counts().reindex(
        ["Low", "Medium", "High"], fill_value=0
    )
    plt.figure(figsize=(7, 5))
    plt.bar(risk_counts.index, risk_counts.values)
    plt.title("Customers by AI Churn-Risk Level")
    plt.xlabel("Risk Level")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(OUTPUT / "customer_risk_distribution.png", dpi=160)
    plt.close()


def create_ai_insights(df, importance, risk, metrics):
    top_features = importance.head(5)["feature"].tolist()
    high_risk = int((risk["risk_level"] == "High").sum())
    high_risk_value = risk.loc[risk["risk_level"] == "High", "annual_value"].sum()

    lines = [
        "AI-ASSISTED BUSINESS INSIGHTS",
        "=" * 42,
        "",
        f"1. Overall churn rate: {df['churn'].mean()*100:.2f}%.",
        f"2. {high_risk:,} customers are classified as HIGH churn risk.",
        f"3. Estimated annual customer value associated with the high-risk group: "
        f"{high_risk_value:,.2f}.",
        "",
        "4. Strongest model signals by absolute coefficient:",
    ]
    lines.extend([f"   - {x}" for x in top_features])
    lines += [
        "",
        "5. Recommended analytical actions:",
        "   - Prioritize high-risk customers for retention outreach.",
        "   - Investigate contract type and payment behavior among high-risk users.",
        "   - Use support-ticket volume as a signal for proactive service recovery.",
        "   - Monitor model performance on new customer data before operational deployment.",
        "",
        "6. Model validation:",
        f"   Accuracy: {metrics['Accuracy']:.3f}",
        f"   Precision: {metrics['Precision']:.3f}",
        f"   Recall: {metrics['Recall']:.3f}",
        f"   F1 Score: {metrics['F1 Score']:.3f}",
        f"   ROC-AUC: {metrics['ROC-AUC']:.3f}",
        "",
        "Note: This is a synthetic portfolio dataset. The model is for demonstration and should "
        "not be treated as a production decision system without validation, monitoring and fairness checks."
    ]
    (OUTPUT / "ai_business_insights.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("Generating synthetic customer data...")
    raw = generate_customer_data()

    print("Cleaning data...")
    df = clean_data(raw)

    print("Training AI churn model...")
    model, X_test, y_test, probabilities, predictions, metrics = train_model(df)

    print("Extracting model drivers...")
    importance = feature_importance(model)

    print("Scoring customer churn risk...")
    risk = create_customer_risk_table(df, model)

    print("Creating KPI and SQL outputs...")
    kpis = create_kpis(df, metrics)
    sql_result = create_sql_analysis(df)

    df.to_csv(OUTPUT / "customer_data_cleaned.csv", index=False)
    risk.to_csv(OUTPUT / "customer_churn_risk_scores.csv", index=False)
    importance.to_csv(OUTPUT / "ai_feature_importance.csv", index=False)
    kpis.to_csv(OUTPUT / "kpi_summary.csv", index=False)
    sql_result.to_csv(OUTPUT / "sql_contract_analysis.csv", index=False)

    cm = confusion_matrix(y_test, predictions)
    pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    ).to_csv(OUTPUT / "confusion_matrix.csv")

    create_charts(df, risk, importance)
    create_ai_insights(df, importance, risk, metrics)

    print("\nAI DATA ANALYTICS PROJECT COMPLETED")
    print("=" * 48)
    print(kpis.to_string(index=False))
    print("\nTop model drivers:")
    print(importance.head(8)[["feature", "coefficient"]].to_string(index=False))
    print(f"\nOutputs saved in: {OUTPUT}")


if __name__ == "__main__":
    main()
