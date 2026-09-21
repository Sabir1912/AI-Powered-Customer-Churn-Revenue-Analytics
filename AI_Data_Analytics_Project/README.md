# AI-Powered Customer Churn & Revenue Analytics

## Project Overview
An end-to-end Data Analytics + AI/ML project that transforms customer data into business insights and predicts which customers are at risk of churn.

## Why this project is different
Instead of only reporting historical sales, this project adds a predictive AI layer. It:
- analyzes customer behavior,
- predicts churn probability,
- assigns Low/Medium/High risk,
- identifies important churn drivers,
- estimates annual customer value associated with high-risk customers,
- generates actionable business insights.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib and SQLite.

## AI/ML Component
A Logistic Regression classifier is trained using:
- Tenure
- Monthly charges
- Support tickets
- Late payments
- Usage hours
- Contract
- Payment method
- Region
- Plan
- Support plan

The model outputs a churn probability for every customer.

## Analytics Component
The project calculates:
- Total customers
- Churn rate
- Average monthly charges
- Annual customer value at risk
- Contract-level churn
- AI feature importance
- Customer risk distribution
- Model evaluation metrics

## How to Run

```bash
pip install -r requirements.txt
python ai_customer_churn_analytics.py
```

## Output Files
The `output/` folder contains:
- `customer_data_cleaned.csv`
- `customer_churn_risk_scores.csv`
- `ai_feature_importance.csv`
- `kpi_summary.csv`
- `sql_contract_analysis.csv`
- `confusion_matrix.csv`
- `customer_analytics.db`
- `churn_by_contract.png`
- `ai_feature_importance.png`
- `customer_risk_distribution.png`
- `ai_business_insights.txt`

## Important Note
The data is synthetic and created specifically for portfolio/demo purposes. The AI model should be validated on real company data before any real business decisions.

## Resume Description
**AI-Powered Customer Churn & Revenue Analytics | Python, SQL, Pandas, Scikit-learn**
- Built an end-to-end analytics pipeline for customer behavior analysis and churn prediction.
- Trained and evaluated a Logistic Regression model to generate customer-level churn probabilities and risk segments.
- Combined KPI analysis, SQL aggregation, model interpretation and automated business insights to identify potential revenue-at-risk segments.
