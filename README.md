# EmberRisk

EmberRisk is a lightweight machine-learning and MLOps project that builds a wildfire-risk classification pipeline from heterogeneous public environmental data.

## Project Goal

The project demonstrates the complete ML data lifecycle:

- Public environmental data ingestion
- Data validation and transformation
- Feature engineering
- Machine-learning model training
- Model evaluation
- Experiment tracking with MLflow
- Model registration
- Pipeline orchestration with Apache Airflow
- Risk prediction
- Streamlit visualization
- Basic data/model monitoring

## Planned Models

- Logistic Regression
- Random Forest
- XGBoost

## Planned Stack

- Python
- DuckDB
- pandas
- scikit-learn
- XGBoost
- MLflow
- Apache Airflow
- Streamlit
- Git/GitHub

## Hardware Constraints

The project is designed to run on a resource-constrained local machine:

- 8 GB RAM
- Dual-core CPU
- No dedicated GPU

Therefore, the project intentionally avoids deep learning, Spark, Kubernetes, and unnecessary cloud infrastructure.

## Project Status

Phase 0 — Project definition and setup.

Data sources and the final prediction target have not yet been finalized. They will be verified before ingestion development begins.

## Scope Boundary

EmberRisk is an independent wildfire-risk ML/MLOps portfolio project. It is also intended to provide transferable skills for the future FloodSense FYP, but EmberRisk will not use FloodSense's Pakistan flood-risk problem or become a copy of that project.
