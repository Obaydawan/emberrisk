# EmberRisk

EmberRisk is a lightweight machine-learning,MLOps and Data project for wildfire-risk classification using public environmental data.

The project combines historical wildfire detections with weather and environmental data to build a cell-day dataset and predict whether a wildfire will occur within a future time horizon.

## Project Goal

The project covers the main stages of a practical ML pipeline:

- Public environmental data ingestion
- Data validation and standardization
- Cell-based spatial data processing
- Temporal feature engineering
- Future wildfire target generation
- Machine-learning model training
- Model evaluation
- Experiment tracking with MLflow
- Model registration
- Pipeline orchestration with Apache Airflow
- Risk prediction
- Streamlit visualization
- Basic data and model monitoring

## Data

EmberRisk uses two main environmental data sources:

- NASA FIRMS wildfire detection data
- NASA POWER weather and environmental data

The study area is represented using a canonical grid of 323 cells. Data is transformed into a cell-day format covering January 2018 through December 2025.

The prediction targets are defined for three future horizons:

- 3 days
- 7 days
- 14 days

Target labels are kept null at the end of the dataset where the required future observation period is not available.

## Current Pipeline

The current data pipeline:

1. Ingests raw FIRMS and POWER data
2. Standardizes both datasets
3. Maps observations to the canonical spatial grid
4. Builds a complete cell-day scaffold
5. Creates historical fire features
6. Joins fire and weather features
7. Generates future-fire targets
8. Runs validation checks
9. Stores the processed datasets as Parquet files

Phase 3 produces a validated cell-day dataset containing 943,806 rows across 323 cells and 8 years of modeling data.

## Planned Models

- Logistic Regression
- Random Forest
- XGBoost

## Planned Stack

- Python
- pandas
- DuckDB
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

Because of these constraints, the project intentionally avoids deep learning, Spark, Kubernetes, and unnecessary cloud infrastructure.

## Project Status

### Completed

- Project setup and repository structure
- FIRMS data ingestion
- POWER data ingestion
- Data validation
- Spatial grid construction
- Fire-history feature engineering
- POWER standardization
- Future-fire target generation
- Cell-day dataset construction
- Phase 3 validation

The Phase 3 processing pipeline currently passes 74 automated tests and produces validated datasets for the 3-day, 7-day, and 14-day prediction horizons.

### Next

- Exploratory data analysis
- Feature analysis and selection
- Train/validation/test split
- Baseline model
- Model comparison
- Experiment tracking
- Model registration
- Pipeline orchestration
- Risk prediction interface
- Monitoring

## Scope

EmberRisk is an independent wildfire-risk ML/MLOps portfolio project.

The focus is on building a complete, reproducible machine-learning pipeline while keeping the system practical enough to develop and run locally.
