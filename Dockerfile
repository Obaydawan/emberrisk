# EmberRisk API -- standalone Dockerfile for deployment (Render, etc.)
#
# This is separate from docker-compose.airflow.yml, which builds the
# orchestration stack. This Dockerfile builds ONLY the FastAPI serving
# layer (api/) plus the demo UI (api/static/) -- what's needed to run
# the live public demo, nothing else from the project.
#
# The trained model artifact (models/gradient_boosting_locked.joblib)
# is committed to the repo specifically for this deployment path -- see
# docs/phase13-deployment.md for why this is a deliberate exception to
# the "models/ is gitignored" rule used everywhere else in this project.

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only copy what the API actually needs at runtime -- not the full
# repo (ingestion/, processing/, dags/, docs/, tests/ are irrelevant
# to serving predictions and would only bloat the image).
COPY api/ ./api/
COPY ml/ ./ml/
COPY models/ ./models/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
