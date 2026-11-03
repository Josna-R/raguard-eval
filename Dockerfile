FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project.
COPY . .

# Sanity-check the install (fails the build if the package is broken).
RUN python -m raguard.cli run sample_data/eval_cases.json

EXPOSE 8501

# Default: run the CI regression gate against the committed baseline.
# Override the command to run the dashboard instead, e.g.:
#   docker run -p 8501:8501 raguard streamlit run raguard/dashboard.py \
#       --server.address=0.0.0.0
CMD ["python", "-m", "raguard.cli", "check", "sample_data/eval_cases.json", "--baseline", "sample_data/baseline.json"]
