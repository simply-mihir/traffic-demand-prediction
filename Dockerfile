FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tests/ tests/

# Default: run the smoke test to prove the pipeline works
CMD ["python", "tests/make_synth_and_check.py"]
