FROM python:3.12-slim

WORKDIR /app/python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# legacy/ is mounted at /app/legacy, which is where trajectory_algorithms/paths.py expects it
CMD ["python", "-m", "scripts.import_data"]
