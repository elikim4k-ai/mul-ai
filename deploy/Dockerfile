FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY mul.config.yaml ./

EXPOSE 8080
CMD ["uvicorn", "--factory", "mul.gateway.app:create_app", "--host", "0.0.0.0", "--port", "8080"]
