#Base image
From python:3.11-slim

# Set working directory
WORKDIR /app

COPY . /app

RUN pip install poetry

# Install dependencies (without virtualenvs)
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root


EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]