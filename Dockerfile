# Use an official Python runtime as a parent image
FROM python:3.9

# Labels / Metadata
LABEL org.opencontainers.image.title="Generative AI Backend Using FastAPI" \
      org.opencontainers.image.description="This image is used to run my FastAPI application" \
      org.opencontainers.image.authors="Subhomoy Roy Choudhury <subhomoyrchoudhury@gmail.com>"

# Set environment varibles
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /code

# Install Poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false

# install project dependencies
COPY pyproject.toml poetry.lock* /code/
RUN poetry install --no-dev --no-interaction --no-ansi

COPY ./app /code/app

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]