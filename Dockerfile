FROM python:3.9

WORKDIR /code

# Install Poetry
RUN pip install poetry
RUN poetry config virtualenvs.create false

# install project dependencies
COPY pyproject.toml poetry.lock* /code/
RUN poetry install --no-dev --no-interaction --no-ansi

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]