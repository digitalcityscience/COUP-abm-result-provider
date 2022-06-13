FROM python:3.9

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app

CMD ["uvicorn", "api:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]