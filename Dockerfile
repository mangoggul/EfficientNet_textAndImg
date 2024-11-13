FROM python:3.12-slim

WORKDIR /app

RUN pip install numpy matplotlib

COPY docker_test.py /app/docker_test.py

CMD ["python", "docker_test.py"]
