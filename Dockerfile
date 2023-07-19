# app/Dockerfile

FROM python:3.7.6-slim
EXPOSE 8501
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /app



COPY . ./


ENTRYPOINT ["streamlit", "run", "interface.py", "--server.port=8501", "--server.address=0.0.0.0"]