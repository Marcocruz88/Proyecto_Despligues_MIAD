FROM python:3.11-slim

WORKDIR /opt/desercion-api

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY api.py variables.py model.pkl run.sh ./

ENV PORT=8001
EXPOSE 8001

CMD ["bash", "./run.sh"]