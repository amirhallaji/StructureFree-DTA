FROM docker.arvancloud.ir/python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir nvitop
RUN pip config set global.timeout 300
RUN pip install torch==2.4.0+cu124 
RUN pip install packaging
RUN pip install bitsandbytes peft transformers trl
RUN pip install -r requirements.txt
COPY . .
RUN chmod +x train.sh
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["./train.sh"]
CMD ["config.yaml"] 