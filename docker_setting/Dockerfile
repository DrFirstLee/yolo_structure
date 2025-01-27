# sudo docker build -t anaconda-yolo .
# sudo docker run -it -p 8505:8888 anaconda-yolo
# sudo docker run -it -p 8505:8888 -v /home/smartride/DrFirst:/app anaconda-yolo
# sudo docker run -it --gpus all --shm-size=128g -p 8505:8888 -v /home/smartride/DrFirst:/app anaconda-yolo
### sudo docker run -it --gpus all --shm-size=128g -p 8505:8888 -v /usr/share/ollama/.ollama/models:/root/.ollama/models -v /home/smartride/DrFirst:/app anaconda-yolo
FROM continuumio/anaconda3:latest

# Working directory
WORKDIR /app

# Copy requirements.txt
COPY ./models /root/.ollama/models
COPY requirements.txt .

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libxrender1 \
    libxext6 \
    nano \
    vim \
    cron \
    && apt-get clean && \
    conda install -y -c conda-forge \
    opencv \
    matplotlib \
    && pip install -r requirements.txt \
    && conda clean -a -y

# Set up crontab (example)
#RUN echo "* * * * * echo 'Hello from Docker' >> /var/log/cron.log 2>&1" > /etc/cron.d/example-cron
#RUN chmod 0644 /etc/cron.d/example-cron && \
#    crontab /etc/cron.d/example-cron

# Ollama 설치
RUN curl -fsSL https://ollama.com/install.sh | bash

# Llama 3.2 Vision 모델 다운로드
# RUN ollama pull llama3.2-vision


# Ensure cron runs in the background
RUN mkdir -p /var/log && touch /var/log/cron.log

# Expose Jupyter port
EXPOSE 8888
EXPOSE 11434

# Default command
CMD service cron start && \
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root & \
    ollama serve
