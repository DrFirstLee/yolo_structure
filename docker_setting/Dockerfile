# Base image
FROM continuumio/anaconda3:latest

# Set working directory
WORKDIR /app

# Copy requirements and models
COPY ./models /root/.ollama/models
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1 \
    libxrender1 \
    libxext6 \
    nano \
    vim \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libegl1-mesa-dev \
    libgstreamer1.0-dev \
    libgtk-3-dev \
    libv4l-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenexr-dev \
    libopenjp2-7-dev \
    cron \
    && apt-get clean

# Install CMake separately to ensure it's available
RUN apt-get install -y cmake

# Download and build OpenCV with GStreamer support
RUN git clone --depth=1 https://github.com/opencv/opencv.git && \
    git clone --depth=1 https://github.com/opencv/opencv_contrib.git && \
    mkdir -p opencv/build && cd opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_GSTREAMER=ON \
          -D OPENCV_EXTRA_MODULES_PATH=/app/opencv_contrib/modules \
          -D WITH_FFMPEG=ON \
          -D BUILD_EXAMPLES=OFF .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd /app && rm -rf opencv opencv_contrib

# Install Python dependencies
RUN conda install -y -c conda-forge \
    opencv \
    matplotlib \
    && pip install -r requirements.txt \
    && conda clean -a -y

# Set timezone to Asia/Seoul
RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    echo "Asia/Seoul" > /etc/timezone

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | bash

# Ensure cron runs in the background
RUN mkdir -p /var/log && touch /var/log/cron.log

# Expose ports
EXPOSE 8888
EXPOSE 11434

# Use JSON format for CMD to avoid OS signal issues
CMD ["bash", "-c", "service cron start && \
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root & \
    ollama serve"]
