# CCTV Detection Program for Korea Road Traffic Corporation

## Project Name
Korea Road Traffic Corporation CCTV Detection Program

## Objective
The goal of this project is to detect various emergency situations in real-time, such as people, stationary vehicles, fallen objects, and moving objects, based on CCTV video footage of roads.

## Methodology
- **Object Detection**: The initial object detection is performed using YOLO (You Only Look Once), which identifies various objects in the CCTV video.
- **Status Detection**: After detecting the objects, the VLM (Visual Language Model) is used to assess the specific status of each object.
- **Data Processing and Event Detection**: The processed data is then finalized, and the identified event is transmitted to Kafka for further use and analysis.

## Key Features
- Real-time object detection from CCTV footage
- Detection of emergency situations such as people, stationary vehicles, and moving objects
- Use of YOLO for initial detection and VLM for status recognition
- Integration with Kafka for event transmission and processing

## How to Run code
To set up this project, clone the repository:

```bash
sudo docker run -it --gpus all --shm-size=128g -p 8505:8888
```

This command will:

- Use all available GPUs (`--gpus all`).
- Allocate 128GB of shared memory (`--shm-size=128g`).
- Expose port `8505` for access to the Jupyter Notebook interface.
- Mount the local `/home/smartride/DrFirst` directory to `/app` in the container for seamless file access.

### 3. Access Jupyter Notebook

Once the container is running, open a web browser and navigate to:

http://localhost:8505

This will give you access to the Jupyter Notebook interface, where you can run the detection scripts.

### 4. Usage

1. Prepare your CCTV footage data.
2. Run the object detection script:

```bash
python /app/yolo_structure/src/yolo_fast_stream.py
```

3. The final processed event will be transmitted to Kafka.
