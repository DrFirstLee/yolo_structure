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

## Installation

To set up this project, clone the repository:

```bash
git clone https://github.com/your-repository-url.git
```
