# Radian Object Detector

Radian Object Detector is an AI-based real-time object detection application developed by **Radiantech-Systems** for embedded NVIDIA platforms.

The application uses **YOLO26** with **NVIDIA DeepStream** to detect objects from video streams in real time. It provides annotated live video, timestamped snapshots, and automatic event video generation.

## Overview

The system processes live video through a DeepStream-based pipeline and performs real-time object detection using the YOLO26 model.

Detected objects can be visualized directly on the video stream, while snapshots and event videos are generated for later monitoring and analysis.

## Features

- Real-time object detection using YOLO26
- NVIDIA DeepStream integration
- TensorRT-based inference
- GStreamer-based video processing
- Annotated live video stream
- Timestamped JPEG snapshots
- Automatic event video generation
- Flask-based snapshot server
- OpenCV integration
- RTSP streaming support
- ARM64 / AArch64 support
- systemd service integration

## Architecture

```text
                 Video Input
                     │
                     ▼
            ┌──────────────────┐
            │  GStreamer /     │
            │  DeepStream      │
            └────────┬─────────┘
                     │
                     ▼
              ┌──────────────┐
              │    YOLO26    │
              │ Object       │
              │ Detection    │
              └──────┬───────┘
                     │
          ┌──────────┼───────────┐
          │          │           │
          ▼          ▼           ▼
      Annotated   Snapshots   Event Videos
       Stream
          │          │           │
          ▼          ▼           ▼
       RTSP       Snapshot     Local
       Stream      Server      Storage
```

## Object Detection

The application uses the **YOLO26** model for real-time object detection.

The model files include:

- `yolo26s.onnx`
- `yolo26s.onnx.data`
- `labels.txt`

A custom DeepStream inference parser is included to integrate the YOLO26 model with the DeepStream pipeline.

## Live Video

The object detector processes the incoming video stream and generates an annotated output containing detected objects and their corresponding bounding boxes.

The processed video can be provided through an RTSP stream for monitoring applications and clients.

## Snapshots

The application generates timestamped JPEG snapshots during object detection.

A lightweight Flask-based snapshot server provides access to the generated snapshots for monitoring and dashboard applications.

## Event Video Generation

The project includes an event video generator that creates video clips when detection events occur.

This allows important detection events to be preserved as separate video footage for later review and analysis.

## Services

The application provides two system services:

- `object-detector.service`
- `snapshot-server.service`

The object detector service runs the real-time detection pipeline, while the snapshot server provides access to generated detection snapshots.

## Project Structure

```text
Asrock_object-detecor/
│
├── files/
│   ├── model/
│   │   ├── yolo26s.onnx
│   │   ├── yolo26s.onnx.data
│   │   └── labels.txt
│   │
│   ├── src/
│   │   ├── object-detector.cpp
│   │   ├── video_event_generator.cpp
│   │   └── nvdsinfer_custom_impl_Yolo/
│   │
│   ├── config_infer_primary_yolo26.txt
│   ├── snapshot_server.py
│   │
│   └── service/
│       ├── object-detector.service
│       └── snapshot-server.service
│
├── debian/
│
├── radian-object-detector.bb
│
└── README.md
```

## Application Components

### Object Detector

The main object detection application is implemented in C++ and uses NVIDIA DeepStream, GStreamer, TensorRT, and OpenCV for real-time video analytics.

### YOLO26 Custom Parser

The custom DeepStream parser connects the YOLO26 model output with the DeepStream inference pipeline.

### Event Video Generator

The event video generator creates dedicated video clips for detection events, allowing important events to be stored and reviewed independently.

### Snapshot Server

The Flask-based snapshot server provides access to generated JPEG snapshots through a lightweight HTTP service.

clone :
git clone https://github.com/Radiantech-Systems/Asrock_object-detecor.git

Install Debian build dependencies:
sudo apt update
sudo apt install -y build-essential debhelper devscripts

build:
dpkg-buildpackage -us -uc -b
## Purpose

The main purpose of this project is to provide a complete **AI-powered video analytics solution** for embedded systems.

It combines real-time object detection, live annotated video, snapshots, and event recording into a single integrated application.

## Organization

Developed and maintained by **Radiantech-Systems**.

**GitHub:**  
https://github.com/Radiantech-Systems
