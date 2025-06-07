# Thermal Imaging-Based Data Capture and Analysis Tool

## 📌 Project Overview

This project provides a robust solution to extract, analyze, and store thermal data from video footage—without accessing backend or raw sensor data—by using image processing and Optical Character Recognition (OCR) techniques. Built for thermal cameras such as FLUKE, it transforms visual thermal frames into structured temperature datasets.

## 🎯 Objectives

- Capture the thermal signature of subjects.
- Extract individual frames from thermal videos.
- Apply OCR to extract temperature data from frames.
- Store structured data in CSV/Excel format for analysis and visualization.

## 🛠️ Tech Stack

| Area               | Tool/Library             |
|--------------------|--------------------------|
| OCR                | Pytesseract (Tesseract OCR) |
| Image Processing   | OpenCV                   |
| Backend            | Python + Flask           |
| Frontend           | HTML, CSS, JavaScript    |
| Data Storage       | Pandas + Excel           |
| Canvas Rendering   | HTML5 Canvas + Fabric.js |
| Version Control    | Git + GitHub             |
| Local Server       | Flask Dev Server         |

## 🧪 How It Works – Process Flow

1. **Record the thermal video** (with crosshair markers added).
2. **Enhance video visibility** by adjusting brightness/contrast.
3. **Extract frames** using OpenCV.
4. **Run OCR** using Tesseract to detect temperature overlays.
5. **Clean & store data** using Pandas in CSV/Excel format.
6. **Visualize** the data through a simple HTML+JS interface.
