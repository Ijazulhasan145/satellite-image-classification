# Real-Time Satellite Image Scraping and Land Classification

This project automates the scraping of satellite images using GPS coordinates and classifies the captured images into land categories like Water, Desert, Industrial, Barren, and Crop, using a deep learning model.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Applications](#applications)
- [Future Work](#future-work)
- [License](#license)
- [Contact](#contact)

---

## About the Project

In today's world, satellite imagery is an important tool for analysis of geographical regions.  
This project bridges two major parts:

- **(1) Scraping Satellite Images Automatically**  
- **(2) Classifying Land Types Using Deep Learning**

Main idea was to develop an automated system which:

- Takes **GPS coordinates**
- **Captures real-time satellite images**
- **Processes the images** using a deep learning model
- **Outputs percentage** of different land types detected

The project was specially focused on **Lahore** area for proof of concept.

---

## Tech Stack

This project leverages several powerful technologies:

- **Python** 🐍 - Core programming language for the whole project.
- **Selenium** 🌐 - Used for browser automation to scrape images from Google Earth website.
- **PyTorch** 🔥 - Deep learning framework for training and testing the land classification model.
- **OpenCV** 📸 - For image handling and processing.
- **Google Earth (Web version)** 🌎 - Source platform for real-time satellite images.

---

## Features

✅ Fully Automated Satellite Image Scraping  
✅ Dynamic Zoom Adjustments for Clarity  
✅ Real-Time Image Capturing and Saving  
✅ Deep Learning Based Land Classification  
✅ Output Predictions in Percentage Format  
✅ Adaptable to Different Cities or Countries

---

## Getting Started

### Prerequisites

Before running the project, you need to install the following Python packages:

```bash
pip install selenium opencv-python torch torchvision
