# LunaCane

LunaCane is an ESP32-S3-based edge-AI smart mobility system designed for elderly users and people with mobility difficulties. It combines embedded sensing, fall detection, voice interaction, GPS/BeiDou positioning, and cloud intelligence to provide safer and more responsive mobility assistance in daily aging-care scenarios.

## Overview

Traditional walking aids mainly provide physical support, but they usually cannot understand user status, detect emergencies, or communicate with caregivers. LunaCane explores how a mobility aid can become an intelligent assistive device by integrating perception, interaction, positioning, and safety support into one portable system.

The system collects motion and positioning data through onboard sensors, performs lightweight edge processing on embedded hardware, and connects with cloud services when more advanced interaction or support is needed. This allows LunaCane to respond quickly to abnormal events while still supporting flexible voice-based user interaction.

## Key Features

- **Fall detection**: Identifies potential falls and abnormal movement patterns using motion sensor data.
- **Edge AI processing**: Runs lightweight sensing and decision-making tasks on the ESP32-S3 platform.
- **Voice interaction**: Supports speech-based communication through ASR and TTS capabilities.
- **GPS/BeiDou positioning**: Provides location information for mobility support and emergency scenarios.
- **Cloud intelligence**: Connects local sensing with cloud-based language model services for more flexible assistance.
- **Caregiver support**: Helps deliver safety-related information when abnormal events are detected.

## System Components

- ESP32-S3 embedded control module
- IMU-based motion sensing module
- Microphone and speaker module
- GPS/BeiDou positioning module
- Wireless communication module
- Edge fall-detection model
- Cloud interaction service

## Project Goal

The goal of LunaCane is to improve mobility safety and daily independence for elderly users and mobility-impaired groups. By combining embedded AI, sensing, positioning, and voice interaction, LunaCane aims to make assistive mobility devices more intelligent, accessible, and human-centered.
