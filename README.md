# PoseEngine: Skeleton-Based Fall Detection

A practical, real-time fall detection system built for indoor environments. This project prioritizes **user privacy** and **computational efficiency** by utilizing handcrafted features extracted from human pose landmarks rather than raw video storage.

## 🚀 The Vision

A real-time fall-detection system for the elderly. This is the second module of the five module Ambient Assisted Living (AAL) system for the elderly.

## 🛠 Tech Stack

  * **Frontend:** React.js + TypeScript
  * **Pose Estimation:** [MediaPipe Pose](https://www.google.com/search?q=https://google.github.io/mediapipe/solutions/pose.html)
  * **Styling:** Tailwind CSS
  * **Backend: ** Connected to the main AAL system; NodeJS + Express

## 📊 Handcrafted Features

Rather than using "black-box" Deep Learning, the system uses specific mathematical heuristics to identify falls:

  * **Vertical Displacement ($\Delta y$):** Monitoring the sudden drop of the **Mid-Hip** (Center of Mass) and **Nose** landmarks.
  * **Height-Collapse Ratio:** Comparing current Nose-to-Ankle distance against an initial standing baseline.
  * **Velocity Thresholds:** Calculating pixels-per-frame movement to distinguish between a fall and a controlled "Activities of Daily Living" (ADL) such as sitting.

## 🔒 Privacy & Performance

  * **No Video Storage:** Frames are processed in RAM and discarded immediately.
  * **Low Latency:** Heuristic math runs in $O(1)$ time complexity once landmarks are extracted.
  * **Offline Capable:** The core detection engine requires no active internet connection once the MediaPipe models are cached.

## 🏗️ Getting Started

1.  Clone the repo: `git clone https://github.com/amariee19/pose-estimation-alias.git`
2.  Install dependencies: `npm install`
3.  Run the development server: `npm run dev`
4.  Click **"Enable Prediction"** to begin the tracking engine.

-----



