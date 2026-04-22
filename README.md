
---

# PoseEngine — Skeleton-Based Fall Detection (Module II)

This document details the architecture and logic flow of the **PoseEngine**, a module within the **Ambient Assisted Living (AAL)** ecosystem designed for elderly monitoring.

---

## 1. System Overview

PoseEngine is a privacy-first, real-time fall detection system. It abstracts human movement into lightweight skeletal landmarks, moving the "intelligence" to the edge (client-side) to eliminate invasive video storage.

- **Frontend**: React 18 (TypeScript) + Vite
- **Engine**: MediaPipe Pose (BlazePose GHUM Hybrid)
- **Logic**: Handcrafted kinematic heuristics
- **Optimization**: Dynamic buffer thresholding

---

## 2. Detection Pipeline

The system interprets movement through a multi-stage pipeline that ensures data is processed and discarded in real-time.

### The Logic Flow:
1.  **Extraction**: MediaPipe identifies 33 key body landmarks from the video stream.
2.  **Transformation**: Raw coordinates are converted into relative distances and angular velocities.
3.  **Voting**: Each frame is evaluated against specific handcrafted features (e.g., Vertical Displacement, Height-Collapse).
4.  **Temporal Analysis**: A sliding window buffer aggregates these "votes" to ensure the movement is a sustained fall rather than a momentary sensor glitch.
5.  **State Management**: If the confidence threshold is crossed, a fall state is triggered.

---

## 3. Handcrafted Feature Matrix

The engine uses human-defined heuristics to distinguish a fall from Activities of Daily Living (ADL):

| Feature | Description | Sensitivity |
| :--- | :--- | :--- |
| **Vertical Displacement ($\Delta y$)** | Tracks the sudden drop of the **Mid-Hip** (Center of Mass) relative to an initial baseline. | High |
| **Height-Collapse Ratio** | Monitors the vertical distance between the **Nose** and **Ankles** to detect a "crumbling" posture. | Medium |
| **Kinetic Velocity** | Measures the acceleration of the head and torso to capture the "impact" signature. | High |

---

## 4. Privacy & Performance

### Privacy-by-Design
* **Zero-Persistence**: All video frames are processed in volatile RAM and discarded instantly.
* **Abstract Output**: The system only outputs coordinate vectors, ensuring the user’s visual identity is never stored or transmitted.

### Performance Optimization
* **$O(1)$ Logic**: Handcrafted math ensures the detection logic adds near-zero overhead to the pose estimation.
* **Asynchronous Processing**: By decoupling the detection engine from the React render cycle, the app maintains a consistent 30+ FPS on consumer-grade hardware.

---

## 5. Development Status

- [x] Initial MediaPipe Integration (Full Model)
- [x] Handcrafted Feature Engine (Base Heuristics)
- [ ] **Dynamic Buffer Implementation**: Refactoring the threshold logic to use quadratic growth patterns, improving accuracy across variable camera distances.
- [ ] Module III Integration (Emergency Alert Trigger)

---

## 6. Setup

```bash
git clone https://github.com/amariee19/pose-estimation-alias.git
npm install
npm run dev
```

---

