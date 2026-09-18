# VisionBrush-AI
A real-time computer vision drawing tool that tracks hand landmarks and gestures to paint and create shapes in mid-air via webcam using OpenCV and MediaPipe.
# 🎨 AirCanvas AI: Real-Time Hand Gesture Painting

VisionBrush-AI is a real-time computer vision drawing system built with **OpenCV**, **MediaPipe**, and **NumPy**. It tracks hand landmarks in 3D space via a standard webcam, translating natural finger gestures into strokes, geometric primitives, and UI selections without physical touching or specialized hardware.

---

## 🚀 Key Features

- **Accurate 3D Hand Tracking:** Leverages MediaPipe Hands to detect 21 key landmarks with low latency and high precision.
- **Adaptive Handedness Interface:** Configures the on-screen toolbar dynamically based on dominant hand preference (Left or Right-handed).
- **Comprehensive Drawing Engine:**
  - **Freehand Mode:** Smooth stroke drawing following index fingertip coordinates.
  - **Geometric Shapes:** Instant line, rectangle, and circle generation with real-time bounding preview.
  - **Text Input:** Place custom text overlays directly on screen at pinpointed coordinates.
  - **Eraser Tool:** Clear specific areas on the canvas with customizable radius.
- **Palette & Tool Selection:** Intuitive pinch gesture selection for 7 primary colors and tools.
- **Session Export:** Built-in triggers to save PNG snapshots or record painting sessions into MP4 video files.

---

## 🛠️ Tech Stack

- **Computer Vision:** OpenCV (`cv2`)
- **Machine Learning Inference:** MediaPipe
- **Array Operations & Blending:** NumPy
- **Language:** Python 3.12+
