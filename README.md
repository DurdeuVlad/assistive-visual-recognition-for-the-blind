# Assistive Vision Pipeline — dotLumen Challenge Submission

This repository contains the source code for my submission to the dotLumen Software Challenge 2. The challenge was a great opportunity to explore practical applications of computer vision and machine learning to assist visually impaired individuals in navigating the world more safely and independently.


> ⚠️ **Note:** Example videos have been excluded from this repository due to **privacy concerns**. If you’re evaluating this project and need a demonstration, please reach out directly and I’ll see what can be done with mock or synthetic data.

> 📘 The full development story, iteration details, and patch notes can be found in [**FULL_STORY.md**](FULL_STORY.md)
---

## Overview

This pipeline processes RGB and depth video streams using a combination of:

- **YOLOv8** for object detection.
- **SegFormer** for semantic segmentation (optional).
- Depth estimation for calculating object distances.
- Visual annotation of detected objects, classes, and estimated distances.

It produces an annotated video that highlights critical information for navigation assistance.

---

## Development Process

The pipeline went through **seven iterations**, each focused on refining detection accuracy, visual clarity, and overall robustness. Here's a quick breakdown of the journey:

### Version 1: Proof of Concept
- First working demo, very rough.
- Detection worked, but results were jittery.
- Learned the importance of trimming videos and managing rendering time.

### Version 2: Early Usability
- Limited to 30 seconds for quicker iteration.
- Confidence filtering added to remove garbage boxes.
- Much cleaner output, more usable.

### Version 3: Scaling Up
- Switched to a larger YOLO model for better detection.
- Performance hit hard, but object count improved.
- Realized I needed to balance speed vs accuracy.

### Version 4: Segmentation Enters
- Integrated SegFormer for street scene segmentation.
- Lots of bugs: forgot to convert BGR → RGB (classic OpenCV mistake).
- Initial segmentation quality was poor but promising.

### Version 5: Smarter UX
- Added EMA smoothing to stabilize distance labels.
- Box colors mapped to safety: green (safe), yellow/red (warning).
- Tweaked visuals for readability — no more floating white text on bright scenes.

### Version 6: Exploratory Debug
- Tried identifying sidewalk/road boundaries more precisely.
- Found real-world mislabeling issues, especially in parks or urban edges.
- Discovered SegFormer’s sidewalk vs road accuracy depends heavily on camera angle.

### Version 7: Final Touch
- Visualized **all** segmentation classes in color with contours and centroids.
- Replaced hardcoded colors with a repeatable hashing trick (modulus-based).
- Labeled everything clearly — looks chaotic, but proved that it works.

> This wasn’t about getting it perfect — it was about proving the concept, iterating quickly, and learning from mistakes. That goal was met.

---

## Installation

```bash
pip install -r requirements.txt
```

Make sure you also download a YOLOv8 model from [Ultralytics](https://github.com/ultralytics/ultralytics) and place it in the working directory.

---

## How to Run

```bash
python script.py --rgb path/to/rgb.mp4 --depth path/to/depth.mp4 --out path/to/output.mp4 --duration 30 --conf 0.3 --seg segformer
```

### CLI Arguments

| Flag        | Description                                                  |
|-------------|--------------------------------------------------------------|
| `--rgb`     | Path to the RGB video file (required)                        |
| `--depth`   | Path to the depth video file (required)                      |
| `--out`     | Path to the output annotated video (required)                |
| `--duration`| Duration to process in seconds (default: 30)                 |
| `--conf`    | Confidence threshold for detection (default: 0.3)            |
| `--seg`     | Segmentation model: `none` or `segformer` (default: none)    |

---

## Code Structure

- **`Config`** – Parses and stores settings.
- **`RGBCamera` / `DepthCamera`** – Handles video input.
- **`ObjectDetector`** – Runs YOLOv8.
- **`DistanceEstimator`** – Computes depth estimates.
- **`Reporter`** – Renders visual output.
- **`SegFormerDetector`** – Handles semantic segmentation.
- **`VideoPipeline`** – Ties everything together.

---

## Limitations

- Only processes a time-limited clip to save compute.
- Requires pre-recorded RGB + Depth videos (RealSense recommended).
- Segmentation output is imperfect — side effects from real-world scene noise.

---

## Development Time

- ~5 hours of coding.
- ~3 hours of research and documentation.
- Built and tested over 2 days in focused evening work sessions.

---

## References

See the original PDF submission for full academic and project references. Key technologies:

- YOLOv8 (Ultralytics)
- SegFormer (Hugging Face)
- Intel RealSense depth input
- OpenCV, Torch, Transformers

---

## Author

**Vlad-Ioan Durdeu**

This challenge was an awesome excuse to go deep on something I care about: building tech that helps people. Thanks for reading.

---