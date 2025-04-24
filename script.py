# === Imports ===
import cv2
import numpy as np
import argparse
import os
import torch
from ultralytics import YOLO
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation


class Config:
    """
    Stores everything we need: paths, duration, confidence, and segmentation type
    """
    def __init__(self, rgb_path, depth_path, output_path, duration, confidence, segmentation):
        """
        Initialize config values from args
        """
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.output_path = output_path
        self.duration = duration
        self.confidence = confidence
        self.segmentation = segmentation  # 'none' or 'segformer'

    @staticmethod
    def from_args():
        """
        Parse CLI args and return a Config object
        """
        parser = argparse.ArgumentParser(description='Assistive Vision Pipeline')
        parser.add_argument('--rgb', required=True, help='Path to RGB video')
        parser.add_argument('--depth', required=True, help='Path to depth video')
        parser.add_argument('--out', required=True, help='Path to output video')
        parser.add_argument('--duration', type=float, default=30.0, help='Max processing duration in seconds')
        parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold for detection')
        parser.add_argument('--seg', choices=['none','segformer'], default='none', help='Which segmentation model to use')
        args = parser.parse_args()
        return Config(args.rgb, args.depth, args.out, args.duration, args.conf, args.seg)


class RGBCamera:
    """
    Handles reading frames from the RGB video
    """
    def __init__(self, video_path):
        """
        Open the video and grab metadata (fps, size)
        """
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open RGB video: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        """
        Return the next frame
        """
        return self.cap.read()

    def release(self):
        """
        Release the video file
        """
        self.cap.release()


class DepthCamera:
    """
    Handles reading and normalizing frames from the depth video
    """
    def __init__(self, video_path):
        """
        Open the depth video file
        """
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open depth video: {video_path}")

    def read(self):
        """
        Read next frame and scale it to 0.0–10.0 meters
        """
        ret, frame = self.cap.read()
        if not ret:
            return ret, None
        depth_frame = frame.astype(np.float32) / 255.0 * 10.0
        return True, depth_frame

    def release(self):
        """
        Release the video file
        """
        self.cap.release()


class FrameAligner:
    """
    Stub class to align RGB and depth frames (currently a passthrough)
    """
    @staticmethod
    def align(rgb_frame, depth_frame):
        """
        Return frames as-is
        """
        return rgb_frame, depth_frame


class ObjectDetector:
    """
    YOLOv8 wrapper for detecting objects in a frame
    """
    def __init__(self, model_path='yolov8n.pt'):
        """
        Load the YOLO model
        """
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Run inference and return boxes, classes, confidences
        """
        results = self.model(frame)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        return boxes, classes, confidences


class DistanceEstimator:
    """
    Estimate distance to the center of each bounding box using depth info
    """
    def estimate(self, boxes, depth_frame):
        """
        For each box, average a small patch in the depth map at the box center
        """
        distances = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            patch = depth_frame[max(cy-2, 0):min(cy+2, depth_frame.shape[0]),
                                 max(cx-2, 0):min(cx+2, depth_frame.shape[1])]
            dist = float(np.nanmean(patch)) if patch.size else float('nan')
            distances.append(dist)
        return distances


class Reporter:
    """
    Handles drawing boxes and distances on frames, and writing output video
    """

    def __init__(self, output_path, frame_size, fps):
        """
        Set up video writer and EMA smoothing state
        """

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.ema = {}
        self.alpha = 0.4

    def _color_for_distance(self, d):
        """
        Pick a box color based on distance
        """

        if d < 1.0:
            return (0, 0, 255)
        if d < 2.5:
            return (0, 255, 255)
        return (0, 255, 0)

    def draw(self, frame, boxes, classes, distances, confidences, class_names, min_conf):
        """
        Draw boxes, distance labels, and smooth distance output using EMA
        """

        for i, (box, cls, dist, conf) in enumerate(zip(boxes, classes, distances, confidences)):
            if conf < min_conf:
                continue
            x1, y1, x2, y2 = box.astype(int)
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            prev = self.ema.get(i, dist)
            dist_s = prev * (1 - self.alpha) + dist * self.alpha
            self.ema[i] = dist_s
            color = self._color_for_distance(dist_s)
            label = f"{class_names[cls]} {dist_s:.1f}m"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), (0, 0, 0), cv2.FILLED)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def write(self, frame):
        """
        Write the annotated frame to the output video
        """

        self.writer.write(frame)

    def release(self):
        """
        Release the writer
        """

        self.writer.release()


class SegFormerDetector:
    """
    Handles loading and running the SegFormer segmentation model
    """

    def __init__(self, device=None, backbone='b2'):
        """
        Load the model and put it into eval mode
        """

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = f'nvidia/segformer-{backbone}-finetuned-cityscapes-1024-1024'
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def segment(self, frame: np.ndarray):
        """
        Run segmentation on an RGB frame and return resized segmentation map
        """

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.feature_extractor(images=rgb, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        seg_map = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

        unique_ids = np.unique(seg_map)
        print("Unique seg classes:", unique_ids)
        side_id = next(k for k, v in self.model.config.id2label.items() if v == 'sidewalk')
        print("Sidewalk pixels:", np.sum(seg_map == side_id))

        seg_map = cv2.resize(seg_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((2,2), np.uint8)
        seg_map = cv2.morphologyEx(seg_map, cv2.MORPH_CLOSE, kernel)

        return seg_map


def overlay_segmentation(frame: np.ndarray, seg_map: np.ndarray, id2label: dict):
    """
    Draw contours for each segmented class and label them on the frame
    """
    overlay = frame.copy()
    for class_id, class_name in id2label.items():
        mask = (seg_map == class_id).astype(np.uint8)
        if cv2.countNonZero(mask) == 0:
            continue
        color = (
            int((class_id * 47) % 256),
            int((class_id * 97) % 256),
            int((class_id * 167) % 256),
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(contours[0])
            cx, cy = x + w // 2, y + h // 2
        (tw, th), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(overlay, (cx, cy - th - 5), (cx + tw + 4, cy), (0, 0, 0), cv2.FILLED)
        cv2.putText(overlay, class_name, (cx + 2, cy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


class VideoPipeline:
    """
    The main pipeline that ties together detection, segmentation, and video output
    """

    def __init__(self, config: Config):
        """
        Initialize all components and prepare for processing
        """

        self.rgb_cam = RGBCamera(config.rgb_path)
        self.depth_cam = DepthCamera(config.depth_path)
        self.detector = ObjectDetector()
        self.estimator = DistanceEstimator()
        self.reporter = Reporter(
            config.output_path,
            (self.rgb_cam.width, self.rgb_cam.height),
            self.rgb_cam.fps
        )
        self.max_frames = int(self.rgb_cam.fps * config.duration)
        self.conf = config.confidence
        self.segmentation = config.segmentation
        if self.segmentation == 'segformer':
            self.seg_detector = SegFormerDetector()

    def run(self):
        """
        Run the full pipeline frame by frame
        """

        frame_count = 0
        class_names = self.detector.model.names
        while frame_count < self.max_frames:
            ret_rgb, rgb_frame = self.rgb_cam.read()
            ret_dep, depth_frame = self.depth_cam.read()
            if not ret_rgb or not ret_dep:
                break
            rgb_aligned, depth_aligned = FrameAligner.align(rgb_frame, depth_frame)
            boxes, classes, confs = self.detector.detect(rgb_aligned)
            dists = self.estimator.estimate(boxes, depth_aligned)
            annotated = self.reporter.draw(
                rgb_aligned, boxes, classes, dists,
                confs, class_names, self.conf
            )
            if self.segmentation == 'segformer':
                seg_map = self.seg_detector.segment(rgb_aligned)
                annotated = overlay_segmentation(
                    annotated, seg_map,
                    self.seg_detector.model.config.id2label
                )
            self.reporter.write(annotated)
            frame_count += 1
        self.release()
        print(f"Processed {frame_count} frames ({frame_count/self.rgb_cam.fps:.1f}s)")

    def release(self):
        """
        Release all resources
        """

        self.rgb_cam.release()
        self.depth_cam.release()
        self.reporter.release()


if __name__ == '__main__':
    """
    CLI entry point to run the pipeline from command line args
    """
    cfg = Config.from_args()
    os.makedirs(os.path.dirname(cfg.output_path) or '.', exist_ok=True)
    pipeline = VideoPipeline(cfg)
    pipeline.run()
