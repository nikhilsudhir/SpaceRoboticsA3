#!/usr/bin/env python3
"""
Perception 1: first-pass auto-sort for the flat pile of frame_<timestamp>.png images saved by
dataset_dir (see README's "Collecting a dataset" section) into per-label subfolders, using the
same detection pipeline cave_explorer.py runs live (all HSV profiles + the stop-sign cascade +
suppress_overlapping_labels - NOT build_classifier_dataset.py's single-known-label version,
since here the label isn't known yet - that's the whole point of this step).

This is a SUGGESTION, not ground truth - the live detector has known blind spots (that's why
we're building a trained classifier at all), so its own output can't be trusted as the final
label. Two buckets exist specifically so you don't silently inherit those blind spots into the
sorted dataset:
    - review_no_detection/  - detector found nothing. Could be a genuine negative, or an
      artefact the detector missed (angle/lighting) - check these by eye.
    - review_multiple/      - detector found more than one artefact in frame (or disagreed
      with itself) - also check these by eye and resolve manually.
A confident single detection gets filed straight into its label's folder. Original files are
COPIED, never moved - the raw flat folder is left untouched in case you want to re-run this.

Usage:
    ros2 run cave_explorer sort_dataset_images -- --input ~/cave_dataset --output ~/cave_dataset_sorted
"""

import argparse
import os
import shutil

import cv2

from cave_explorer.cave_explorer import (
    ARTIFACT_COLOR_PROFILES, merge_overlapping_boxes, suppress_overlapping_labels, Detection,
)
from cave_explorer.build_classifier_dataset import load_stop_sign_cascade

REVIEW_NO_DETECTION = 'review_no_detection'
REVIEW_MULTIPLE = 'review_multiple'


def detect_color_artifacts(image_rgb):
    """Standalone equivalent of cave_explorer.py's CaveExplorer.detect_color_artifacts()"""

    hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    detections = []
    for profile in ARTIFACT_COLOR_PROFILES:
        mask = cv2.inRange(hsv_image, profile['hsv_lower'], profile['hsv_upper'])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= profile['min_area']]
        for bbox in merge_overlapping_boxes(boxes):
            detections.append(Detection(profile['label'], bbox, profile['color_rgb']))

    return suppress_overlapping_labels(detections)


def detect_stop_signs(image_bgr, cascade):
    raw = cascade.detectMultiScale(image_bgr, minSize=(20, 20))
    return [Detection('stop_sign', (x, y, w, h), (0, 255, 0)) for (x, y, w, h) in raw]


def classify_frame(image_bgr, cascade):
    """Returns a suggested label: a real artefact label, REVIEW_NO_DETECTION, or REVIEW_MULTIPLE"""

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detections = suppress_overlapping_labels(
        detect_stop_signs(image_bgr, cascade) + detect_color_artifacts(image_rgb),
        priority_labels={'stop_sign'})

    if len(detections) == 0:
        return REVIEW_NO_DETECTION
    if len(detections) > 1:
        return REVIEW_MULTIPLE
    return detections[0].label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Flat folder of frame_*.png files (your dataset_dir)')
    parser.add_argument('--output', required=True, help='Where to create suggested per-label subfolders')
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input)
    output_dir = os.path.expanduser(args.output)
    os.makedirs(output_dir, exist_ok=True)

    cascade = load_stop_sign_cascade()
    counts = {}

    filenames = sorted(f for f in os.listdir(input_dir) if f.lower().endswith('.png'))
    for filename in filenames:
        image_bgr = cv2.imread(os.path.join(input_dir, filename))
        if image_bgr is None:
            continue

        label = classify_frame(image_bgr, cascade)
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        shutil.copy2(os.path.join(input_dir, filename), os.path.join(label_dir, filename))
        counts[label] = counts.get(label, 0) + 1

    print(f'{len(filenames)} images sorted into {output_dir}:\n')
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        flag = '  <- check these by eye' if label in (REVIEW_NO_DETECTION, REVIEW_MULTIPLE) else ''
        print(f'  {label:<20} {count:>4}{flag}')
    print(f'\nNext: skim each folder and move anything misfiled, resolve the two review_'
          f'folders by eye, then delete near-duplicate/blurry shots (quality over quantity - '
          f'see Perception 1 in the brief). Once happy, that\'s your sorted dataset ready for '
          f'build_classifier_dataset.')


if __name__ == '__main__':
    main()
