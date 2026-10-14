#!/usr/bin/env python3
"""
Perception 1 -> Advanced 1 prep: turn the raw, sorted-by-folder dataset frames (whole camera
images, one subfolder per artefact label - see README's "Collecting a dataset" section) into
cropped images ready to upload to Google Teachable Machine (https://teachablemachine.withgoogle.com/),
one output subfolder per label, matching its expected upload format.

Each raw frame is already known to belong to a specific label (its folder), so unlike
cave_explorer.py's live detect_color_artifacts()/detect_stop_signs() this doesn't need to figure
out WHICH artefact type matched - it just needs the best bounding box for that one type (reusing
the exact same HSV profile/cascade classifier the live node uses, so the trained classifier sees
crops consistent with what it'll actually be fed at runtime - see cave_explorer.py's planned
classify_crop() integration). A 'negative' folder (frames with no artefact in view) is passed
through unchanged - there's nothing to crop, and Teachable Machine needs full-frame examples of
"none of these" regardless.

Usage:
    ros2 run cave_explorer build_classifier_dataset -- --input ~/cave_dataset --output ~/cave_dataset_cropped

('--' separates ros2 run's own args from this script's - needed because of the leading '-i'/'-o').
Then drag each labelled subfolder of --output into Teachable Machine as a separate class.
"""

import argparse
import os

import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory

from cave_explorer.cave_explorer import ARTIFACT_COLOR_PROFILES, merge_overlapping_boxes

# How much (pixels) to pad each crop beyond the detected box - a classifier benefits from a
# little surrounding context, and this also forgives a slightly-too-tight detection.
CROP_MARGIN_PX = 15

# Passed through uncropped (see module docstring) rather than run through any detector.
NEGATIVE_LABEL = 'negative'

# Crop-extraction-only overrides, looser than the live ARTIFACT_COLOR_PROFILES entry for that
# label. Only safe to use here, never live: live detection needs a tight profile so it doesn't
# false-positive against OTHER labels it's competing with in the same frame (see
# suppress_overlapping_labels()) - but here, the folder a frame came from already tells us the
# true label with certainty (it was manually verified - see sort_dataset_images.py's review
# piles), so there's no competing label to accidentally match instead, and a looser box just
# means recovering genuine-but-fainter/further sightings of the same confirmed object that the
# strict live profile would (correctly, for live use) have ignored. white_sphere needed this:
# its live profile (S<=140, V>=200) only yielded a usable crop for 5/27 manually-confirmed
# white_sphere photos - most were real but distant/dim sightings sitting just outside that
# tight live-safe window, not false matches.
CROP_HSV_OVERRIDES = {
    'white_sphere': {'hsv_lower': (90, 10, 80), 'hsv_upper': (120, 255, 255), 'min_area': 20},
}


def detect_color_label(image, label):
    """
    Perception 2's HSV approach, restricted to the one profile matching 'label' (we already
    know the true label from the folder it came from - no need for suppress_overlapping_labels'
    cross-label disambiguation here). Returns the largest merged box, or None if nothing matched.
    """

    profile = next((p for p in ARTIFACT_COLOR_PROFILES if p['label'] == label), None)
    if profile is None:
        return None
    profile = {**profile, **CROP_HSV_OVERRIDES.get(label, {})}

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.inRange(hsv_image, profile['hsv_lower'], profile['hsv_upper'])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= profile['min_area']]
    if not boxes:
        return None

    merged = merge_overlapping_boxes(boxes)
    return max(merged, key=lambda b: b[2] * b[3])


def detect_stop_sign(image, cascade):
    """Same cascade classifier cave_explorer.py uses, restricted to the single largest hit."""

    detections = cascade.detectMultiScale(image, minSize=(20, 20))
    if len(detections) == 0:
        return None
    x, y, w, h = max(detections, key=lambda d: d[2] * d[3])
    return (x, y, w, h)


def crop_with_margin(image, box, margin=CROP_MARGIN_PX):
    x, y, w, h = box
    height, width = image.shape[:2]
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(width, x + w + margin), min(height, y + h + margin)
    return image[y0:y1, x0:x1]


def load_stop_sign_cascade():
    cascade_path = os.path.join(
        get_package_share_directory('cave_explorer'), 'config', 'stop_data.xml')
    return cv2.CascadeClassifier(cascade_path)


def process_label_folder(label, input_dir, output_dir, cascade):
    out_label_dir = os.path.join(output_dir, label)
    os.makedirs(out_label_dir, exist_ok=True)

    filenames = sorted(f for f in os.listdir(input_dir) if f.lower().endswith('.png'))
    saved, skipped = 0, 0

    for filename in filenames:
        image_bgr = cv2.imread(os.path.join(input_dir, filename))
        if image_bgr is None:
            skipped += 1
            continue

        if label == NEGATIVE_LABEL:
            cv2.imwrite(os.path.join(out_label_dir, filename), image_bgr)
            saved += 1
            continue

        # Saved dataset frames are RGB (see cave_explorer.py's save_image_to_dataset, which
        # converts RGB->BGR before cv2.imwrite) - convert back for the RGB-space HSV detector.
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        box = detect_stop_sign(image_bgr, cascade) if label == 'stop_sign' \
            else detect_color_label(image_rgb, label)

        if box is None:
            skipped += 1
            continue

        crop = crop_with_margin(image_bgr, box)
        cv2.imwrite(os.path.join(out_label_dir, filename), crop)
        saved += 1

    return saved, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True,
                         help='Raw dataset directory, one subfolder per label (from the dataset_dir launch argument + manual sorting)')
    parser.add_argument('--output', required=True,
                         help='Where to write cropped images, one subfolder per label - upload each as a class to Teachable Machine')
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input)
    output_dir = os.path.expanduser(args.output)
    os.makedirs(output_dir, exist_ok=True)

    known_labels = {p['label'] for p in ARTIFACT_COLOR_PROFILES} | {'stop_sign', NEGATIVE_LABEL}
    cascade = load_stop_sign_cascade()

    print(f'{"label":<16} {"saved":>6} {"skipped":>8}')
    total_saved, total_skipped = 0, 0
    for label in sorted(os.listdir(input_dir)):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            continue
        if label not in known_labels:
            print(f'{label:<16} (unrecognised label - skipping folder entirely)')
            continue

        saved, skipped = process_label_folder(label, label_dir, output_dir, cascade)
        total_saved += saved
        total_skipped += skipped
        print(f'{label:<16} {saved:>6} {skipped:>8}')

    print(f'\n{total_saved} crops saved to {output_dir}, {total_skipped} frames skipped '
          f'(no detection found in that folder\'s expected profile - worth a manual look if '
          f'this is a large fraction of a label\'s frames).')


if __name__ == '__main__':
    main()
