import os.path
import glob
import math
from copy import copy

import numpy as np
from matplotlib import pyplot as plt
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from sklearn.externals import joblib

from features import extract_features
from search import apply_threshold, draw_bboxes, find_cars
from train import generate_X, generate_y, train_model


def process_imager(svc, X_scaler,
                   x_start_stop, y_start_stop,
                   xy_overlap, xy_window,
                   orient, pix_per_cell,
                   cell_per_block,
                   spatial_size, hist_bins,
                   scale=1.5,
                   threshold=1,
                   processor=None,
                   trace=False):
    """Create function that processes one frame of video"""

    def _process_image(img):
        draw_img, heat_map = find_cars(
            img,
            svc=svc, X_scaler=X_scaler,
            x_start_stop=x_start_stop,
            y_start_stop=y_start_stop,
            xy_overlap=xy_overlap,
            xy_window=xy_window,
            orient=orient,
            pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            spatial_size=spatial_size,
            hist_bins=hist_bins,
            scale=scale,
            color=(0,255,0)
        )
        new_map = apply_threshold(heat_map, threshold=threshold)
        labels = label(new_map)
        processor.update(labels)

        if trace:
            out_img = np.copy(draw_img)
        else:
            out_img = np.copy(img)

        return draw_bboxes(
            out_img,
            processor.bboxes
        )

    return _process_image


def visualize(fig, rows, cols, imgs, titles, cmap=None):
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)

        if cmap is None:
            if img_dims < 3:
                cmap = 'hot'
            else:
                cmap = 'gray'

        plt.imshow(img, cmap=cmap)
        plt.title(titles[i])


class VideoProcessor(object):

    def __init__(self, x_max, y_max,
                 in_margins=100,
                 out_margins=50,
                 close_dist=25):
        self.frame = 0
        self.x_max = x_max
        self.y_max = y_max
        self.in_margins = in_margins
        self.out_margins = out_margins
        self.close_dist = close_dist
        self.bboxes = []

    def find_match(self, bbox, potential_matches):
        """
        Checks to see if new box is valid.
        It's valid if it is near an edge, or if close to one of the previous boxes
        """
        matches = []

        for idx, b in enumerate(potential_matches):
            dist = self.distance(bbox, b)
            print('..checking dist between {} and {} = {}'.format(
                bbox, b, dist))

            if dist <= self.close_dist:
                matches.append(idx)

        return matches

    def distance(self, bbox, potential_match):
        (x_min, y_min), (x_max, y_max) = bbox
        (x_min_2, y_min_2), (x_max_2, y_max_2) = potential_match

        center = self.get_center(bbox)
        match_center = self.get_center(potential_match)

        x_dist = None
        y_dist = None

        # bbox is to right of potential_match
        if x_min > x_max_2:
            x_dist = x_min - x_max_2
        # bbox is to left of potential match
        elif x_min_2 > x_max:
            x_dist = x_min_2 - x_max
        # boxes overlap
        else:
            x_dist = 0
            # x_dist = abs(match_center[0] - center[0])

        # bbox is on top of potential_match
        if y_min > y_max_2:
            y_dist = y_min - y_max_2
        # bbox is below potential match
        elif y_min_2 > y_max:
            y_dist = y_min_2 - y_max
        # boxes overlap
        else:
            y_dist = 0
            # y_dist = abs(match_center[1] - center[1])

        return math.sqrt((x_dist ** 2) + (y_dist ** 2))

    def near_bounds(self, bbox, margins):
        (x_min, y_min), (x_max, y_max) = bbox
        # center = self.get_center(bbox)

        # Close to 0s
        # if (center[0] <= self.margins) or (center[1] <= self.margins):
        if (x_min <= margins) or (y_min <= margins):
            return True
        # Close to x_max
        # if (self.x_max - center[0]) <= self.margins:
        if (self.x_max - x_max) <= margins:
            return True
        # Close to y_max
        # if (self.y_max - center[1]) <= self.margins:
        if (self.y_max - y_max) <= margins:
            return True
        return False

    def smooth_bboxes(self, bboxes):
        """
        Create new bounding box by smoothing over a collection of bounding boxes.
        """
        x_mins, x_maxes = [], []
        y_mins, y_maxes = [], []

        for bbox in bboxes:
            (x_min, y_min), (x_max, y_max) = bbox

            x_mins.append(x_min)
            x_maxes.append(x_max)
            y_mins.append(y_min)
            y_maxes.append(y_max)

        return (
            (int(np.average(x_mins)), int(np.average(y_mins))),
            (int(np.average(x_maxes)), int(np.average(y_maxes)))
        )

    def label_bboxes(self, labels):
        bboxes = []

        for car_number in range(1, labels[1] + 1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                    (np.max(nonzerox), np.max(nonzeroy)))
            bboxes.append(bbox)

        return bboxes

    def update(self, labels):
        """
        Checks new array of boxes against previous boxes.
        Removes invalid boxes.
        """
        new_bboxes = copy(self.bboxes)
        label_bboxes = self.label_bboxes(labels)

        add_bboxes = []
        add_label_idx = set([])
        prune_idx = set([])

        match_map = {}

        # Prune bboxes that don't appear in labels, and are close to margins
        print('>>> Updating {}'.format(self.frame))
        print('  > Pruning old boxes ({})'.format(len(new_bboxes)))
        for i, bbox in enumerate(new_bboxes):
            match_idx = self.find_match(bbox, label_bboxes)

            if not match_idx and self.near_bounds(
                    bbox, margins=self.out_margins):
                prune_idx.add(i)

        # Process labeled bboxes
        print('  > Processing labels ({})'.format(len(labels)))
        for i, bbox in enumerate(label_bboxes):
            match_idx = self.find_match(bbox, new_bboxes)

            if match_idx:

                # Store matches for processing
                for idx in match_idx:
                    if not match_map.get(idx):
                        match_map[idx] = []
                    match_map[idx].append(i)

            # Otherwise add if first frame or near bounds
            elif self.frame == 0 or self.near_bounds(
                    bbox, margins=self.in_margins):
                add_label_idx.add(i)

        print('  > Processing matches ({})'.format(len(match_map)))
        for idx, matches in match_map.items():
            prune_idx.add(idx)  # remove original regardless

            # If multiple labels to existing boxes, break into labels
            if len(matches) > 1:
                add_label_idx |= set(matches)

            # If 1-1 label-bbox mapping, smooth into new box
            else:
                _grouped_bboxes = [new_bboxes[idx], label_bboxes[matches[0]]]
                smoothed_bbox = self.smooth_bboxes(_grouped_bboxes)
                add_bboxes.append(smoothed_bbox)

        # Prune and add bboxes
        print('  > Updating bboxes')
        new_bboxes = [
            v for i, v in enumerate(new_bboxes)
            if i not in prune_idx
        ]
        for idx in add_label_idx:
            add_bboxes.append(label_bboxes[idx])

        new_bboxes += add_bboxes
        self.bboxes = new_bboxes
        self.frame += 1

    def get_center(self, bbox):
        (x_min, y_min), (x_max, y_max) = bbox

        return (float(x_max - x_min) / 2 + x_min,
                float(y_max - y_min) / 2 + y_min)


if __name__ == '__main__':

    # ==========
    # Parameters
    # ==========

    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off

    x_start_stop = [None, None]
    y_start_stop = [400, 656] # Min and max in y to search in slide_window()
    xy_overlap = (0.5, 0.5)
    xy_window = (128, 128)
    scale = 1.5
    threshold = 2
    output_path = 'detailed_processed_video.mp4'

    model_path = 'model.pkl'
    X_scaler_path = 'X_scaler.pkl'

    # ================
    # Train classifier
    # ================
    model = None
    X_scaler = None

    if os.path.isfile(model_path):
        model = joblib.load(model_path)

    if os.path.isfile(X_scaler_path):
        X_scaler = joblib.load(X_scaler_path)

    if model is None:
        # Get data
        cars = glob.glob('./data/vehicles/**/*.png')
        notcars = glob.glob('./data/non-vehicles/**/*.png')

        # Extract features
        print('Extracting car features...')
        car_features = extract_features(
            cars,
            color_space=color_space,
            spatial_size=spatial_size, hist_bins=hist_bins,
            orient=orient, pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            hog_channel=hog_channel, spatial_feat=spatial_feat,
            hist_feat=hist_feat, hog_feat=hog_feat
        )

        print('Extracting not car features...')
        notcar_features = extract_features(
            notcars,
            color_space=color_space,
            spatial_size=spatial_size, hist_bins=hist_bins,
            orient=orient, pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            hog_channel=hog_channel, spatial_feat=spatial_feat,
            hist_feat=hist_feat, hog_feat=hog_feat
        )

        # Load or create scaler
        save_scaler = (True if X_scaler is None else False)

        X, X_scaler = generate_X(
            car_features, notcar_features,
            X_scaler=X_scaler
        )
        y = generate_y(car_features, notcar_features)
        if save_scaler:
            joblib.dump(X_scaler, X_scaler_path)

        model = train_model(X, y)
        joblib.dump(model, model_path)

    # =============
    # Process video
    # =============

    clip = VideoFileClip('project_video.mp4')#.subclip(27, -1)
    # clip = VideoFileClip('project_video.mp4').subclip(27, 37)
    frame = clip.get_frame(0)

    processor = VideoProcessor(
        x_max=frame.shape[1],
        y_max=frame.shape[0]
    )
    process_image = process_imager(
        svc=model,
        X_scaler=X_scaler,
        x_start_stop=x_start_stop,
        y_start_stop=y_start_stop,
        xy_overlap=xy_overlap,
        xy_window=xy_window,
        orient=orient,
        pix_per_cell=pix_per_cell,
        cell_per_block=cell_per_block,
        spatial_size=spatial_size,
        hist_bins=hist_bins,
        scale=scale,
        threshold=threshold,
        processor=processor,
        trace=False
    )

    print('Processing video...')
    out_clip = clip.fl_image(process_image)
    out_clip.write_videofile(output_path, audio=False)
