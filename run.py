import glob
import math
from copy import copy

import numpy as np
from matplotlib import pyplot as plt
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

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
                   processor=None):
    """Create function that processes one frame of video"""

    def _process_image(img):
        out_img, heat_map = find_cars(
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
            scale=scale
        )
        new_map = apply_threshold(heat_map, threshold=threshold)
        labels = label(new_map)
        processor.update(labels)

        return draw_bboxes(np.copy(img), processor.bboxes)

    return _process_image


def visualize(fig, rows, cols, imgs, titles):
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)

        if img_dims < 3:
            plt.imshow(img, cmap='hot')
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])


class VideoProcessor(object):

    def __init__(self, x_max, y_max, margins=100, close_dist=50):
        self.x_max = x_max
        self.y_max = y_max
        self.margins = margins
        self.close_dist = close_dist
        self.bboxes = []

    def find_match(self, bbox):
        """
        Checks to see if new box is valid.
        It's valid if it is near an edge, or if close to one of the previous boxes
        """
        matches = []
        center = self.get_center(bbox)

        for idx, b in enumerate(self.bboxes):
            b_center = self.get_center(b)
            dist = self.distance(center, b_center)
            print('..checking dist between {} and {}'.format(center, b_center))
            print('..= {}'.format(dist))

            if dist <= self.close_dist:
                matches.append(idx)

        return matches

    def distance(self, p0, p1):
        return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

    def near_bounds(self, bbox):
        center = self.get_center(bbox)

        # Close to 0s
        if (center[0] <= self.margins) or (center[1] <= self.margins):
            return True
        # Close to x_max
        if (self.x_max - center[0]) <= self.margins:
            return True
        # Close to y_max
        if (self.y_max - center[1]) <= self.margins:
            return True
        return False

    def smooth_boxes(self, bboxes):
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

    def update(self, labels):
        """
        Checks new array of boxes against previous boxes.
        Removes invalid boxes.
        """
        if len(labels) > 0:
            new_boxes = copy(self.bboxes)
            # grouped_labels = group_labels(labels)

            for car_number in range(1, labels[1] + 1):
                nonzero = (labels[0] == car_number).nonzero()
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])
                bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                        (np.max(nonzerox), np.max(nonzeroy)))

                print(bbox)
                match_idx = self.find_match(bbox)

                # Check to see if it's new car coming into frame
                if match_idx:
                    print('>>> Found matches: {}'.format(match_idx))

                    # Create smooth box
                    matched_bboxes = [new_boxes[idx] for idx in match_idx]
                    print(matched_bboxes)
                    matched_bboxes.append(bbox)
                    smoothed_box = self.smooth_boxes(matched_bboxes)

                    # Pop old boxes
                    new_boxes = [
                        v for i, v in enumerate(new_boxes)
                        if i not in match_idx
                    ]
                    # Append smoothed box
                    new_boxes.append(smoothed_box)
                else:
                    # Otherwise check to see if it's near bounds as a new car
                    is_new = self.near_bounds(bbox)

                    if is_new:
                        print('>>> Near bounds!')
                        new_boxes.append(bbox)

            self.bboxes = new_boxes

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
    output_path = 'processed_video.mp4'

    # ================
    # Train classifier
    # ================

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

    X, X_scaler = generate_X(car_features, notcar_features)
    y = generate_y(car_features, notcar_features)

    model = train_model(X, y)

    # =============
    # Process video
    # =============

    clip = VideoFileClip('project_video.mp4').subclip(5, 7)
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
        processor=processor
    )

    print('Processing video...')
    out_clip = clip.fl_image(process_image)
    out_clip.write_videofile(output_path, audio=False)
