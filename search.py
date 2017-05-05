from __future__ import absolute_import

import numpy as np

import cv2
from features import bin_spatial, color_hist, convert_color, get_hog_features


def find_cars(img, svc, X_scaler,
              x_start_stop, y_start_stop,
              xy_overlap, xy_window,
              orient, pix_per_cell,
              cell_per_block,
              spatial_size, hist_bins,
              scale=1.5):
    """
    Extract all features for given image in one go.
    """
    img_boxes = []
    count = 0

    draw_img = np.copy(img)
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    img = img.astype(np.float32) / 255

    # Heatmap
    heatmap = np.zeros_like(img[:,:,0])

    # Search windows
    img_tosearch = img[
        y_start_stop[0]:y_start_stop[1],
        x_start_stop[0]:x_start_stop[1],
        :
    ]
    ctrans_tosearch = convert_color(img_tosearch, color_space='YCrCb')

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(
            ctrans_tosearch,
            (np.int(imshape[1]/scale),
            (np.int(imshape[0]/scale)))
        )

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2 # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for entire image
    hog1 = get_hog_features(
        ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(
        ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(
        ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    ystart = y_start_stop[0]

    for xb in range(nxsteps):
        for yb in range(nysteps):
            count += 1
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            hog_feat1 = hog1[
                ypos:ypos+nblocks_per_window,
                xpos:xpos+nblocks_per_window
            ].ravel()
            hog_feat2 = hog2[
                ypos:ypos+nblocks_per_window,
                xpos:xpos+nblocks_per_window
            ].ravel()
            hog_feat3 = hog3[
                ypos:ypos+nblocks_per_window,
                xpos:xpos+nblocks_per_window
            ].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract image patch
            subimg = cv2.resize(
                ctrans_tosearch[
                    ytop:ytop+window,
                    xleft:xleft+window
                ],
                (64, 64)
            )

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make prediction
            test_features = X_scaler.transform(
                np.hstack(
                    (spatial_features, hist_features, hog_features)
                ).reshape(1, -1)
            )
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                start_coords = (xbox_left, ytop_draw+ystart)
                end_coords = (xbox_left+win_draw, ytop_draw+win_draw+ystart)

                cv2.rectangle(
                    draw_img,
                    start_coords,
                    end_coords,
                    color=(0, 0, 255)
                )
                img_boxes.append(
                    (start_coords, end_coords)
                )
                heatmap[
                    ytop_draw+ystart:ytop_draw+win_draw+ystart,
                    xbox_left:xbox_left+win_draw
                ] += 1

    return draw_img, heatmap


def apply_threshold(heat_map, threshold):
    new_map = np.copy(heat_map)
    # Zero out pixels below the threshold
    new_map[new_map < threshold] = 0
    # Return thresholded map
    return new_map


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


def draw_bboxes(img, bboxes):
    """Draw bounding boxes"""
    for box in bboxes:
        cv2.rectangle(img, box[0], box[1], (0,0,255), 6)

    return img
