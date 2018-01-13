""" This module implements the analysis of the transformed binary image """
import numpy as np
import cv2


class LaneDetector(object):
    def __init__(self, filter_len=30, polyfilter_len=20):
        # The length of the smoothing filter
        self._filter_len = filter_len
        self._polyfilter_len = polyfilter_len

        # Array for storing the last n detections of the first peaks
        self._detections = []

        # Array of storing the last n lane polys
        self._poly_detections = []

    def __add_to_poly_detections(self, polynomials):
        """
        Helper function to push an element in filter sequence
        """
        if len(self._poly_detections) >= self._polyfilter_len:
            # Shift all items if filter is filled
            for i in range(len(self._poly_detections) - 1):
                self._poly_detections[i] = self._poly_detections[i + 1]
            self._poly_detections[-1] = polynomials
        else:
            self._poly_detections.append(polynomials)

    def __get_mean_of_polynomials(self):
        """
        Helper function to get the mean values from filter sequence
        """
        if len(self._poly_detections) > 0:
            return np.mean([x[0] for x in self._poly_detections], axis=0), \
                   np.mean([x[1] for x in self._poly_detections], axis=0)

        return None, None

    def __add_to_detections(self, detection):
        """
        Helper function to push an element in filter sequence
        """
        if len(self._detections) >= self._filter_len:
            # Shift all items if filter is filled
            for i in range(len(self._detections)-1):
                self._detections[i] = self._detections[i + 1]
            self._detections[-1] = detection
        else:
            self._detections.append([detection[0], detection[1]])

    def __get_mean_of_detections(self):
        """
        Helper function to get the mean values from filter sequence
        """
        if len(self._detections) > 0:
            mean_values = [int(np.mean([x[0] for x in self._detections])),
                           int(np.mean([x[1] for x in self._detections]))]
            return mean_values
        return None

    def find_lanes_hist_peaks_initial(self, bin_warped_img, signal_threshold=10000, signal_filter_len=30):
        """
        This function will search for initial peak points in the bird-eye-view-warped binary image.
        :returns A list of peaks that correspond to potential lane lines.
        """
        # Compute Histogram over the lower half of image
        col_sum = np.sum(bin_warped_img[bin_warped_img.shape[0] // 2:, :], axis=0)
        col_sum_bin = np.zeros_like(col_sum)

        # Signal filtering
        for i in range(col_sum.shape[0]):
            col_sum_bin[i] = 1 if np.max(col_sum[i:i + signal_filter_len]) > signal_threshold else 0

        # Detect Peaks
        peaks = []
        # A detected peak has an offset of filter-length / 2
        peak_offset = signal_filter_len // 2
        last_lowhigh_index = 0
        for i in range(col_sum.shape[0]):
            if i > 0 and col_sum_bin[i] > col_sum_bin[i - 1]:  # Going from LOW to HIGH - remember this event
                last_lowhigh_index = i
            if i > 0 and col_sum_bin[i] < col_sum_bin[i - 1]:  # Going from HIGH to LOW - mean it with last LH for peak
                if last_lowhigh_index is None:
                    raise ValueError('No LOW to HIGH transition detected for a peak.')
                peaks.append(((last_lowhigh_index + i) // 2) + peak_offset)
                last_lowhigh_index = None

        # Check for Signal at end of sequence
        if last_lowhigh_index is not None:
            peaks.append(((last_lowhigh_index + int(col_sum.shape[0])) // 2) + peak_offset)

        return peaks

    def find_lanes_hist_peaks_filter(self, bin_warped_img, peak_list, lane_seed_peak_threshold=75):
        """
        This function takes a peak list as starting points for filtering these start points.
        It will look in self._detections for last seed-peaks for the search.
        """
        # If there are peaks in the detections list, use them and look if a new peak is near them.
        # If a new peak is near enough, it is mean-ed with the last detection point.
        current_seed_peak_detection = [None, None]

        # Maybe we can pre-fill the current_seed_peak_detection
        if len(self._detections) > 0:
            # Yes, we have some older detections
            current_seed_peak_detection = self._detections[-1]
            # Check for new peaks
            for peak in peak_list:
                # get the last detected seed peaks and compare to new peaks
                for i, last_seed_peaks in enumerate(current_seed_peak_detection):
                    if np.abs(last_seed_peaks - peak) <= lane_seed_peak_threshold:
                        current_seed_peak_detection[i] = (last_seed_peaks + peak) // 2
        else:
            # Oh no, there are no pre-detections
            # Lets simply take the nearest peak to the middle, in the left half of the image.
            # Do the same for the right peak.
            mid_of_view = bin_warped_img.shape[1] // 2

            left_peaks = []
            right_peaks = []
            for peak in peak_list:
                abs_dist = np.abs(mid_of_view-peak)
                if peak > mid_of_view:
                    right_peaks.append((peak, abs_dist))
                else:
                    left_peaks.append((peak, abs_dist))

            # Abort if no peaks were found
            if len(left_peaks) <= 0 or len(right_peaks) <= 0:
                return None

            # Otherwise, we have our candidates!
            current_seed_peak_detection[0] = sorted(left_peaks, key=lambda x: x[1])[0][0]
            current_seed_peak_detection[1] = sorted(right_peaks, key=lambda x: x[1])[0][0]

            self.__add_to_detections(current_seed_peak_detection)
            current_seed_peak_detection = self.__get_mean_of_detections()

        return current_seed_peak_detection

    def find_lanes_hist_peaks_lane_positions(self,
                                             bin_warped_img,
                                             seed_peaks,
                                             intensity_threshold=0.1,
                                             shift_threshold=35,
                                             win_slide_size=(42, 42)):
        """
        This function will detect the lane pixels from seed peak points.
        :param bin_warped_img: The warped binary image.
        :param seed_peaks: Exactly two peaks as seeds.
        :return: mask_left_lane, mask_right_lane
        """
        mask_left_lane = np.zeros_like(bin_warped_img)
        mask_right_lane = np.zeros_like(bin_warped_img)

        if seed_peaks is None:
            return None, None

        if len(seed_peaks) != 2:
            raise Exception('Expecting exactly 2 seed peaks.')

        abs_intensity_threshold = win_slide_size[0] * win_slide_size[1] * intensity_threshold

        # A sliding window will now start at the seed peaks to find local maxima.
        # This is the array with 2 arrays, containing the detected positions,
        # it is initialized with the bottom-seed peaks:
        lane_positions = [[(bin_warped_img.shape[0], seed_peaks[0])], [(bin_warped_img.shape[0], seed_peaks[1])]]

        for i, seed_peak in enumerate(seed_peaks):
            for height in reversed(range(0, bin_warped_img.shape[0] + win_slide_size[0], win_slide_size[0])):
                max_signal = -1
                max_width = -1
                for width in range(seed_peak - shift_threshold, seed_peak):
                    # look for the strongest signal in this horizontal slice
                    non_zeros = np.count_nonzero(bin_warped_img[height:height+win_slide_size[0],
                                                                width:width+win_slide_size[1]])
                    if non_zeros > abs_intensity_threshold and non_zeros > max_signal:
                        max_signal = non_zeros
                        max_width = width

                if max_signal > 0:
                    seed_peak = max_width + (win_slide_size[1] // 2)
                    lane_positions[i].append([height, max_width + (win_slide_size[1] // 2)])

        return lane_positions

    def find_lanes_hist_peaks_lane_pixels(self, bin_warped_img, lane_positions, win_slide_size):
        # For every detected position, the underlying pixels are declared as the lane mask
        lane_pixels = [np.zeros_like(bin_warped_img),
                       np.zeros_like(bin_warped_img)]

        for i, pos in enumerate(lane_positions):
            drawing_rects = np.zeros_like(bin_warped_img)
            for point in pos:
                pt1 = (point[1] - (win_slide_size[0] // 2), point[0] - (win_slide_size[0] // 2))
                pt2 = (point[1] + (win_slide_size[0] // 2), point[0] + (win_slide_size[0] // 2))
                cv2.rectangle(drawing_rects, pt1, pt2, 255, -1)

            lane_pixels[i][(drawing_rects > 0) & (bin_warped_img > 0)] = 255

        return lane_pixels[0], lane_pixels[1]

    def fit_poly(self, mask_left, mask_right):
        """
        This function will fit the polys into the pixels of the lanes.
        :param mask_left: a mask of the left lane pixels.
        :param mask_right: a mask of the right lane pixels.
        :return: polynomials as array
        """
        left_pxs = np.where(mask_left > 0)
        rigth_pxs = np.where(mask_right > 0)

        if len(rigth_pxs[0]) <= 0 or len(left_pxs[0]) <= 0:
            return None, None

        left_poly = np.polyfit(left_pxs[0], left_pxs[1], 2)
        right_poly = np.polyfit(rigth_pxs[0], rigth_pxs[1], 2)

        return left_poly, right_poly

    def draw_poly(self, img, poly):
        """
        This function paints all points of the polynomial in the image.
        :param img: A grayscale image.
        :param poly: The polynomial.
        :return: Image with inpainted polynomial.
        """
        if poly is None:
            return img

        ys = np.linspace(0, img.shape[0] - 1, img.shape[0])
        xs = poly[0] * ys ** 2 + poly[1] * ys + poly[2]
        for pt in zip(xs, ys):
            if int(pt[0]) < img.shape[1] and int(pt[1]) < img.shape[0] and int(pt[0]) >= 0 and int(pt[1]) > 0:
                img[int(pt[1]), int(pt[0])] = 255

        return img

    def filter_poly(self, poly_left, poly_right):
        # Push the current polynomials to filter sequence
        if poly_left is not None and poly_right is not None:
            self.__add_to_poly_detections([poly_left, poly_right])

        # Get Mean of last n polynomials
        return self.__get_mean_of_polynomials()

    def calc_radius_offset_poly(self, poly_left, poly_right, image_width, image_height):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720    # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Calculate the offset of the car to center
        left_pos = (poly_left[0] * image_height ** 2 + poly_left[1] * image_height + poly_left[2]) - (image_width / 2)
        right_pos = (poly_right[0] * image_height ** 2 + poly_right[1] * image_height + poly_right[2]) - (image_width / 2)

        # Check if vehicle is left or right from center
        if abs(left_pos) > abs(right_pos):
            offset = -abs(right_pos-abs(left_pos))*xm_per_pix
        else:
            offset = abs(right_pos-abs(left_pos))*xm_per_pix

        # Convert polys to world space by recalculating them
        ys = np.linspace(0, image_height, num=image_height)
        xs_left = []
        xs_right = []
        for y in ys:
            xs_left.append(poly_left[0] * y ** 2 + poly_left[1] * y + poly_left[2])
            xs_right.append(poly_right[0] * y ** 2 + poly_right[1] * y + poly_right[2])

        # Create hstack from points for opencv fillPoly
        pts_left = np.array([np.transpose(np.vstack([xs_left, ys]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([xs_right, ys])))])
        poly = np.hstack((pts_left, pts_right))

        # Fit new polynomials to x,y in world space
        poly_left_meters = np.polyfit(ys * ym_per_pix, np.array(xs_left) * xm_per_pix, 2)
        poly_right_meters = np.polyfit(ys * ym_per_pix, np.array(xs_right) * xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * poly_left_meters[0] * image_height * ym_per_pix + poly_left_meters[1]) ** 2) ** 1.5) \
                        / np.absolute(2 * poly_left_meters[0])
        right_curverad = ((1 + (2 * poly_right_meters[0] * image_height * ym_per_pix + poly_right_meters[1]) ** 2)
                          ** 1.5) / np.absolute(2 * poly_right_meters[0])

        radius = (left_curverad + right_curverad) / 2
        return radius, offset, poly

    def sanity_check_poly(self, poly_left, poly_right, t_A=None, t_B=0.2):
        """
        This function performs a basic sanity check on the 2 polynomials of f(y) = Ay2 * By + C
        :param poly_left: The polynomial of the left lane.
        :param poly_right: The polynomial of the right lane.
        :param t_A: If the absolute difference of A_left and A_right is greater than t_A, None, None is returned.
        :param t_B: The same as t_A but for B_left and B_right
        :return: None, None if thresholds were exceeded.
        """
        # Check if polys are None
        if poly_left is None or poly_right is None:
            return None, None

        abs_diffs = []
        for i in range(len(poly_left)):
            abs_diffs.append(int(abs(poly_left[i] - poly_right[i]) * 100) / 100.0)

        if t_A is not None and abs_diffs[0] > t_A:
            return None, None

        if t_B is not None and abs_diffs[1] > t_B:
            return None, None

        return poly_left, poly_right


    def find_lanes_hist_peaks(self, bin_warped_img):
        # Window size for detection and inpainting
        win_size = (42, 42)

        # Current polynomials for the lanes
        poly_left = None
        poly_right = None
        poly = None
        radius_of_curvature = -1
        offset_to_center = -1

        mask_left = np.zeros_like(bin_warped_img)
        mask_right = np.zeros_like(bin_warped_img)

        # First step is to identify histogram peaks in the histogram of sums of the lower image part
        bare_peaks = self.find_lanes_hist_peaks_initial(bin_warped_img)

        # Then these currently detected peaks are fed into a smoothing function that checks previous detections
        filt_peaks = self.find_lanes_hist_peaks_filter(bin_warped_img, bare_peaks)

        # With the filtered 2 peaks, the lanes are identified by using the sliding window approach
        lane_posit = self.find_lanes_hist_peaks_lane_positions(bin_warped_img, filt_peaks, intensity_threshold=0.05, win_slide_size=win_size)
        if lane_posit[0] is not None:
            # With the rough positions, the pixels are identified
            mask_left, mask_right = self.find_lanes_hist_peaks_lane_pixels(bin_warped_img, lane_posit, win_slide_size=win_size)

            if mask_left is not None and mask_right is not None:

                # Fit the polynomials into the mask pixels
                poly_left, poly_right = self.fit_poly(mask_left, mask_right)

                # Do a basic sanity check for polynomials
                poly_left, poly_right = self.sanity_check_poly(poly_left, poly_right)

        # Check if pipeline failed.
        new_polys = poly_left is not None and poly_right is not None

        # Filter the polynomials
        poly_left, poly_right = self.filter_poly(poly_left, poly_right)

        # Compute the radius of curvature and the position with respect to center
        if poly_left is not None and poly_right is not None:
            radius_of_curvature, offset_to_center, poly = self.calc_radius_offset_poly(poly_left,
                                                                                       poly_right,
                                                                                       bin_warped_img.shape[1],
                                                                                       bin_warped_img.shape[0])

        # Prepare the image for the inpainted green lane
        image_with_lane = np.zeros(shape=(bin_warped_img.shape[0], bin_warped_img.shape[1], 3), dtype=np.uint8)
        if poly is not None:
            color = (0, 255, 0) if new_polys else (0, 255, 196)
            cv2.fillPoly(image_with_lane, np.int_([poly]), color)

        image_with_polys = np.zeros_like(bin_warped_img)
        image_with_polys = self.draw_poly(image_with_polys, poly_left)
        image_with_polys = self.draw_poly(image_with_polys, poly_right)
        image_with_polys = cv2.dilate(image_with_polys, np.ones((3, 3), np.uint8), 1)

        # A nice looking image is created here
        inpainted_lanes = np.zeros(shape=(bin_warped_img.shape[0], bin_warped_img.shape[1], 3), dtype=np.uint8)
        inpainted_lanes[:, :, :][bin_warped_img > 0] = 255
        inpainted_lanes[:, :, :][(mask_left > 0) | (mask_right > 0)] = 0
        inpainted_lanes[:, :, 2][mask_left > 0] = 255
        inpainted_lanes[:, :, 0][mask_right > 0] = 255
        inpainted_lanes[:, :, :][image_with_polys > 0] = [0, 255, 255]

        for pos in lane_posit:
            if pos:
                for point in pos:
                    pt1 = (point[1] - (win_size[0] // 2), point[0] - (win_size[0] // 2))
                    pt2 = (point[1] + (win_size[0] // 2), point[0] + (win_size[0] // 2))
                    cv2.rectangle(inpainted_lanes, pt1, pt2, (0, 128, 0), 1)

        return inpainted_lanes, image_with_lane, radius_of_curvature, offset_to_center


