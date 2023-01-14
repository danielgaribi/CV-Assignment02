"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def calc_ssdd(left_image: np.ndarray,
                  right_image: np.ndarray,
                  win_size: int,
                  dsp_range: int,
                  row: int,
                  col: int,
                  d: int) -> int:
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        diparity = d - dsp_range

        left_min_row = row - win_size//2
        left_max_row = row + win_size//2
        left_min_col = col - win_size//2
        left_max_col = col + win_size//2

        right_min_row = row - win_size//2
        right_max_row = row + win_size//2
        right_min_col = dsp_range + col - win_size//2 + diparity
        right_max_col = dsp_range + col + win_size//2 + diparity

        return np.sum((left_image[left_min_row:left_max_row+1, left_min_col:left_max_col+1] -
                right_image[right_min_row:right_max_row+1, right_min_col:right_max_col+1] ) ** 2)


    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        """INSERT YOUR CODE HERE"""
        left_image_pad = np.pad(left_image, ((win_size//2, win_size//2), (win_size//2, win_size//2), (0,0)) , mode='constant')
        right_image_pad = np.pad(right_image, ((win_size//2, win_size//2), (win_size//2+dsp_range, win_size//2+dsp_range), (0,0)) , mode='constant')
        for row in range(win_size//2, num_of_rows+win_size//2):
            for col in range(win_size//2, num_of_cols+win_size//2):
                for d in range(dsp_range * 2 + 1):
                    ssdd_tensor[row-win_size//2, col-win_size//2, d] = Solution.calc_ssdd(left_image_pad, right_image_pad, win_size,
                                                                  dsp_range, row, col, d)

        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        # you can erase the label_no_smooth initialization.
        label_no_smooth = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
        """INSERT YOUR CODE HERE"""
        label_no_smooth = ssdd_tensor.argmin(2)
        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        """INSERT YOUR CODE HERE"""
        l_slice = np.copy(c_slice)
        for c in range(1, num_of_cols):
            min_prev_col = min([l_slice[d_prev, c-1] for d_prev in range(num_labels)])
            for d in range(num_labels):
                l_slice[d, c] += min(l_slice[d, c-1],
                                     p1 + min([l_slice[d+i, c-1] for i in [1, -1] if 0<=d+i<num_labels]),
                                     p2 + min([l_slice[d+k, c-1] for k in range(-d, num_labels-d) if abs(k)>=2]))\
                                 - min_prev_col
        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        for i, l_slice in enumerate(ssdd_tensor):
            l[i, :, :] = self.dp_grade_slice(l_slice.T, p1, p2).T
        return self.naive_labeling(l)


    def extract_ssdd_slice(self,
                           ssdd_tensor: np.ndarray,
                           row: int,
                           col: int,
                           offset: int,
                           direction: int):
        nof_rows, nof_cols, _ = ssdd_tensor.shape
        # if direction == 1:
        #     return ssdd_tensor[row, :col, :]
        # elif direction == 2:
        #     offset = col-row
        #     return ssdd_tensor.diagonal(offset).T[:row if offset>=0 else row+offset, :]
        # elif direction == 3:
        #     return ssdd_tensor[:row, col, :]
        # elif direction == 4:
        #     offset = (nof_col-col-1)-row
        #     return np.fliplr(ssdd_tensor).diagonal(offset).T[:row if offset>=0 else row+offset, :]
        # elif direction == 5:
        #     return np.flip(ssdd_tensor[row, col+1:, :], axis=0)
        # elif direction == 6:
        #     offset = col - row
        #     return np.flip(ssdd_tensor.diagonal(offset).T[(row if offset>=0 else row+offset)+1:, :], axis=0)
        # elif direction == 7:
        #     return np.flip(ssdd_tensor[row+1:, col, :], axis=0)
        # elif direction == 8:
        #     offset = (nof_col - col - 1) - row
        #     return np.flip(np.fliplr(ssdd_tensor).diagonal(offset).T[(row if offset>=0 else row+offset)+1:, :], axis=0)

        rows = np.arange(0, nof_rows)
        cols = np.arange(0, nof_cols)
        cols, rows = np.meshgrid(cols, rows)

        if direction == 1:
            return ssdd_tensor[row, :, :],\
                   rows[row, :],\
                   cols[row, :]
        elif direction == 2:
            return ssdd_tensor.diagonal(offset).T[:, :],\
                   rows.diagonal(offset).T[:],\
                   cols.diagonal(offset).T[:]
        elif direction == 3:
            return ssdd_tensor[:, col, :],\
                   rows[:, col],\
                   cols[:, col]
        elif direction == 4:
            return np.fliplr(ssdd_tensor).diagonal(offset).T[:, :],\
                   np.fliplr(rows).diagonal(offset).T[:],\
                   np.fliplr(cols).diagonal(offset).T[:]
        elif direction == 5:
            return np.flip(ssdd_tensor[row, :, :], axis=0),\
                   np.flip(rows[row, :], axis=0),\
                   np.flip(cols[row, :], axis=0)
        elif direction == 6:
            return np.flip(ssdd_tensor.diagonal(offset).T[:, :], axis=0),\
                   np.flip(rows.diagonal(offset).T[:], axis=0),\
                   np.flip(cols.diagonal(offset).T[:], axis=0)
        elif direction == 7:
            return np.flip(ssdd_tensor[:, col, :], axis=0),\
                   np.flip(rows[:, col], axis=0),\
                   np.flip(cols[:, col], axis=0)
        elif direction == 8:
            return np.flip(np.fliplr(ssdd_tensor).diagonal(offset).T[:, :], axis=0),\
                   np.flip(np.fliplr(rows).diagonal(offset).T[:], axis=0),\
                   np.flip(np.fliplr(cols).diagonal(offset).T[:], axis=0)
        else:
            raise ValueError("Invalid direction")

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        """INSERT YOUR CODE HERE"""
        nof_rows, nof_cols, _ = ssdd_tensor.shape
        ls = {direction: np.zeros_like(ssdd_tensor) for direction in range(1, num_of_directions+1)}
        row = 0
        col = 0
        for i in range(nof_rows+nof_cols-1):
            for direction in range(1, num_of_directions+1):
                offset = -(nof_rows-1) + row + col
                slices, rows, cols = self.extract_ssdd_slice(ssdd_tensor, row, col, offset, direction)
                ls[direction][rows, cols, :] = self.dp_grade_slice(slices.T, p1, p2).T
            if row < nof_rows-1:
                row += 1
            else:
                col += 1

        direction_to_slice = {direction: self.naive_labeling(ls[direction]) for direction in range(1, num_of_directions+1)}

        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        """INSERT YOUR CODE HERE"""
        return self.naive_labeling(l)

