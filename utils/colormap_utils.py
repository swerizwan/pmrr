"""An awesome colormap for really neat visualizations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

def colormap(rgb=False):
    """
    Generates a colormap array with predefined colors.

    Parameters:
    rgb (bool): If True, the colors are in RGB format. 
                If False, the colors are in BGR format (default is False).

    Returns:
    np.ndarray: An array of colors in the specified format.
    """

    # Define a list of colors as an array. Each color is represented by three values (R, G, B).
    color_list = np.array(
        [
            # Format: [R, G, B, R, G, B, ...]
            0.000, 0.447, 0.741,  # blue
            0.850, 0.325, 0.098,  # red
            0.929, 0.694, 0.125,  # yellow
            0.494, 0.184, 0.556,  # purple
            0.466, 0.674, 0.188,  # green
            0.301, 0.745, 0.933,  # light blue
            0.635, 0.078, 0.184,  # dark red
            0.300, 0.300, 0.300,  # grey
            0.600, 0.600, 0.600,  # light grey
            1.000, 0.000, 0.000,  # bright red
            1.000, 0.500, 0.000,  # orange
            0.749, 0.749, 0.000,  # yellow-green
            0.000, 1.000, 0.000,  # bright green
            0.000, 0.000, 1.000,  # bright blue
            0.667, 0.000, 1.000,  # purple
            0.333, 0.333, 0.000,  # olive
            0.333, 0.667, 0.000,  # green
            0.333, 1.000, 0.000,  # lime green
            0.667, 0.333, 0.000,  # brown
            0.667, 0.667, 0.000,  # mustard
            0.667, 1.000, 0.000,  # yellow-green
            1.000, 0.333, 0.000,  # orange-red
            1.000, 0.667, 0.000,  # gold
            1.000, 1.000, 0.000,  # yellow
            0.000, 0.333, 0.500,  # teal
            0.000, 0.667, 0.500,  # green-blue
            0.000, 1.000, 0.500,  # bright green-blue
            0.333, 0.000, 0.500,  # dark purple
            0.333, 0.333, 0.500,  # grey-blue
            0.333, 0.667, 0.500,  # teal
            0.333, 1.000, 0.500,  # green-teal
            0.667, 0.000, 0.500,  # purple-pink
            0.667, 0.333, 0.500,  # mauve
            0.667, 0.667, 0.500,  # olive-green
            0.667, 1.000, 0.500,  # light green
            1.000, 0.000, 0.500,  # pink
            1.000, 0.333, 0.500,  # light pink
            1.000, 0.667, 0.500,  # peach
            1.000, 1.000, 0.500,  # light yellow
            0.000, 0.333, 1.000,  # light blue
            0.000, 0.667, 1.000,  # sky blue
            0.000, 1.000, 1.000,  # cyan
            0.333, 0.000, 1.000,  # purple-blue
            0.333, 0.333, 1.000,  # blue-grey
            0.333, 0.667, 1.000,  # light blue
            0.333, 1.000, 1.000,  # cyan
            0.667, 0.000, 1.000,  # purple
            0.667, 0.333, 1.000,  # mauve
            0.667, 0.667, 1.000,  # light purple
            0.667, 1.000, 1.000,  # light cyan
            1.000, 0.000, 1.000,  # magenta
            1.000, 0.333, 1.000,  # light magenta
            1.000, 0.667, 1.000,  # pink
            0.167, 0.000, 0.000,  # dark red
            0.333, 0.000, 0.000,  # darker red
            0.500, 0.000, 0.000,  # even darker red
            0.667, 0.000, 0.000,  # even darker red
            0.833, 0.000, 0.000,  # even darker red
            1.000, 0.000, 0.000,  # red
            0.000, 0.167, 0.000,  # dark green
            0.000, 0.333, 0.000,  # darker green
            0.000, 0.500, 0.000,  # even darker green
            0.000, 0.667, 0.000,  # even darker green
            0.000, 0.833, 0.000,  # even darker green
            0.000, 1.000, 0.000,  # green
            0.000, 0.000, 0.167,  # dark blue
            0.000, 0.000, 0.333,  # darker blue
            0.000, 0.000, 0.500,  # even darker blue
            0.000, 0.000, 0.667,  # even darker blue
            0.000, 0.000, 0.833,  # even darker blue
            0.000, 0.000, 1.000,  # blue
            0.000, 0.000, 0.000,  # black
            0.143, 0.143, 0.143,  # dark grey
            0.286, 0.286, 0.286,  # darker grey
            0.429, 0.429, 0.429,  # even darker grey
            0.571, 0.571, 0.571,  # even darker grey
            0.714, 0.714, 0.714,  # even darker grey
            0.857, 0.857, 0.857,  # light grey
            1.000, 1.000, 1.000   # white
        ]
    ).astype(np.float32)

    # Reshape the color list to have each color as a separate sub-array and scale values to 255
    color_list = color_list.reshape((-1, 3)) * 255

    # If RGB is False, switch the order to BGR
    if not rgb:
        color_list = color_list[:, ::-1]

    return color_list
