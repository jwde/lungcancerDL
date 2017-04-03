import numpy as np
def hu_to_visual_features(img, low, high):
    """
    Parameters:
        img := 3D array of hu units
        low := everything less than low gets maped to low
        high := everything more than high gets mapped to high
    """

    # Make a deep copy to avoid np pointer craziness...
    # TODO: does this need to happen?
    new_image = np.copy(img)

    # Threshold the values
    new_image[new_image < low] = low
    new_image[new_image > high] = high

    # Scale the values
    new_image -= low
    new_image = new_image / float(high - low)

    return new_image
