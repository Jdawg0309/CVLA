"""Adapter to make ImageData compatible with domain ImageMatrix interface."""


class ImageDataAdapter:
    """Wraps ImageData to provide ImageMatrix-compatible interface for domain functions."""

    def __init__(self, image_data):
        self.data = image_data.pixels
        self.name = image_data.name

    def as_matrix(self):
        if len(self.data.shape) == 2:
            return self.data
        return (0.299 * self.data[:, :, 0] +
                0.587 * self.data[:, :, 1] +
                0.114 * self.data[:, :, 2])

    @property
    def is_grayscale(self):
        return len(self.data.shape) == 2

    @property
    def height(self):
        return self.data.shape[0]

    @property
    def width(self):
        return self.data.shape[1]

    @property
    def history(self):
        return []
