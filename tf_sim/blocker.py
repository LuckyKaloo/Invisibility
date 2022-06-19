import tensorflow as tf


class ShapeBlocker:
    def blocks(self, ray_pos):
        raise NotImplementedError("To be implemented in subclass")


class RectangleBlocker(ShapeBlocker):
    def __init__(self, x, y, z, w, h):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.h = h

        self.x_min = x - w / 2
        self.x_max = x + w / 2
        self.y_min = y - h / 2
        self.y_max = y + h / 2

    def get_config(self):
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "w": self.w,
            "h": self.h
        }

    def blocks(self, ray_pos):
        return tf.logical_and(
            tf.logical_and(self.x_min <= ray_pos[:, 0], ray_pos[:, 0] <= self.x_max),
            tf.logical_and(self.y_min <= ray_pos[:, 1], ray_pos[:, 1] <= self.y_max),
        )
