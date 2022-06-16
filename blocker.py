import numpy as np

class ShapeBlocker:
    def blocks(self, ray_pos):
        pass


class RectangleBlocker(ShapeBlocker):
    def __init__(self, x, y, z, w, h):
        self.z = z

        self.x_min = x - w / 2
        self.x_max = x + w / 2
        self.y_min = y - h / 2
        self.y_max = y + h / 2

    def blocks(self, ray_pos):
        return self.x_min <= ray_pos[0] <= self.x_max and self.y_min <= ray_pos[1] <= self.y_max


class TiltedRectangleBlocker(RectangleBlocker):
    def __init__(self, x, y, z, w, h, angle):
        RectangleBlocker.__init__(self, x, y, z, w, h)
        self.cos = np.cos(angle)
        self.sin = np.sin(angle)

    def blocks(self, ray_pos):
        return self.x_min <= ray_pos[0] * self.cos - ray_pos[1] * self.sin <= self.x_max and self.y_min <= ray_pos[0] * self.sin + ray_pos[1] * self.cos <= self.y_max


class EllipseBlocker(ShapeBlocker):
    def __init__(self, x, y, z, w, h):
        self.x = x
        self.y = y
        self.z = z

        self.a2 = w * w / 4
        self.b2 = h * h / 4

    def blocks(self, ray_pos):
        return (ray_pos[0] - self.x) * (ray_pos[0] - self.x) / self.a2 + (ray_pos[1] - self.y) * (ray_pos[1] - self.y) / self.b2 <= 1



