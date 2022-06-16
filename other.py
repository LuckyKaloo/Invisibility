import numpy as np


class Pinhole:
    def __init__(self, pos, radius):
        self.pos = pos
        self.radius = radius

        self.diff = 3

        self.c1 = np.array([self.diff * self.radius, self.diff * self.radius, pos[2]])
        self.c2 = np.array([- self.diff * self.radius, - self.diff * self.radius, pos[2]])

    def blocks(self, pos_ray, vec_ray):
        final_pos = pos_ray + vec_ray * (self.pos[2] - pos_ray[2]) / vec_ray[2]
        d_vec = final_pos - self.pos
        d = np.linalg.norm(d_vec)

        return d > self.radius

class Screen:
    def __init__(self, z):
        self.z = z

    def intersection(self, ray_pos, ray_vec):
        t = (self.z - ray_pos[2]) / ray_vec[2]
        intersection = ray_pos + t * ray_vec

        return intersection[0:2]

class ScreenBounded:
    def __init__(self, x, y, z, w, h):
        self.z = z

        self.x_min = x - w / 2
        self.x_max = x + w / 2
        self.y_min = y - h / 2
        self.y_max = y + h / 2

    def intersection(self, ray_pos, ray_vec):
        t = (self.z - ray_pos[2]) / ray_vec[2]
        intersection = ray_pos + t * ray_vec

        if self.x_min <= intersection[0] <= self.x_max and self.y_min <= intersection[1] <= self.y_max:
            return intersection[0:2]
        else:
            return None
