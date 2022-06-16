import numpy as np


class ShapeSource:
    def generate_pos(self):
        pass


class RectangleSource(ShapeSource):
    def __init__(self, x, y, z, w, h):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.h = h

    def generate_pos(self):
        rand_x = (np.random.rand() - 0.5) * self.w + self.x
        rand_y = (np.random.rand() - 0.5) * self.h + self.y
        return np.array([rand_x, rand_y, self.z])


class EllipseSource(ShapeSource):
    def __init__(self, x, y, z, w, h):
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.h = h

    def generate_pos(self):
        while True:
            rand_x_frac = (np.random.rand() - 0.5)
            rand_y_frac = (np.random.rand() - 0.5)

            if rand_x_frac * rand_x_frac + rand_y_frac + rand_y_frac < 1:
                rand_x = rand_x_frac * self.w + self.x
                rand_y = rand_y_frac * self.h + self.y

                return np.array([rand_x, rand_y, self.z])


class CrossSource(ShapeSource):
    def __init__(self, x, y, z, w_h, h_h, w_v, h_v):
        self.rect1 = RectangleSource(x, y, z, w_h, h_h)
        self.rect2 = RectangleSource(x, y + (h_v + h_h) / 4, z, w_v, (h_v - h_h) / 2)
        self.rect3 = RectangleSource(x, y - (h_v + h_h) / 4, z, w_v, (h_v - h_h) / 2)

        area1 = w_h * h_h
        area2 = area3 = w_v * (h_v - h_h) / 2
        total_area = area1 + area2 + area3

        self.thresh1 = area1 / total_area
        self.thresh2 = area2 / total_area + self.thresh1

    def generate_pos(self):
        n = np.random.rand()
        if n < self.thresh1:
            return self.rect1.generate_pos()
        elif n < self.thresh2:
            return self.rect2.generate_pos()
        else:
            return self.rect3.generate_pos()


class Laser:
    def __init__(self, x, y, z, R):
        self.ellipse = EllipseSource(x, y, z, 2 * R, 2 * R)

    def generate_ray(self):
        vec = np.array([0, 0, 1])
        vec /= np.linalg.norm(vec)
        return self.ellipse.generate_pos(), vec
        