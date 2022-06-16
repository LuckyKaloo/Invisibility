import numpy as np

from other import Pinhole

class Camera:
    def __init__(self, z_camera, camera_radius, z_pupil, f_number, focal_length):
        self.pupil_radius = focal_length / (2 * f_number)

        # self.thin_lens = ThinLens(0, 0, z_camera + z0, lens_radius, focal_length, n)
        self.pupil_pinhole = Pinhole(np.array([0, 0, z_camera + z_pupil]), self.pupil_radius)
        self.camera_pinhole = Pinhole(np.array([0, 0, z_camera]), camera_radius)

    def projection(self, ray_pos, ray_vec):
        if not self.pupil_pinhole.blocks(ray_pos, ray_vec):
            return ray_pos[0:2]
        else:
            return None