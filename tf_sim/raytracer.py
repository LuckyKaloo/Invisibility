import tensorflow as tf
from tqdm import tqdm
import numpy as np

from tf_sim.blocker import RectangleBlocker
from tf_sim.camera import Camera
from tf_sim.lens import LenticularLens
from tf_sim.source import RectangleSource


def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


@tf.function
def apply_blur(img):
    blur = _gaussian_kernel(9, 5, 1, img.dtype)
    img = tf.nn.conv2d(img, blur, [1, 1, 1, 1], 'SAME')
    return img


class RayTracer:
    def __init__(self, config: dict, lens=None, camera=None, source=None, blocker=None):
        """
        Initializes raytracer with custom objects

        :param lens:
        :param camera:
        :param source:
        """

        if lens is None:
            lens_config = config["lens"]
            if "class" in lens_config.keys():
                lens_class = lens_config.pop("class")
                lens = lens_class(**lens_config)
            else:
                lens = LenticularLens(**lens_config)

        if camera is None:
            camera_config = config["camera"]
            if "class" in camera_config.keys():
                camera_class = camera_config.pop("class")
                camera = camera_class(**camera_config)
            else:
                camera = Camera(**camera_config)

        if source is None:
            source_config = config["source"]
            if "class" in source_config.keys():
                source_class = source_config.pop("class")
                source = source_class(**source_config)
            else:
                source = RectangleSource(**source_config)

        if blocker is None:
            if "blocker" in config.keys():
                blocker_config = config["blocker"]
                if "class" in blocker_config.keys():
                    blocker_class = blocker_config.pop("class")
                    blocker = blocker_class(**blocker_config)
                else:
                    blocker = RectangleBlocker(**blocker_config)

        self.config = config
        self.lens = lens
        self.camera = camera
        self.source = source
        self.blocker = blocker

    @tf.function
    def trace_for_rays(self):
        """
        Traces rays

        :return: (n, 4 tensor): initial and final position of rays, (n,) tensor: weight of rays
        """
        initial_pos = self.source.generate_pos(self.config["batch_size"] * 10)
        phi_min, phi_max = self.lens.angle_bounds_phi_camera(
            initial_pos, self.camera.pupil_pinhole.pos, tf.convert_to_tensor(self.camera.pupil_pinhole.r),
            self.camera.camera_pinhole.pos, tf.convert_to_tensor(self.camera.camera_pinhole.r)
        )
        theta_min, theta_max = self.lens.angle_bounds_theta(initial_pos)
        weight = (phi_max - phi_min) * (theta_max - theta_min) / self.config["average_steradians"]

        idx = tf.where(weight > 0)

        initial_pos = tf.gather(initial_pos, tf.squeeze(idx), axis=0)[:self.config["batch_size"]]
        phi_min = tf.gather(phi_min, idx)[:self.config["batch_size"]]
        phi_max = tf.gather(phi_max, idx)[:self.config["batch_size"]]
        theta_min = tf.gather(theta_min, idx)[:self.config["batch_size"]]
        theta_max = tf.gather(theta_max, idx)[:self.config["batch_size"]]
        weight = tf.gather(weight, idx)[:self.config["batch_size"]]

        # batch = tf.stack(
        #     [*tf.unstack(initial_pos, axis=-1), phi_max, phi_min, theta_max, theta_min, weight], axis=-1)
        # batch = tf.gather(batch, tf.where(weight >= 0), axis=0)[:self.config["batch_size"], :, :]
        # _x, _y, _z, phi_max, phi_min, theta_max, theta_min, weight = tf.unstack(batch, axis=-1)
        # initial_pos = tf.concat((_x, _y, _z), axis=-1)

        n = tf.shape(weight)[0]

        phi_diff = phi_max - phi_min
        theta_diff = theta_max - theta_min

        phi_ave = (phi_max + phi_min) / 2
        theta_ave = (theta_max + theta_min) / 2

        initial_phi = tf.random.uniform(shape=(n, self.config["rays_per_pos"]), minval=-0.5,
                                        maxval=0.5) * phi_diff + phi_ave

        initial_theta = tf.random.uniform(shape=(n, self.config["rays_per_pos"]), minval=-0.5,
                                          maxval=0.5) * theta_diff + theta_ave

        initial_vec = tf.stack([
            tf.cos(initial_phi) * tf.sin(initial_theta),
            tf.sin(initial_phi),
            tf.cos(initial_phi) * tf.cos(initial_theta)
        ], axis=-1)

        initial_vec = tf.reshape(initial_vec, (n * self.config["rays_per_pos"], 3))

        # so the thing doesn't die
        initial_vec = initial_vec - 1e-8 * (tf.sign(initial_vec) + 1) * (tf.sign(initial_vec) - 1)

        initial_pos = tf.repeat(initial_pos, (self.config["rays_per_pos"]), axis=0)

        pos_out, vec_out, intersect = self.lens.refract(initial_pos, initial_vec)

        c_intersect, arr = self.camera.projection(pos_out, vec_out)
        intersect = intersect * tf.cast(c_intersect, tf.float32)

        if self.blocker is not None:
            intersect = intersect * tf.cast(not self.blocker.blocks(initial_pos), tf.float32)

        full_arr = tf.concat([
            initial_pos[:, :2],
            arr[:, :2]
        ], axis=-1)

        weight = tf.repeat(weight, 8796, axis=0)

        intersect = tf.expand_dims(intersect, -1)

        n_rays = tf.reduce_sum(weight * intersect)

        return full_arr, intersect, weight, n_rays

    @tf.function
    def trace_for_pdf(self, pdf):
        _, h, w, _ = pdf.shape
        full_arr, intersect, weight, n_rays = self.trace_for_rays()
        binned = tf.cast(tf.math.floordiv(full_arr[:, 2:4] + self.lens.w / 2,
                                          self.lens.w / h), tf.int32)
        # todo change this if lens is not square

        binned = binned[2] + binned[3] * h
        binned = tf.concat([binned, tf.range(h ** 2, dtype=tf.int32)], axis=0)
        values = tf.concat([
            weight * intersect,
            tf.zeros(shape=(h ** 2, 1), dtype=tf.float32)
        ], axis=0)
        # todo this might become very slow for larger images

        binned = tf.clip_by_value(binned, 0, h ** 2 - 1)

        idx = tf.argsort(binned)
        binned = tf.gather(binned, idx)
        values = tf.gather(values, idx)

        # magic aggregation
        segments = tf.unique(binned)[1]
        aggregated = tf.math.segment_sum(values, segments)

        pdf = pdf + apply_blur(tf.reshape(aggregated, (1, h, w, 1)))
        return pdf, n_rays
