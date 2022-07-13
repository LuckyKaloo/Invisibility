import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

from blocker import RectangleBlocker
from camera import Camera
from lens import LenticularLens
from raytracer import RayTracer
from source import RectangleSource



def sim(p_standard=None, R_standard=None):
    constant_p = constant_R = False
    file_root = ""

    if p_standard is not None:
        constant_p = True
        file_root = "p={p:.5f}".format(p=p_standard)

    if R_standard is not None:
        constant_R = True
        file_root = "R={R:.5f}".format(R=R_standard)

    if constant_p and constant_R:
        return

    if not os.path.exists(file_root):
        os.mkdir(file_root)
        print("made dir " + file_root)

    for ratio in np.arange(0.2, 0.6, 0.05):
        try:
            lens_config = {
                "class": LenticularLens,
                "zo": 0.3,
                "w": 0.09,
                "h": 0.09,
                "R": R_standard if constant_R else p_standard / ratio,
                "p": p_standard if constant_p else R_standard * ratio,
                "t": 0.003,
                "n": 1.6357
            }
            camera_config = {
                "class": Camera,
                "z_camera": 0.878,
                "camera_radius": 0.031,
                "z_pupil": 0.026,
                "pupil_radius": 0.004
            }
            source_config = {
                "class": RectangleSource,
                "x": 0,
                "y": 0.0,
                "z": 0,
                "w": 0.8,
                "h": 0.5
            }
            blocker_config = {
                "class": RectangleBlocker,
                "x": 0,
                "y": 0,
                "z": 0,
                "w": 0.05,
                "h": 0.05
            }

            config = {
                "batch_size": 1000,
                "rays_per_pos": int(2 * np.pi * 2 * 10 ** 5 * 0.007),
                "average_steradians": 0.007,
                "lens": lens_config,
                "camera": camera_config,
                "source": source_config,
                "blocker": blocker_config
            }
            NUM_TO_PASS = 5 * 10 ** 7

            IMG_RESOLUTION = 256

            raytracer = RayTracer(config=config)

            # rolling_arr = np.zeros(shape=(1, 5), dtype="float32")

            s = tf.constant(0.0)
            pdf = tf.zeros(shape=(1, IMG_RESOLUTION, IMG_RESOLUTION, 1))

            # with open("output/{ratio:.2f}.csv".format(ratio=np.round(ratio, 2)), mode='w') as f:
            with tqdm(total=NUM_TO_PASS) as pbar:
                while s.numpy() < NUM_TO_PASS:
                    # full_arr, weight, n_rays = raytracer.trace_for_rays()
                    pdf, n_rays = raytracer.trace_for_pdf(pdf)
                    s += n_rays
                    # if rolling_arr.shape[0] > 10**6:
                    #     np.savetxt(f, rolling_arr, delimiter=",", fmt="%.4f")
                    #     rolling_arr = data
                    # else:
                    #     rolling_arr = np.vstack([rolling_arr, data])
                    pbar.update(n_rays.numpy())
            #
            # plt.imshow(pdf / np.max(pdf))
            # plt.show()

            horizontally_averaged = tf.reduce_mean(pdf, axis=2)
            np.savetxt(file_root + "/{ratio:.2f}.csv".format(ratio=np.round(ratio, 2)), horizontally_averaged.numpy()[0, :, 0], fmt="%.4f")

            # plt.imshow(tf.cast(pdf / tf.reduce_max(pdf) * 256, tf.uint8).numpy()[0])
            # plt.show()

            img = tf.io.encode_png(tf.reshape(tf.cast(pdf / tf.reduce_max(pdf) * 256, tf.uint8),
                                              (IMG_RESOLUTION, IMG_RESOLUTION, 1)))
            tf.io.write_file(file_root + "/{ratio:.2f}.png".format(ratio=np.round(ratio, 2)), img)

        except Exception:
            continue


def main():
#     p_standard_list = (0.00015776, 0.0014111)
#     p_standard_list = (0.0002)
#     R_standard_list = (0.0001, 0.001232)

    sim(p_standard=0.0002)

#     for p_standard in p_standard_list:
#         sim(p_standard=p_standard)
#     for R_standard in R_standard_list:
#         sim(R_standard=R_standard)


if __name__ == "__main__":
    main()

# @tf.function
# def ray_trace(initial_pos, phi_max, phi_min, theta_max, theta_min, camera, lenticular_lens: LenticularLens):
#     """
#     very bad
#
#     :param initial_pos: (n, 3) tensor
#     :param phi_max: (n,) tensor
#     :param phi_min: (n,) tensor
#     :param theta_max: (n,) tensor
#     :param theta_min: (n,) tensor
#     :param camera: python object
#     :param lenticular_lens: python object
#     :return: (n, 4) tensor and scalar tensor
#     """
#
#     n = tf.shape(phi_max)[0]
#
#     phi_diff = phi_max - phi_min
#     theta_diff = theta_max - theta_min
#
#     phi_ave = (phi_max + phi_min) / 2
#     theta_ave = (theta_max + theta_min) / 2
#
#     initial_phi = tf.random.uniform(shape=(n, RAYS_PER_POSITION), minval=-0.5, maxval=0.5) * phi_diff + phi_ave
#
#     initial_theta = tf.random.uniform(shape=(n, RAYS_PER_POSITION), minval=-0.5, maxval=0.5) * theta_diff + theta_ave
#
#     initial_vec = tf.stack([
#         tf.cos(initial_phi) * tf.sin(initial_theta),
#         tf.sin(initial_phi),
#         tf.cos(initial_phi) * tf.cos(initial_theta)
#     ], axis=-1)
#
#     initial_vec = tf.reshape(initial_vec, (n * RAYS_PER_POSITION, 3))
#
#     print(initial_pos.shape)
#
#     initial_pos = tf.repeat(initial_pos, (RAYS_PER_POSITION), axis=0)
#
#     pos_out, vec_out, intersect = lenticular_lens.refract(initial_pos, initial_vec)
#
#     c_intersect, arr = camera.projection(pos_out, vec_out)
#     intersect = tf.logical_and(intersect, c_intersect)
#
#     full_arr = tf.concat([
#         initial_pos[:, :2],
#         arr[:, :2]
#     ], axis=-1)
#
#     return intersect, full_arr
#
#
# def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
#     x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
#     g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
#     g_norm2d = tf.pow(tf.reduce_sum(g), 2)
#     g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
#     g_kernel = tf.expand_dims(g_kernel, axis=-1)
#     return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)
#
#
# @tf.function
# def apply_blur(img):
#     blur = _gaussian_kernel(3, 2, 1, img.dtype)
#     img = tf.nn.conv2d(img, blur, [1,1,1,1], 'SAME')
#     return img
#
#
# def serialize(weight, intersect, full_arr):
#     feature_dict = {
#         "weight": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(weight).numpy()])),
#         "intersect": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(intersect).numpy()])),
#         "full_arr": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(full_arr).numpy()]))
#     }
#     example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
#     return example_proto.SerializeToString()
#
#
# def sliding_window(x, window_size, stride, axis=0):
#     n_in = tf.shape(x)[axis]
#     n_out = (n_in - window_size) // stride + 1
#     # Just in case n_in < window_size
#     n_out = tf.math.maximum(n_out, 0)
#     r = tf.expand_dims(tf.range(n_out), 1)
#     idx = r * stride + tf.range(window_size)
#     return tf.gather(x, idx, axis=axis)
#
#
# if __name__ == "__main__":
#     # ray tracing parameters
#     NUM_TO_PASS = 10 ** 8
#     WINDOW_SIZE = 1000
#
#     RAYS_PER_STERADIAN = 2 * 10 ** 5
#     RAYS_PER_POSITION = int(2 * np.pi * RAYS_PER_STERADIAN * 0.007)
#     IMG_RESOLUTION = 128
#
#     N_POS_TRIAL = 100
#
#     # lenticular lens, lenticules are aligned along the y-axis
#     z_lenticular = 0.3
#     w_lenticular = 0.09  # width of lens along x-axis
#     h_lenticular = 0.09  # height of lens along y-axis
#     # R_lenticular = 9.95195E-05
#     # p_lenticular = 0.025/161
#     # t_lenticular = 0.000245625
#     # n_lenticular = 1.6357
#
#     # lpi = 161
#
#     # camera
#     z_camera = 0.878
#     camera_radius = 0.031
#     z_pupil = 0.026
#     f_number = 2.8
#     focal_length = 0.0733
#
#     # source
#     x_source = z_source = 0
#     y_source = 0.1
#     w_source = 0.8
#     h_source = 0.5
#
#     # blocker
#     x_blocker = y_blocker = z_blocker = 0
#     w_blocker = 0.03
#     h_blocker = 0.1
#
#     # list of lenticular lenses stuff... R, p, t, n, lpi
#
#     """
#     LUKE you can edit this to include fewer arrays cos i think it'll take quite long still the important ones are the ones with the last number being 18, 50 and 161
#     """
#     # lenticular_list = [
#     #     [0.001231864, 0.025/18, 0.00303, 1.58654, 18],
#     #     # [0.001708775, 0.025/25, 0.00392, 1.58654, 25],
#     #     # [0.001204897, 0.025/32, 0.00316, 1.58654, 32],
#     #     # [0.000784645, 0.025/42, 0.00201, 1.58654, 42],
#     #     [0.000270681, 0.025/50, 0.000589495, 1.6357, 50],
#     #     # [0.000227893, 0.025/75, 0.000528137, 1.6357, 75],
#     #     # [0.000149347, 0.025/100, 0.000407783, 1.6357, 100],
#     #     [9.95195E-05, 0.025/161, 0.000245625, 1.6357, 161]
#     # ]
#
#     p_standard = 0.0005
#     lenticular_list = []
#
#     for ratio in np.arange(0.2, 0.21, 0.05):
#         lenticular_list.append([p_standard / ratio, p_standard, p_standard * 2, 1.6357, 0])
#
#     for (i, parameters) in enumerate(lenticular_list):
#         R_lenticular, p_lenticular, t_lenticular, n_lenticular, lpi = parameters
#
#         # initialising components
#         lenticular_lens = LenticularLens(z_lenticular, w_lenticular, h_lenticular, R_lenticular, p_lenticular,
#                                          t_lenticular, n_lenticular)
#         camera = Camera(z_camera, camera_radius, z_pupil, f_number, focal_length)
#         source = RectangleSource(x_source, y_source, z_source, w_source, h_source)
#         # blocker = RectangleBlocker(x_blocker, y_blocker, z_blocker, w_blocker, h_blocker)
#
#         total_rays = 0
#         total_passed = 0
#         n_positions = 0
#         # screen_positions = []
#         data = []
#
#         n_run = 0
#         # initial_pos = source.generate_pos(int(NUM_TO_PASS / 1000))
#         # phi_min, phi_max = lenticular_lens.angle_bounds_phi_camera(initial_pos, camera.pupil_pinhole.pos,
#         #                                                            tf.convert_to_tensor(camera.pupil_pinhole.r),
#         #                                                            camera.camera_pinhole.pos,
#         #                                                            tf.convert_to_tensor(camera.camera_pinhole.r))
#         #
#         # theta_min, theta_max = lenticular_lens.angle_bounds_theta(initial_pos)
#         # weight = (phi_max - phi_min) * (theta_max - theta_min) / 0.007
#         # stacked = tf.stack([*tf.unstack(initial_pos, axis=-1), phi_max, phi_min, theta_max, theta_min, weight], axis=-1)
#         #
#         # stacked = tf.gather(stacked, tf.where(weight >= 0), axis=0)
#         # batched = sliding_window(stacked, WINDOW_SIZE, WINDOW_SIZE)
#         print("test")
#         s = tf.constant(0.0)
#         start_time = time.time()
#         j = 0
#
#         pdf = tf.zeros(shape=(1, IMG_RESOLUTION, IMG_RESOLUTION, 1), dtype=tf.float32)
#
#         rolling_arr = np.zeros(shape=(1, 5), dtype="float32")
#
#         # with tf.io.TFRecordWriter(f"out_{i}.tfrecord") as writer:
#         with open("out.csv", mode='a') as f:
#             with (pbar := tqdm(total=NUM_TO_PASS)):
#                 while s.numpy() < NUM_TO_PASS:
#                     initial_pos = source.generate_pos(WINDOW_SIZE * 10)  # todo this will probably cause errors
#                     phi_min, phi_max = lenticular_lens.angle_bounds_phi_camera(initial_pos, camera.pupil_pinhole.pos,
#                                                                                tf.convert_to_tensor(
#                                                                                    camera.pupil_pinhole.r),
#                                                                                camera.camera_pinhole.pos,
#                                                                                tf.convert_to_tensor(
#                                                                                    camera.camera_pinhole.r))
#
#                     theta_min, theta_max = lenticular_lens.angle_bounds_theta(initial_pos)
#                     weight = (phi_max - phi_min) * (theta_max - theta_min) / 0.007
#                     batch = tf.stack(
#                         [*tf.unstack(initial_pos, axis=-1), phi_max, phi_min, theta_max, theta_min, weight], axis=-1)
#                     batch = tf.gather(batch, tf.where(weight >= 0), axis=0)[:WINDOW_SIZE, :, :]
#                     # todo try to put as much of this into a tf.function possible
#                     _x, _y, _z, *args, weight = tf.unstack(batch, axis=-1)
#                     intersect, full_arr = ray_trace(tf.concat((_x, _y, _z), axis=-1), *args, camera, lenticular_lens)
#                     weight = tf.repeat(weight, 8796, axis=0)
#                     intersect = tf.expand_dims(intersect, -1)
#                     rays = tf.reduce_sum(tf.cast(intersect, tf.float32) * weight)
#
#                     binned = tf.cast(tf.math.floordiv(full_arr[:, 2:4] + w_lenticular / 2,
#                                                       w_lenticular / IMG_RESOLUTION), tf.int32)
#                     # todo change this if lens is not square
#
#                     binned = binned[2] + binned[3] * IMG_RESOLUTION
#                     binned = tf.concat([binned, tf.range(IMG_RESOLUTION ** 2, dtype=tf.int32)], axis=0)
#                     values = tf.concat([
#                         weight * tf.cast(intersect, tf.float32),
#                         tf.zeros(shape=(IMG_RESOLUTION ** 2, 1), dtype=tf.float32)
#                     ], axis=0)
#                     # todo this might become very slow for larger images
#
#                     binned = tf.clip_by_value(binned, 0, IMG_RESOLUTION ** 2 - 1)
#
#                     idx = tf.argsort(binned)
#                     binned = tf.gather(binned, idx)
#                     values = tf.gather(values, idx)
#
#                     # magic aggregation
#                     segments = tf.unique(binned)[1]
#                     aggregated = tf.math.segment_sum(values, segments)
#
#                     pdf = pdf + apply_blur(tf.reshape(aggregated, (1, IMG_RESOLUTION, IMG_RESOLUTION, 1)))
#
#                     s += rays
#                     #
#                     # data = np.hstack([weight.numpy(), full_arr.numpy()])
#                     # data = data[np.squeeze(intersect.numpy())]
#                     #
#                     # if rolling_arr.shape[0] > 10**6:
#                     #     np.savetxt(f, rolling_arr, delimiter=",", fmt="%.4f")
#                     #     rolling_arr = data
#                     # else:
#                     #     rolling_arr = np.vstack([rolling_arr, data])
#                     # writer.write(serialize(weight, intersect, full_arr))
#                     pbar.update(rays.numpy())
#                     j += 1
#
#         print(time.time()-start_time)
#         print(tf.reduce_max(pdf))
#         print(j)
#         img = tf.io.encode_png(tf.reshape(tf.cast(pdf / tf.reduce_max(pdf) * 256, tf.uint8), (IMG_RESOLUTION, IMG_RESOLUTION, 1)))
#         tf.io.write_file("out.png", img)
