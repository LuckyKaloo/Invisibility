import tensorflow as tf
from camera import *
from lens import *
from source import *
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
import os


@tf.function
def ray_trace(initial_pos, phi_max, phi_min, theta_max, theta_min, camera, lenticular_lens: LenticularLens):
    """
    very bad

    :param initial_pos: (n, 3) tensor
    :param phi_max: (n,) tensor
    :param phi_min: (n,) tensor
    :param theta_max: (n,) tensor
    :param theta_min: (n,) tensor
    :param camera: python object
    :param lenticular_lens: python object
    :return: (n, 4) tensor and scalar tensor
    """

    n = tf.shape(phi_max)[0]

    phi_diff = phi_max - phi_min
    theta_diff = theta_max - theta_min

    phi_ave = (phi_max + phi_min) / 2
    theta_ave = (theta_max + theta_min) / 2

    initial_phi = tf.random.uniform(shape=(n, RAYS_PER_POSITION), minval=-0.5, maxval=0.5) * phi_diff + phi_ave

    initial_theta = tf.random.uniform(shape=(n, RAYS_PER_POSITION), minval=-0.5, maxval=0.5) * theta_diff + theta_ave

    initial_vec = tf.stack([
        tf.cos(initial_phi) * tf.sin(initial_theta),
        tf.sin(initial_phi),
        tf.cos(initial_phi) * tf.cos(initial_theta)
    ], axis=-1)

    initial_vec = tf.reshape(initial_vec, (n * RAYS_PER_POSITION, 3))

    print(initial_pos.shape)

    initial_positions = tf.repeat(initial_pos, (RAYS_PER_POSITION), axis=0)

    pos_out, vec_out, weights = lenticular_lens.refract(initial_positions, initial_vec)

    c_intersect, arr = camera.projection(pos_out, vec_out)
    weights = weights * tf.cast(c_intersect, tf.float32)

    full_arr = tf.concat([
        initial_positions[:, :2],
        arr[:, :2]
    ], axis=-1)

    return weights, full_arr


def serialize(weight, intersect, full_arr):
    feature_dict = {
        "weight": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(weight).numpy()])),
        "intersect": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(intersect).numpy()])),
        "full_arr": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(full_arr).numpy()]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example_proto.SerializeToString()


def sliding_window(x, window_size, stride, axis=0):
    n_in = tf.shape(x)[axis]
    n_out = (n_in - window_size) // stride + 1
    # Just in case n_in < window_size
    n_out = tf.math.maximum(n_out, 0)
    r = tf.expand_dims(tf.range(n_out), 1)
    idx = r * stride + tf.range(window_size)
    return tf.gather(x, idx, axis=axis)


if __name__ == "__main__":
    # ray tracing parameters
    NUM_TO_PASS = 10 ** 7
    WINDOW_SIZE = 400

    RAYS_PER_STERADIAN = 2 * 10 ** 5
    RAYS_PER_POSITION = int(2 * np.pi * RAYS_PER_STERADIAN * 0.007)

    N_POS_TRIAL = 100

    # lenticular lens, lenticules are aligned along the y-axis
    z_lenticular = 0.3
    w_lenticular = 0.09  # width of lens along x-axis
    h_lenticular = 0.09  # height of lens along y-axis
    # R_lenticular = 9.95195E-05
    # p_lenticular = 0.025/161
    # t_lenticular = 0.000245625
    # n_lenticular = 1.6357

    # lpi = 161

    # camera
    z_camera = 0.878
    camera_radius = 0.031
    z_pupil = 0.026
    pupil_radius = 0.002
    f_number = 2.8
    focal_length = 0.0733

    # source
    x_source = z_source = 0
    y_source = 0.1
    w_source = 0.8
    h_source = 0.5

    # blocker
    x_blocker = y_blocker = z_blocker = 0
    w_blocker = 0.03
    h_blocker = 0.1

    # list of lenticular lenses stuff... R, p, t, n, lpi

    """
    LUKE you can edit this to include fewer arrays cos i think it'll take quite long still the important ones are the ones with the last number being 18, 50 and 161
    """
    # lenticular_list = [
    #     [0.001231864, 0.025/18, 0.00303, 1.58654, 18],
    #     # [0.001708775, 0.025/25, 0.00392, 1.58654, 25],
    #     # [0.001204897, 0.025/32, 0.00316, 1.58654, 32],
    #     # [0.000784645, 0.025/42, 0.00201, 1.58654, 42],
    #     [0.000270681, 0.025/50, 0.000589495, 1.6357, 50],
    #     # [0.000227893, 0.025/75, 0.000528137, 1.6357, 75],
    #     # [0.000149347, 0.025/100, 0.000407783, 1.6357, 100],
    #     [9.95195E-05, 0.025/161, 0.000245625, 1.6357, 161]
    # ]

    p_standard = 0.0005
    lenticular_list = []

    for ratio in np.arange(0.2, 0.21, 0.2):
        lenticular_list.append([p_standard / ratio, p_standard, p_standard * 2, 1.6357, 0])

    for (i, parameters) in enumerate(lenticular_list):
        R_lenticular, p_lenticular, t_lenticular, n_lenticular, lpi = parameters

        # initialising components
        lenticular_lens = LenticularLens(z_lenticular, w_lenticular, h_lenticular, R_lenticular, p_lenticular,
                                         t_lenticular, n_lenticular)
        camera = Camera(z_camera, camera_radius, z_pupil, pupil_radius)
        source = RectangleSource(x_source, y_source, z_source, w_source, h_source)
        # blocker = RectangleBlocker(x_blocker, y_blocker, z_blocker, w_blocker, h_blocker)

        total_rays = 0
        total_passed = 0
        n_positions = 0
        # screen_positions = []
        data = []

        n_run = 0
        initial_pos = source.generate_pos(NUM_TO_PASS)
        phi_min, phi_max = lenticular_lens.angle_bounds_phi_camera(initial_pos, camera.pupil_pinhole.pos,
                                                                   tf.convert_to_tensor(camera.pupil_pinhole.r),
                                                                   camera.camera_pinhole.pos,
                                                                   tf.convert_to_tensor(camera.camera_pinhole.r))

        theta_min, theta_max = lenticular_lens.angle_bounds_theta(initial_pos)
        weight = (phi_max - phi_min) * (theta_max - theta_min) / 0.007
        stacked = tf.stack([*tf.unstack(initial_pos, axis=-1), phi_max, phi_min, theta_max, theta_min, weight], axis=-1)

        stacked = tf.gather(stacked, tf.where(weight >= 0), axis=0)
        batched = sliding_window(stacked, WINDOW_SIZE, WINDOW_SIZE)
        print("test")
        s = tf.constant(0.0)
        start_time = time.time()
        j = 0

        rolling_arr = np.zeros(shape=(1, 5), dtype="float32")

        # with tf.io.TFRecordWriter(f"out_{i}.tfrecord") as writer:
        with open("out.csv", mode='a') as f:
            f.truncate(0)

            with (pbar := tqdm(total=NUM_TO_PASS)):
                while s.numpy() < NUM_TO_PASS:
                    batch = batched[j]
                    _x, _y, _z, *args, weight = tf.unstack(batch, axis=-1)
                    weights, full_arr = ray_trace(tf.concat((_x, _y, _z), axis=-1), *args, camera, lenticular_lens)
                    weight = tf.repeat(weight, 8796, axis=0)
                    weights = tf.expand_dims(weights, -1)
                    rays = tf.reduce_sum(tf.sign(weights) * weight)
                    s += rays
                    # j += 1
                    # np.save(f"weight_{i}_{j}.npy", weight.numpy())
                    # np.save(f"intersect{i}_{j}.npy", intersect.numpy())
                    # np.save(f"full_arr{i}_{j},npy", full_arr.numpy())
                    # print(intersect.shape)
                    # print(tf.reduce_max(full_arr * tf.cast(tf.expand_dims(intersect, -1), tf.float32)))
                    weight = weight * weights
                    data = np.hstack([weight.numpy(), full_arr.numpy()])
                    data = data[np.squeeze(tf.cast(tf.sign(weight), tf.bool).numpy())]

                    if rolling_arr.shape[0] > 10**6:
                        np.savetxt(f, rolling_arr, delimiter=",", fmt="%.4f")
                        rolling_arr = data
                    else:
                        rolling_arr = np.vstack([rolling_arr, data])
                    # writer.write(serialize(weight, intersect, full_arr))
                    pbar.update(rays.numpy())

                j += 1