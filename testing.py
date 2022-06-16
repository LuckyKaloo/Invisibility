import numpy as np

from lens import LenticularLens
from other import Pinhole
from source import *
from camera import Camera

from blocker import TiltedRectangleBlocker


def lens_testing():
    z_lenticular = 0.3
    w_lenticular = 0.1  # width of lens along x-axis
    h_lenticular = 0.1  # height of lens along y-axis
    f_lenticular = 0.00006
    pitch = 0.00016
    thickness = 0.0001
    n_lenticular = 1.6

    lenticular_lens = LenticularLens(z_lenticular, w_lenticular, h_lenticular, f_lenticular, pitch, thickness, n_lenticular)

    initial_pos = np.array([0.18882672, -0.05527891, 0])
    initial_vec = np.array([-0.57168044, 0.03874726, 0.81956093])
    # initial_vec /= np.linalg.norm(initial_vec)

    full_arr = np.array(lenticular_lens.refract_more_info(initial_pos, initial_vec))
    print(full_arr)
    np.savetxt('testing.csv', full_arr, delimiter=',', fmt='%.10f', newline='\n')


def theta_bounds_testing():
    zo = 0.1
    w = 0.1  # width of lens along x-axis
    h = 0.1  # height of lens along y-axis
    f = 0.006
    p = 0.017
    t = 0.01
    n = 1.3

    lens = LenticularLens(zo, w, h, f, p, t, n)
    print(lens.angle_bounds_theta(np.array([0.05, 0, 0])))


def phi_bounds_testing():
    zo = 0.1
    w = 0.1  # width of lens along x-axis
    h = 0.1  # height of lens along y-axis
    f = 0.006
    p = 0.017
    t = 0.01
    n = 1.3

    lens = LenticularLens(zo, w, h, f, p, t, n)

    pinhole = Pinhole(np.array([0, 0.1, 0.1]), 0.005)

    print(lens.angle_bounds_phi(np.array([0, 0.3, 0]), pinhole))


def pinhole_testing():
    pinhole = Pinhole(np.array([0, 0, 0.1]), 0.005)
    print(pinhole.angle_bounds_phi(np.array([0, 0, 0])))
    print(pinhole.angle_bounds_phi(np.array([1, 0, 0])))
    print(pinhole.angle_bounds_phi(np.array([0, 0.05, 0])))


def shape_testing():
    zo = 0.1
    w = 0.1  # width of lens along x-axis
    h = 0.1  # height of lens along y-axis
    f = 0.006
    p = 0.017
    t = 0.01
    n = 1.3

    lens = LenticularLens(zo, w, h, f, p, t, n)

    shape = RectangleSource(0, 0, 0, 0.1, 0.1)
    pinhole = Pinhole(np.array([0, 0, 0.1]), 0.005)

    for i in range(100):
        initial_pos = shape.generate_pos()
        phi_min, phi_max = pinhole.angle_bounds_phi(initial_pos)
        theta_min, theta_max = lens.angle_bounds_theta(initial_pos)

        initial_phi = (np.random.rand() - 0.5) * (phi_max - phi_min) + (phi_max + phi_min) / 2  # vertical angle, decides y
        initial_theta = (np.random.rand() - 0.5) * (theta_max - theta_min) + (theta_max + theta_min) / 2  # horizontal angle, decides x
        initial_vec = np.array([
            np.cos(initial_phi) * np.sin(initial_theta),
            np.sin(initial_phi),
            np.cos(initial_phi) * np.cos(initial_theta)
        ])

        print(initial_pos, np.array([phi_min, phi_max]), np.array([theta_min, theta_max]), initial_phi, initial_theta, initial_vec)


def refract_testing():
    z_lenticular = 0.5
    w_lenticular = 1  # width of lens along x-axis
    h_lenticular = 1  # height of lens along y-axis
    f_lenticular = 0.00006
    pitch = 0.00016
    thickness = 0.0001
    n_lenticular = 1.6

    lenticular_lens = LenticularLens(z_lenticular, w_lenticular, h_lenticular, f_lenticular, pitch, thickness, n_lenticular)

    data = []

    for i in range(100000):
        pos = np.array([(np.random.rand() - 0.5)/100, (np.random.rand() - 0.5)/100, 0])
        vec = np.array([(np.random.rand() - 0.5)/3, (np.random.rand() - 0.5)/3, np.random.rand()/5 + 0.8])
        vec /= np.linalg.norm(vec)
        try:
            arr = np.array([lenticular_lens.refract_more_info(pos, vec)])[0]
            vec = arr[1]
            vec_out = arr[5]
            if vec_out[2] != 0 and vec[2] != 0:
                data.append([vec_out[1] / vec_out[2], vec[1] / vec[2]])

            # print(vec[1] / vec[2], refract_vec[1] / refract_vec[2], (refract_vec[1] / refract_vec[2]) / (vec[1] / vec[2]))
            # print(vec[0] / refract_vec[0], vec[1] / refract_vec[1], vec[2] / refract_vec[2])
            # data.append((refract_vec[1] / refract_vec[2]) / (vec[1] / vec[2]))

            # print(vec, refract_vec)
            # print((vec[1] / vec[2]) / (refract_vec[1] / refract_vec[2]))
            # data.append(arr)
        except Exception:
            pass

    np.savetxt('ray_refract_data.csv', data, delimiter=',')


def thick_lens_refract_testing():
    camera = Camera(0.2, 0.02, 0.02, 0.07, 0.031, 0.07, 1.52)
    data = []
    vec = np.array([0.34, 0, 1])
    vec /= np.linalg.norm(vec)

    num = 6

    for i in range(num):
        for j in range(num):
            if i == num/2 and j == num/2:
                continue
            pos = np.array([0.01 * (i-num/2), 0.01 * (j-num/2), 0])

            s = camera.thick_lens_refract(pos, vec)
            data.append(np.array(np.concatenate([pos, vec, s[0], s[1]])))
    np.savetxt('ray_refract_data.csv', data, delimiter=',', newline='\n')


def file_size_testing():
    # lpi_list = [18, 50, 161]
    # for lpi in lpi_list:
    #     s = sum(1 for line in open('50cm to lens, 0 degrees/' + str(lpi) + ' lpi.csv'))
    #     print(s)

    s = sum(1 for line in open('50cm to lens, 0 degrees/18 lpi.csv'))
    print(s)


def resize_file():
    target_s = 9998148

    s = sum(1 for line in open('50cm to lens, 0 degrees/18 lpi.csv'))
    print(s)

    with open('50cm to lens, 0 degrees/18 lpi.csv', 'r') as f:
        with open('50cm to lens, 0 degrees/18 lpi.csv', 'a') as f_write:
            f_write.truncate(0)

            for i in range(s - target_s):
                f.readline()

            f_write.write(f.read())


def outside_range_testing():
    total_count = 0
    outside_count = 0

    with open('Actual Data/Unblocked Files/Changing LPI/18 lpi, 0.3 cm.csv') as f:
        for line in f:
            lst = line.split(',')
            x = float(lst[2])
            y = float(lst[3])

            if not (-0.035 <= x <= 0.035 and -0.035 <= y <= 0.035):
                outside_count += 1

            total_count += 1

    print(total_count, outside_count)


def tilted_testing():
    blocker = TiltedRectangleBlocker(0, 0, 0, 0.05, 0.03, -0.1)
    data = []
    for row in np.arange(-0.1, 0.1, 0.001):
        for col in np.arange(-0.1, 0.1, 0.001):
            if blocker.blocks([col, row]):
                data.append([col, row])

    with open('tilted_testing.csv', 'w') as f:
        np.savetxt(f, data, fmt='%.3f', delimiter=',')



if __name__ == "__main__":
    # thick_lens_refract_testing()
    # refract_testing()
    # lens_testing()
    # file_size_testing()
    # resize_file()
    # outside_range_testing()
    tilted_testing()