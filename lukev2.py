import numpy as np

from lens import LenticularLens
from camera import Camera
from source import *
from joblib import Parallel, delayed
import time

Cores = 4  # number of cores
lpi = 18

def f(n_pos, k):
    # data to be stored internally until return
    data_f = []  # passed rays
    total_f = 0  # total rays simulated


    for i in range(n_pos):
        # randomising the position of the rays generated
        initial_pos = source.generate_pos()

        # phi_min, phi_max = lenticular_lens.angle_bounds_phi_lens(initial_pos)
        phi_min, phi_max = lenticular_lens.angle_bounds_phi_camera(initial_pos, camera)
        theta_min, theta_max = lenticular_lens.angle_bounds_theta(initial_pos)

        phi_diff = phi_max - phi_min
        theta_diff = theta_max - theta_min
        n_rays = int(RAYS_PER_STERADIAN * phi_diff * theta_diff)
        if n_rays <= 0:
            continue

        total_f += n_rays

        for j in range(n_rays):
            phi_ave = (phi_max + phi_min) / 2
            theta_ave = (theta_max + theta_min) / 2

            initial_phi = (np.random.rand() - 0.5) * phi_diff + phi_ave  # vertical angle, decides y
            initial_theta = (np.random.rand() - 0.5) * theta_diff + theta_ave  # horizontal angle, decides x
            initial_vec = np.array([
                np.cos(initial_phi) * np.sin(initial_theta),
                np.sin(initial_phi),
                np.cos(initial_phi) * np.cos(initial_theta)
            ])

            # ray tracing
            try:
                arr = lenticular_lens.refract(initial_pos, initial_vec)
                if arr is None:
                    continue
                len_lens_pos, len_lens_vec = arr

                arr = camera.projection(len_lens_pos, len_lens_vec)
                if arr is not None:
                    full_arr = np.array([initial_pos[0], initial_pos[1], arr[0], arr[1]])
                    data_f.append(full_arr)
            except:
                pass

    return data_f, total_f

# ray tracing parameters
NUM_TO_PASS = 10 ** 5
NUM_UPDATE = 100

RAYS_PER_STERADIAN = 2 * 10 ** 5
RAYS_PER_POSITION = int(2 * np.pi * RAYS_PER_STERADIAN)

N_POS_TRIAL = 100

# lenticular lens, lenticules are aligned along the y-axis
z_lenticular = 0.5
w_lenticular = 0.09  # width of lens along x-axis
h_lenticular = 0.09  # height of lens along y-axis
R_lenticular = 9.95 * 10 ** -5
p_lenticular = 0.025 / 161
t_lenticular = 2.46 * 10 ** -4
n_lenticular = 1.6357

# camera
z_camera = 0.878
camera_radius = 0.031
z_pupil = 0.026
f_number = 2.8
focal_length = 0.0733

# source
x_source = z_source = 0
y_source = 0.1
w_source = 0.7
h_source = 0.5

# blocker
x_blocker = y_blocker = z_blocker = 0
w_blocker = 0.03
h_blocker = 0.1

# list of lenticular lenses stuff... R, p, t, n, lpi


"""
LUKE you can edit this to include fewer arrays cos i think it'll take quite long still the important ones are the ones with the last number being 18, 50 and 161
"""
lenticular_list = [
    [0.001231864, 0.025/18, 0.00303, 1.58654, 18],
    # [0.001708775, 0.025/25, 0.00392, 1.58654, 25],
    # [0.001204897, 0.025/32, 0.00316, 1.58654, 32],
    # [0.000784645, 0.025/42, 0.00201, 1.58654, 42],
    [0.000270681, 0.025/50, 0.000589495, 1.6357, 50],
    # [0.000227893, 0.025/75, 0.000528137, 1.6357, 75],
    # [0.000149347, 0.025/100, 0.000407783, 1.6357, 100],
    [9.95195E-05, 0.025/161, 0.000245625, 1.6357, 161]
]

for lenticular_parameters in lenticular_list:
    R_lenticular, p_lenticular, t_lenticular, n_lenticular, lpi = lenticular_parameters
    # initialising components
    lenticular_lens = LenticularLens(z_lenticular, w_lenticular, h_lenticular, R_lenticular, p_lenticular, t_lenticular, n_lenticular)
    camera = Camera(z_camera, camera_radius, z_pupil, f_number, focal_length)
    source = RectangleSource(x_source, y_source, z_source, w_source, h_source)
    # blocker = RectangleBlocker(x_blocker, y_blocker, z_blocker, w_blocker, h_blocker)

    total_rays = 0
    total_passed = 0
    n_positions = 0
    # screen_positions = []
    data = []

    n_run = 0

    with open('50cm to lens, 0 degrees/{lpi} lpi test.csv'.format(lpi=lpi), 'a') as f_rays:
        f_rays.truncate(0)

        start_time = curr_start_time = time.time()

        for n_run in range(0, NUM_UPDATE):
            if total_rays == 0:  # trial run
                print('trial')
                data2 = Parallel(n_jobs=Cores, verbose = 3)(delayed(f)(N_POS_TRIAL, k) for k in range(Cores))
                n_positions += Cores * N_POS_TRIAL
            else:
                n_pos_to_sim = (NUM_TO_PASS - total_passed) * n_positions / total_passed / (NUM_UPDATE - n_run)
                data2 = Parallel(n_jobs=Cores, verbose = 0)(delayed(f)(int(n_pos_to_sim / Cores), k) for k in range(Cores))
                n_positions += int(n_pos_to_sim / Cores) * Cores

            for dataset in data2:
                total_passed += len(dataset[0])
                total_rays += dataset[1]

                for i in dataset[0]:
                    data.append(i)

            # exporting all the current data to a file and resetting everything so that the list remains small
            np.savetxt(f_rays, data, delimiter=',', fmt='%.4f', newline='\n')

            print(
                ('lpi = {lpi}, ' +
                 'total rays = {total_rays}, time = {time_taken:.1f} s, total time = {total_time:.1f} min, eta = {eta:.1f} min, ' +
                 '% finished = {percent:.1f}, ' +
                 '% passed = {percent_passed:.1f}, ' +
                 'avg angles = {average_angles}, total pos = {total_positions}, expected pos = {expected_pos}')

                    .format(lpi=lpi,
                            total_rays=total_rays, time_taken=time.time() - curr_start_time,
                            total_time=(time.time() - start_time) / 60,
                            eta=(time.time() - start_time) / 60 * (NUM_TO_PASS / total_passed - 1),
                            percent=total_passed/NUM_TO_PASS*100,
                            percent_passed=total_passed / total_rays * 100,
                            average_angles=int(total_rays/n_positions), total_positions=n_positions, expected_pos=int(n_positions * NUM_TO_PASS / total_passed))
            )

            data = []
            curr_start_time = time.time()

    with open('50cm to lens, 0 degrees/{lpi} lpi info test.csv'.format(lpi=lpi), 'w') as f_info:
        f_info.write(str(int(n_positions * RAYS_PER_POSITION)) + ',' + str(total_passed))
