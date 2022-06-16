from lens import LenticularLens
from camera import Camera
from source import *
from blocker import *

import time

def main():
    # ray tracing parameters
    NUM_TO_PASS_TOTAL = 10 ** 7
    NUM_UPDATE = 1000

    RAYS_PER_STERADIAN = 10 ** 5
    RAYS_PER_POSITION = int(2 * np.pi * RAYS_PER_STERADIAN)

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

        with open('50cm to lens, 0 degrees/{lpi} lpi info.csv'.format(lpi=lpi), 'r') as f_info:
            full_total_rays = int(f_info.readline())

        total_rays = 0
        total_passed = 0
        current_rays = 0
        current_passed = 0
        n_positions = 0
        # screen_positions = []
        data = []

        filename = '50cm to lens, 0 degrees/{lpi} lpi.csv'.format(lpi=lpi)
        NUM_TO_PASS = NUM_TO_PASS_TOTAL - sum(1 for line in open(filename))
        N_CRIT = NUM_TO_PASS / NUM_UPDATE
        print(NUM_TO_PASS)

        with open(filename, 'a') as f_rays:
        # with open('passed_rays.csv', 'a') as f_rays)

            # f_rays.truncate(0)
            start_time = curr_start_time = time.time()

            while total_passed < NUM_TO_PASS:
                # randomising the position of the rays generated
                initial_pos = source.generate_pos()
                # if blocker.blocks(initial_pos):
                #     continue

                # phi_min, phi_max = lenticular_lens.angle_bounds_phi_lens(initial_pos)
                phi_min, phi_max = lenticular_lens.angle_bounds_phi_camera(initial_pos, camera)
                theta_min, theta_max = lenticular_lens.angle_bounds_theta(initial_pos)

                phi_diff = phi_max - phi_min
                theta_diff = theta_max - theta_min
                n_rays = int(RAYS_PER_STERADIAN * phi_diff * theta_diff)
                if n_rays <= 0:
                    continue

                phi_ave = (phi_max + phi_min) / 2
                theta_ave = (theta_max + theta_min) / 2

                n_positions += 1

                # simulating all rays originating from the randomised point
                for i in range(n_rays):
                    # randomising the initial angle of the ray
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
                            # screen_positions.append(arr)
                            data.append(full_arr)
                            current_passed += 1
                    except:
                        continue

                current_rays += n_rays
                full_total_rays += RAYS_PER_POSITION

                # exporting all the current data to a file and resetting everything so that the list remains small
                if current_passed >= N_CRIT:
                    total_passed += current_passed
                    np.savetxt(f_rays, data, delimiter=',', fmt='%.4f', newline='\n')

                    with open('50cm to lens, 0 degrees/{lpi} lpi info.csv'.format(lpi=lpi), 'w') as f_info:
                        f_info.write(str(full_total_rays))

                    total_rays += current_rays

                    print(
                        ('lpi = {lpi}, ' +
                         'total rays = {total_rays}, time = {time_taken:.1f} s, total time = {total_time:.1f} min, eta = {eta:.1f} min, ' +
                         '% finished = {percent:.1f}, ' +
                         '% passed = {percent_passed:.1f}, ' +
                         'avg angles = {average_angles}, expected pos = {expected_pos}')

                            .format(lpi=lpi,
                                    total_rays=total_rays, time_taken=time.time() - curr_start_time,
                                    total_time=(time.time() - start_time) / 60,
                                    eta=(time.time() - start_time) / 60 * (NUM_TO_PASS / total_passed - 1),
                                    percent=total_passed/NUM_TO_PASS*100,
                                    percent_passed=total_passed / total_rays * 100,
                                    average_angles=int(total_rays/n_positions), total_positions=n_positions, expected_pos=int(n_positions * NUM_TO_PASS / total_passed))
                    )

                    current_passed = 0
                    current_rays = 0
                    # screen_positions = 0
                    data = []
                    curr_start_time = time.time()


if __name__ == '__main__':
    main()