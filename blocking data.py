import os

import numpy as np
from blocker import RectangleBlocker, TiltedRectangleBlocker

from joblib import Parallel, delayed

def f(f_read_name, f_write_name, blocker, overwrite=False):
    if not overwrite:
        if os.path.exists(f_write_name):
            return


    with open(f_read_name, 'r') as f_read:
        with open(f_write_name, 'a') as f_write:
            f_write.truncate(0)

            for line in f_read:
                if line == '\n':
                    continue

                split = line.split(',')

                arr = [float(split[0]), float(split[1])]
                if not blocker.blocks(arr):
                    f_write.write(split[2] + ',' + split[3])


def changing_width(Cores, lpi_list, w_list):
    f_read_name = 'Actual Data/Unblocked Files/Changing LPI/{lpi} lpi, 0.3 cm.csv'
    f_write_name = 'Actual Data/Changing Width/{lpi} LPI - 3cm/3x{i}.csv'

    Parallel(n_jobs=Cores, verbose=5)(delayed(f)(f_read_name.format(lpi=lpi), f_write_name.format(i=np.round(w, 2), lpi=lpi), RectangleBlocker(0, 0, 0, w / 100, 0.03)) for w in w_list for lpi in lpi_list)


def changing_height(Cores, lpi_list, h_list):
    f_read_name = 'Actual Data/Unblocked Files/Changing LPI/{lpi} lpi, 0.3 cm.csv'
    f_write_name = 'Actual Data/Changing Height/{lpi} LPI - 3cm/3x{i}.csv'

    Parallel(n_jobs=Cores, verbose=5)(delayed(f)(f_read_name.format(lpi=lpi), f_write_name.format(i=np.round(h, 2), lpi=lpi), RectangleBlocker(0, 0, 0, 0.03, h / 100)) for h in h_list for lpi in lpi_list)


def changing_height_width(Cores, h_list, w_list):
    f_read_name = 'Actual Data/Unblocked Files/Changing LPI/161 lpi, 0.3 cm.csv'
    f_write_name = 'Actual Data/Changing Height/161 LPI - {w}cm/{w}x{h}.csv'

    Parallel(n_jobs=Cores, verbose=5)(delayed(f)(f_read_name, f_write_name.format(w=np.round(w, 2), h=np.round(h, 2)), RectangleBlocker(0, 0, 0, w / 100, h / 100)) for w in w_list for h in h_list)


def changing_angle(Cores, lpi_list, angle_list):
    f_read_name = 'Actual Data/Unblocked Files/Changing LPI/{lpi} lpi, 0.3 cm.csv'
    f_write_name = 'Actual Data/Changing Angle/{lpi} LPI - 5x5/{angle:.1f} rad.csv'

    Parallel(n_jobs=Cores, verbose=5)(delayed(f)(f_read_name.format(lpi=lpi), f_write_name.format(lpi=lpi, angle=angle), TiltedRectangleBlocker(0, 0, 0, 0.05, 0.05, angle)) for lpi in lpi_list for angle in angle_list)


def changing_distance(Cores, d_list):
    f_read_name = 'Actual Data/Unblocked Files/Changing Distance/161 lpi, {d:.2f} cm.csv'
    f_write_name = 'Actual Data/Changing Distance/{d:.2f}.csv'

    Parallel(n_jobs=Cores, verbose=5)(delayed(f)(f_read_name.format(d=np.round(d, 2)), f_write_name.format(d=np.round(d, 2)), RectangleBlocker(0, 0, 0, 0.05, 0.03)) for d in d_list)


def changing_lens(Cores, mu_list, n):
    f_read_name = 'Actual Data/Unblocked Files/Changing Lens/{ratio:.2f} mu, {n} n.csv'
    f_write_name = 'Actual Data/Changing Lens/{ratio:.2f}.csv'

    Parallel(n_jobs=Cores, verbose=5)(delayed(f)(f_read_name.format(ratio=np.round(mu, 2), n=n), f_write_name.format(ratio=np.round(mu, 2)), RectangleBlocker(0, 0, 0, 0.05, 0.05), overwrite=True) for mu in mu_list)


def main():
    Cores = 4

    lpi_list = [18, 161]
    i_list = np.arange(2, 17, 0.2)
    h_list = np.arange(2, 18, 1)
    w_list = [1, 3, 5]

    angle_list = np.arange(-1.8, 1.9, 0.1)

    d_list = np.arange(0.1, 0.7, 0.02)

    mu_list = np.arange(0.2, 1.8, 0.2)
    n = 1.6357

    # changing_width(Cores, lpi_list, i_list)
    # changing_height(Cores, lpi_list, i_list)
    # changing_height_width(Cores, h_list, w_list)
    # changing_angle(Cores, lpi_list, angle_list)
    # changing_distance(Cores, d_list)
    changing_lens(Cores, mu_list, n)


if __name__ == '__main__':
    main()