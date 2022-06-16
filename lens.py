import numpy as np


class LenticularLens:
    def __init__(self, zo, w, h, R, p, t, n):
        self.zo = zo

        self.w = w
        self.h = h
        self.p = p
        self.t = t
        self.R = R
        self.f = R - np.sqrt(4 * R * R - p * p) / 2

        self.n = n
        self.mu = 1 / self.n


        self.c1 = np.array([w / 2, h / 2, zo])
        self.c2 = np.array([-w / 2, -h / 2, zo])

        self.r_min = 1
        self.r_max = 4

    def refract(self, pos, vec):
        # pos and dir are in the format (x, y, z), pos is position vector of initial point of release, vec is direction

        """
        calculate the intersection of the incoming ray with the circle part
        """

        # getting intersection of ray and x-axis
        xi = pos[0]
        zi = pos[2]

        # point at which ray is tangent
        if vec[0] != 0:
            m = vec[2] / vec[0]
            v = 1 / np.sqrt(1 + m * m)
            x_tan = np.clip(m * self.R * v, -self.p / 2, self.p / 2)
            z_tan = self.zo + self.f + self.R - np.sqrt(self.R * self.R - x_tan * x_tan)

            # intersection of moved ray with x-axis
            x1 = x_tan - (z_tan - zi) / m - (self.p if m > 0 else 0)  # intersection of leftmost tangent ray with x-axis
            num = np.ceil((x1 - xi) / self.p)  # number of lenticules to move the actual ray to the right

            xf = xi + num * self.p  # final x coordinate of intersection of moved ray with x-axis

            # intersection of ray with lenticular lens
            root = np.sqrt(m * m * (self.R - xf) * (self.R + xf) - self.f * self.f - 2 * m * xf * (self.R + self.zo - zi) -
                           (self.zo - zi) * (2 * self.R + self.zo - zi) + 2 * self.f * (self.R + m * xf + self.zo - zi)) * np.sign(m)
            x_moved = (m * (self.R - self.f + m * xf + self.zo) - root) * v * v  # intersection of ray with lens, moved to primary lenticule
            x_intersect = x_moved - num * self.p

            pos_in = pos + (x_intersect - pos[0]) / vec[0] * vec  # position at which ray enters lens
        else:
            x1 = -self.p / 2
            num = np.ceil((x1 - xi) / self.p)  # number of lenticules to move the actual ray to the right

            x_moved = xi + num * self.p  # final x coordinate of intersection of moved ray with x-axis
            z_intersect = -self.f + self.R - np.sqrt(self.R * self.R - x_moved * x_moved) + self.zo

            pos_in = pos + (z_intersect - pos[2]) / vec[2] * vec


        # refraction
        n_vec_in = np.array([-x_moved, 0, self.zo - self.f + self.R - pos_in[2]])
        n_vec_in /= np.linalg.norm(n_vec_in)

        dot_in = np.dot(n_vec_in, vec)
        vec_in = np.sqrt(1 - self.mu * self.mu * (1 - dot_in * dot_in)) * n_vec_in + self.mu * (vec - dot_in * n_vec_in)


        """
        calculate the intersection of the ray inside the thing with the flat part
        """

        n_vec_out = np.array([0, 0, 1], dtype=float)
        dot_out = np.dot(n_vec_out, vec_in)
        root = 1 - self.n * self.n * (1 - dot_out * dot_out)

        if root < 0:
            return None

        vec_out = np.sqrt(root) * n_vec_out + self.n * (vec_in - dot_out * n_vec_out)

        pos_out = pos_in + vec_in * (self.zo + self.t - self.f - pos_in[2]) / vec_in[2]

        return pos_out, vec_out

    def angle_bounds_theta(self, ray_pos):
        vec_c1 = self.c1 - ray_pos
        theta_max = np.arctan(vec_c1[0] / vec_c1[2])

        vec_c2 = self.c2 - ray_pos
        theta_min = np.arctan(vec_c2[0] / vec_c2[2])

        return theta_min, theta_max

    def angle_bounds_phi_lens(self, ray_pos):
        vec_c1 = self.c1 - ray_pos
        phi_max = np.arctan(vec_c1[1] / vec_c1[2])

        vec_c2 = self.c2 - ray_pos
        phi_min = np.arctan(vec_c2[1] / vec_c2[2])

        return phi_min, phi_max

    def calc_m(self, yo, zo, yp, zp, r):
        return (yo - yp) / ((r - 1) * self.zo + zo - r * zp)

    def angle_bounds_phi_pinhole(self, ray_pos, pinhole):
        yo = ray_pos[1]
        zo = ray_pos[2]
        zp = pinhole.pos[2]
        radius = pinhole.radius

        m_min_min = self.calc_m(yo, zo, - radius, zp, self.r_min)
        m_min_max = self.calc_m(yo, zo, radius, zp, self.r_min)
        m_max_min = self.calc_m(yo, zo, - radius, zp, self.r_max)
        m_max_max = self.calc_m(yo, zo, radius, zp, self.r_max)

        phi_min = np.arctan(min(m_min_min, m_min_max, m_max_min, m_max_max))
        phi_max = np.arctan(max(m_min_min, m_min_max, m_max_min, m_max_max))

        return phi_min, phi_max

    def angle_bounds_phi_camera(self, ray_pos, camera):
        phi_min_pupil, phi_max_pupil = self.angle_bounds_phi_pinhole(ray_pos, camera.pupil_pinhole)
        phi_min_camera, phi_max_camera = self.angle_bounds_phi_pinhole(ray_pos, camera.camera_pinhole)
        phi_min_lens, phi_max_lens = self.angle_bounds_phi_lens(ray_pos)

        # print(phi_min_pupil, phi_max_pupil, phi_min_camera, phi_max_camera, phi_min_lens, phi_max_lens)

        return max(phi_min_pupil, phi_min_camera, phi_min_lens), min(phi_max_pupil, phi_max_camera, phi_max_lens)


    def refract_more_info(self, pos, vec):
        # pos and dir are in the format (x, y, z), pos is position vector of initial point of release, vec is direction

        """
        calculate the intersection of the incoming ray with the circle part
        """

        # getting intersection of ray and x-axis
        xi = pos[0]
        zi = pos[2]

        # point at which ray is tangent
        if vec[0] != 0:
            m = vec[2] / vec[0]
            v = 1 / np.sqrt(1 + m * m)
            x_tan = np.clip(m * self.R * v, -self.p / 2, self.p / 2)
            z_tan = min(self.R * (1 - v) - self.f + self.zo, 10)

            # intersection of moved ray with x-axis
            x1 = x_tan - (z_tan - zi) / m - (self.p if m > 0 else 0)  # intersection of leftmost tangent ray with x-axis
            num = np.ceil((x1 - xi) / self.p)  # number of lenticules to move the actual ray to the right

            xf = xi + num * self.p  # final x coordinate of intersection of moved ray with x-axis

            # intersection of ray with lenticular lens
            root = np.sqrt(m * m * (self.R - xf) * (self.R + xf) - self.f * self.f - 2 * m * xf * (self.R + self.zo - zi) -
                           (self.zo - zi) * (2 * self.R + self.zo - zi) + 2 * self.f * (self.R + m * xf + self.zo - zi)) * np.sign(m)
            x_moved = (m * (self.R - self.f + m * xf + self.zo) - root) * v * v  # intersection of ray with lens, moved to primary lenticule
            x_intersect = x_moved - num * self.p

            pos_in = pos + (x_intersect - pos[0]) / vec[0] * vec  # position at which ray enters lens
        else:
            x1 = -self.p / 2
            num = np.ceil((x1 - xi) / self.p)  # number of lenticules to move the actual ray to the right

            x_moved = xi + num * self.p  # final x coordinate of intersection of moved ray with x-axis
            z_intersect = -self.f + self.R - np.sqrt(self.R * self.R - x_moved * x_moved) + self.zo

            pos_in = pos + (z_intersect - pos[2]) / vec[2] * vec


        # refraction
        n_vec_in = np.array([-x_moved, 0, self.zo - self.f + self.R - pos_in[2]])
        n_vec_in /= np.linalg.norm(n_vec_in)

        dot_in = np.dot(n_vec_in, vec)
        vec_in = np.sqrt(1 - self.mu * self.mu * (1 - dot_in * dot_in)) * n_vec_in + self.mu * (vec - dot_in * n_vec_in)


        """
        calculate the intersection of the ray inside the thing with the flat part
        """

        n_vec_out = np.array([0, 0, 1], dtype=float)
        dot_out = np.dot(n_vec_out, vec_in)

        root = 1 - self.n * self.n * (1 - dot_out * dot_out)

        print(root)

        if root < 0:
            return pos, vec, pos_in, vec_in, np.array([0, 0, 0]), np.array([0, 0, 0])

        vec_out = np.sqrt(root) * n_vec_out + self.n * (vec_in - dot_out * n_vec_out)

        pos_out = pos_in + vec_in * (self.zo + self.t - self.f - pos_in[2]) / vec_in[2]

        return pos, vec, pos_in, vec_in, pos_out, vec_out


class ThinLens:
    def __init__(self, x, y, z, R, f, n):
        self.x = x
        self.y = y
        self.z = z
        self.R = R

        self.n = n
        self.roc = 2 * (n - 1) * f

        self.centre_pos = np.array([x, y, z])
        self.curve_centre_pos = self.centre_pos + np.array([0, 0, self.roc])

    def refract(self, ray_pos, ray_vec):
        t = (self.z - ray_pos[2]) / ray_vec[2]
        hit_pos = ray_pos + t * ray_vec

        if np.linalg.norm(self.centre_pos - hit_pos) > self.R:
            return None

        normal1 = self.curve_centre_pos - hit_pos
        normal1 /= np.linalg.norm(normal1)
        normal2 = normal1 * np.array([-1, -1, 1])

        k2 = self.boundary_refract(ray_vec, normal1, 1, self.n)
        k3 = self.boundary_refract(k2, normal2, self.n, 1)

        return hit_pos, k3

    def boundary_refract(self, inc_vec, normal_vec, n1, n2):
        mu = n1 / n2
        dot = np.linalg.norm(np.dot(inc_vec, normal_vec))

        return mu * inc_vec + (np.sqrt(1 - mu * mu * (1 - dot * dot)) - mu * dot) * normal_vec
