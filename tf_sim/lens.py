import tensorflow as tf


class LenticularLens:
    def __init__(self, zo, w, h, R, p, t, n):
        """
        Creates a lenticular lens
        idk what any of these params are
        do not change them after the object is initialized

        :param zo: python scalar
        :param w: python scalar
        :param h: python scalar
        :param R: python scalar
        :param p: python scalar
        :param t: python scalar
        :param n: python scalar
        """
        # todo i could convert all these to tf.variables
        print("using tf lens")
        self.zo = float(zo)

        self.w = float(w)
        self.h = float(h)
        self.p = float(p)
        self.t = float(t)
        self.R = float(R)
        self.f = float(R) - tf.sqrt(4 * self.R * self.R - self.p * self.p) / 2

        self.n = float(n)
        self.mu = 1 / self.n

        self.c1 = tf.constant([w / 2, h / 2, zo])
        self.c2 = tf.constant([-w / 2, -h / 2, zo])

        self.r_min = 1.0
        self.r_max = 4.0

    def get_config(self):
        return {
            "zo": self.zo,
            "w": self.w,
            "h": self.h,
            "R": self.R,
            "p": self.p,
            "t": self.t,
            "n": self.n
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 3)), tf.TensorSpec(shape=(None, 3))])
    def refract(self, pos, vec):
        """
        outputs refracted ray

        :param pos: (n, 3) tensor
        :param vec: (n, 3) tensor
        :return: 2 (n, 3) tensors
        """
        print("LenticularLens: refract tracing")
        n = tf.shape(pos)[0]

        # getting intersection of ray and x-axis
        xi = pos[:, 0]
        zi = pos[:, 2]

        m = vec[:, 2] / vec[:, 0]
        v = tf.math.rsqrt(1 + m * m)
        x_tan = tf.clip_by_value(m * self.R * v, -self.p / 2, self.p / 2)
        z_tan = self.zo + self.f + self.R - tf.sqrt(self.R * self.R - x_tan * x_tan)

        # intersection of moved ray with x-axis
        x1 = x_tan - (z_tan - zi) / m - self.p * tf.maximum(tf.sign(m), 0.0)
        num = tf.math.ceil((x1 - xi) / self.p)  # number of lenticules to move the actual ray to the right

        xf = xi + num * self.p  # final x coordinate of intersection of moved ray with x-axis
        # tf.print(tf.reduce_sum(xf), tf.reduce_sum(m), tf.reduce_sum(zi))

        # intersection of ray with lenticular lens
        root = tf.sqrt(m * m * (self.R - xf) * (self.R + xf) - self.f * self.f - 2 * m * xf * (self.R + self.zo - zi) -
                       (self.zo - zi) * (2 * self.R + self.zo - zi) + 2 * self.f * (
                               self.R + m * xf + self.zo - zi)) * tf.sign(m)  # wtf

        # intersection of ray with lens, moved to primary lenticule
        x_moved = (m * (
                self.R - self.f + m * xf + self.zo) - root) * v * v
        x_intersect = x_moved - num * self.p
        pos_in = pos + tf.expand_dims((x_intersect - pos[:, 0]) / vec[:, 0], -1) * vec

        n_vec_in = tf.stack([
            - x_moved,
            tf.zeros(shape=(n,)),
            self.zo - self.f + self.R - pos_in[:, 2]
        ], axis=-1)

        n_vec_in = n_vec_in / tf.norm(n_vec_in, axis=-1, keepdims=True)

        # dot_in = tf.reduce_sum(n_vec_in * vec, axis=-1)
        # vec_in = tf.expand_dims(tf.sqrt(1 - self.mu * self.mu * (1 - dot_in * dot_in)), axis=-1) * n_vec_in + \
        #          self.mu * (vec - tf.expand_dims(dot_in, axis=-1) * n_vec_in)
        dot_in = tf.reduce_sum(n_vec_in * vec, axis=-1)

        theta_f_root = tf.sqrt(1 - self.mu * self.mu * (1 - dot_in * dot_in))

        weights_1 = 1 - (tf.pow((dot_in - self.n * theta_f_root) / (dot_in + self.n * theta_f_root), 2)
                         + tf.pow((theta_f_root - self.n * dot_in) / (theta_f_root + self.n * dot_in), 2)) / 2

        vec_in = (tf.expand_dims(theta_f_root, axis=-1) * n_vec_in +
                  self.mu * (vec - tf.expand_dims(dot_in, axis=-1) * n_vec_in))

        # calculate the intersection of the ray inside the thing with the flat part
        n_vec_out = tf.constant([[0.0, 0.0, 1.0]])
        dot_out = vec_in[:, 2]
        root = 1 - self.n * self.n * (1 - dot_out * dot_out)

        vec_out = (tf.expand_dims(tf.sqrt(tf.nn.relu(root)), -1) * n_vec_out +
                   self.n * (vec_in - tf.expand_dims(dot_out, -1) * n_vec_out))

        pos_out = pos_in + vec_in * tf.expand_dims((self.zo + self.t - self.f - pos_in[:, 2]) / vec_in[:, 2], -1)

        return pos_out, vec_out, tf.cast(root >= 0, tf.float32) * weights_1

    @tf.function
    def calc_m(self, yo, zo, yp, zp, r):
        """
        ???

        :param yo: (n,) tensor
        :param zo: (n,) tensor
        :param yp: scalar tensor
        :param zp: scalar tensor
        :param r: scalar tensor
        :return: (n,) tensor
        """
        print("LenticularLens: calc_m tracing")
        return (yo - yp) / ((r - 1) * self.zo + zo - r * zp)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 3))])
    def angle_bounds_theta(self, ray_pos):
        """
        idk what this does

        :param ray_pos: (n, 3) tensor
        :return: 2 (n,) tensors
        """
        print("LenticularLens: angle_bounds_phi_theta tracing")
        vec_c1 = self.c1 - ray_pos
        theta_max = tf.math.atan(vec_c1[:, 0] / vec_c1[:, 2])

        vec_c2 = self.c2 - ray_pos
        theta_min = tf.math.atan(vec_c2[:, 0] / vec_c2[:, 2])

        return theta_min, theta_max

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 3))])
    def angle_bounds_phi_lens(self, ray_pos):
        """
        idk what this does

        :param ray_pos: (n, 3) tensor
        :return: 2 (n,) tensors
        """
        print("LenticularLens: angle_bounds_phi_lens tracing")
        vec_c1 = self.c1 - ray_pos
        phi_max = tf.math.atan(vec_c1[:, 1] / vec_c1[:, 2])

        vec_c2 = self.c2 - ray_pos
        phi_min = tf.math.atan(vec_c2[:, 1] / vec_c2[:, 2])

        return phi_min, phi_max

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 3)), tf.TensorSpec(shape=(3,)),
                                  tf.TensorSpec(shape=())])
    def angle_bounds_phi_pinhole(self, ray_pos, pinhole_pos, pinhole_rad):
        """
        idk what this does

        :param ray_pos: (n, 3) tensor where n is batch size
        :param pinhole_rad: scalar tensor
        :param pinhole_pos: (3,) tensor
        :return: phi bounds (whatever phi is) (2 (n,) tensors)
        """
        print("LenticularLens: angle_bound_phi_pinhole tracing")
        yo = ray_pos[:, 1]
        zo = ray_pos[:, 2]
        zp = pinhole_pos[2]
        radius = pinhole_rad

        m = tf.stack([
            self.calc_m(yo, zo, - radius, zp, self.r_min),  # m_min_min
            self.calc_m(yo, zo, radius, zp, self.r_min),  # m_min_max
            self.calc_m(yo, zo, - radius, zp, self.r_max),  # m_max_min
            self.calc_m(yo, zo, radius, zp, self.r_max),  # m_max_max
        ], axis=-1)

        phi_min = tf.math.atan(tf.reduce_min(m, axis=-1))
        phi_max = tf.math.atan(tf.reduce_max(m, axis=-1))

        return phi_min, phi_max

    @tf.function  # im not writing an input signature for this
    def angle_bounds_phi_camera(self, ray_pos, pupil_pos, pupil_rad, camera_pos, camera_rad):
        """
        idk what this does

        :param ray_pos: (n, 3) tensor
        :param pupil_pos: (3,) tensor
        :param pupil_rad: scalar tensor
        :param camera_pos: (3,) tensor
        :param camera_rad: scalar tensor
        :return: 2 (n,) tensors
        """
        print("LenticularLens: angle_bound_phi_camera tracing")
        phi_min_pupil, phi_max_pupil = self.angle_bounds_phi_pinhole(ray_pos, pupil_pos, pupil_rad)
        phi_min_camera, phi_max_camera = self.angle_bounds_phi_pinhole(ray_pos, camera_pos, camera_rad)
        phi_min_lens, phi_max_lens = self.angle_bounds_phi_lens(ray_pos)

        phi_min = tf.stack((phi_min_pupil, phi_min_lens, phi_min_camera), axis=-1)
        phi_max = tf.stack((phi_max_lens, phi_max_camera, phi_max_pupil), axis=-1)

        return tf.reduce_max(phi_min, axis=-1), tf.reduce_min(phi_max, axis=-1)


# testing
# if __name__ == "__main__":
#     lens = LenticularLens(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
#     ray_pos = tf.random.uniform(shape=(10, 3))
#     ray_vec = tf.random.uniform(shape=(10, 3))
#     pupil_pos = tf.random.uniform(shape=(3,))
#     pupil_rad = tf.random.uniform(shape=())
#     camera_pos = tf.random.uniform(shape=(3,))
#     camera_rad = tf.random.uniform(shape=())
#     print(lens.angle_bounds_phi_camera(ray_pos, pupil_pos, pupil_rad, camera_pos, camera_rad))
#     print(lens.angle_bounds_theta(ray_pos))
#     print(lens.refract(ray_pos, ray_vec))
