import tensorflow as tf


class Pinhole:
    def __init__(self, pos, r):
        """
        idk what this does

        :param pos: python 3-vector
        :param r: python scalar
        """
        self.pos = tf.convert_to_tensor(pos, dtype=tf.float32)
        self.r = r

        # idk what this is (actually idk what any of these are)
        self.diff = 3.0

        self.c1 = tf.constant([self.diff * r, self.diff * self.r, pos[2]])
        self.c2 = tf.constant([- self.diff * r, - self.diff * self.r, pos[2]])

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 3)), tf.TensorSpec(shape=(None, 3))])
    def blocks(self, pos_ray, vec_ray):
        """
        idk what this does

        :param pos_ray: (n, 3) tensor, n is batch size
        :param vec_ray: (n, 3) tensor n is batch size
        :return: (n,) tensor, whether each ray intersects with pinhole
        """
        final_pos = pos_ray + vec_ray * (self.pos[2] - tf.expand_dims(pos_ray[:, 2], axis=-1)) / tf.expand_dims(
            vec_ray[:, 2], axis=-1)
        d_vec = final_pos - self.pos
        d = tf.linalg.norm(d_vec, axis=-1)

        return d > self.r

    def get_config(self):
        return {
            "pos": self.pos,
            "r": self.r
        }


class Camera:
    def __init__(self, z_camera, camera_radius, z_pupil, pupil_radius):
        """
        Creates a camera

        :param z_camera: z position
        :param camera_radius: self-explanatory
        :param z_pupil: distance to the pupil (whatever that is)
        :param pupil_radius: self-explanatory
        """
        self.z_camera = z_camera
        self.camera_radius = camera_radius
        self.z_pupil = z_pupil

        self.pupil_radius = pupil_radius

        # self.thin_lens = ThinLens(0, 0, z_camera + z0, lens_radius, focal_length, n)
        self.pupil_pinhole = Pinhole([0, 0, z_camera + z_pupil], self.pupil_radius)
        self.camera_pinhole = Pinhole([0, 0, z_camera], camera_radius)

    def projection(self, ray_pos, ray_vec):
        """
        check if ray enters camera (?)

        :param ray_pos: (n, 3) tensor, n is batch size
        :param ray_vec: (n, 3) tensor, n is batch size
        :return: x, y position if ray intersects
        """
        return (not self.pupil_pinhole.blocks(ray_pos, ray_vec)), ray_pos[:, :2]

    def get_config(self):
        return {
            "z_camera": self.z_camera,
            "camera_radius": self.camera_radius,
            "z_pupil": self.z_pupil,
            "f_number": self.f_number,
            "focal_length": self.focal_length
        }


# testing
# if __name__ == "__main__":
#     c = Camera(0, 1, 1, 1, 1)
#     print(c.projection(tf.random.uniform(shape=(10, 3)), tf.random.uniform(shape=(10, 3))))
