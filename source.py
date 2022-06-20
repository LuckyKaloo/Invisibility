import tensorflow as tf


class Source:
    def generate_pos(self, n_batches):
        raise NotImplementedError("To be implemented in subclass")


class RectangleSource(Source):
    # x,y are center
    def __init__(self, x, y, z, w, h):
        """
        Creates a rectangle source
        do not try to change these values after the __init__ has been called
        it will not work trust me (unless you use tf.Variable's)

        :param x: x value of center (float)
        :param y: y value of center (float)
        :param z: z position of plane (float)
        :param w: width of plane (float)
        :param h: height of plane (float)
        """
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.h = h

        self.x_min = x - w / 2
        self.x_max = x + w / 2
        self.y_min = y - h / 2
        self.y_max = y + h / 2
        self.z = float(z)

    def get_config(self):
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "w": self.w,
            "h": self.h
        }

    @tf.function
    def generate_pos(self, n_batches):
        """
        Generates a batch of rays

        :param n_batches: batch size (int, ideally scalar tensor)
        :return: A batch of rays
        """
        print("RectangeSource: generate_pos tracing")
        x = tf.random.uniform(shape=(n_batches,), minval=self.x_min, maxval=self.x_max)
        y = tf.random.uniform(shape=(n_batches,), minval=self.y_min, maxval=self.y_max)
        z = tf.fill(dims=(n_batches,), value=self.z)
        return tf.stack((x, y, z), axis=-1)


# testing
if __name__ == "__main__":
    source = RectangleSource(0, 0, 0, 1, 1)
    print(source.generate_pos(10))
