import tensorflow as tf


class RunningVars:

    def __init__(self):
        self.count = None
        self.sum = None
        self.sum_of_squares = None

    def append(self, x):
        if type(self.count) == type(None):
            self.count = tf.zeros_like(x)
            self.sum = tf.zeros_like(x)
            self.sum_of_squares = tf.zeros_like(x)
        self.sum += x
        self.sum_of_squares += tf.square(x)
        self.count += 1

    def reset(self, resets):
        resets = tf.convert_to_tensor(resets)
        resets = tf.where(resets, tf.zeros_like(resets, dtype=self.count.dtype),
                          tf.ones_like(resets, dtype=self.count.dtype))
        resets = tf.expand_dims(resets, axis=-1)
        self.count = tf.multiply(resets, self.count)
        self.sum = tf.multiply(resets, self.sum)
        self.sum_of_squares = tf.multiply(resets, self.sum_of_squares)

    def remove(self, x):
        self.sum -= x
        self.sum_of_squares -= tf.square(x)

    def mean(self):
        return self.sum / self.count

    def variance(self):
        return (self.sum_of_squares / self.count) - tf.square(self.mean())

    def stdev(self):
        return tf.sqrt(self.variance())


if __name__ == "__main__":
    var_shape = (5, 3)
    rv = RunningVars(var_shape)
    for i in range(10000):
        var = tf.random.normal(var_shape, dtype=tf.float32)
        rv.append(var)
    print("example")
    print(var)
    print("mean")
    print(rv.mean())
    print("stdev")
    print(rv.stdev())
    print("count")
    print(rv.count)
    resets = [True, False, True, False, True]
    rv.reset(resets)
    print("mean")
    print(rv.mean())
    print("stdev")
    print(rv.stdev())
    print("count")
    print(rv.count)
    rv.append(var)
    print("mean")
    print(rv.mean())
    print("stdev")
    print(rv.stdev())
    print("count")
    print(rv.count)
