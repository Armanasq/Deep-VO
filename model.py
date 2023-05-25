def Quat_error_angle(y_true, y_pred):
    """
    The function takes in two quaternions, normalizes the predicted quaternion, and then calculates the
    angle between the two quaternions

    :param y_true: the true quaternion
    :param y_pred: the predicted quaternion
    :return: The angle between the two quaternions.
    """
    y_pred = tf.linalg.normalize(y_pred, ord='euclidean', axis=1)[0]
    w0, x0, y0, z0 = tf.split(
        (tf.multiply(y_pred, [1., -1, -1, -1]),), num_or_size_splits=4, axis=-1)
    w1, x1, y1, z1 = tf.split(y_true, num_or_size_splits=4, axis=-1)
    w = w0*w1 - x0*x1 - y0*y1 - z0*z1
    angle = (tf.abs(2 * tf.math.acos(tf.math.sqrt(tf.math.square(w))))) * 180/np.pi
    return tf.clip_by_value(angle, -1e3, 1e3)


def Quat_mult(y_true, y_pred):
    """
    The function takes in two quaternions, normalizes the first one, and then multiplies the two
    quaternions together.

    The function returns the absolute value of the vector part of the resulting quaternion.

    The reason for this is that the vector part of the quaternion is the axis of rotation, and the
    absolute value of the vector part is the angle of rotation.

    The reason for normalizing the first quaternion is that the first quaternion is the predicted
    quaternion, and the predicted quaternion is not always normalized.

    The reason for returning the absolute value of the vector part of the resulting quaternion is that
    the angle of rotation is always positive.

    The reason for returning the vector part of the resulting quaternion is that the axis of rotation is
    always a vector.

    :param y_true: the ground truth quaternion
    :param y_pred: the predicted quaternion
    :return: The absolute value of the quaternion multiplication of the predicted and true quaternions.
    """
    y_pred = tf.linalg.normalize(y_pred, ord='euclidean', axis=1)[0]
    w0, x0, y0, z0 = tf.split(
        (tf.multiply(y_pred, [1., -1, -1, -1]),), num_or_size_splits=4, axis=-1)
    w1, x1, y1, z1 = tf.split(y_true, num_or_size_splits=4, axis=-1)
    # w = w0*w1 - x0*x1 - y0*y1 - z0*z1
    x = w0*x1 + x0*w1 + y0*z1 - z0*y1
    y = w0*y1 - x0*z1 + y0*w1 + z0*x1
    z = w0*z1 + x0*y1 - y0*x1 + z0*w1
    return (tf.abs(tf.multiply(2.0, tf.concat(values=[x, y, z], axis=-1))))


def quaternion_mean_multiplicative_error(y_true, y_pred):
    return tf.reduce_mean(Quat_mult(y_true, y_pred))

# Custom loss layer


class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        # def __init__(self, nb_outputs=3, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(
            ys_pred) == self.nb_outputs
        loss = 0

        # for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
        #    precision = K.exp(-log_var[0])
        #    loss += K.sum(precision * (y_true - y_pred)**2., -1) + log_var[0]

        precision = K.exp(-self.log_vars[0][0])
        loss += precision * \
            mean_absolute_error(ys_true[0], ys_pred[0]) + self.log_vars[0][0]
        precision = K.exp(-self.log_vars[1][0])
        loss += precision * \
            quaternion_mean_multiplicative_error(
                ys_true[1], ys_pred[1]) + self.log_vars[1][0]
        # loss += precision * quaternion_phi_4_error(ys_true[1], ys_pred[1]) + self.log_vars[1][0]
 
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)

