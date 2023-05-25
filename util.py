import numpy as np

def dcm2quat(dcm):
    traces = np.trace(dcm, axis1=1, axis2=2)  # Calculate trace along the last two axes
    indices = traces > 0

    qw = np.zeros_like(traces)
    qx = np.zeros_like(traces)
    qy = np.zeros_like(traces)
    qz = np.zeros_like(traces)

    S = np.sqrt(traces[indices] + 1.0) * 2
    qw[indices] = 0.25 * S
    qx[indices] = (dcm[indices, 2, 1] - dcm[indices, 1, 2]) / S
    qy[indices] = (dcm[indices, 0, 2] - dcm[indices, 2, 0]) / S
    qz[indices] = (dcm[indices, 1, 0] - dcm[indices, 0, 1]) / S

    condition_1 = np.logical_and(~indices, dcm[:, 0, 0] > dcm[:, 1, 1])
    condition_2 = np.logical_and(condition_1, dcm[:, 0, 0] > dcm[:, 2, 2])

    S = np.sqrt(1.0 + dcm[condition_2, 0, 0] - dcm[condition_2, 1, 1] - dcm[condition_2, 2, 2]) * 2
    qw[condition_2] = (dcm[condition_2, 2, 1] - dcm[condition_2, 1, 2]) / S
    qx[condition_2] = 0.25 * S
    qy[condition_2] = (dcm[condition_2, 0, 1] + dcm[condition_2, 1, 0]) / S
    qz[condition_2] = (dcm[condition_2, 0, 2] + dcm[condition_2, 2, 0]) / S

    condition_3 = np.logical_and(~indices, ~condition_2, dcm[:, 1, 1] > dcm[:, 2, 2])

    S = np.sqrt(1.0 + dcm[condition_3, 1, 1] - dcm[condition_3, 0, 0] - dcm[condition_3, 2, 2]) * 2
    qw[condition_3] = (dcm[condition_3, 0, 2] - dcm[condition_3, 2, 0]) / S
    qx[condition_3] = (dcm[condition_3, 0, 1] + dcm[condition_3, 1, 0]) / S
    qy[condition_3] = 0.25 * S
    qz[condition_3] = (dcm[condition_3, 1, 2] + dcm[condition_3, 2, 1]) / S

    condition_4 = np.logical_and(~indices, ~condition_2, ~condition_3)

    S = np.sqrt(1.0 + dcm[condition_4, 2, 2] - dcm[condition_4, 0, 0] - dcm[condition_4, 1, 1]) * 2
    qw[condition_4] = (dcm[condition_4, 1, 0] - dcm[condition_4, 0, 1]) / S
    qx[condition_4] = (dcm[condition_4, 0, 2] + dcm[condition_4, 2, 0]) / S
    qy[condition_4] = (dcm[condition_4, 1, 2] + dcm[condition_4, 2, 1]) / S
    qz[condition_4] = 0.25 * S
    q = np.stack((qw, qx, qy, qz), axis=1)

    return q

def quat2eul(q):
    """
    The function takes in a quaternion and returns the roll, pitch, and yaw angles.

    :param q: quaternion
    :return: the roll, pitch and yaw angles of the quaternion.
    """
    normalized_array = q/np.linalg.norm(q, axis=1).reshape(len(q), 1)
    w, x, y, z = np.hsplit(normalized_array, 4)
    roll_x = (np.arctan2(2*(w[:, 0]*x[:, 0] + y[:, 0]*z[:, 0]),
                         (1-2*(x[:, 0]*x[:, 0] + y[:, 0]*y[:, 0]))))
    pitch_y = (np.arcsin(2*(w[:, 0]*y[:, 0] - x[:, 0]*z[:, 0])))
    yaw_z = (np.arctan2(2*(w[:, 0]*z[:, 0] + x[:, 0]*y[:, 0]),
                        (1-2*(y[:, 0]*y[:, 0] + z[:, 0]*z[:, 0]))))
    roll_x.reshape(len(roll_x), 1), pitch_y.reshape(len(roll_x), 1), yaw_z.reshape(len(roll_x), 1)
    return np.column_stack((roll_x, pitch_y, yaw_z))

def dcm2eul(dcm):
    """
    Converts a Direction Cosine Matrix (DCM) to Euler angles.

    :param dcm: Direction Cosine Matrix (n x 3 x 3)
    :return: Euler angles (n x 3) representing roll, pitch, and yaw
    """
    roll = np.arctan2(dcm[:, 2, 1], dcm[:, 2, 2])
    pitch = np.arctan2(-dcm[:, 2, 0], np.sqrt(dcm[:, 2, 1]**2 + dcm[:, 2, 2]**2))
    yaw = np.arctan2(dcm[:, 1, 0], dcm[:, 0, 0])

    return np.column_stack((roll, pitch, yaw))



