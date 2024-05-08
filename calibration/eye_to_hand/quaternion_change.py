import quaternion
import numpy as np

#t_pb_to_co = [.0417737, -0.0400657, -0.0430157]
#q_pb_to_co = [0.712855, 0.013033, 0.014284, 0.701044 ]

t_pb_to_co = [0.00741169, -0.0619934, 0.0449459]

q_pb_to_co = [0.0154384, 0.00615987, 0.372608, 0.92784]

t_co_to_clink = [0.015, 0, 0]
q_co_to_clink = [0.500, -0.497, 0.501,  0.502]


def change_tf_input_to_T(t, q):
    Q = quaternion.as_quat_array(q)
    R = quaternion.as_rotation_matrix(Q)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

T_qb_to_co = change_tf_input_to_T(t_pb_to_co, q_pb_to_co)
T_co_to_clink = change_tf_input_to_T(t_co_to_clink, q_co_to_clink)

print(T_qb_to_co)
T_qb_to_clink = T_qb_to_co @ T_co_to_clink
print(T_qb_to_clink[:3, 3])
print(quaternion.from_rotation_matrix(T_qb_to_clink[:3, :3]))
