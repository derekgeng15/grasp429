import numpy as np, tf

t_pb_to_co = [0.00741169, -0.0619934, 0.0449459]

q_pb_to_co = [0.0154384, 0.00615987, 0.372608, 0.92784]

t_co_to_clink = [0.015, 0.0, 0.0]
q_co_to_clink = [0.500, -0.497, 0.501,  0.502]

pb_to_clink = np.array(t_pb_to_co)+np.array(t_co_to_clink), tf.transformations.quaternion_multiply(q_pb_to_co,q_co_to_clink)
print(pb_to_clink[0])
print(pb_to_clink[1])

