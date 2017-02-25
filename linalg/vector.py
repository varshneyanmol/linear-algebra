import linalg.generalop as g
import math


'''
column vector is represented as:
V = [[1],
     [-4],
     [8]]

and row vector is represented as:
V = [1, -4, 8]
'''


def dot(v1, v2):
    return g.multiply(g.transpose(v1), v2)[0][0]


def norm(v):
    return math.sqrt(dot(v, v))


def projection(of_v, on_v):
    temp = dot(of_v, on_v)/(norm(on_v)**2)
    return g.multiply_by_scalar(on_v, temp)


def transform_to_column_vector(v):
    return [[v_i] for v_i in v]


def transform_to_row_vector(v):
    return [v_i[0] for v_i in v]


def orthonormal(v):
    n = norm(v)
    return [[v_i[0]/n] for v_i in v]
