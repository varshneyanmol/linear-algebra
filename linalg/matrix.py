import linalg.vector as vector
import linalg.generalop as g
import copy

import numpy as np

'''
a (2 * 3) matrix is of the form:
D = [[2, 4, 5],
     [3, 6, -8]]
'''


def zero_matrix(num_rows, num_cols):
    M = []
    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            row.append(0)
        M.append(row)
    return M


def identity_matrix(size):
    I = []
    for i in range(size):
        row = []
        for j in range(size):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        I.append(row)
    return I


def trace(matrix):
    # return sum of diagonal elements of a square matrix
    size = check_for_square(matrix)
    if size == -1:
        print("can not find trace of a non-square matrix")
        return -1
    tr = 0
    for i in range(size):
        for j in range(size):
            if i == j:
                tr += matrix[i][j]
    return tr


def check_for_square(matrix):
    rows, cols = g.shape(matrix)
    if rows != cols:
        return -1
    else:
        return rows


def minor_sub_matrix(matrix, i, j):
    # return a sub-matrix for size (n-1, n-1) by leaving i_th row and j_th column of the argument matrix
    # only for square matrix
    size = check_for_square(matrix)
    if size == -1:
        print("can not find minor sub matrix of a non-square matrix!");
        return -1
    _minor_sub_matrix = []
    for row in range(size):
        _minor_row = []
        for col in range(size):

            if not (row == i or col == j):
                _minor_row.append(matrix[row][col])

        if row != i:
            _minor_sub_matrix.append(_minor_row)

    return _minor_sub_matrix


def determinant(matrix, along_row=0):
    size = check_for_square(matrix)

    # if not a square matrix
    if size == -1:
        print("can not find determinant of non-square matrix!")
        return -1

    # terminating condition for recursion
    if size == 1:
        return matrix[0][0]

    d = 0  # determinant
    for i in range(size):
        d += ((-1) ** (along_row + i)) * (matrix[along_row][i] * determinant(minor_sub_matrix(matrix, along_row, i)))
    return d


def minor(matrix, i, j):
    size = check_for_square(matrix)
    if size == -1:
        print("can not find minor of a non-square matrix!")
        return -1
    return determinant(minor_sub_matrix(matrix, i, j))


def cofactor(matrix, i, j):
    size = check_for_square(matrix)
    if size == -1:
        print("can not find cofactor of non-square matrix!")
        return -1
    return ((-1) ** (i + j)) * minor(matrix, i, j)


def cofactor_matrix(matrix):
    size = check_for_square(matrix)
    if size == -1:
        print("can not find cofactor matrix of a non-square matrix")
        return -1
    C = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(cofactor(matrix, i, j))
        C.append(row)
    return C


def adjoint(matrix):
    if check_for_square(matrix) == -1:
        print("can not find adjoint of a non-square matrix")
        return -1
    return g.transpose(cofactor_matrix(matrix))


def inverse(matrix):
    if check_for_square(matrix) == -1:
        print("can not find inverse of a non-square matrix")
        return -1

    det = determinant(matrix)
    if det == 0:
        print("can not find inverse of a singular matrix")
        return -1

    return g.divide_by_scalar(adjoint(matrix), det)


def orthonormalize(matrix):
    # this function expects an already orthogonal matrix and thus does not check for the same
    Tmatrix = g.transpose(matrix)
    cols, rows = g.shape(Tmatrix)
    M = []
    for col in range(cols):
        col_v = vector.orthonormal(vector.transform_to_column_vector(Tmatrix[col]))
        M.append(vector.transform_to_row_vector(col_v))
    return g.transpose(M)


def _get_column(matrix, from_row, col):
    rows, cols = g.shape(matrix)
    u = [matrix[i][col] for i in range(from_row, rows)]
    return u


def _row_operation(matrix, pivot_row, pivot_column, on_row):
    if matrix[on_row][pivot_column] == 0:
        return

    u = [matrix[pivot_row][i] for i in range(len(matrix[0]))]
    multiplier = matrix[on_row][pivot_column]
    for i in range(len(u)):
        u[i] = u[i] * multiplier
        matrix[on_row][i] = matrix[on_row][i] - u[i]


def _swap_rows(matrix, i, j):
    u = matrix[i]
    matrix[i] = matrix[j]
    matrix[j] = u


def row_echelon_form(matrix):
    rows, cols = g.shape(matrix)
    M = copy.deepcopy(matrix)
    for i in range(rows):
        for j in range(cols):
            u = _get_column(M, i, j)
            if u == [0] * len(u):
                continue

            # find out which row has the first non-zero entry
            r = -1
            for k in range(len(u)):
                if u[k] != 0:
                    r = i + k
                    break

            # swap row 'i' with row 'r'
            if i != r:
                _swap_rows(M, i, r)

            pivot = M[i][j]
            if i == rows - 1 and abs(pivot) < 0.000001:
                # this is kind of '*jugaad*' here done to avoid floating point errors
                # for example: instead of 0.0 pivot for the last row may come out to be 0.0000000000316 then
                # the next for loop is going to make it 1 which will disturb the solution
                # so explicitly make it 0.0
                # and break so to avoid 'divide by zero' error
                M[i][j] = 0.0
                break

            # make pivot element 1
            for k in range(cols):
                M[i][k] = M[i][k] / pivot

            # make all elements below pivot zero in that column
            for k in range(i + 1, rows):
                _row_operation(M, i, j, k)

            break
    return M


def reduced_row_echelon_form(matrix):
    rows, cols = g.shape(matrix)
    M = copy.deepcopy(matrix)
    for i in range(rows):
        for j in range(cols):
            u = _get_column(M, i, j)
            if u == [0] * len(u):
                continue

            # find out which row has the first non-zero entry
            r = -1
            for k in range(len(u)):
                if u[k] != 0:
                    r = i + k
                    break

            # swap row 'i' with row 'r'
            if i != r:
                _swap_rows(M, i, r)

            pivot = M[i][j]
            if i == rows - 1 and pivot < 0.000001:
                # this is kind of '*jugaad*' here done to avoid floating point errors
                # for example: instead of 0.0 pivot for the last row may come out to be 0.0000000000316 then
                # the next for loop is going to make it 1 which will disturb the solution
                # so explicitly make it 0.0
                # and break so to avoid 'divide by zero' error
                M[i][j] = 0.0
                break

            # make pivot element 1
            for k in range(cols):
                M[i][k] = M[i][k] / pivot

            # make all elements below pivot zero in that column
            for k in range(0, rows):
                if k == i:
                    continue
                _row_operation(M, i, j, k)

            break
    return M


def gauss_elimination(matrix):
    # *only one free variable is allowed*
    # it means number of equations must be equal to the number of unknowns
    # expects an augmented matrix
    # works well on homogeneous system. why? see reduced_row_echelon_from() or row_echelon_form() functions

    # M = reduced_row_echelon_form(matrix)
    M = row_echelon_form(matrix)
    rows, cols = g.shape(M)
    x = [0] * rows

    last_row = M[rows - 1]
    flag = True  # assuming the system has no solution or infinite solutions
    for i in range(len(last_row) - 1):
        if last_row[i] != 0:
            flag = False
            break

    if flag:
        if last_row[-1] == 0:  # system has infinitely many solutions
            x[rows - 1] = 1
        else:
            return None  # system has no solution

    for i in range(rows - 1, -1, -1):
        if x[i] != 0:  # case of infinitely many solution where x[rows-1] is manually set to 1
            continue

        x[i] = M[i][cols - 1]
        for j in range(rows):
            if j != i:
                x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]

    return x


def gram_schmidt_orthonormalize(matrix):
    Tmatrix = g.transpose(matrix)
    # no. of cols in 'matrix' are no. of rows in 'Tmatrix'
    # and no. of rows in 'matrix' are no. of cols in 'Tmatrix'
    cols, rows = g.shape(Tmatrix)
    M = []
    M.append(Tmatrix[0])
    for col in range(1, cols):
        col_v = vector.transform_to_column_vector(Tmatrix[col])
        for m_col in range(len(M)):
            m_col_v = vector.transform_to_column_vector(M[m_col])
            col_v = g.subtract(col_v, vector.projection(col_v, m_col_v))

        M.append(vector.transform_to_row_vector(col_v))
    return orthonormalize(g.transpose(M))


def qr_gram_schmidt(matrix):
    Q = gram_schmidt_orthonormalize(matrix)
    R = g.multiply(g.transpose(Q), matrix)  # as Q is orthonormal matrix, so, transpose(Q) = inverse(Q)
    return Q, R


def _add_identity_matrix_to_householder(H, I_size):
    if I_size == 0:
        return H
    I = identity_matrix(I_size)
    H_size = check_for_square(H)  # H is a square matrix
    M = []
    for i in range(I_size + H_size):
        row = []
        for j in range(I_size + H_size):
            if i < I_size and j < I_size:
                row.append(I[i][j])
            elif i >= I_size and j >= I_size:
                row.append(H[i - I_size][j - I_size])
            else:
                row.append(0)

        M.append(row)
    return M


def tridiagonal(matrix):
    size = check_for_square(matrix)
    if size == -1:
        print("can not tridiagonalize a non-square matrix")
        return

    # H = I - w*transpose(w)
    # w = (u - v)/norm(u - v)
    # since the matrix is symmetric, we can extract 'u' vector from matrix rows and then transform it to column vector
    for i in range(size - 2):
        u = vector.transform_to_column_vector([matrix[i][u_i] for u_i in range(i + 1, size)])
        v = vector.transform_to_column_vector([vector.norm(u) if v_i == 0 else 0 for v_i in range(len(u))])
        w = g.divide_by_scalar(g.subtract(u, v), vector.norm(g.subtract(u, v)))
        temp = g.multiply_by_scalar(g.multiply(w, g.transpose(w)), 2)
        H = g.subtract(identity_matrix(len(w)), temp)
        H = _add_identity_matrix_to_householder(H, size - len(H))
        matrix = g.multiply(H, g.multiply(matrix, H))

    return matrix


def _get_householder_vectors(matrix, col):
    Tmatrix = g.transpose(matrix)
    _, rows = g.shape(Tmatrix)
    u = vector.transform_to_column_vector([Tmatrix[col][u_i] for u_i in range(col, rows)])
    v = vector.transform_to_column_vector([vector.norm(u) if v_i == 0 else 0 for v_i in range(len(u))])
    return u, v


def qr_householder(matrix):
    rows, cols = g.shape(matrix)
    Q = identity_matrix(rows)
    for i in range(cols):
        u, v = _get_householder_vectors(matrix, i)
        if not u or u == v:  # 'u' and 'v' may be empty when called for the last column. hint: check for square matrix
            continue
        w = g.divide_by_scalar(g.subtract(u, v), vector.norm(g.subtract(u, v)))
        temp = g.multiply_by_scalar(g.multiply(w, g.transpose(w)), 2)
        H = g.subtract(identity_matrix(len(w)), temp)
        H = _add_identity_matrix_to_householder(H, i)
        Q = g.multiply(H, Q)
        matrix = g.multiply(H, matrix)

    return g.transpose(Q), matrix


def diagonal(matrix):
    rows, cols = g.shape(matrix)
    if rows > cols:
        diag = [matrix[i][i] for i in range(cols)]
    else:
        diag = [matrix[i][i] for i in range(rows)]

    return vector.transform_to_column_vector(diag)


def _add_column_of_zeros(matrix):
    for i in range(len(matrix)):
        matrix[i].append(0)


def eigenvalues(matrix, algo='householder'):
    # returns a row vector of eigenvalues
    if algo == 'householder':
        qr_algo = qr_householder
    else:
        qr_algo = qr_gram_schmidt

    size = check_for_square(matrix)
    if size == -1:
        print("can not find eigenvalues of a non-square matrix!")
        return

    change = 1
    E = diagonal(matrix)
    steps = 0
    while change > 0.00000001:
        if steps > 50000:    # eigenvalues are not converging
            return None
        Eold = E
        Q, R = qr_algo(matrix)
        matrix = g.multiply(R, Q)
        E = diagonal(matrix)
        change = vector.norm(g.subtract(E, Eold))
        steps += 1

    return vector.transform_to_row_vector(diagonal(matrix))


def eig(matrix, algo='householder'):
    # works only for real eigenvalues and eigenvectors
    # returns a row vector of eigenvalues and a matrix having eigenvectors along its columns
    # if can not converge eigenvalues it returns ev = -1 and ew = [[0]]

    size = check_for_square(matrix)
    if size == -1:
        print("can not find eigenvalues and eigenvectors of a non-square matrix")
        return

    ev = eigenvalues(matrix, algo)  # ev is a row vector of eigenvalues
    if ev is None:  # return 0 eigenvector
        return -1, [[0]]

    ew = []  # ew is the matrix that has eigenvectors in its column
    for i in range(len(ev)):
        M = g.subtract(matrix, g.multiply_by_scalar(identity_matrix(size), ev[i]))
        _add_column_of_zeros(M)
        M = reduced_row_echelon_form(M)
        x = gauss_elimination(M)  # x is one eigen vector corrosponding to ev[i]
        x = vector.orthonormal(vector.transform_to_column_vector(x))  # making x a unit vector
        ew.append(vector.transform_to_row_vector(x))

    ew = g.transpose(ew)
    return ev, ew


if __name__ == '__main__':

    E = [[2, 3, -1, 0],
         [1, -1, -2, 0],
         [3, 1, 3, 0]]

    S = [[1, -2, 3],
         [1, 2, 1],
         [-1, 2, -3]]

    G = [[5, 4, 3, 9, 7],
         [4, 1, 8, 4, 2],
         [3, 8, 12, 14, 9],
         [9, 4, 14, 5, 10],
         [7, 2, 9, 10, 3]]

    _P = [[0, 0, -2],
          [0, -2, 0],
          [-2, 0, 3]]

    Z = [[3, -1, 2, 0],
         [4, 3, 3, 0],
         [5, 7, 4, 0]]

    _Z = [[2, -1, 2, 0],
          [5, 3, -1, 0],
          [1, 5, -5, 0]]

    K = [[2, 3, -1, 0],
         [1, -1, -2, 0],
         [3, 1, 3, 0]]

    M = [[4, 1, -2],
         [1, 1, 0],
         [-2, -5, 3],
         [2, 1, -2]]

    Y = [[0.8147, 0.0975, 0.1576],
         [0.9058, 0.2785, 0.9706],
         [0.1270, 0.5469, 0.9572],
         [0.9134, 0.9575, 0.4854],
         [0.6324, 0.9649, 0.8003]]

    _Y = [[0.8147, 0.0975, 0.1576],
          [0.9058, 0.2785, 0.9706],
          [0.1270, 0.5469, 0.9572]]

    A = [[1, 2, 3, 4, 5],
         [0, 2, 3, 4, 5],
         [0, 0, 3, 4, 5],
         [0, 0, 0, 4, 5],
         [0, 0, 0, 0, 5]]

    H = [[0, 1, 2],
         [1, 2, 1],
         [2, 7, 8]]

    J = [[1, 2],
         [4, 3]]

    N = [[2, 4, 6, 0],
         [4, 5, 6, 0],
         [3, 1, -2, 0]]

    P = [[25, 5, 1, 106.8],
         [64, 8, 1, 177.2],
         [144, 12, 1, 279.2]]

    _O = [[1, 2, 3, 2, 0],
          [1, 3, 5, 5, 0],
          [2, 4, 7, 1, 0],
          [-1, -2, -6, 7, 0]]

    M = [[1, 4, 6],     # this matrix has complex eigenvalues and eigenvectors
         [5, 7, 8],
         [0, 7, 8]]

    print("calculated by numpy: ")
    a = np.array(G)
    val_, vec_ = np.linalg.eig(a)
    print(val_)
    print()
    g.show_matrix(vec_)
    print()

    print("calculated by me:")
    val, vec = eig(G)
    print(val)
    print()
    g.show_matrix(vec)
    print()
