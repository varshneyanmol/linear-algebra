def shape(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    return num_rows, num_cols


def show_matrix(matrix):
    for row in matrix:
        print(row)


def multiply_by_scalar(matrix, k):
    rows, cols = shape(matrix)
    M = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(k*matrix[i][j])
        M.append(row)
    return M


def divide_by_scalar(matrix, k):
    rows, cols = shape(matrix)
    D = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(matrix[i][j] / k)
        D.append(row)
    return D


def transpose(matrix):
    num_rows, num_cols = shape(matrix)
    T = []
    for j in range(num_cols):
        transposed_row = []
        for i in range(num_rows):
            transposed_row.append(matrix[i][j])
        T.append(transposed_row)

    return T


def add(matrix1, matrix2):
    rows1, cols1 = shape(matrix1)
    rows2, cols2 = shape(matrix2)
    if rows1 != rows2 or cols1 != cols2:
        print("can not add matrices as dimentions not matched")
        return -1
    Add = []
    for i in range(rows1):
        row = []
        for j in range(cols1):
            row.append(matrix1[i][j] + matrix2[i][j])
        Add.append(row)
    return Add


def subtract(matrix1, matrix2):
    rows1, cols1 = shape(matrix1)
    rows2, cols2 = shape(matrix2)
    if rows1 != rows2 or cols1 != cols2:
        print("can not subtract matrices as dimentions not matched")
        return -1
    Sub = []
    for i in range(rows1):
        row = []
        for j in range(cols1):
            row.append(matrix1[i][j] - matrix2[i][j])
        Sub.append(row)
    return Sub


def multiply(matrix1, matrix2):
    rows1, cols1 = shape(matrix1)
    rows2, cols2 = shape(matrix2)
    if cols1 != rows2:
        print("dimensions of matrices not matching for multiplications")
        return -1
    M = []
    for i in range(rows1):
        row = []
        for j in range(cols2):
            sum = 0
            for k in range(rows2):
                sum += matrix1[i][k] * matrix2[k][j]
            row.append(sum)
        M.append(row)
    return M
