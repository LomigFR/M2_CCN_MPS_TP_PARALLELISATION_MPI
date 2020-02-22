from mpi4py import MPI
import numpy as np
import math
import time

# Generates a square matrix filled with 0
# As is, the program works for (N+4)%4 == 0
def create_square_zero_matrix(N):
    return np.zeros((N, N))

# Generates a square matrix with 2 paddings of one layer each
def create_square_matrix_pad(matrix, pad_width1, pad_width2, value1, value2):
    temp_matrix = np.pad(matrix, pad_width=pad_width1, mode='constant',
                         constant_values=value1)
    matrix_pad = np.pad(temp_matrix, pad_width=pad_width2, mode='constant',
                         constant_values=value2)
    return matrix_pad

# Generates small matrices filled with 0 and whose size is compatible with
# that of the main matrix
def create_snippet_matrix(matrice_to_scat, node_number):
    tuple_A_pad = matrice_to_scat.shape
    nb_lignes = tuple_A_pad[0]
    nb_colonnes = tuple_A_pad[1]
    snippet = np.zeros([int(nb_lignes/node_number), nb_colonnes])
    return snippet

# Performs convolution calculations and fills a secondaray matrix
# with the results
def calculate_convolution(matrix_snippet):
    temp_matrix = np.zeros((matrix_snippet.shape[0]-2, 
                            matrix_snippet.shape[1]-2))
    for i in range(temp_matrix.shape[0]):
        for j in range(temp_matrix.shape[1]):
            new_i = i + 1
            new_j = j + 1
            temp_matrix[i, j] = 0.25*(new_snippet[new_i+1, new_j]
            + new_snippet[new_i-1, new_j]
            + new_snippet[new_i, new_j+1]
            + new_snippet[new_i, new_j-1])
    return temp_matrix

# Function that calculates the normalized difference in the way of the example
# of code provided
def calculate_sqrt_diff_norm(matrix_snippet, B_matrix):
    square_diff_norm = 0.0
    for i in range(B_matrix.shape[0]):
        for j in range(B_matrix.shape[1]):
            new_i = i + 1
            new_j = j + 1
            square_diff_norm += math.sqrt(
                    (B_matrix[i, j] - matrix_snippet[new_i, new_j])
                    *(B_matrix[i, j] - matrix_snippet[new_i, new_j]))
    return square_diff_norm

# Calculates the normalized difference via the absolute value. This function
# was created with the objective of comparing the performance of this
# program vs. the calculation provided in the example code (using sqrt).
# As it is, the lines concerned are commented for each node.
def calculate_abs_diff_norm(matrix_snippet, B_matrix):
    abs_diff_norm = 0.0
    for i in range(B_matrix.shape[0]):
        for j in range(B_matrix.shape[1]):
            new_i = i + 1
            new_j = j + 1
            abs_diff_norm += abs(
            (B_matrix[i, j] - matrix_snippet[new_i, new_j]))
    return abs_diff_norm

# To copy matrix B into the snippet matrix for the node 0
def copy_B_matrix_into_snippet_0(snippet_node_0, B_matrix):
    for i in range(B_matrix.shape[0]):
        for j in range(B_matrix.shape[1]):
            new_i = i + 1
            new_j = j + 1
            snippet_node_0[new_i, new_j] = B_matrix[i, j]
    return snippet_node_0

# To copy matrix B into the snippet matrix for the middle nodes
def copy_B_matrix_into_other_snippet(snippet, B_matrix):
    nb_columns = B_matrix.shape[1]
    for i in range(B_matrix.shape[0]):
        snippet[i, 1:nb_columns+1] = B_matrix[i, :]
    return snippet

# Calculates the total normalized difference
def calculate_diffnorm_total(value_diff):
    diff_from_1 = comm.recv(source=1)
    diff_from_2 = comm.recv(source=2)
    diff_from_3 = comm.recv(source=3)
    diff_total = value_diff + diff_from_1 + diff_from_2 + diff_from_3
    return diff_total

# Toggles "converge" variable to True if the total normalized difference
# allows it.
def check_diff_total(diff_total, converge):
    if diff_total <= 0.01:
        converge = True
    return converge

# Function that ensures the reconstruction of the final matrix (with only the
# padding of 1), once convergence is complete
def build_final_matrix(new_snippet):
    new_snippet = np.delete(new_snippet, (0), axis=1)
    new_snippet = np.delete(new_snippet, (-1), axis=1)
    new_snippet = np.delete(new_snippet, (0), axis=0)
    new_snippet = np.delete(new_snippet, (-1), axis=0)
    final_matrix = comm.gather(new_snippet, root=0)
    return final_matrix

# To display the final matrix
def display_final_matrix(matrix):
    if rank == 0:
        # print("Final matrix :", matrix)
        temp_matrix = np.insert(matrix[0], -1, matrix[1], axis=0)
        temp_matrix = np.insert(temp_matrix, -1, matrix[2], axis=0)
        temp_matrix = np.insert(temp_matrix, -1, matrix[3], axis=0)
        print("Dimensions of the final matrix (includes previous ones-padding):",
              temp_matrix.shape)
        print("Number of iterations :", iteration_num)

# To display the program execution time
def display_time(start, end):
    print("Execution time for node number", rank, " :", end - start, " seconds")


###############################################################################


comm = MPI.COMM_WORLD
node_number = comm.Get_size()
rank = comm.Get_rank()
converge = False
iteration_num = 0

start = time.time()

temp_matrix_to_scat = create_square_zero_matrix(60)
matrix_to_scat = create_square_matrix_pad(temp_matrix_to_scat, 1, 1, 1, 0)
snippet = create_snippet_matrix(matrix_to_scat, node_number)
comm.Scatter(matrix_to_scat, snippet, root=0)

while(converge == False): 
    if rank == 0:
        iteration_num = iteration_num+1
        comm.send(snippet[-1], dest=1)
        received_from_bottom = comm.recv(source=1)
        new_snippet = np.insert(snippet, -1, received_from_bottom, axis=0)

        B_matrix = calculate_convolution(new_snippet)
        square_diff_norm = calculate_sqrt_diff_norm(new_snippet, B_matrix)
#        abs_diff_norm = calculate_abs_diff_norm(new_snippet, B_matrix)
        snippet = copy_B_matrix_into_snippet_0(snippet, B_matrix)

        diff_total = calculate_diffnorm_total(square_diff_norm)
        print("Total normalized difference:", diff_total)

        converge = check_diff_total(diff_total, converge)

    elif rank == 1:
        received_from_top = comm.recv(source=0)
        comm.send(snippet[0], dest=0)
        comm.send(snippet[-1], dest=2)
        received_from_bottom = comm.recv(source=2)
        temp_snippet = np.insert(snippet, 0, received_from_top, axis=0)
        new_snippet = np.insert(temp_snippet, -1, received_from_bottom, axis=0)

        B_matrix = calculate_convolution(new_snippet)
        square_diff_norm = calculate_sqrt_diff_norm(new_snippet, B_matrix)
#        abs_diff_norm = calculate_abs_diff_norm(new_snippet, B_matrix)
        snippet = copy_B_matrix_into_other_snippet(snippet, B_matrix)

        comm.send(square_diff_norm, dest=0)
    
    elif rank == 2:
        comm.send(snippet[-1], dest=3)
        received_from_bottom = comm.recv(source=3)
        received_from_top = comm.recv(source=1)
        comm.send(snippet[0], dest=1)
        temp_snippet = np.insert(snippet, 0, received_from_top, axis=0)
        new_snippet = np.insert(temp_snippet, -1, received_from_bottom, axis=0)

        B_matrix = calculate_convolution(new_snippet)
        square_diff_norm = calculate_sqrt_diff_norm(new_snippet, B_matrix)
#        abs_diff_norm = calculate_abs_diff_norm(new_snippet, B_matrix)
        snippet = copy_B_matrix_into_other_snippet(snippet, B_matrix)

        comm.send(square_diff_norm, dest=0)

    else:
        received_from_top = comm.recv(source=2)
        comm.send(snippet[0], dest=2)
        new_snippet = np.insert(snippet, 0, received_from_top, axis=0)

        B_matrix = calculate_convolution(new_snippet)
        square_diff_norm = calculate_sqrt_diff_norm(new_snippet, B_matrix)
#        abs_diff_norm = calculate_abs_diff_norm(new_snippet, B_matrix)
        snippet = copy_B_matrix_into_other_snippet(snippet, B_matrix)

        comm.send(square_diff_norm, dest=0)

    converge = comm.bcast(converge, 0)


###############################################################################


# Reconstruction of the final matrix and display of its contents and its
# dimensions
final_matrix = build_final_matrix(new_snippet)
display_final_matrix(final_matrix)

end = time.time()
display_time(start, end)