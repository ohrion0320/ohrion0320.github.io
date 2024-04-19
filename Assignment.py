import numpy as np

# Coefficient matrix
A = np.array([[1, -4, -2],
              [2, 1, 2],
              [3, 2, -1]], dtype=float) 
# Right-hand side vector
b = np.array([21, 3, -2], dtype=float) 

# Augmented matrix
Ab = np.hstack((A, b.reshape(len(b), 1)))

# Forward elimination
n = len(b)
for i in range(n):
    # Partial pivoting
    p_row = max(range(i, n), key=lambda j: abs(Ab[j, i]))
    Ab[[i, p_row]] = Ab[[p_row, i]]
    
    # Elimination
    for j in range(i+1, n):
        factor = Ab[j, i] / Ab[i, i]
        Ab[j, :] -= factor * Ab[i, :]

# Back substitution
x = np.zeros(n)
for i in range(n-1, -1, -1):
    x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]

print("Solution:", x)