import numpy as np

def solveLinearSystem(A, b):
    if A.shape[0] != A.shape[1]:
        print("This solver is not intended to solve systems with asymmetric A's! ")
        return

    n = b.shape[0]
    augmentedMatrix = createAugmentedMatrix(A, b)
    echelonFormedMatrix = gaussianEliminationWithPartialPivoting(augmentedMatrix, n)
    solutionVector = backSubstitute(echelonFormedMatrix, n)
    return solutionVector

def createAugmentedMatrix(A, b):
    b_reshaped = b.reshape(-1, 1)
    return np.concatenate((A, b_reshaped), axis=1, dtype=float)


def gaussianEliminationWithPartialPivoting(augmentedMatrix, n):
  current_row = 0
  while current_row < n:
    # Find the pivot row (row with the largest absolute value in the current column)
    pivot_row = current_row
    for candidate_row in range(current_row + 1, n):
      if abs(augmentedMatrix[candidate_row, current_row]) > abs(augmentedMatrix[pivot_row, current_row]):
        pivot_row = candidate_row

    # Swap the current row with the pivot row if needed
    if pivot_row != current_row:
      augmentedMatrix[[pivot_row, current_row]] = augmentedMatrix[[current_row, pivot_row]]

    # Check for division by zero error
    if augmentedMatrix[current_row, current_row] == 0.0:
      raise Exception("Division by 0 error")

    # Eliminate elements below the pivot element
    for j in range(current_row + 1, n):
      scaling_factor = augmentedMatrix[j][current_row] / augmentedMatrix[current_row][current_row]
      augmentedMatrix[j] = augmentedMatrix[j] - (scaling_factor * augmentedMatrix[current_row])

    current_row += 1

  return augmentedMatrix

def backSubstitute(echelonFormedSystem, n):
    solutionVector = np.zeros(n)
    solutionVector[n - 1] = echelonFormedSystem[n - 1][n] / echelonFormedSystem[n - 1][n - 1]
    for k in range(n - 2, -1, -1):
        solutionVector[k] = echelonFormedSystem[k][n]
        for j in range(k + 1, n):
            solutionVector[k] = solutionVector[k] - echelonFormedSystem[k][j] * solutionVector[j]
        solutionVector[k] = solutionVector[k] / echelonFormedSystem[k][k]
    return solutionVector

def isConsistent(A, b):
    augmentedMatrix = createAugmentedMatrix(A, b)
    echelonForm = gaussianEliminationWithPartialPivoting(augmentedMatrix, b.shape[0])
    r = 1e-8
    for row in echelonForm:
        if all(abs(val) <= r for val in row[:-1]) and not (abs(row[-1]) <= r):
            return False
    return True

def solveNutrients():
    A = np.array([[2.5/10, 0.5/10, 1.5/10, 1/10],
                  [0.1/10, 6.5/10, 3/10, 2/10],
                  [0.1/10, 0.5/10, 1.5/10,1.5/10 ],
                  [0.1/10, 0.1/10, 0.5/10, 7/10]])
    b = np.array([[50],
                  [180],
                  [30],
                  [60]])
    x = solveLinearSystem(A,b)
    xRounded = np.round(x)
    return xRounded


