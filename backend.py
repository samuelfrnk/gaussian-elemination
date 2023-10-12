import numpy as np

# Task a)
# Implement the gaussian elimination method, to solve the given system of linear equations;
# Add partial pivoting to increase accuracy and stability of the solution;
# Return the solution for x.
def solveLinearSystem(A, b):
  #print(A)
  #print(b)
  #print(A.shape[1])
  if A.shape[0] != A.shape[1]:
    print("This solver is not intended to solve systems with asymmetric A's! ")
    return

  #if b.shape[0] != A.shape[0] or b.shape[1] != 1:
  #  print("the provided vector is not valid! ")
  #  return

  numberOfVariables = b.shape[0]



  augmentedMatrix = createAugmentedMatrix(A,b)

  augmentedMatrixInEchelonForm = gaussianElimination(augmentedMatrix, numberOfVariables)
  #print(augmentedMatrixInEchelonForm)
  solutionVector = backSubstitute(augmentedMatrixInEchelonForm, numberOfVariables)
  #print(solutionVector)
  return solutionVector

def createAugmentedMatrix(A, b):
  #We are going to create the augmented matrix and set the data type to float
  print(A)
  print(b)
  b_reshaped = b.reshape(-1, 1)
  return np.concatenate((A,b_reshaped),axis=1, dtype=float)

def gaussianElimination(augmentedMatrix, numberOfVariables):
  row = 0
  while row < numberOfVariables:
    for entry in range(row +1, numberOfVariables):
      if abs(augmentedMatrix[row,row]) < abs(augmentedMatrix[entry,row]):
        augmentedMatrix[[entry,row]] = augmentedMatrix[[row,entry]]

    if augmentedMatrix[row, row] == 0.0:
      raise Exception("Division by 0 error")

    for j in range(row+1, numberOfVariables):
      scaling_factor = augmentedMatrix[j][row] / augmentedMatrix[row][row]
      augmentedMatrix[j] = augmentedMatrix[j] - (scaling_factor * augmentedMatrix[row])
      #print(augmentedMatrix)

    row = row + 1
  return augmentedMatrix

def backSubstitute(echelonFormedSystem, numberOfVariables):
  solutionVector = np.zeros(numberOfVariables)
  solutionVector[numberOfVariables-1] =echelonFormedSystem[numberOfVariables-1][numberOfVariables] / echelonFormedSystem[numberOfVariables-1][numberOfVariables-1]
  for k in range(numberOfVariables - 2, -1,-1):
    solutionVector[k] = echelonFormedSystem[k][numberOfVariables]
    for j in range(k+1,numberOfVariables):
      solutionVector[k] = solutionVector[k] - echelonFormedSystem[k][j] * solutionVector[j]
    solutionVector[k] = solutionVector[k] / echelonFormedSystem[k][k]
  return solutionVector




# Task b)
# Implement a method, checking whether the system is consistent or not;
# Obviously, you're not allowed to use any method solving that problem for you.
# Return either true or false
def isConsistent(A,b):
  #print(A)
  return 0


# Task c)
# Implement a method to compute the daily amounts of chicken breast, brown rice, black beans and avocado to eat to achieve the daily nutritional intake described in the exercise;
# Return a vector x with the grams of chicken breast, brown rice, black beans and avocado to eat each day.
def solveNutrients():
  return np.ones(4)


b = np.random.randint(1, 81, size=(3, 1))
A = np.random.randint(1, 81, size=(3, 3))
#print(A)
#print(b)
solveLinearSystem(A,b)

