class SlowMatrix:

	## The constructor
	#@param matrix A 2d Python list containing data
	def __init__(self, matrix):
		self.matrix = matrix
		#pass

	## Matrix multiplication
	# @param self SlowMatrix1
	# @param mat2 SlowMatrix2
	def __matmul__(self, mat2):

		result = [[sum(a * b for a, b in zip(A_row, B_col)) 
						for B_col in zip(*mat2)]
                            for A_row in self.matrix]
		
		return result
		#pass

	## Element wise multiplication
	# @param self SlowMatrix1
	# @param mat2 SlowMatrix2
	def __mul__(self, mat2):

		result = []

		for l1, l2 in zip(self.matrix, mat2):
			res = [a*b for a, b in zip(l1, l2)]
			result.append(res)
		

		return result
		#pass

	## Element wise addition
	# @param self SlowMatrix1
	# @param mat2 SlowMatrix2
	def __add__(self, mat2):

		result = []

		for l1, l2 in zip(self.matrix, mat2):
			res = [a+b for a, b in zip(l1, l2)]
			result.append(res)
		

		return result
		#pass

	## Element wise subtraction
	# @param self SlowMatrix1
	# @param mat2 SlowMatrix2
	def __sub__(self, mat2):

		result = []

		for l1, l2 in zip(self.matrix, mat2):
			res = [a-b for a, b in zip(l1, l2)]
			result.append(res)
		

		return result
		#pass

	## Equality operator
	# @param self SlowMatrix1
	# @param mat2 SlowMatrix2
	def __eq__(self, mat2):

		result = []

		for l1, l2 in zip(self.matrix, mat2):
			res = [a == b for a, b in zip(l1, l2)]
			result.append(res)
			
		return result
		#pass

	## Calculate transpose
	def transpose(self):

		result = []

		for l in zip(*self.matrix):
			result.append(l)

		return result
		#pass

	## Creates a SlowMatrix of 1s
	# @param shape A python pair (row, col)
	def ones(shape):
		rows, cols = shape
		A = [[1]*cols]*rows
		
		return A
		#pass

	## Creates a SlowMatrix of 0s
	# @param shape A python pair (row, col)
	def zeros(shape):
		rows, cols = shape
		A = [[0]*cols]*rows
		
		return A
		#pass

	## Returns i,jth element
	# @param key A python pair (i,j)
	def __getitem__(self, key):
		i = key
		return self.matrix[i]
		#pass

	## Sets i,jth element
	# @param key A python pair (i,j)
	# #param value Value to set
	def __setitem__(self, key, value):
		i, j = key
		self.matrix[i][j] = value
		#pass

	## Converts SlowMatrix to a Python string
	def __str__(self):
		A = ''
		for i in range(len(self.matrix[:])):
			for j in range(len(self.matrix[:][0])):
				A += str(self.matrix[i][j])
		
		return A
		#pass
