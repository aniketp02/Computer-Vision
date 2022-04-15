from SlowMatrix import SlowMatrix

def test_mul():
	m1 = SlowMatrix([[1, 2, 3], [4, 1, 6], [2, 1, 8]])
	m2 = SlowMatrix([[3, 4, 5], [4, 2, 2], [1, 2, 5]])

	assert (m1 @ m2) == SlowMatrix([[14, 14, 24], [22, 30, 52], [18, 26, 52]])


	m1 = SlowMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

	assert (m1 @ m2) == m2

	m1 = SlowMatrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
	assert (m1 @ m2) == SlowMatrix([[8, 8, 12], [8, 8, 12], [8, 8, 12]])

def test_add():
	m1 = SlowMatrix([[1, 2, 3], [4, 1, 6], [2, 1, 8]])
	m2 = SlowMatrix([[3, 4, 5], [4, 2, 2], [1, 2, 5]])

	assert (m1 + m2) == SlowMatrix([[4, 6, 8], [8, 3, 8], [3, 3, 13]])