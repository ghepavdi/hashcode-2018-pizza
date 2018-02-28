def possible_kernels(filename):
	if filename == "example.in":
		return [(1, 2), (2, 1)]
	elif filename == "small.in":
		return [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (1, 4), (4, 1), (1, 5), (5, 1)]
	elif filename == "medium.in":
		return [(4, 2), (2, 4), (1, 8), (8, 1), (3, 3), (9, 1), (1, 9), (1, 10), (10, 1), (1, 11), (11, 1), (12, 1), (12, 1)]
	elif filename == "big.in":
		return [(1, 14), (14, 1), (2, 7), (7, 2), (6, 2), (2, 6), (1, 12), (12, 1), (3, 4), (4, 3), (1, 13), (13, 1)]