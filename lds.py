def longest_decreasing_subsequence_ids(maes):
    n = len(maes)
    # Initialize the LDS array and the predecessor array
    lds = [1] * n
    prev = [-1] * n

    # Compute the LDS values and track predecessors
    for i in range(1, n):
        for j in range(i):
            if maes[i] < maes[j] and lds[i] < lds[j] + 1:
                lds[i] = lds[j] + 1
                prev[i] = j

    # Find the index of the maximum value in LDS
    max_length = max(lds)
    max_index = lds.index(max_length)

    # Reconstruct the LDS by backtracking through the prev array
    lds_ids = []
    current_index = max_index
    while current_index != -1:
        lds_ids.append(current_index)
        current_index = prev[current_index]

    lds_ids.reverse()  # Reverse to get the correct order
    lds_seq = [maes[id] for id in lds_ids]  # Get the MAE values for the LDS IDs

    return lds_seq, lds_ids, max_length

# Example usage
maes = [0.1, 0.3, 0.2, 0.15, 0.18, 0.16, 0.14, 0.12, 0.1, 0.09, 0.15, 0.08]
lds_seq, lds_ids, length = longest_decreasing_subsequence_ids(maes)
print("Longest Decreasing Subsequence (non-contiguous):", lds_seq)
print("IDs:", lds_ids)
print("Length:", length)
