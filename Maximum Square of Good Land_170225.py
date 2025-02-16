'''
Maximum Square of Good Land
A farmer wants to farm their land with the maximum area where good land is present.

The land is represented as a binary matrix with 1s and 0s, where:

1 represents good land.
0 represents bad land.
The farmer only wants to farm in a square-shaped area of good land with the maximum area.

Write a program to find the largest square of 1s in the given matrix and return the area of that square.

'''

# def largest_square_of_ones(matrix):
#     if not matrix or not matrix[0]:
#         return 0

#     rows = len(matrix)
#     print('Number of Rows -', rows)
#     cols = len(matrix[0])
#     print('Number of Cols -', cols)
    
#     # dp[i][j] will store the side length of the largest square
#     # that ends at cell (i, j)
#     dp = [[0] * cols for _ in range(rows)]
    
#     max_side = 0
    
#     for i in range(rows):
#         for j in range(cols):
#             if matrix[i][j] == 1:
#                 if i == 0 or j == 0:
#                     # First row or first column can only form a 1x1 square
#                     dp[i][j] = 1
#                 else:
#                     # Take the min of three neighbors + 1
#                     dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
                
#                 # Update maximum side length found
#                 max_side = max(max_side, dp[i][j])
#             else:
#                 dp[i][j] = 0
    
#     # The area of the largest square
#     return max_side * max_side

# # Example usage:
# matrix_example = [
#     [0, 1, 1, 0, 1],
#     [1, 0, 1, 0, 1],
#     [1, 1, 1, 1, 1],
#     [0, 1, 1, 1, 0],
#     [1, 1, 1, 0, 1]
# ]

# print(largest_square_of_ones(matrix_example)) # Output: 4

def maximalRectangle(matrix):
    if not matrix or not matrix[0]:
        return 0
    
    rows, cols = len(matrix), len(matrix[0])
    heights = [0] * cols  # Histogram heights
    #print('Number of heights -', heights)
    max_area = 0

    def largestRectangleArea(heights):
        stack = []  # Stack holds indices of heights
        max_area = 0
        heights.append(0)  # Append a zero height to flush remaining stack at the end
        print('Number of heights -', heights)
        for i, h in enumerate(heights):
            while stack and h < heights[stack[-1]]:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        
        heights.pop()  # Restore original array
        return max_area

    for row in matrix:
        for j in range(cols):
            heights[j] = heights[j] + 1 if row[j] == 1 else 0
        max_area = max(max_area, largestRectangleArea(heights))

    return max_area

# Example Usage
matrix_example = [
    [0, 1, 1, 0, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 0, 1]
]

print(maximalRectangle(matrix_example))  # Output: 10
