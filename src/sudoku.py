# in this file the sudoku class is defined
# 6x6 sudoku is solved using backtracking algorithm

import numpy as np
import random
from copy import deepcopy
import time
import sys
import matplotlib.pyplot as plt

class Sudoku:
    def __init__(self):
        print("Initializing Sudoku...")
        # 6x6 grid initialized to 0 (empty cells)
        self.grid = [[0]*6 for _ in range(6)]
        self.fill()
        self.solvedGrid = deepcopy(self.grid)
        # self.generate_sudoku()


    def getIncreasingLines(self):
        """Get all orthogonally connected lines where the values are increasing. Use the solved grid."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # All 8 directions
    
        rows = len(self.solvedGrid)
        cols = len(self.solvedGrid[0])
        
        # Helper function for checking if a position is within bounds
        def is_valid(x, y):
            return 0 <= x < rows and 0 <= y < cols
        
        # DFS function to explore increasing paths from a given cell
        def dfs(x, y, path):
            current_value = self.solvedGrid[x][y]
            path.append((x, y))  # Add current position to the path
            
            # Track whether a valid larger neighbor is found
            found_larger = False
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny) and self.solvedGrid[nx][ny] > current_value:
                    found_larger = True
                    dfs(nx, ny, path[:])  # Recursively search from this neighbor
            
            # If no larger neighbor is found, append the path if it has more than 1 element
            if not found_larger and len(path) > 1:
                results.append(path)
        
        results = []
        
        # Explore increasing paths starting from each cell
        for i in range(rows):
            for j in range(cols):
                dfs(i, j, [])
        
        return results
    
    def getEvenOddLines(self):
        """Get all orthogonally connected lines where the values alternate between even and odd."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Orthogonal directions (up, down, left, right)
        
        rows = len(self.solvedGrid)
        cols = len(self.solvedGrid[0])
        
        # Helper function for checking if a position is within bounds
        def is_valid(x, y):
            return 0 <= x < rows and 0 <= y < cols
        
        def is_even(num):
            return num % 2 == 0
        
        # DFS function to explore paths from a given cell
        def dfs(x, y, path, visited):
            current_value = self.solvedGrid[x][y]
            path.append((x, y))  # Add current position to the path
            visited[x][y] = True  # Mark the current cell as visited
            
            found_larger = False
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny) and not visited[nx][ny] and is_even(self.solvedGrid[nx][ny]) != is_even(current_value):
                    found_larger = True
                    dfs(nx, ny, path[:], visited)  # Recursively search from this neighbor
            
            # If no valid neighbor is found, append the path if it has more than 1 element
            if not found_larger and len(path) > 1:
                results.append(path)
            
            # Unmark this cell to allow it to be used in future paths starting from different cells
            visited[x][y] = False
        
        results = []
    
        # Initialize a visited matrix to keep track of visited cells during each DFS
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        
        # Start DFS from each cell
        for i in range(rows):
            for j in range(cols):
                dfs(i, j, [], visited)
        
        return results
    
    def getRandomLines(self, lines, num_lines=10, lengthRange=(3, 6)):
        """Get a random sample of lines from the list of lines which dont overlap."""
        visited = [[False]*6 for _ in range(6)]
        random.shuffle(lines)
        result = []
        count = 0
        for line in lines:
            if count >= num_lines:
                break
            if len(line) >= lengthRange[0] and len(line) <= lengthRange[1] and not any(visited[i][j] for i, j in line):
                result.append(line)
                # Mark the cells in the line as visited
                for i, j in line:
                    visited[i][j] = True
                count += 1
        return result


    def count_solutions(self):
        """Count the number of solutions for the current grid."""
        count = 0
        for row in range(6):
            for col in range(6):
                if self.grid[row][col] == 0:
                    for num in range(1, 7):
                        if self.is_valid(self.grid, row, col, num):
                            self.grid[row][col] = num
                            count += self.count_solutions()
                            self.grid[row][col] = 0
                    return count
        return 1

    def is_valid(self, grid, row, col, num):
        """Check if placing num in grid[row][col] is valid."""
        # Check if num is in the current row or column
        for i in range(6):
            if grid[row][i] == num or grid[i][col] == num:
                return False

        # Check if num is in the 2x3 sub-grid (regions)
        start_row, start_col = 2 * (row // 2), 3 * (col // 3)
        for i in range(2):
            for j in range(3):
                if grid[start_row + i][start_col + j] == num:
                    return False
        return True
    
    def fill(self):
        """Recursive backtracking to randomly fill the grid with valid values."""
        for row in range(6):
            for col in range(6):
                if self.grid[row][col] == 0:
                    random_nums = list(range(1, 7))
                    random.shuffle(random_nums)  # Shuffle numbers to ensure randomness
                    
                    for num in random_nums:
                        if self.is_valid(self.grid, row, col, num):
                            self.grid[row][col] = num
                            if self.fill():
                                return True
                            self.grid[row][col] = 0
                    return False
        return True

    def display(self, solved=False):
        """Displays the Sudoku grid."""
        grid_to_display = self.solvedGrid if solved else self.grid
        # nice visual representation of the grid with grid lines
        for i in range(6):
            if i % 2 == 0:
                print("+" + "-----------+"*2)
            # else:
                # print("|" + "       |"*2)
            for j in range(6):
                if j % 3 == 0:
                    print("| ", end=" ")
                print(grid_to_display[i][j] if grid_to_display[i][j] != 0 else "_", end="  ")
            print("|")
        print("+" + "-----------+"*2)

    def displayAsPlot(self, lines=None, solved=False):
        """Displays the Sudoku grid indicating the indices on the edge and displaying the 2x3 grids and values."""
        grid_to_display = self.solvedGrid if solved else self.grid

        fig, ax = plt.subplots(figsize=(6, 6))

        # Set white background and draw grid
        ax.matshow(np.ones((6, 6)), cmap='Greys', vmin=1, vmax=1)  # white background

        # Set ticks at the border to show indices (0 to 5)
        ax.set_xticks(np.arange(6))
        ax.set_yticks(np.arange(6))
        ax.set_xticklabels([f"{i}" for i in range(0, 6)])
        ax.set_yticklabels([f"{i}" for i in range(0, 6)])
        
        # Draw 2x3 subgrid lines with thicker lines
        for i in range(7):
            lw = 3 if i % 2 == 0 else 1  # Thicker horizontal lines for subgrids (every 2 rows)
            ax.axhline(i - 0.5, color='black', lw=lw)
            
            lw = 3 if i % 3 == 0 else 1  # Thicker vertical lines for subgrids (every 3 columns)
            ax.axvline(i - 0.5, color='black', lw=lw)
        
        # Fill the grid with values from the Sudoku matrix
        for i in range(6):
            for j in range(6):
                value = grid_to_display[i][j]
                if value != 0:  # Only display non-zero values
                    ax.text(j, i, str(value), va='center', ha='center', fontsize=16)
        
        # Hide tick marks but keep labels
        ax.tick_params(axis='x', which='both', length=0)  # Remove x-axis tick indicators
        ax.tick_params(axis='y', which='both', left=False)  # Remove y-axis tick indicators

        ax.set_xlim(-0.5, 5.5)  # Limit for the x-axis
        ax.set_ylim(5.5, -0.5)

        # Display lines
        if lines is not None:
            for line in lines:
                for i in range(len(line) - 1):
                    (row1, col1), (row2, col2) = line[i], line[i+1]
                    ax.plot([col1, col2], [row1, row2], color="#cc222299", lw=2)

        # Adjust the grid spacing and flip the y-axis (if desired)
        plt.tight_layout()
        plt.show()

    def countEmpty(self):
        """Count the number of empty cells in the grid."""
        count = 0
        for row in self.grid:
            count += row.count(0)
        return count

    def removeValues(self, num, maxIterations=1000):
        """Remove num random values from the grid if it just has one unique solution."""
        count = 0
        removed = 0
        while removed < num and count < maxIterations:
            row, col = random.randint(0, 5), random.randint(0, 5)
            if self.grid[row][col] != 0:
                temp = self.grid[row][col]
                self.grid[row][col] = 0
                if self.count_solutions() != 1:
                    self.grid[row][col] = temp
                else:
                    removed += 1
            count += 1


class HelloWorld:
    def __init__(self):
        print("Hello World")

# Example usage
# sudoku = Sudoku()
# sudoku.display()

    

        