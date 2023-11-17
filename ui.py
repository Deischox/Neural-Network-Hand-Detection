import pygame
import sys
import numpy as np
from network import predictDrawing

# Constants for the grid
GRID_SIZE = 28
GRID_WIDTH = 400  # Adjust this based on your preference
GRID_HEIGHT = 400  # Adjust this based on your preference
CELL_SIZE = GRID_WIDTH // GRID_SIZE

# Colors
WHITE = (0, 0, 0)
BLACK = (255, 255, 255)
GRAY = (125, 125, 125)

def create_grid():
    grid = [[WHITE] * GRID_SIZE for _ in range(GRID_SIZE)]
    return grid

def draw_grid(screen, grid):
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def saveAsCSV(numpy_array):
    numpy_array = numpy_array[:,:,0].flatten()
    #predictDrawing(numpy_array)
    csv_file_path = "numpy_array_2.csv"
    with open(csv_file_path, "ab") as f:
        # Save the NumPy array to a CSV file
        f.write(b"\n")
        f.write(b"8,")
        np.savetxt(f, numpy_array, fmt='%d', newline=',', delimiter=",")

    print(f"NumPy array saved as {csv_file_path}")

def main():
    pygame.init()
    screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT))
    pygame.display.set_caption("Drawing Grid")

    grid = create_grid()
    drawing = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    saveAsCSV(np.array(grid))
                    grid = [[WHITE] * GRID_SIZE for _ in range(GRID_SIZE)]
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
                x, y = event.pos
                grid_x, grid_y = x // CELL_SIZE, y // CELL_SIZE
                grid[grid_y][grid_x] = BLACK

                # grid[grid_y-1][grid_x+1] = GRAY
                # grid[grid_y][grid_x+1] = GRAY
                # grid[grid_y+1][grid_x+1] = GRAY
                # grid[grid_y-1][grid_x] = GRAY
                # grid[grid_y+1][grid_x] = GRAY
                # grid[grid_y+1][grid_x-1] = GRAY
                # grid[grid_y-1][grid_x-1] = GRAY
                # grid[grid_y][grid_x-1] = GRAY
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                #saveAsCSV(np.array(grid))
            elif event.type == pygame.MOUSEMOTION and drawing:
                x, y = event.pos
                grid_x, grid_y = x // CELL_SIZE, y // CELL_SIZE
                grid[grid_y][grid_x] = BLACK

                # grid[grid_y-1][grid_x+1] = GRAY
                # grid[grid_y][grid_x+1] = GRAY
                # grid[grid_y+1][grid_x+1] = GRAY
                # grid[grid_y-1][grid_x] = GRAY
                # grid[grid_y+1][grid_x] = GRAY
                # grid[grid_y+1][grid_x-1] = GRAY
                # grid[grid_y-1][grid_x-1] = GRAY
                # grid[grid_y][grid_x-1] = GRAY

        screen.fill(WHITE)
        draw_grid(screen, grid)
        pygame.display.update()

if __name__ == "__main__":
    main()
