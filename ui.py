import pygame
import sys
import numpy as np
from network import predictDrawing, onlineTraining
import csv

# Constants for the grid
GRID_SIZE = 28
GRID_WIDTH = 400  # Adjust this based on your preference
GRID_HEIGHT = 400  # Adjust this based on your preference
CELL_SIZE = GRID_WIDTH // GRID_SIZE


#Pygame
LEFT = 1
RIGHT = 3

# Colors
WHITE = (0, 0, 0)
BLACK = (255, 255, 255)
GRAY = (125, 125, 125)

CURRENT_COLOR = WHITE

def create_grid():
    grid = [[WHITE] * GRID_SIZE for _ in range(GRID_SIZE)]
    return grid

def draw_grid(screen, grid):
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

#0=house, 1=car, 2=....
pygame_number_list = [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7]
def saveAsCSV(numpy_array, class_index):
    numpy_array = numpy_array[:,:,0].flatten()
    csv_file_path = "bp.csv"
    with open(csv_file_path, "a", newline='') as f:
        # Save the NumPy array to a CSV file
        writer = csv.writer(f)
        match class_index:
            case pygame.K_0:
                writer.writerow(['0'] + numpy_array.tolist()) #house
            case pygame.K_1:
                writer.writerow(['1'] + numpy_array.tolist()) #car
            case pygame.K_2:
                writer.writerow(['2'] + numpy_array.tolist()) #wireless headphones
            case pygame.K_3:
                writer.writerow(['3'] + numpy_array.tolist()) #bottle
            case pygame.K_4:
                writer.writerow(['4'] + numpy_array.tolist())  #on ear headphones
            case pygame.K_5:
                writer.writerow(['5'] + numpy_array.tolist())  # stick man

    print(f"NumPy array saved as {csv_file_path}")

def predict(numpy_array):
    numpy_array = numpy_array[:,:,0].flatten()
    predictDrawing(numpy_array)
def main():
    print("Draw by pressing left mousebutton and press a key (0-7) to save it and online train the model")#
    print("Pressing the right mousebutton removes the pixel from that location ")
    print("Press d to delete the current drawing and start fresh")
    print("0 --> House\n1-->Car\n2-->Inear headphones\n3-->Bottle\n4-->On ear headphones\n5-->stick man")
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
                if event.key in pygame_number_list:
                    saveAsCSV(np.array(grid), event.key)
                    onlineTraining()
                    grid = [[WHITE] * GRID_SIZE for _ in range(GRID_SIZE)]
                elif event.key == pygame.K_d:
                    grid = [[WHITE] * GRID_SIZE for _ in range(GRID_SIZE)]
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
                x, y = event.pos
                grid_x, grid_y = x // CELL_SIZE, y // CELL_SIZE
                if event.button == LEFT:
                    CURRENT_COLOR = BLACK
                else:
                    CURRENT_COLOR = WHITE
                grid[grid_y][grid_x] = CURRENT_COLOR
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.MOUSEMOTION and drawing:
                x, y = event.pos
                grid_x, grid_y = x // CELL_SIZE, y // CELL_SIZE
                if grid_x >= 0 and grid_x < len(grid) and grid_y >= 0 and grid_y < len(grid[0]):
                    
                    grid[grid_y][grid_x] = CURRENT_COLOR
                    predict(np.array(grid))

        screen.fill(WHITE)
        draw_grid(screen, grid)
        pygame.display.update()

if __name__ == "__main__":
    main()
