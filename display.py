import pygame
import sys
import csv
import numpy as np
import shutil

# Constants for the grid
GRID_SIZE = 28
GRID_WIDTH = 400  # Adjust this based on your preference
GRID_HEIGHT = 400  # Adjust this based on your preference
CELL_SIZE = GRID_WIDTH // GRID_SIZE

# Pygame
LEFT = 1
RIGHT = 3

# Colors
WHITE = (0, 0, 0)
BLACK = (255, 255, 255)
GRAY = (125, 125, 125)

CURRENT_COLOR = WHITE

FILE_PATH = "data/train.csv"

with open(FILE_PATH, "r") as file:
    reader = csv.reader(file)
    csv_file_list = list(reader)


def create_grid():
    grid = [[WHITE] * GRID_SIZE for _ in range(GRID_SIZE)]
    return grid


def draw_grid(screen, grid):
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            pygame.draw.rect(screen, color, (x * CELL_SIZE,
                             y * CELL_SIZE, CELL_SIZE, CELL_SIZE))


def load_grid_from_csv(index):
    single_channel_grid = load_grid(index)
    transformed_grid = transform_grid_to_RGB(single_channel_grid)
    return transformed_grid


def transform_grid_to_RGB(single_channel_grid):
    transformed_grid = []
    for row in single_channel_grid:
        row_elements = []
        for cell in row:
            if cell == '255':
                row_elements.append((255, 255, 255))
            else:
                row_elements.append((0, 0, 0))

        transformed_grid.append(row_elements)
    return transformed_grid


def load_grid(index):
    array_for_index = csv_file_list[index]
    np_array = np.array(array_for_index)
    label = np_array[:1]
    pixel_vector = np.array(array_for_index[1:])
    print(f"loading index: {index} that has label with class {label}")
    rows = np.split(pixel_vector, 28)
    return rows


def remove_indicies_from_csv(to_be_removed_indices):
    new_rows = rows_without_removed_indices(to_be_removed_indices)
    write_new_rows_to_csv(new_rows)


def write_new_rows_to_csv(new_rows):
    with open(FILE_PATH, "w", newline='') as write_file:
        writer = csv.writer(write_file)
        writer.writerows(new_rows)


def rows_without_removed_indices(indices):
    with open(FILE_PATH, "r", newline="") as read_file:
        reader = csv.reader(read_file)
        new_rows = [_ for index, _ in enumerate(
            reader) if index not in indices]
    return new_rows


def main():
    print("--> Navigate with arrow keys")
    print("--> Press 'd' to delete current image")

    pygame.init()
    screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT))
    pygame.display.set_caption("Display Grid")

    grid = create_grid()
    index = 0
    grid = load_grid_from_csv(1)
    indices_to_be_removed = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                remove_indicies_from_csv(indices_to_be_removed)
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and index > 0:
                    index = index - 1
                    grid = load_grid_from_csv(index)
                elif event.key == pygame.K_RIGHT and index < len(csv_file_list):
                    index = index + 1
                    grid = load_grid_from_csv(index)
                elif event.key == pygame.K_d:
                    print(
                        f"Deleting index {index} from {FILE_PATH} after program exits")
                    indices_to_be_removed.append(index)

        screen.fill(WHITE)
        draw_grid(screen, grid)
        pygame.display.update()


if __name__ == "__main__":
    main()
