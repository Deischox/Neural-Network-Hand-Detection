import pygame
import sys
import numpy as np
from network import forward, softmax_numpy
from activation import sigmoid
LABELS = ["House", "Car", "Inear headphones", "Bottle",
          "On ear headphones", "Stick man", "TV-screen", "Sun"]

IMAGE_INDEX = 0


circles_layer_one = []
circles_layer_two = []
circles_layer_three = []
circles_layer_four = []

image_surface = None


# Initialize Pygame
pygame.init()

# Window dimensions
WIDTH, HEIGHT = 1000, 700

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Initialize the window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
window = screen.get_rect()

pygame.display.set_caption("Visualization of Neural Network")

last_layer = None
test_data = np.loadtxt("data/train.csv",
                       delimiter=",")


def init_values(index=0):
    global image_surface
    global circles_layer_one
    global circles_layer_two
    global circles_layer_three
    global circles_layer_four
    global last_layer

    circles_layer_one = []
    circles_layer_two = []
    circles_layer_three = []
    circles_layer_four = []

    my_net = np.load("models/model.npy", allow_pickle=True)
    fac = 0.99 / 255

    test_data = np.loadtxt("data/train.csv",
                           delimiter=",")
    test_data = np.asfarray(test_data[:, 1:]) * fac + 0.01
    O, _ = forward(my_net, test_data[index][:784])

    print(my_net[2].weights)

    O[2] = softmax_numpy(O[2])
    last_layer = O[2]

    test_data = np.loadtxt("data/train.csv",
                           delimiter=",")
    image_surface = pygame.surfarray.make_surface(
        np.reshape(test_data[index][:784], (28, 28)).astype(np.uint8))
    image_surface = pygame.transform.scale(image_surface, (112, 112))
    image_surface = pygame.transform.rotate(image_surface, -90)

    for i in range(784):
        circles_layer_one.append(
            Circle(-10000, -50000+i*110, 50, test_data[index][i], map_value_to_color(test_data[index][i])))

    for i in range(len(O[0])):
        circles_layer_two.append(
            Circle(0, i*900, 400, O[0][i], map_value_to_color(O[0][i])))

    for i in range(len(O[1])):
        circles_layer_three.append(
            Circle(10000, i*900, 400, O[1][i], map_value_to_color(O[1][i])))

    for i in range(len(O[2])):
        circles_layer_four.append(
            Circle(15000, 3600+i*900, 400, O[2][i], map_value_to_color(O[2][i])))


class Circle:
    def __init__(self, x, y, radius, value, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.value = value
        self.text = "{:.3f}".format(value)
        self.color = color
        self.clicked = False

# Function to check if a point is within a circle


def is_point_in_circle(circle, point):
    adjusted_x = int(circle.x * zoom_factor) + pan_x
    adjusted_y = int(circle.y * zoom_factor) + pan_y
    adjusted_radius = int(circle.radius * zoom_factor)
    dist_to_circle = ((point[0] - adjusted_x) ** 2 +
                      (point[1] - adjusted_y) ** 2) ** 0.5
    return dist_to_circle <= adjusted_radius


def map_value_to_color(value):
    # Ensure the value is within the valid range [0, 1]
    value = max(0, min(1, value))

    # Interpolate between white and green
    r = int((1 - value) * 255)
    g = 255
    b = int((1 - value) * 255)

    return (r, g, b)


def map_value_to_color_line(value):
    # Ensure the value is within the valid range [0, 1]
    value = max(0, min(1, value))

    # Interpolate between white and Black
    r = int(max(100, (value) * 255))
    g = int(max(100, (value) * 255))
    b = int(max(100, (value) * 255))

    return pygame.Color(r, g, b, 0)


# Initial pan and zoom values
pan_x, pan_y = 200, 00
zoom_factor = 0.04

clock = pygame.time.Clock()

# Create a font object
font = pygame.font.Font(None, 24)


init_values()
while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                # Load next Image
                if IMAGE_INDEX < len(test_data):
                    IMAGE_INDEX += 1
                    init_values(index=IMAGE_INDEX)
            if event.key == pygame.K_o:
                if IMAGE_INDEX > 0:
                    IMAGE_INDEX -= 1
                    init_values(index=IMAGE_INDEX)

    # Check for mouse click events
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        # Check if the mouse click is within any circle
        mouse_x, mouse_y = pygame.mouse.get_pos()
        world_x = (mouse_x - pan_x) / zoom_factor
        world_y = (mouse_y - pan_y) / zoom_factor

        for circle in circles_layer_two + circles_layer_three + circles_layer_four:
            if is_point_in_circle(circle, (mouse_x, mouse_y)):
                circle.clicked = not circle.clicked
                if circle.clicked:
                    circle.color = RED

    keys = pygame.key.get_pressed()

    # Handle pan
    if keys[pygame.K_LEFT]:
        pan_x += 5
    if keys[pygame.K_RIGHT]:
        pan_x -= 5
    if keys[pygame.K_UP]:
        pan_y += 5
    if keys[pygame.K_DOWN]:
        pan_y -= 5

    # Handle zoom
    if keys[pygame.K_w]:
        zoom_factor *= 1.02
    if keys[pygame.K_s]:
        zoom_factor /= 1.02

    # Clear the screen
    screen.fill(BLACK)

    for circle in circles_layer_two:
        for end_circle in circles_layer_three:
            # Draw the line
            adjusted_point1 = (
                int(circle.x * zoom_factor) + pan_x, int(circle.y * zoom_factor) + pan_y)
            adjusted_point2 = (int(
                end_circle.x * zoom_factor) + pan_x, int(end_circle.y * zoom_factor) + pan_y)
            pygame.draw.line(screen, map_value_to_color_line(circle.value), adjusted_point1,
                             adjusted_point2, max(1, int(circle.value*5)))

    for i, circle in enumerate(circles_layer_four):
        adjusted_x = int(circle.x * zoom_factor) + pan_x
        adjusted_y = int(circle.y * zoom_factor) + pan_y
        # Render text surface
        if last_layer[i] == max(last_layer):
            text_surface = font.render(LABELS[i], True, RED)
        else:
            text_surface = font.render(LABELS[i], True, WHITE)

        text_rect = text_surface.get_rect(
            center=(adjusted_x+2000*zoom_factor, adjusted_y))

        # Blit text onto the screen
        screen.blit(text_surface, text_rect)

    for circle in circles_layer_one:
        for end_circle in circles_layer_two:
            # Draw the line
            if circle.value > 0:
                adjusted_point1 = (
                    int(circle.x * zoom_factor) + pan_x, int(circle.y * zoom_factor) + pan_y)
                adjusted_point2 = (int(
                    end_circle.x * zoom_factor) + pan_x, int(end_circle.y * zoom_factor) + pan_y)
                pygame.draw.line(screen, WHITE, adjusted_point1,
                                 adjusted_point2, 1)

    for circle in circles_layer_three:
        for end_circle in circles_layer_four:
            # Draw the line
            adjusted_point1 = (
                int(circle.x * zoom_factor) + pan_x, int(circle.y * zoom_factor) + pan_y)
            adjusted_point2 = (int(
                end_circle.x * zoom_factor) + pan_x, int(end_circle.y * zoom_factor) + pan_y)
            pygame.draw.line(screen, map_value_to_color_line(circle.value), adjusted_point1,
                             adjusted_point2, max(1, int(circle.value*5)))

    # Draw the circle with pan and zoom adjustments
    for circle in circles_layer_two + circles_layer_three + circles_layer_four:
        adjusted_x = int(circle.x * zoom_factor) + pan_x
        adjusted_y = int(circle.y * zoom_factor) + pan_y
        adjusted_radius = int(circle.radius * zoom_factor)
        pygame.draw.circle(screen, circle.color,
                           (adjusted_x, adjusted_y), adjusted_radius)

        # Render text surface
        text_surface = font.render(circle.text, True, BLACK)
        text_rect = text_surface.get_rect(center=(adjusted_x, adjusted_y))

        # Blit text onto the screen
        screen.blit(text_surface, text_rect)

    for circle in circles_layer_one:
        adjusted_x = int(circle.x * zoom_factor) + pan_x
        adjusted_y = int(circle.y * zoom_factor) + pan_y
        adjusted_radius = int(circle.radius * zoom_factor)
        pygame.draw.circle(screen, circle.color,
                           (adjusted_x, adjusted_y), adjusted_radius)

    screen.blit(image_surface, (1000-112, 700-112))

    text_surface_help = font.render(
        "W/S: Zoom, Arrow Keys: Move, P/O: next/last image", True, WHITE)
    text_rect_help = text_surface_help.get_rect(center=(WIDTH-200, 10))

    # Blit text onto the screen
    screen.blit(text_surface_help, text_rect_help)
    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)
