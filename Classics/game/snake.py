# snake_game.py

import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Constants for the game
CELL_SIZE = 20
NUM_CELLS_WIDTH = 40
NUM_CELLS_HEIGHT = 30
SCREEN_WIDTH = NUM_CELLS_WIDTH * CELL_SIZE
SCREEN_HEIGHT = NUM_CELLS_HEIGHT * CELL_SIZE
WHITE = (255, 255, 255)
RED = (255, 0, 0)
DARK = (0, 0, 0)
FPS = 10

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

class SnakeGame:
    def __init__(self):
        self.snake_pos = [[SCREEN_WIDTH//2, SCREEN_HEIGHT//2]]
        self.snake_direction = 'UP'
        self.prey_pos = [random.randint(0, NUM_CELLS_WIDTH-1) * CELL_SIZE, random.randint(0, NUM_CELLS_HEIGHT-1) * CELL_SIZE]
        self.score = 0

    def reset(self):
        self.__init__()

    def move_snake(self):
        if self.snake_direction == 'UP':
            new_head = [self.snake_pos[0][0], self.snake_pos[0][1] - CELL_SIZE]
        elif self.snake_direction == 'DOWN':
            new_head = [self.snake_pos[0][0], self.snake_pos[0][1] + CELL_SIZE]
        elif self.snake_direction == 'LEFT':
            new_head = [self.snake_pos[0][0] - CELL_SIZE, self.snake_pos[0][1]]
        else:  # RIGHT
            new_head = [self.snake_pos[0][0] + CELL_SIZE, self.snake_pos[0][1]]

        self.snake_pos.insert(0, new_head)
        if self.snake_pos[0] == self.prey_pos:
            self.score += 1
            self.prey_pos = [random.randint(0, NUM_CELLS_WIDTH-1) * CELL_SIZE, random.randint(0, NUM_CELLS_HEIGHT-1) * CELL_SIZE]
        else:
            self.snake_pos.pop()

    def check_collisions(self):
        # Check boundary collision
        if (self.snake_pos[0][0] >= SCREEN_WIDTH or self.snake_pos[0][0] < 0 or
                self.snake_pos[0][1] >= SCREEN_HEIGHT or self.snake_pos[0][1] < 0):
            return True
        # Check self collision
        if self.snake_pos[0] in self.snake_pos[1:]:
            return True
        return False

    def draw_elements(self):
        screen.fill(DARK)
        for pos in self.snake_pos:
            pygame.draw.rect(screen, WHITE, pygame.Rect(pos[0], pos[1], CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, RED, pygame.Rect(self.prey_pos[0], self.prey_pos[1], CELL_SIZE, CELL_SIZE))
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and self.snake_direction != 'DOWN':
                        self.snake_direction = 'UP'
                    elif event.key == pygame.K_DOWN and self.snake_direction != 'UP':
                        self.snake_direction = 'DOWN'
                    elif event.key == pygame.K_LEFT and self.snake_direction != 'RIGHT':
                        self.snake_direction = 'LEFT'
                    elif event.key == pygame.K_RIGHT and self.snake_direction != 'LEFT':
                        self.snake_direction = 'RIGHT'

            self.move_snake()
            if self.check_collisions():
                running = False
            self.draw_elements()
            clock.tick(FPS)

        pygame.quit()
        sys
