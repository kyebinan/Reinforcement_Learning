import pygame
import random
import numpy as np
from game.game import Game 


class SnakeGame(Game):
    def __init__(self, width=40, height=30):
        super().__init__()
        self.width = width
        self.height = height
        self.cell_size = 20
        self.screen = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
        pygame.display.set_caption('Snake Game')
        self.directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.delta = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
        self.reset()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2), (self.width // 2, 1 + self.height // 2)]
        self.direction = 'UP'
        self.done = False
        self.food = self._spawn_food()
        return self.get_state()

    def _spawn_food(self):
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake:
                return food

    def step(self, action, grid=False):
        # Get the next direction based on the action
        next_direction = self.directions[action]
        
        # Prevent the snake from reversing direction
        if (self.direction == 'UP' and next_direction != 'DOWN') or \
        (self.direction == 'DOWN' and next_direction != 'UP') or \
        (self.direction == 'LEFT' and next_direction != 'RIGHT') or \
        (self.direction == 'RIGHT' and next_direction != 'LEFT'):
            self.direction = next_direction

        # Calculate new head position
        head_x, head_y = self.snake[0]
        delta_x, delta_y = self.delta[self.direction]
        new_head = (head_x + delta_x, head_y + delta_y)

        # Check for game over conditions
        if (new_head in self.snake or
            new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height):
            self.done = True
            reward = -100  # Penalty for losing
            return self.get_state(), reward, self.done

        # Check if snake eats food
        if new_head == self.food:
            self.food = self._spawn_food()
            reward = 50  # Reward for eating food
            # Do not remove the tail, effectively growing the snake
        else:
            # Move the snake by removing the tail
            self.snake.pop()
            reward = -1

        # Insert the new head to move the snake
        self.snake.insert(0, new_head)

        if not grid:
            return self.get_state(), reward, self.done
        else:
            return self.get_state_grid(), reward, self.done
    
    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        # Check for dangers
        danger_left = (head_x - 1, head_y) in self.snake or head_x - 1 < 0
        danger_right = (head_x + 1, head_y) in self.snake or head_x + 1 >= self.width
        danger_up = (head_x, head_y - 1) in self.snake or head_y - 1 < 0
        danger_down = (head_x, head_y + 1) in self.snake or head_y + 1 >= self.height

        # Check for food direction
        food_right = food_x > head_x
        food_left = food_x < head_x
        food_up = food_y < head_y
        food_down = food_y > head_y
        bools = [danger_left, danger_right, danger_up, danger_down, food_right, food_left, food_up, food_down]
        state = booleans_to_int(bools)
        return state
    
    def get_state_grid(self):
        # Create an empty grid initialized with 0s
        grid = np.zeros((self.height, self.width))

        
        # Place the food in the grid
        food_x, food_y = self.food
        grid[food_y][food_x] = 1
        
        # Place the snake in the grid
        for i, (x, y) in enumerate(self.snake):
            if i == 0:  # Head of the snake
                grid[y][x] = 2
            else:  # Body of the snake
                grid[y][x] = 3
        print(grid)
        return grid.flatten()

    def render(self):
        self.screen.fill((0, 0, 0))
        for block in self.snake:
            rect = pygame.Rect(block[0] * self.cell_size, block[1] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 255, 255), rect)
        food_rect = pygame.Rect(self.food[0] * self.cell_size, self.food[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (213, 50, 80), food_rect)
        pygame.display.flip()


def booleans_to_int(bools):
    """
    Convert a tuple of 8 boolean values to an integer, treating each boolean
    as a bit in a binary number.
    """
    binary_string = ''.join(['1' if b else '0' for b in bools])
    return int(binary_string, 2)

def main():
    game = SnakeGame(40, 30)
    clock = pygame.time.Clock()
    while not game.done:
        action = random.randint(0, 3)
        grid, reward, done = game.step(action)
        game.render()
        clock.tick(10)  # Control the speed of the game

        if game.done:
            print("Game Over")
            #game.reset()


if __name__ == "__main__":
    main()