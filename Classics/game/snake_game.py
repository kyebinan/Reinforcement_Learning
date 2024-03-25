import pygame
import random
from game import Game 


class SnakeGame(Game):
    def __init__(self, width=40, height=30):
        super().__init__()
        self.width = width
        self.height = height
        self.directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.delta = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
        self.reset()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = 'UP'
        self.score = 0
        self.done = False
        self.food = self._spawn_food()
        self.frame_iteration = 0
        return self.get_state()

    def _spawn_food(self):
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake:
                return food

    def step(self, action):
        self.frame_iteration += 1
        # Convert action to direction
        self.direction = self.directions[action]

        # Move the snake
        head_x, head_y = self.snake[0]
        delta_x, delta_y = self.delta[self.direction]
        new_head = (head_x + delta_x, head_y + delta_y)

        # Check for game over conditions
        if (new_head in self.snake or
                new_head[0] < 0 or new_head[0] >= self.width or
                new_head[1] < 0 or new_head[1] >= self.height or
                self.frame_iteration > 100*len(self.snake)):
            self.done = True
            reward = -10  # Penalty for losing
            return self.get_state(), reward, self.done

        # Check if snake eats food
        if new_head == self.food:
            self.score += 1
            self.food = self._spawn_food()
            reward = 10  # Reward for eating food
        else:
            self.snake.pop()  # Remove the tail
            reward = 0

        self.snake.insert(0, new_head)  # Add new head

        return self.get_state(), reward, self.done

    def get_state(self):
        # Implement state representation
        # For simplicity, we're not implementing a full state representation here.
        # A typical state could include the direction of the snake, distance to food,
        # and information about the immediate surroundings.
        return (self.snake, self.food, self.direction)

    def render(self):
        pygame.init()
        cell_size = 20  # Size of each square
        screen = pygame.display.set_mode((self.width * cell_size, self.height * cell_size))
        pygame.display.set_caption('Snake Game')

        # Colors
        black = (0, 0, 0)
        white = (255, 255, 255)
        red = (213, 50, 80)

        # Game loop control flag
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill(black)

            # Draw the snake
            for block in self.snake:
                rect = pygame.Rect(block[0] * cell_size, block[1] * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, white, rect)

            # Draw the food
            food_rect = pygame.Rect(self.food[0] * cell_size, self.food[1] * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, red, food_rect)

            pygame.display.flip()
            pygame.time.wait(100)

        pygame.quit()



def main():
    game = SnakeGame(40, 30)

    # Game loop
    while not game.done:
        # For simplicity, actions are random. Replace with your RL agent's actions.
        action = 0 #random.randint(0, 3)  # Random action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        state, reward, done = game.step(action)
        game.render()  # Render the game state
        
        if done:
            print("Game Over")
            game.reset()  # Reset the game to start over

if __name__ == "__main__":
    main()