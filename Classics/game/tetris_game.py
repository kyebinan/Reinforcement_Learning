import numpy as np
import random
from game import Game 
import pygame


class TetrisGame(Game):
    def __init__(self, width=10, height=20):
        super().__init__()
        self.width = width
        self.height = height
        self.cell_size = 30
        self.screen = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
        pygame.display.set_caption('Snake Game')
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.current_piece = None
        self.next_piece = self._get_new_piece()
        self.game_over = False
        self._reset()

    def _get_new_piece(self):
        # Define tetromino shapes
        shapes = [
            [[1, 1, 1, 1]],  # I
            [[1, 1, 1], [0, 1, 0]],  # T
            [[1, 1], [1, 1]],  # O
            [[0, 1, 1], [1, 1, 0]],  # S
            [[1, 1, 0], [0, 1, 1]],  # Z
            [[1, 0, 0], [1, 1, 1]],  # J
            [[0, 0, 1], [1, 1, 1]]   # L
        ]
        return random.choice(shapes)

    def _reset(self):
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.score = 0
        self.done = False
        self.current_piece = self.next_piece
        self.next_piece = self._get_new_piece()
        self.piece_position = [0, self.width // 2 - len(self.current_piece[0]) // 2]
        self.frame_iteration = 0

    def reset(self):
        self._reset()
        return self.get_state()

    def step(self, action):
        # Handle left movement with boundary check
        if action == 0:  # Move left
            if not self.check_collision(self.current_piece, (self.piece_position[0] - 1, self.piece_position[1])) and \
            self.is_within_boundaries(self.current_piece, (self.piece_position[0] - 1, self.piece_position[1])):
                self.piece_position[0] -= 1

        # Handle right movement with boundary check
        elif action == 1:  # Move right
            if not self.check_collision(self.current_piece, (self.piece_position[0] + 1, self.piece_position[1])) and \
            self.is_within_boundaries(self.current_piece, (self.piece_position[0] + 1, self.piece_position[1])):
                self.piece_position[0] += 1

        # Handle rotation
        elif action == 2:  # Rotate
            rotated_piece = self.rotate_piece(self.current_piece)
            if not self.check_collision(rotated_piece, self.piece_position) and \
            self.is_within_boundaries(rotated_piece, self.piece_position):
                self.current_piece = rotated_piece

        # Automatic drop (or manual if action == 3)
        # The piece falls one step
        if not self.check_collision(self.current_piece, (self.piece_position[0], self.piece_position[1] + 1)):
            self.piece_position[1] += 1
        else:
            # Place the piece on the board
            self.add_piece_to_board(self.current_piece, self.piece_position)
            self.clear_lines()
            # Move to the next piece
            self.current_piece = self.next_piece
            self.next_piece = self._get_new_piece()
            self.piece_position = [self.width // 2 - len(self.current_piece[0]) // 2, 0]
            # Check for game over
            if self.check_collision(self.current_piece, self.piece_position):
                self.game_over = True
                self.done = True
                return self.get_state(), -10, self.done  # Game over penalty

        return self.get_state(), 0, self.done  # Return default reward for other actions

    def is_within_boundaries(self, piece, offset):
        off_x, off_y = offset
        for y, row in enumerate(piece):
            for x, cell in enumerate(row):
                if cell:
                    # Check if the cell is outside the board's width
                    if x + off_x < 0 or x + off_x >= self.width:
                        return False
        return True

    def render(self):
        self.screen.fill((0, 0, 0))
        # Draw the board
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i][j] == 1:
                    pygame.draw.rect(self.screen, (255, 255, 255), [j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size])

        # Draw the current piece
        for i, row in enumerate(self.current_piece):
            for j, cell in enumerate(row):
                if cell == 1:
                    pygame.draw.rect(self.screen, (255, 165, 0), [(self.piece_position[0] + j) * self.cell_size, (self.piece_position[1] + i) * self.cell_size, self.cell_size, self.cell_size])

        pygame.display.flip()

    def get_state(self):
        # Implement state representation (omitted for brevity)
        # Could be the raw board, a flattened board, or additional features
        return self.board.flatten()

    def check_collision(self, piece, offset):
        off_x, off_y = offset
        for y, row in enumerate(piece):
            for x, cell in enumerate(row):
                try:
                    if cell and self.board[y + off_y][x + off_x]:
                        return True
                except IndexError:
                    return True
        return False

    def add_piece_to_board(self, piece, offset):
        off_x, off_y = offset
        for y, row in enumerate(piece):
            for x, cell in enumerate(row):
                if cell:
                    self.board[y + off_y][x + off_x] = cell

    def clear_lines(self):
        lines_to_clear = [index for index, row in enumerate(self.board) if all(row)]
        for line in lines_to_clear:
            del self.board[line]
            self.board.insert(0, [0 for _ in range(self.width)])
            self.score += 1

    def rotate_piece(self, piece):
        return [list(row) for row in zip(*piece[::-1])]
    

def main():
    pygame.init()
    game = TetrisGame(10, 20)
    clock = pygame.time.Clock()

    while not game.done:
        action = random.randint(0, 3)
        grid, reward, done = game.step(action)
        game.render()
        clock.tick(1)  # Control the speed of the game

        if game.done:
            print("Game Over")
            #game.reset()

if __name__ == "__main__":
    main()