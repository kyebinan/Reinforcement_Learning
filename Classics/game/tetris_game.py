import numpy as np
import random
from game import Game 
import pygame


class TetrisGame(Game):
    def __init__(self, width=10, height=20):
        super().__init__()
        self.width = width
        self.height = height
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
        self.frame_iteration += 1
        if action == 0:  # Move left
            if not self.check_collision(self.current_piece, (self.piece_position[0] - 1, self.piece_position[1])):
                self.piece_position[0] -= 1
        elif action == 1:  # Move right
            if not self.check_collision(self.current_piece, (self.piece_position[0] + 1, self.piece_position[1])):
                self.piece_position[0] += 1
        elif action == 2:  # Rotate
            rotated_piece = self.rotate_piece(self.current_piece)
            if not self.check_collision(rotated_piece, self.piece_position):
                self.current_piece = rotated_piece
        elif action == 3:  # Drop
            while not self.check_collision(self.current_piece, (self.piece_position[0], self.piece_position[1] + 1)):
                self.piece_position[1] += 1
            self.add_piece_to_board(self.current_piece, self.piece_position)
            self.clear_lines()
            self.current_piece = self.next_piece
            self.next_piece = self._get_new_piece()
            self.piece_position = [self.width // 2 - len(self.current_piece[0]) // 2, 0]
            if self.check_collision(self.current_piece, self.piece_position):
                self.game_over = True
                self.done = True

        # Implement move down and game over check
        if not self.done:
            if self.check_collision(self.current_piece, (self.piece_position[0], self.piece_position[1] + 1)):
                self.add_piece_to_board(self.current_piece, self.piece_position)
                self.clear_lines()
                self.current_piece = self.next_piece
                self.next_piece = self._get_new_piece()
                self.piece_position = [self.width // 2 - len(self.current_piece[0]) // 2, 0]
                if self.check_collision(self.current_piece, self.piece_position):
                    self.game_over = True
                    self.done = True
            else:
                self.piece_position[1] += 1

        return self.get_state(), self.score, self.done

    def render(self, screen):
        cell_size = 40
        screen.fill((0, 0, 0))

        # Draw the board
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i][j] == 1:
                    pygame.draw.rect(screen, (255, 255, 255), [j * cell_size, i * cell_size, cell_size, cell_size])

        # Draw the current piece
        for i, row in enumerate(self.current_piece):
            for j, cell in enumerate(row):
                if cell == 1:
                    pygame.draw.rect(screen, (255, 165, 0), [(self.piece_position[0] + j) * cell_size, (self.piece_position[1] + i) * cell_size, cell_size, cell_size])

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
    screen = pygame.display.set_mode((game.width * 40, game.height * 40))
    pygame.display.set_caption("Tetris")
    clock = pygame.time.Clock()

    fall_time = 0
    fall_speed = 1000  # Milliseconds between each automatic step down

    running = True
    while running:
        # The time passed since the last tick, in milliseconds
        delta_time = clock.tick(60)  # Run at max 60 frames per second
        
        fall_time += delta_time

        if fall_time >= fall_speed:
            fall_time = 0
            game.step(3)  # Drop piece over time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    game.step(0)
                elif event.key == pygame.K_RIGHT:
                    game.step(1)
                elif event.key == pygame.K_UP:
                    game.step(2)
                elif event.key == pygame.K_DOWN:
                    # For an immediate drop, increase fall_speed temporarily or call game.step(3) multiple times.
                    game.step(3)

        screen.fill((0, 0, 0))
        game.render(screen)
        pygame.display.flip()

        if game.game_over:
            print("Game Over. Press any key to restart.")
            waiting_for_input = True
            while waiting_for_input:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting_for_input = False
                    if event.type == pygame.KEYDOWN:
                        game.reset()
                        waiting_for_input = False

    pygame.quit()

if __name__ == "__main__":
    main()