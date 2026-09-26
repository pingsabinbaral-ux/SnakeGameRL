# import required libraries
import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple


# movement directions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# 2d coordinate point
Point = namedtuple('Point', 'x y')

# board configuration and constants
BLOCK_SIZE = 20
SPEED = 40
MAX_STEPS = 400

# ui color definitions
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)


# snake game environment for rl agent
class SnakeGameAI:
    def __init__(self, w=400, h=400):
        self.w = w
        self.h = h

        # initialize pygame and display
        pygame.init()
        self.font = pygame.font.Font(None, 25)
        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # tracking metrics
        self.n_games = 0
        self.record = 0
        self.epsilon = 1.0

        # start first episode
        self.reset()

    # reset board state for a new game
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y),
        ]
        self.score = 0
        self.frame_iteration = 0
        self._place_food()

    # spawn food on an unoccupied grid tile
    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    # execute one step given an action [straight, right, left]
    def play_step(self, action):
        self.frame_iteration += 1

        # handle quit events
        for e in pygame.event.get(pump=False):
            if e.type == pygame.QUIT:
                pygame.quit()
                quit()

        # move snake head
        prev = self.head
        self._move(action)
        self.snake.insert(0, self.head)

        # check for game over condition
        if self.is_collision() or self.frame_iteration > MAX_STEPS * len(self.snake):
            self._update_ui()
            return -10, True, self.score

        # food consumption and reward shaping
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            d_prev = abs(prev.x - self.food.x) + abs(prev.y - self.food.y)
            d_new = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            reward = 0.1 if d_new < d_prev else -0.1

        # render frame
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, False, self.score

    # check boundaries and self-collision
    def is_collision(self, pt=None):
        pt = pt or self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        return pt in self.snake[1:]

    # draw game elements and hud
    def _update_ui(self):
        self.display.fill(BLACK)

        # draw snake body
        for p in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(p.x, p.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(p.x + 4, p.y + 4, 12, 12))

        # draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # draw heads-up display text
        hud = self.font.render(
            f"Game: {self.n_games}   Score: {self.score}   Record: {self.record}   Eps: {self.epsilon:.2f}",
            True, WHITE
        )
        self.display.blit(hud, [5, 5])

        pygame.event.pump()
        pygame.display.flip()

    # update snake direction and position
    def _move(self, action):
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        # decode relative steering action
        if np.array_equal(action, [1, 0, 0]):
            self.direction = clockwise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            self.direction = clockwise[(idx + 1) % 4]
        else:
            self.direction = clockwise[(idx - 1) % 4]

        # calculate next head coordinate
        x, y = self.head
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        else:
            y -= BLOCK_SIZE
        self.head = Point(x, y)
