# import required libraries
import random
import numpy as np
import torch
from collections import deque

# import environment and model classes
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import QNet, QTrainer

# hyperparameters and training configuration
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9
STATE_SIZE = 13
HIDDEN = 256
ACTIONS = 3
TARGET_UPDATE_FREQ = 1000
EPS_MIN = 0.01


# prompt user for exploration decay parameter
def get_eps_decay():
    while True:
        try:
            value = int(input("Enter epsilon decay rate (between 30 and 100): "))
            if 30 <= value <= 100:
                return value
            print("Please enter a number between 30 and 100.")
        except ValueError:
            print("Invalid input. Please enter an integer.")


# reinforcement learning agent
class Agent:
    def __init__(self, eps_decay=50):
        self.n_games = 0
        self.epsilon = 1.0
        self.gamma = GAMMA
        self.eps_decay = eps_decay
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = QNet(STATE_SIZE, HIDDEN, ACTIONS)
        self.trainer = QTrainer(self.model, LR, self.gamma, TARGET_UPDATE_FREQ)

    # build 13-element state representation
    def get_state(self, game):
        head = game.snake[0]

        # adjacent points around the head
        pt_left = Point(head.x - BLOCK_SIZE, head.y)
        pt_right = Point(head.x + BLOCK_SIZE, head.y)
        pt_up = Point(head.x, head.y - BLOCK_SIZE)
        pt_down = Point(head.x, head.y + BLOCK_SIZE)

        # current direction flags
        going_left = game.direction == Direction.LEFT
        going_right = game.direction == Direction.RIGHT
        going_up = game.direction == Direction.UP
        going_down = game.direction == Direction.DOWN

        col = game.is_collision

        # danger relative to current heading
        danger_straight = (
            (going_right and col(pt_right)) or
            (going_left and col(pt_left)) or
            (going_up and col(pt_up)) or
            (going_down and col(pt_down))
        )
        danger_right = (
            (going_up and col(pt_right)) or
            (going_down and col(pt_left)) or
            (going_left and col(pt_up)) or
            (going_right and col(pt_down))
        )
        danger_left = (
            (going_down and col(pt_right)) or
            (going_up and col(pt_left)) or
            (going_right and col(pt_up)) or
            (going_left and col(pt_down))
        )

        # state features
        state = [
            danger_straight,
            danger_right,
            danger_left,
            going_left,
            going_right,
            going_up,
            going_down,
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y,
            (game.food.x - head.x) / game.w,
            (game.food.y - head.y) / game.h,
        ]
        return np.array(state, dtype=float)

    # store experience transition in replay buffer
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # train on a random sample from replay memory
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE)
        else:
            batch = list(self.memory)
        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # train on the most recent step
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # select action with epsilon-greedy policy
    def get_action(self, state):
        self.epsilon = max(EPS_MIN, 1.0 - self.n_games / self.eps_decay)
        move = [0, 0, 0]

        # explore or exploit
        if random.random() < self.epsilon:
            move[random.randint(0, 2)] = 1
        else:
            with torch.no_grad():
                t = torch.tensor(state, dtype=torch.float)
                prediction = self.model(t)
                move[torch.argmax(prediction).item()] = 1
        return move


# main training loop
def train():
    record = 0

    # prompt user for epsilon decay rate
    eps_decay = get_eps_decay()

    agent = Agent(eps_decay=eps_decay)
    game = SnakeGameAI()

    while True:
        # get current state and chosen action
        state = agent.get_state(game)
        action = agent.get_action(state)

        # advance game by one step
        reward, done, score = game.play_step(action)
        next_state = agent.get_state(game)

        # learn from the transition
        agent.train_short_memory(state, action, reward, next_state, done)
        agent.remember(state, action, reward, next_state, done)

        # handle game over
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # track high score without saving to disk
            if score > record:
                record = score

            # sync stats to game hud
            game.n_games = agent.n_games
            game.record = record
            game.epsilon = agent.epsilon

            print(f'Game {agent.n_games}  Score {score}  Record {record}  Epsilon {agent.epsilon:.3f}')


if __name__ == '__main__':
    train()
