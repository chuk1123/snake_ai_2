import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
NEW_TRAINING = False # Do this when the Snake is not learning
MODEL_PATH = "model/new_model.pth"

GAMES = 0
BLOCK_SIZE = 20

def check_collide_with_itself(num_blocks, game):
    head = game.snake[0]
    point_l = Point(head.x - BLOCK_SIZE*num_blocks, head.y)
    point_r = Point(head.x + BLOCK_SIZE*num_blocks, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE*num_blocks)
    point_d = Point(head.x, head.y + BLOCK_SIZE*num_blocks)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    result = [
        # Danger straight - collide with itself
        (dir_r and point_r in game.snake) or 
        (dir_l and point_l in game.snake) or 
        (dir_u and point_u in game.snake) or 
        (dir_d and point_d in game.snake),

        # Danger right - collide with itself
        (dir_u and point_r in game.snake) or 
        (dir_d and point_l in game.snake) or 
        (dir_l and point_u in game.snake) or 
        (dir_r and point_d in game.snake),

        # Danger left - collide with itself
        (dir_d and point_r in game.snake) or 
        (dir_u and point_l in game.snake) or 
        (dir_r and point_u in game.snake) or 
        (dir_l and point_d in game.snake)
    ]

    return result
    

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        
        #MODEL 3
        #self.model = Linear_QNet(20, 128, 128, 3) #20 inputs
        #self.model.load_state_dict(torch.load('model/model3.pth'))

        #MODEL 2
        self.model = Linear_QNet(26, 256, 3) #26 inputs
        if not NEW_TRAINING:
            self.model.load_state_dict(torch.load(MODEL_PATH))

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.new_training = NEW_TRAINING #IMPORTANT! SPECIFY BEFORE TRAINING

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        danger_1_block = [status for status in check_collide_with_itself(1, game)]
        danger_2_block = [status for status in check_collide_with_itself(2, game)]
        danger_3_block = [status for status in check_collide_with_itself(3, game)]
        danger_4_block = [status for status in check_collide_with_itself(4, game)]
        danger_5_block = [status for status in check_collide_with_itself(5, game)]

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            danger_1_block[0], #Collide with itself (n blocks away)
            danger_1_block[1],
            danger_1_block[2],

            danger_2_block[0],
            danger_2_block[1],
            danger_2_block[2],
            
            danger_3_block[0], #Straight
            danger_3_block[1], #Right
            danger_3_block[2], #Left

            danger_4_block[0],
            danger_4_block[1],
            danger_4_block[2],
            
            danger_5_block[0],
            danger_5_block[1],
            danger_5_block[2],
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        if self.new_training:
            self.epsilon = 80 - self.n_games
        else:
            self.epsilon = 0

        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    global GAMES
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    total_reward = 0
    while GAMES < 5000:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        try:
            reward, done, score = game.play_step(final_move)
        except:
            save_model = input("Save model? (y/n) ")
            if save_model == 'y':
                agent.model.save()
                print("saved model!")
                quit()
            else:
                quit()

        state_new = agent.get_state(game)
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            GAMES += 1
            agent.train_long_memory()

            if score > record:
                record = score
                #agent.model.save()
                #print("saved model!")

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()