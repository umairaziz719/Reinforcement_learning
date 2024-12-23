import numpy as np
import pygame
import random
import tensorflow as tf
from keras import layers


from collections import deque
import time

# Hyperparameters
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration factor
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 64
replay_buffer_size = 10000
target_update_frequency = 10
max_steps = 10000
max_epsilon_steps = 1000

# Game Settings
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
BLOCK_SIZE = 20
SCREEN_DIM = (SCREEN_WIDTH, SCREEN_HEIGHT)
FPS = 15  # Frames per second

# Action Constants
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]

# Define the Q-Network
class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu', input_dim=state_size)
        self.dense2 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(action_size, activation='linear')
    
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

# Snake Game Class (Using Pygame)
class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode(SCREEN_DIM)
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = [(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)]
        self.snake_dir = RIGHT
        self.food = self.generate_food()
        self.score = 0
        self.game_over = False
        self.steps = 0
        return self.get_state()

    def generate_food(self):
        return (random.randrange(0, SCREEN_WIDTH, BLOCK_SIZE), random.randrange(0, SCREEN_HEIGHT, BLOCK_SIZE))

    def get_state(self):
        # Create a grid state of zeros
        state = np.zeros((SCREEN_HEIGHT // BLOCK_SIZE, SCREEN_WIDTH // BLOCK_SIZE), dtype=int)
        
        for segment in self.snake:
            # Calculate grid positions, ensuring they are within bounds
            x_idx = max(0, min((segment[0] // BLOCK_SIZE), (SCREEN_WIDTH // BLOCK_SIZE) - 1))
            y_idx = max(0, min((segment[1] // BLOCK_SIZE), (SCREEN_HEIGHT // BLOCK_SIZE) - 1))
            
            # Set the snake's position in the state grid
            state[y_idx, x_idx] = 1
    
        # Mark food on the grid
        food_x_idx = max(0, min((self.food[0] // BLOCK_SIZE), (SCREEN_WIDTH // BLOCK_SIZE) - 1))
        food_y_idx = max(0, min((self.food[1] // BLOCK_SIZE), (SCREEN_HEIGHT // BLOCK_SIZE) - 1))
        state[food_y_idx, food_x_idx] = 2  # 2 represents the food
        
        return state.flatten()  # Return flattened state


    def step(self, action):
        if action == UP:
            self.snake_dir = UP
        elif action == DOWN:
            self.snake_dir = DOWN
        elif action == LEFT:
            self.snake_dir = LEFT
        elif action == RIGHT:
            self.snake_dir = RIGHT

        head_x, head_y = self.snake[0]

        if self.snake_dir == UP:
            head_y -= BLOCK_SIZE
        elif self.snake_dir == DOWN:
            head_y += BLOCK_SIZE
        elif self.snake_dir == LEFT:
            head_x -= BLOCK_SIZE
        elif self.snake_dir == RIGHT:
            head_x += BLOCK_SIZE

        new_head = (head_x, head_y)
        self.snake = [new_head] + self.snake[:-1]

        # Check for collisions with wall or itself
        if (head_x < 0 or head_x >= SCREEN_WIDTH or head_y < 0 or head_y >= SCREEN_HEIGHT or new_head in self.snake[1:]):
            self.game_over = True
            return self.get_state(), -10, True  # Penalty for dying

        # Check if the snake eats food
        reward = 0
        if new_head == self.food:
            self.snake.append(self.snake[-1])  # Grow the snake
            self.food = self.generate_food()  # Generate new food
            reward = 10  # Reward for eating food

        self.steps += 1
        if self.steps >= max_steps:
            self.game_over = True

        return self.get_state(), reward, self.game_over

    def render(self):
        self.screen.fill((0, 0, 0))
        for segment in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), (segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()
        self.clock.tick(FPS)

# DQN Agent Class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(replay_buffer_size)
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.epsilon = epsilon

    def get_action(self, state):
        state = np.reshape(state, (1, -1))
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model(state)
        return np.argmax(q_values[0])

    def train(self):
        if self.memory.size() < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        next_q_values = self.target_model(next_states)
        next_q_value = np.max(next_q_values, axis=1)
        target_q_values = rewards + (gamma * next_q_value * (1 - dones))

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_value = tf.gather(q_values, actions, axis=1, batch_dims=1)
            loss = tf.reduce_mean(tf.square(target_q_values - q_value))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def decrease_epsilon(self):
        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay

# Main Training Loop
def train_snake():
    pygame.init()
    env = SnakeGame()
    state_size = env.get_state().shape[0]
    action_size = len(ACTIONS)
    agent = DQNAgent(state_size, action_size)

    episode = 0
    while episode < max_epsilon_steps:
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.memory.add((state, action, reward, next_state, done))
            agent.train()
            state = next_state
            total_reward += reward
            agent.decrease_epsilon()
            env.render()  # Update the GUI
            time.sleep(0.1)

        # Update the target model periodically
        if episode % target_update_frequency == 0:
            agent.update_target_model()

        print(f"Episode {episode + 1}/{max_epsilon_steps}, Total Reward: {total_reward}")
        episode += 1

    pygame.quit()

# Start Training
train_snake()
