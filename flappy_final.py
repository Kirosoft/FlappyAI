import pygame, sys, random, numpy as np, os
from pygame.locals import *
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize pygame and the mixer for sound
pygame.init()
pygame.mixer.init()

# Global flag to disable sounds during training.
TRAINING_MODE = False

# Load MP3 files for sound effects (ensure these files exist in the same folder)
try:
    flap_sound = pygame.mixer.Sound("flap.mp3")
    score_sound = pygame.mixer.Sound("score.mp3")
    hit_sound = pygame.mixer.Sound("hit.mp3")
except Exception as e:
    print("Error loading MP3 files:", e)
    flap_sound = score_sound = hit_sound = None

# Constants and settings
WIDTH, HEIGHT = 400, 600
FPS = 30
GRAVITY = 1
FLAP_STRENGTH = -12
PIPE_GAP = 150
PIPE_WIDTH = 80
PIPE_SPEED = 4

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Setup display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Novel Twist Flappy Bird with DQN")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

# Device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent with Experience Replay
class DQNAgent:
    def __init__(self):
        self.input_dim = 4  # [bird_y, norm_velocity, norm_pipe_distance, norm_pipe_gap_center]
        self.output_dim = 2  # [do nothing, flap]
        self.policy_net = DQN(self.input_dim, self.output_dim).to(device)
        self.target_net = DQN(self.input_dim, self.output_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.update_target_every = 10  # episodes
        self.episodes_done = 0

    def select_action(self, state):
        # state is a numpy array of shape (4,)
        if np.random.rand() < self.epsilon:
            return random.randrange(self.output_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # Current Q values
        q_values = self.policy_net(states).gather(1, actions)
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        # Compute target
        target = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Global DQN agent instance
agent = DQNAgent()

# Function to get continuous, normalized state from bird and pipe
def get_state(bird, pipe):
    state = np.array([
        bird.y / HEIGHT,
        (bird.velocity + 20) / 40,         # Normalize velocity (expected roughly -20 to +20)
        (pipe.x - bird.x) / WIDTH,
        pipe.gap_center / HEIGHT
    ], dtype=np.float32)
    return state

# Classes for game objects
class Bird:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.x = 50
        self.y = HEIGHT // 2
        self.velocity = 0
        self.radius = 20
        
    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity
        
    def flap(self):
        self.velocity = FLAP_STRENGTH
        if not TRAINING_MODE and flap_sound:
            flap_sound.play()
            
    def draw(self, surface, score):
        # Bird color changes based on score
        color = (min(255, 100 + int(score * 5) % 155), 100, 200)
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), self.radius)
        
class Pipe:
    def __init__(self, x):
        self.x = x
        self.set_height()
        
    def set_height(self):
        self.gap_center = random.randint(100, HEIGHT - 100)
        self.top = self.gap_center - PIPE_GAP // 2
        self.bottom = self.gap_center + PIPE_GAP // 2
        
    def update(self):
        self.x -= PIPE_SPEED
        
    def draw(self, surface):
        pygame.draw.rect(surface, (34, 139, 34), (self.x, 0, PIPE_WIDTH, self.top))
        pygame.draw.rect(surface, (34, 139, 34), (self.x, self.bottom, PIPE_WIDTH, HEIGHT - self.bottom))
        
    def collide(self, bird):
        if bird.x + bird.radius > self.x and bird.x - bird.radius < self.x + PIPE_WIDTH:
            if bird.y - bird.radius < self.top or bird.y + bird.radius > self.bottom:
                return True
        return False

# Global game variables for play modes
bird = Bird()
pipes = []
score = 0.0
game_over = False

def reset_game():
    global bird, pipes, score, game_over
    bird.reset()
    pipes = [Pipe(WIDTH + 100)]
    score = 0.0
    game_over = False

def draw_text(text, size, color, center):
    font_obj = pygame.font.SysFont("Arial", size)
    text_surface = font_obj.render(text, True, color)
    rect = text_surface.get_rect(center=center)
    screen.blit(text_surface, rect)

def main_menu():
    while True:
        screen.fill(WHITE)
        draw_text("Novel Flappy Bird Twist", 32, BLACK, (WIDTH//2, HEIGHT//4))
        draw_text("1. Train AI (DQN)", 28, BLACK, (WIDTH//2, HEIGHT//2 - 40))
        draw_text("2. Play with AI", 28, BLACK, (WIDTH//2, HEIGHT//2))
        draw_text("3. Play (Human)", 28, BLACK, (WIDTH//2, HEIGHT//2 + 40))
        draw_text("Press 1, 2, or 3", 24, BLACK, (WIDTH//2, HEIGHT//2 + 100))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_1:
                    train_dqn()
                    return
                elif event.key == K_2:
                    play_with_dqn()
                    return
                elif event.key == K_3:
                    play_human()
                    return

def train_dqn():
    global TRAINING_MODE
    print("Training with Deep Q-Learning started...")
    TRAINING_MODE = True  # disable sounds during training
    num_episodes = 500
    max_steps = 1000
    episode_scores = []
    
    for ep in range(num_episodes):
        local_bird = Bird()
        local_pipe = Pipe(WIDTH + 100)
        local_score = 0.0
        state = get_state(local_bird, local_pipe)
        done = False
        
        for step in range(max_steps):
            action = agent.select_action(state)
            if action == 1:
                local_bird.flap()
            local_bird.update()
            local_pipe.update()
            
            reward = 1.0 / FPS  # reward for staying alive
            
            # Check if pipe passed
            if local_pipe.x + PIPE_WIDTH < 0:
                local_pipe = Pipe(WIDTH + 100)
                local_score += 1
                reward += 10
                
            # Check for collision or out-of-bound
            if (local_bird.y - local_bird.radius < 0 or 
                local_bird.y + local_bird.radius > HEIGHT or 
                local_pipe.collide(local_bird)):
                reward = -100
                done = True
                
            next_state = get_state(local_bird, local_pipe)
            agent.store_experience(state, action, reward, next_state, done)
            agent.optimize_model()
            state = next_state
            local_score += 1.0 / FPS
            
            if done:
                break
        
        episode_scores.append(local_score)
        # Decay epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        # Update target network every few episodes
        agent.episodes_done += 1
        if agent.episodes_done % agent.update_target_every == 0:
            agent.update_target_network()
        if (ep + 1) % 50 == 0:
            avg_score = sum(episode_scores[-50:]) / 50
            print(f"Episode {ep+1}/{num_episodes}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    # Save the trained model
    torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
    print("Training complete! Model saved to 'dqn_model.pth'")
    TRAINING_MODE = False
    main_menu()

def play_with_dqn():
    global bird, pipes, score, game_over
    reset_game()
    pipes = [Pipe(WIDTH + 100)]
    # If a trained model exists, load it
    if os.path.exists("dqn_model.pth"):
        agent.policy_net.load_state_dict(torch.load("dqn_model.pth", map_location=device))
        agent.policy_net.eval()
        print("Loaded trained DQN model.")
    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            # When game is over, press space to return to main menu.
            if game_over and event.type == KEYDOWN and event.key == K_SPACE:
                main_menu()
                return
                
        if not game_over:
            current_pipe = pipes[0]
            state = get_state(bird, current_pipe)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = agent.policy_net(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
            if action == 1:
                bird.flap()
            bird.update()
            score += 1.0 / FPS
            
            if pipes[-1].x < WIDTH - 300:
                pipes.append(Pipe(WIDTH))
            if pipes[0].x + PIPE_WIDTH < 0:
                pipes.pop(0)
                score += 1
                if score_sound and not TRAINING_MODE:
                    score_sound.play()
                    
            for pipe in pipes:
                pipe.update()
                if pipe.collide(bird):
                    game_over = True
                    if hit_sound and not TRAINING_MODE:
                        hit_sound.play()
            if bird.y - bird.radius < 0 or bird.y + bird.radius > HEIGHT:
                game_over = True
                if hit_sound and not TRAINING_MODE:
                    hit_sound.play()
        
        bg_color = ((int(score)*5)%256, (int(score)*3)%256, (int(score)*7)%256)
        screen.fill(bg_color)
        for pipe in pipes:
            pipe.draw(screen)
        bird.draw(screen, score)
        draw_text(f"Score: {int(score)}", 24, WHITE, (WIDTH//2, 30))
        if game_over:
            draw_text("Game Over! Press SPACE to return to Main Menu", 28, WHITE, (WIDTH//2, HEIGHT//2))
        pygame.display.update()

def play_human():
    global bird, pipes, score, game_over
    reset_game()
    pipes = [Pipe(WIDTH + 100)]
    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            # When game is over, press space to return to main menu.
            if game_over and event.type == KEYDOWN and event.key == K_SPACE:
                main_menu()
                return
            if event.type == KEYDOWN:
                if event.key == K_SPACE and not game_over:
                    bird.flap()
                    
        if not game_over:
            bird.update()
            score += 1.0 / FPS
            if pipes[-1].x < WIDTH - 300:
                pipes.append(Pipe(WIDTH))
            if pipes[0].x + PIPE_WIDTH < 0:
                pipes.pop(0)
                score += 1
                if score_sound and not TRAINING_MODE:
                    score_sound.play()
            for pipe in pipes:
                pipe.update()
                if pipe.collide(bird):
                    game_over = True
                    if hit_sound and not TRAINING_MODE:
                        hit_sound.play()
            if bird.y - bird.radius < 0 or bird.y + bird.radius > HEIGHT:
                game_over = True
                if hit_sound and not TRAINING_MODE:
                    hit_sound.play()
                    
        bg_color = ((int(score)*5)%256, (int(score)*3)%256, (int(score)*7)%256)
        screen.fill(bg_color)
        for pipe in pipes:
            pipe.draw(screen)
        bird.draw(screen, score)
        draw_text(f"Score: {int(score)}", 24, WHITE, (WIDTH//2, 30))
        if game_over:
            draw_text("Game Over! Press SPACE to return to Main Menu", 28, WHITE, (WIDTH//2, HEIGHT//2))
        pygame.display.update()

# Optionally load an existing DQN model at startup
if os.path.exists("dqn_model.pth"):
    agent.policy_net.load_state_dict(torch.load("dqn_model.pth", map_location=device))
    agent.policy_net.eval()
    print("Loaded existing DQN model from 'dqn_model.pth'.")

if __name__ == "__main__":
    main_menu()
