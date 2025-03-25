import pygame, sys, random, math, pickle, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ----- Constants -----
WIDTH, HEIGHT = 400, 600
FPS = 60
GRAVITY = 0.5
JUMP_VELOCITY = -10
PIPE_GAP = 150
PIPE_SPACING = 250  # increased horizontal spacing for easier gameplay

# ----- Colors -----
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SKY_BLUE = (135, 206, 235)
GREEN = (0, 255, 0)

# ----- Pygame Initialization -----
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Novel Flappy Bird with AI")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

# ----- Load Sound Effects -----
jump_sound = pygame.mixer.Sound("jump.wav") if os.path.exists("jump.wav") else None
hit_sound = pygame.mixer.Sound("hit.wav") if os.path.exists("hit.wav") else None
score_sound = pygame.mixer.Sound("score.wav") if os.path.exists("score.wav") else None

# ----- Game Classes -----
class Bird:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = WIDTH // 4
        self.y = HEIGHT // 2
        self.vel = 0
        self.radius = 20

    def jump(self):
        self.vel = JUMP_VELOCITY
        if jump_sound: jump_sound.play()

    def update(self):
        self.vel += GRAVITY
        self.y += self.vel

    def draw(self, surf):
        pygame.draw.circle(surf, BLACK, (int(self.x), int(self.y)), self.radius)

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius,
                           self.radius * 2, self.radius * 2)

class Pipe:
    def __init__(self, x):
        self.x = x
        self.gap_y = random.randint(100, HEIGHT - 100)
        self.width = 50

    def update(self):
        self.x -= 2  # speed

    def draw(self, surf):
        # top pipe
        top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y - PIPE_GAP // 2)
        # bottom pipe
        bottom_rect = pygame.Rect(self.x, self.gap_y + PIPE_GAP // 2, self.width, HEIGHT - self.gap_y)
        pygame.draw.rect(surf, GREEN, top_rect)
        pygame.draw.rect(surf, GREEN, bottom_rect)

    def collides(self, bird):
        bird_rect = bird.get_rect()
        top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y - PIPE_GAP // 2)
        bottom_rect = pygame.Rect(self.x, self.gap_y + PIPE_GAP // 2, self.width, HEIGHT - self.gap_y)
        return bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect)

# ----- DQN Agent and Network -----
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.memory = deque(maxlen=5000)
        self.gamma = 0.99
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.lr = 0.001
        self.batch_size = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # load model if exists
        if os.path.exists("dqn_model_one_shot.pth"):
            self.model.load_state_dict(torch.load("dqn_model_one_shot.pth"))
        if os.path.exists("dqn_memory_one_shot.pkl"):
            with open("dqn_memory_one_shot.pkl", "rb") as f:
                self.memory = pickle.load(f)

    def get_state(self, bird, pipes):
        # Use relative position to next pipe as state.
        # Find next pipe
        next_pipe = None
        for pipe in pipes:
            if pipe.x + pipe.width > bird.x:
                next_pipe = pipe
                break
        if not next_pipe:
            next_pipe = Pipe(WIDTH + PIPE_SPACING)

        # Normalize state values
        state = np.array([
            bird.y / HEIGHT,
            bird.vel / 10,
            (next_pipe.x - bird.x) / WIDTH,
            (next_pipe.gap_y - bird.y) / HEIGHT
        ], dtype=np.float32)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        action_batch = torch.LongTensor(np.array([b[1] for b in batch])).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(np.array([b[2] for b in batch])).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        done_batch = torch.FloatTensor(np.array([b[4] for b in batch])).to(self.device)

        # current Q values
        q_values = self.model(state_batch).gather(1, action_batch).squeeze()

        # next Q values
        next_q_values = self.model(next_state_batch).max(1)[0]
        target = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        loss = self.criterion(q_values, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        torch.save(self.model.state_dict(), "dqn_model_one_shot.pth")
        with open("dqn_memory_one_shot.pkl", "wb") as f:
            pickle.dump(self.memory, f)

# ----- Game Loop Functions -----
def game_loop(mode="manual", agent=None, train_mode=False):
    bird = Bird()
    pipes = [Pipe(WIDTH + i * PIPE_SPACING) for i in range(3)]
    score = 0
    run = True
    frame_count = 0
    reward = 0

    while run:
        clock.tick(FPS)
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if agent: agent.save()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if mode == "manual" and event.key == pygame.K_SPACE:
                    bird.jump()

        # If AI is playing, decide action
        state = agent.get_state(bird, pipes) if agent else None
        action = 0
        if mode == "ai" and agent:
            action = agent.act(state)
            if action == 1:
                bird.jump()
        elif train_mode and agent:
            # In training mode we use the agent's action to interact
            action = agent.act(state)
            if action == 1:
                bird.jump()

        bird.update()

        # update pipes
        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.update()
            if pipe.collides(bird):
                if hit_sound: hit_sound.play()
                reward = -100
                if agent:
                    next_state = agent.get_state(bird, pipes)
                    agent.remember(state, action, reward, next_state, True)
                run = False
            if pipe.x + pipe.width < 0:
                rem.append(pipe)
                add_pipe = True
            # Score when passing a pipe
            if pipe.x + pipe.width < bird.x and not hasattr(pipe, 'scored'):
                score += 1
                reward = 10
                if score_sound: score_sound.play()
                pipe.scored = True

        for r in rem:
            pipes.remove(r)
        if add_pipe:
            pipes.append(Pipe(WIDTH))

        # Check if bird hits floor or goes off-screen
        if bird.y - bird.radius < 0 or bird.y + bird.radius > HEIGHT:
            if hit_sound: hit_sound.play()
            reward = -100
            if agent:
                next_state = agent.get_state(bird, pipes)
                agent.remember(state, action, reward, next_state, True)
            run = False

        # In training mode, get next state and remember transition
        if agent and (mode=="ai" or train_mode):
            next_state = agent.get_state(bird, pipes)
            agent.remember(state, action, reward, next_state, False)
            agent.replay()

        # Draw everything
        screen.fill(SKY_BLUE)
        for pipe in pipes:
            pipe.draw(screen)
        bird.draw(screen)
        score_text = font.render("Score: " + str(score), True, BLACK)
        screen.blit(score_text, (10, 10))
        pygame.display.flip()
        frame_count += 1

    # End of game: show game over screen
    game_over_text = font.render("Game Over! Press SPACE to go back", True, BLACK)
    screen.blit(game_over_text, (20, HEIGHT // 2))
    pygame.display.flip()
    # Wait for SPACE to return
    waiting = True
    while waiting:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if agent: agent.save()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False

def main_menu():
    menu_options = ["Play Manually", "Play with AI", "Train"]
    selected = 0
    agent = DQNAgent(state_dim=4, action_dim=2)  # 0: do nothing, 1: jump

    while True:
        screen.fill(WHITE)
        title_text = font.render("Novel Flappy Bird", True, BLACK)
        screen.blit(title_text, (WIDTH//2 - title_text.get_width()//2, 50))
        for i, option in enumerate(menu_options):
            color = BLACK if i == selected else (100,100,100)
            text = font.render(option, True, color)
            screen.blit(text, (WIDTH//2 - text.get_width()//2, 150 + i*50))
        info_text = font.render("Use UP/DOWN arrows and ENTER", True, BLACK)
        screen.blit(info_text, (WIDTH//2 - info_text.get_width()//2, 400))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                agent.save()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(menu_options)
                if event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(menu_options)
                if event.key == pygame.K_RETURN:
                    # Option selected
                    if menu_options[selected] == "Play Manually":
                        game_loop(mode="manual")
                    elif menu_options[selected] == "Play with AI":
                        game_loop(mode="ai", agent=agent)
                    elif menu_options[selected] == "Train":
                        # In training mode, run several episodes
                        episodes = 50
                        for ep in range(episodes):
                            game_loop(mode="train", agent=agent, train_mode=True)
                            print(f"Episode {ep+1}/{episodes} complete.")
                        agent.save()
        clock.tick(FPS)

if __name__ == "__main__":
    main_menu()
