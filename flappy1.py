import pygame, sys, random, math, numpy as np
from pygame.locals import *

# Initialize pygame
pygame.init()

# Constants and settings
WIDTH, HEIGHT = 400, 600
FPS = 30
GRAVITY = 1
FLAP_STRENGTH = -12
PIPE_GAP = 150
PIPE_WIDTH = 80
PIPE_SPEED = 4

# Q-learning settings
STATE_BINS = (10, 10, 10, 10)  # discretization bins for (bird_y, bird_velocity, pipe_dist, pipe_gap_center)
LEARNING_RATE = 0.7
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1  # exploration rate
TRAIN_EPISODES = 500

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Setup display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Novel Twist Flappy Bird with AI")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

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

    def draw(self, surface):
        # Novel twist: bird color changes with score (or time)
        color = (min(255, 100 + score * 5 % 155), 100, 200)
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), self.radius)

class Pipe:
    def __init__(self, x):
        self.x = x
        self.set_height()

    def set_height(self):
        # Randomly set the pipe gap vertical center
        self.gap_center = random.randint(100, HEIGHT - 100)
        self.top = self.gap_center - PIPE_GAP // 2
        self.bottom = self.gap_center + PIPE_GAP // 2

    def update(self):
        self.x -= PIPE_SPEED

    def draw(self, surface):
        pygame.draw.rect(surface, (34, 139, 34), (self.x, 0, PIPE_WIDTH, self.top))
        pygame.draw.rect(surface, (34, 139, 34), (self.x, self.bottom, PIPE_WIDTH, HEIGHT - self.bottom))

    def collide(self, bird):
        # Check collision with bird (circle vs rect)
        if bird.x + bird.radius > self.x and bird.x - bird.radius < self.x + PIPE_WIDTH:
            if bird.y - bird.radius < self.top or bird.y + bird.radius > self.bottom:
                return True
        return False

# Q-Learning Agent for Flappy Bird
class QLearningAgent:
    def __init__(self):
        # Q-table dimensions based on discretized state and 2 actions: flap or not
        self.q_table = np.zeros(STATE_BINS + (2,))

    def discretize(self, bird, pipe):
        # Normalize and discretize continuous values
        # bird y: 0 to HEIGHT, velocity: -20 to +20, pipe distance: 0 to WIDTH, pipe gap center: 0 to HEIGHT
        bird_y = int(np.clip(bird.y / HEIGHT * (STATE_BINS[0] - 1), 0, STATE_BINS[0] - 1))
        bird_vel = int(np.clip((bird.velocity + 20) / 40 * (STATE_BINS[1] - 1), 0, STATE_BINS[1] - 1))
        pipe_dist = int(np.clip((pipe.x - bird.x) / WIDTH * (STATE_BINS[2] - 1), 0, STATE_BINS[2] - 1))
        pipe_gap = int(np.clip(pipe.gap_center / HEIGHT * (STATE_BINS[3] - 1), 0, STATE_BINS[3] - 1))
        return (bird_y, bird_vel, pipe_dist, pipe_gap)

    def choose_action(self, state, epsilon=EPSILON):
        if np.random.random() < epsilon:
            return np.random.choice([0, 1])
        else:
            return np.argmax(self.q_table[state])

    def update_q(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        self.q_table[state + (action,)] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * best_next - self.q_table[state + (action,)])

# Global Q-agent (will be trained via "Train AI" option)
agent = QLearningAgent()

# Global game variables
bird = Bird()
pipes = []
pipe_timer = 0
score = 0
game_over = False

def reset_game():
    global bird, pipes, pipe_timer, score, game_over
    bird.reset()
    pipes = [Pipe(WIDTH + 100)]
    pipe_timer = 0
    score = 0
    game_over = False

def draw_text(text, size, color, center):
    font_obj = pygame.font.SysFont("Arial", size)
    text_surface = font_obj.render(text, True, color)
    rect = text_surface.get_rect(center=center)
    screen.blit(text_surface, rect)

def main_menu():
    while True:
        screen.fill(WHITE)
        draw_text("Novel Flappy Bird Twist", 32, BLACK, (WIDTH // 2, HEIGHT // 4))
        draw_text("1. Train AI", 28, BLACK, (WIDTH // 2, HEIGHT // 2 - 20))
        draw_text("2. Play with AI", 28, BLACK, (WIDTH // 2, HEIGHT // 2 + 20))
        draw_text("3. Play", 28, BLACK, (WIDTH // 2, HEIGHT // 2 + 60))
        draw_text("Press 1, 2, or 3", 24, BLACK, (WIDTH // 2, HEIGHT // 2 + 120))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_1:
                    train_ai()
                    return
                elif event.key == K_2:
                    play_with_ai()
                    return
                elif event.key == K_3:
                    play_human()
                    return

def train_ai():
    global agent
    print("Training started...")
    episodes = TRAIN_EPISODES
    for ep in range(episodes):
        # Initialize a new episode
        local_bird = Bird()
        local_pipe = Pipe(WIDTH + 100)
        local_score = 0
        done = False

        while not done:
            state = agent.discretize(local_bird, local_pipe)
            action = agent.choose_action(state, epsilon=EPSILON)
            # action 1 = flap, 0 = do nothing
            if action == 1:
                local_bird.flap()
            local_bird.update()
            local_pipe.update()

            # If pipe moved off screen, reset it and increment score
            if local_pipe.x + PIPE_WIDTH < 0:
                local_pipe = Pipe(WIDTH + 100)
                local_score += 1
                reward = 10  # reward for passing a pipe
            else:
                reward = 0.1  # small reward for staying alive

            # Collision or hitting ground/ceiling gives heavy penalty and ends episode
            if local_bird.y - local_bird.radius < 0 or local_bird.y + local_bird.radius > HEIGHT or local_pipe.collide(local_bird):
                reward = -100
                done = True

            next_state = agent.discretize(local_bird, local_pipe)
            agent.update_q(state, action, reward, next_state)
    print("Training complete! Episodes trained:", episodes)
    # After training, return to main menu
    main_menu()

def play_with_ai():
    global bird, pipes, score, game_over, pipe_timer
    reset_game()
    pipes = [Pipe(WIDTH + 100)]
    running = True
    while running:
        clock.tick(FPS)
        # Event handling for restart
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_r and game_over:
                    reset_game()

        if not game_over:
            # Use AI to decide action
            current_pipe = pipes[0]
            state = agent.discretize(bird, current_pipe)
            action = np.argmax(agent.q_table[state])
            if action == 1:
                bird.flap()
            bird.update()

            # Create new pipe when needed
            if pipes[-1].x < WIDTH - 200:
                pipes.append(Pipe(WIDTH))
            # Remove off-screen pipes
            if pipes[0].x + PIPE_WIDTH < 0:
                pipes.pop(0)
                score += 1

            for pipe in pipes:
                pipe.update()
                if pipe.collide(bird):
                    game_over = True

            # Check boundaries
            if bird.y - bird.radius < 0 or bird.y + bird.radius > HEIGHT:
                game_over = True

        # Draw everything
        # Novel twist: change background based on score (cycles colors)
        bg_color = ((score * 5) % 256, (score * 3) % 256, (score * 7) % 256)
        screen.fill(bg_color)
        for pipe in pipes:
            pipe.draw(screen)
        bird.draw(screen)
        draw_text(f"Score: {score}", 24, WHITE, (WIDTH // 2, 30))
        if game_over:
            draw_text("Game Over! Press R to Restart", 28, WHITE, (WIDTH // 2, HEIGHT // 2))
        pygame.display.update()

def play_human():
    global bird, pipes, score, game_over, pipe_timer
    reset_game()
    pipes = [Pipe(WIDTH + 100)]
    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type == KEYDOWN:
                # Use space bar for flapping
                if event.key == K_SPACE and not game_over:
                    bird.flap()
                if event.key == K_r and game_over:
                    reset_game()

        if not game_over:
            bird.update()
            if pipes[-1].x < WIDTH - 200:
                pipes.append(Pipe(WIDTH))
            if pipes[0].x + PIPE_WIDTH < 0:
                pipes.pop(0)
                score += 1

            for pipe in pipes:
                pipe.update()
                if pipe.collide(bird):
                    game_over = True

            if bird.y - bird.radius < 0 or bird.y + bird.radius > HEIGHT:
                game_over = True

        # Draw everything
        bg_color = ((score * 5) % 256, (score * 3) % 256, (score * 7) % 256)
        screen.fill(bg_color)
        for pipe in pipes:
            pipe.draw(screen)
        bird.draw(screen)
        draw_text(f"Score: {score}", 24, WHITE, (WIDTH // 2, 30))
        if game_over:
            draw_text("Game Over! Press R to Restart", 28, WHITE, (WIDTH // 2, HEIGHT // 2))
        pygame.display.update()

if __name__ == "__main__":
    main_menu()
