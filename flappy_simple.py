import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Game settings
WIDTH = 400
HEIGHT = 600
FPS = 60
GRAVITY = 0.25
JUMP_STRENGTH = -5
PIPE_WIDTH = 50
PIPE_HEIGHT = 500
PIPE_GAP = 150
BIRD_WIDTH = 40
BIRD_HEIGHT = 40

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Create the screen object
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird")

# Fonts
font = pygame.font.SysFont('Arial', 30)

# Game objects
bird = pygame.Rect(100, HEIGHT // 2, BIRD_WIDTH, BIRD_HEIGHT)
bird_velocity = 0

# Pipes list
pipes = []

# Clock to control the frame rate
clock = pygame.time.Clock()

# Function to create a new pipe
def create_pipe():
    height = random.randint(100, HEIGHT - PIPE_GAP - 100)
    top_pipe = pygame.Rect(WIDTH, 0, PIPE_WIDTH, height)
    bottom_pipe = pygame.Rect(WIDTH, height + PIPE_GAP, PIPE_WIDTH, HEIGHT - (height + PIPE_GAP))
    return top_pipe, bottom_pipe

# Function to display the bird
def draw_bird():
    pygame.draw.rect(screen, BLUE, bird)

# Function to display the pipes
def draw_pipes():
    for pipe in pipes:
        pygame.draw.rect(screen, GREEN, pipe)

# Function to check for collisions
def check_collisions():
    if bird.top <= 0 or bird.bottom >= HEIGHT:
        return True
    for pipe in pipes:
        if bird.colliderect(pipe):
            return True
    return False

# Function to display the score
def display_score(score):
    score_text = font.render(f"Score: {score}", True, BLACK)
    screen.blit(score_text, (10, 10))

# Game loop
def game_loop():
    global pipes  # Declare pipes as global to modify the global variable
    bird.y = HEIGHT // 2
    bird_velocity = 0
    pipes.clear()  # Now it's recognized as the global variable
    score = 0
    pipe_timer = 0
    game_over = False

    while True:
        screen.fill(WHITE)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not game_over:
                    bird_velocity = JUMP_STRENGTH

        if not game_over:
            # Update bird position
            bird_velocity += GRAVITY
            bird.y += bird_velocity

            # Update pipes
            pipe_timer += 1
            if pipe_timer >= 90:  # Create a new pipe every 90 frames
                pipes.extend(create_pipe())
                pipe_timer = 0

            # Move pipes to the left
            pipes = [pipe.move(-3, 0) for pipe in pipes if pipe.right > 0]

            # Check for collisions
            if check_collisions():
                game_over = True

            # Update score
            for pipe in pipes:
                if pipe.right == bird.left:
                    score += 1

        # Draw game objects
        draw_bird()
        draw_pipes()
        display_score(score)

        if game_over:
            game_over_text = font.render("Game Over! Press R to Restart", True, BLACK)
            screen.blit(game_over_text, (WIDTH // 3, HEIGHT // 2))

            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                game_loop()

        pygame.display.update()
        clock.tick(FPS)


# Start the game
game_loop()
