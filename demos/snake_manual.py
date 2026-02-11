import pygame
import random
from collections import namedtuple

# --- Data Structures ---
State = namedtuple("State", ["snake", "direction", "food", "score", "game_over"])
# snake: list of (x, y) positions, head is first element.
# direction: (dx, dy)
# food: (x, y)
# score: int
# game_over: bool

# Constants
WIDTH, HEIGHT = 400, 400
BLOCK = 20
GRID_WIDTH = WIDTH // BLOCK
GRID_HEIGHT = HEIGHT // BLOCK


# --- Functional Helpers ---
def add_vectors(a, b):
    """Associative vector addition (Semigroup); identity is (0,0)."""
    return (a[0] + b[0], a[1] + b[1])


# Simple Functor to chain state transformations
class Functor:
    def __init__(self, value):
        self.value = value

    def map(self, func):
        return Functor(func(self.value))

    def get(self):
        return self.value


# --- Pure Game Logic Functions ---
def handle_input(state, events):
    """Processes events to update direction."""
    key_map = {
        pygame.K_UP: (0, -1),
        pygame.K_DOWN: (0, 1),
        pygame.K_LEFT: (-1, 0),
        pygame.K_RIGHT: (1, 0),
    }
    for event in events:
        if event.type == pygame.KEYDOWN:
            new_dir = key_map.get(event.key)
            if new_dir:
                # Prevent reversing: new_dir + current != (0,0)
                if add_vectors(state.direction, new_dir) != (0, 0):
                    state = state._replace(direction=new_dir)
    return state


def move_snake(state):
    """Moves the snake. If the new head is on the food, the snake grows; else, it moves normally."""
    new_head = add_vectors(state.snake[0], state.direction)
    # Determine if we should grow (i.e. if apple is eaten)
    if new_head == state.food:
        new_snake = [new_head] + state.snake  # Keep entire body: growth
    else:
        new_snake = [new_head] + state.snake[:-1]  # Regular move: drop tail
    return state._replace(snake=new_snake)


def check_collisions(state):
    """Checks for wall or self collisions."""
    head = state.snake[0]
    x, y = head
    if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
        return state._replace(game_over=True)
    # Check self collision (only if snake length > 1)
    if len(state.snake) > 1 and head in state.snake[1:]:
        return state._replace(game_over=True)
    return state


def update_food_and_score(state):
    """If the snake has eaten the food, update score and generate new food."""
    if state.snake[0] == state.food:
        new_food = (
            random.randint(0, GRID_WIDTH - 1),
            random.randint(0, GRID_HEIGHT - 1),
        )
        return state._replace(food=new_food, score=state.score + 1)
    return state


def game_tick(state, events):
    """Chains pure functions to update the game state."""
    return (
        Functor(state)
        .map(lambda s: handle_input(s, events))
        .map(move_snake)
        .map(check_collisions)
        .map(lambda s: update_food_and_score(s) if not s.game_over else s)
        .get()
    )


# --- Rendering ---
def draw_state(screen, state):
    screen.fill((0, 0, 0))
    # Draw snake
    for block in state.snake:
        rect = pygame.Rect(block[0] * BLOCK, block[1] * BLOCK, BLOCK, BLOCK)
        pygame.draw.rect(screen, (0, 255, 0), rect)
    # Draw food
    food_rect = pygame.Rect(state.food[0] * BLOCK, state.food[1] * BLOCK, BLOCK, BLOCK)
    pygame.draw.rect(screen, (255, 0, 0), food_rect)
    pygame.display.flip()


# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    # Initial game state: snake starts at center moving right.
    init_snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
    state = State(
        snake=init_snake,
        direction=(1, 0),
        food=(random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)),
        score=0,
        game_over=False,
    )

    while not state.game_over:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                state = state._replace(game_over=True)
        state = game_tick(state, events)
        draw_state(screen, state)
        clock.tick(10)  # 10 frames per second

    pygame.quit()
    print("Game Over! Score:", state.score)


if __name__ == "__main__":
    main()
