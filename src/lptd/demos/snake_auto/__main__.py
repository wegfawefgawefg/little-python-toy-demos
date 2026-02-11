import pygame, sys, random
from collections import deque

# --- Configuration ---
CELL_SIZE = 20
GRID_WIDTH, GRID_HEIGHT = 30, 20
SCREEN_SIZE = (CELL_SIZE * GRID_WIDTH, CELL_SIZE * GRID_HEIGHT)
FPS = 14400
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)


# --- BFS Pathfinding ---
def bfs(start, goal, obstacles):
    queue = deque([[start]])
    visited = set([start])
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if (x, y) == goal:
            return path[1:]  # exclude starting cell
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                if (nx, ny) not in obstacles and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(path + [(nx, ny)])
    return None


# --- Snake Class ---
class Snake:
    def __init__(self, color, init_pos):
        self.color = color
        self.body = deque([init_pos])
        self.direction = (1, 0)
        self.next_move = None

    def head(self):
        return self.body[0]

    def update_direction(self, apple, obstacles):
        # Compute obstacles (exclude tail because it moves)
        obs = set(obstacles)
        if len(self.body) > 1:
            obs -= {self.body[-1]}
        path = bfs(self.head(), apple, obs)
        if path:
            # next move from path
            self.next_move = path[0]
        else:
            # fallback: try all directions
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = self.head()[0] + dx, self.head()[1] + dy
                if (
                    0 <= nx < GRID_WIDTH
                    and 0 <= ny < GRID_HEIGHT
                    and (nx, ny) not in obs
                ):
                    self.next_move = (nx, ny)
                    return
            self.next_move = self.head()  # no valid move

    def move(self, apple_eaten):
        # Insert next move at head; if not eaten, pop tail.
        if self.next_move:
            self.body.appendleft(self.next_move)
            if not apple_eaten:
                self.body.pop()


# --- Main Game ---
def main():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    clock = pygame.time.Clock()

    # Initialize two snakes in different corners
    snake1 = Snake(GREEN, (5, 5))
    snake2 = Snake(BLUE, (GRID_WIDTH - 6, GRID_HEIGHT - 6))
    snakes = [snake1, snake2]

    # Place apple at random free cell
    def new_apple():
        while True:
            pos = (
                random.randint(0, GRID_WIDTH - 1),
                random.randint(0, GRID_HEIGHT - 1),
            )
            occupied = any(pos in s.body for s in snakes)
            if not occupied:
                return pos

    apple = new_apple()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Build obstacles: positions occupied by snakes
        obstacles = set()
        for s in snakes:
            obstacles.update(s.body)

        # Each snake picks its next move
        for s in snakes:
            # For pathfinding, obstacles include the other snake’s body
            other_obs = obstacles - {s.body[-1]}
            # allow its own tail cell to be free
            s.update_direction(apple, other_obs)

        # Conflict resolution: if both snakes plan the same next cell, let snake2 try alternate
        if snake1.next_move == snake2.next_move:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                alt = (snake2.head()[0] + dx, snake2.head()[1] + dy)
                if (
                    0 <= alt[0] < GRID_WIDTH
                    and 0 <= alt[1] < GRID_HEIGHT
                    and alt not in obstacles
                ):
                    snake2.next_move = alt
                    break
            else:
                # if no alternate, snake2 stays in place
                snake2.next_move = snake2.head()

        # Check if any snake eats the apple
        apple_eaten = [False, False]
        for idx, s in enumerate(snakes):
            if s.next_move == apple:
                apple_eaten[idx] = True

        # Move snakes (each grows if it ate the apple, else moves normally)
        for idx, s in enumerate(snakes):
            s.move(apple_eaten[idx])

        # If any snake ate the apple, reposition apple
        if any(apple_eaten):
            apple = new_apple()

        # Draw everything
        screen.fill(BLACK)
        # Draw apple
        pygame.draw.rect(
            screen,
            RED,
            (apple[0] * CELL_SIZE, apple[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
        )
        # Draw snakes
        for s in snakes:
            for seg in s.body:
                pygame.draw.rect(
                    screen,
                    s.color,
                    (seg[0] * CELL_SIZE, seg[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                )
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
