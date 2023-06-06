import pygame
import random

# Инициализация Pygame
pygame.init()

# Размер окна
window_width = 800
window_height = 600

# Цвета
black = (0, 0, 0)
green = (0, 255, 0)
red = (255, 0, 0)

# Создание игрового окна
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Змейка")

# Размер ячейки и скорость змейки
cell_size = 20
snake_speed = 10

# Позиция и начальное направление змейки
snake_x = window_width // 2
snake_y = window_height // 2
snake_dx = 0
snake_dy = 0

# Инициализация змейки
snake = [(snake_x, snake_y)]

# Создание первой яблоко
apple_x = random.randint(0, window_width - cell_size) // cell_size * cell_size
apple_y = random.randint(0, window_height - cell_size) // cell_size * cell_size

# Основной игровой цикл
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and snake_dy != cell_size:
                snake_dx = 0
                snake_dy = -cell_size
            elif event.key == pygame.K_DOWN and snake_dy != -cell_size:
                snake_dx = 0
                snake_dy = cell_size
            elif event.key == pygame.K_LEFT and snake_dx != cell_size:
                snake_dx = -cell_size
                snake_dy = 0
            elif event.key == pygame.K_RIGHT and snake_dx != -cell_size:
                snake_dx = cell_size
                snake_dy = 0

    # Перемещение змейки
    snake_x += snake_dx
    snake_y += snake_dy

    # Проверка столкновения с границами окна
    if snake_x < 0 or snake_x >= window_width or snake_y < 0 or snake_y >= window_height:
        running = False

    # Проверка столкновения с самой собой
    if (snake_x, snake_y) in snake[:-1]:
        running = False

    # Добавление головы змейки
    snake.append((snake_x, snake_y))

    # Обрезка хвоста змейки
    if len(snake) > 1:
        snake = snake[-1 * len(snake):]

    # Проверка столкновения с яблоком
    if snake_x == apple_x and snake_y == apple_y:
        # Создание нового яблока
        apple_x = random.randint(0, window_width - cell_size) // cell_size * cell_size
        apple_y = random.randint(0, window_height - cell_size) // cell_size * cell_size
    else:
        # Удаление последнего сегмента хвоста
        snake = snake[:-1]

    # Очистка экрана
    window.fill(black)

    # Отрисовка змейки
    for segment in snake:
        pygame.draw.rect(window, green, (segment[0], segment[1], cell_size, cell_size))

    # Отрисовка яблока
    pygame.draw.rect(window, red, (apple_x, apple_y, cell_size, cell_size))

    # Обновление экрана
    pygame.display.update()

    # Ограничение скорости
    clock.tick(snake_speed)

# Завершение Pygame
pygame.quit()
