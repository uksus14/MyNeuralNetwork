from NNnumber_train import get_nn, train, save_nn, load_nn

nn = get_nn(20, 20)
train_for = 128
ANSWERS = list(range(10))+["Nothing"]

import pygame as pg
pg.init()

FPS = 60

SIDEBAR_WIDTH, WINDOW_HEIGHT = 300, 750
GAP = 4
CIRCLE_RADIUS = 25
GRID_HEIGHT = 5
GRID_WIDTH = 3
FONT_SIZE = 30
assert nn.i.width == GRID_HEIGHT * GRID_WIDTH
assert nn.o.width == len(ANSWERS)

BACKGROUND = (150, 150, 150)
EMPTY_CELL = (255, 255, 255)
FILLED_CELL = (0, 0, 0)
CIRCLE = (0, 0, 0)

CELL_SIZE = WINDOW_HEIGHT // GRID_HEIGHT - GAP
WINDOW_WIDTH = (CELL_SIZE + GAP) * GRID_WIDTH + SIDEBAR_WIDTH
ANSWER_OFFSET_Y = WINDOW_HEIGHT // (len(ANSWERS) + 1)
ANSWER_OFFSET_X = WINDOW_HEIGHT - SIDEBAR_WIDTH + ANSWER_OFFSET_Y

screen = pg.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])
clock = pg.time.Clock()
pg.font.init()
impact = pg.font.SysFont('impact', FONT_SIZE)
small_impact = pg.font.SysFont('impact', FONT_SIZE//2)

def draw_cell(x: int, y: int, cell: bool):
    left = GAP//2 + (GAP + CELL_SIZE)*x
    top = GAP//2 + (GAP + CELL_SIZE)*y
    rect = ((left, top), (CELL_SIZE, CELL_SIZE))
    color = EMPTY_CELL
    if cell:
        color = FILLED_CELL
    pg.draw.rect(screen, color, rect)

def draw_grid():
    grid = nn.inputs
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            draw_cell(x, y, grid[0, y*GRID_WIDTH+x])

def draw_answer(answer, i):
    top = ANSWER_OFFSET_Y*(i+1)
    pg.draw.circle(screen, FILLED_CELL, (ANSWER_OFFSET_X, top), answer*CIRCLE_RADIUS)
    text = impact.render(f'{ANSWERS[i]}', False, FILLED_CELL)
    screen.blit(text, (ANSWER_OFFSET_X+ANSWER_OFFSET_Y, top-FONT_SIZE//2))

def draw_answers():
    answers = nn.answers()
    for i in range(len(ANSWERS)):
        draw_answer(answers[0, i], i)

def update():
    screen.fill(BACKGROUND)
    draw_grid()
    draw_answers()
    text = small_impact.render(f"train: {train_for}", False, FILLED_CELL)
    screen.blit(text, (ANSWER_OFFSET_X+2*ANSWER_OFFSET_Y, int(11.5*ANSWER_OFFSET_Y)-FONT_SIZE//4))
    pg.display.flip()

PG_NUMBERS = [pg.K_KP0, pg.K_KP1, pg.K_KP2, pg.K_KP3, pg.K_KP4, pg.K_KP5, pg.K_KP6, pg.K_KP7, pg.K_KP8, pg.K_KP9, pg.K_BACKSPACE][:len(ANSWERS)]

def to_grid(x: int, y: int):
    gx = x//(CELL_SIZE+GAP)
    gy = y//(CELL_SIZE+GAP)
    if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
        return gx, gy
    return None

def pressing_key():
    if key is not None:
        nn.update(key)
def pressing_mouse():
    if mouse is not None:
        cell = to_grid(*pg.mouse.get_pos())
        if cell is None:
            return
        x, y = cell
        nn.change_input(y*GRID_WIDTH+x, mouse)
def pressing_update():
    pressing_key()
    pressing_mouse()

key = None
mouse = None
running = True
while running:
    clock.tick(FPS)
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                nn.clear_input()
                continue
            elif event.key == pg.K_t:
                train(nn, train_for)
                continue
            elif event.key == pg.K_s:
                save_nn(nn)
                continue
            elif event.key == pg.K_l:
                nn = load_nn()
                continue
            try:
                pressed_key = PG_NUMBERS.index(event.key)
            except ValueError:
                continue
            key = pressed_key
        elif event.type == pg.KEYUP:
            key = None
        elif event.type == pg.MOUSEBUTTONDOWN:
            cell = to_grid(*pg.mouse.get_pos())
            if cell is None:
                continue
            x, y = cell
            mouse = 1-nn.inputs[0, y*GRID_WIDTH+x]
        elif event.type == pg.MOUSEBUTTONUP:
            mouse = None
        elif event.type == pg.MOUSEWHEEL:
            if event.y>0:
                train_for *= 2
            elif train_for > 1:
                train_for //= 2

    pressing_update()
    update()

pg.quit()
