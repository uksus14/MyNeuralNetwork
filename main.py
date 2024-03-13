from NNnumber_train import get_nn, train, save_nn, load_nn, draw
from scheme import scheme

nn = get_nn(scheme)
train_for = 128
ANSWERS = list(range(10))

import pygame as pg
pg.init()

FPS = 60

SIDEBAR_WIDTH, WINDOW_HEIGHT = 300, 756
GAP = 4
CIRCLE_RADIUS = 25
GRID_HEIGHT = 28
GRID_WIDTH = 28
FONT_SIZE = 30
nn.clear_input(GRID_HEIGHT, GRID_WIDTH)
assert nn.o.width == len(ANSWERS)

DRAWING_NEIGHBORHOOD = (            (0, -1, .4),
                        (-1, 0, .4), (0, 0, 1),  (1, 0, .4),
                                    (0, 1, .4))

BACKGROUND = (150, 150, 150)
EMPTY_CELL = (255, 255, 255)
FILLED_CELL = (0, 0, 0)
CIRCLE = (0, 0, 0)

CELL_SIZE = WINDOW_HEIGHT // GRID_HEIGHT - GAP
WINDOW_WIDTH = (CELL_SIZE + GAP) * GRID_WIDTH + SIDEBAR_WIDTH
ANSWER_OFFSET_Y = WINDOW_HEIGHT // (len(ANSWERS) + 1)
ANSWER_OFFSET_X = WINDOW_WIDTH - SIDEBAR_WIDTH + ANSWER_OFFSET_Y

screen = pg.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])
clock = pg.time.Clock()
pg.font.init()
impact = pg.font.SysFont('impact', FONT_SIZE)
small_impact = pg.font.SysFont('impact', FONT_SIZE//2)

def draw_cell(x: int, y: int, cell: float):
    left = GAP//2 + (GAP + CELL_SIZE)*x
    top = GAP//2 + (GAP + CELL_SIZE)*y
    rect = ((left, top), (CELL_SIZE, CELL_SIZE))
    color = tuple(int(empty*(1-cell)+filled*cell) for empty, filled in zip(EMPTY_CELL, FILLED_CELL))
    pg.draw.rect(screen, color, rect)

def draw_grid():
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            draw_cell(x, y, nn.inputs[y, x, 0])

def draw_answer(answer: float, i: int):
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
    screen.blit(text, (ANSWER_OFFSET_X+ANSWER_OFFSET_Y, int(10*ANSWER_OFFSET_Y)-FONT_SIZE//4))
    pg.display.flip()

PG_KP_NUMBERS = [pg.K_KP0, pg.K_KP1, pg.K_KP2, pg.K_KP3, pg.K_KP4, pg.K_KP5, pg.K_KP6, pg.K_KP7, pg.K_KP8, pg.K_KP9, pg.K_BACKSPACE][:len(ANSWERS)]
PG_NUMBERS = [pg.K_0, pg.K_1, pg.K_2, pg.K_3, pg.K_4, pg.K_5, pg.K_6, pg.K_7, pg.K_8, pg.K_9, pg.K_BACKSPACE][:len(ANSWERS)]

def to_grid(x: int, y: int):
    gx = x//(CELL_SIZE+GAP)
    gy = y//(CELL_SIZE+GAP)
    if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
        return gx, gy
    return None

def drawing(x: int, y: int):
    for dx, dy, add in DRAWING_NEIGHBORHOOD:
        cell = nn.inputs[y+dy, x+dx]
        if cell is not None:
            cell = min(cell + add, 1)
            nn.change_input(cell, x+dx, y+dy)

def pressing_key(frame: int):
    if key is not None:
        nn.update(key)
def pressing_mouse(frame: int):
    if mouse is None:return
    mouse_coords = pg.mouse.get_pos()
    cell = to_grid(*mouse_coords)
    if cell is None:
        return
    x, y = cell
    if mouse:
        drawing(x, y)
    else:
        nn.change_input(0, x, y)
def pressing_update(frame: int):
    pressing_key(frame)
    pressing_mouse(frame)

key = None
mouse = None
running = True
frame = 0
alt = "0"
while running:
    clock.tick(FPS)
    frame = (frame + 1)%60
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                running = False
            if event.key in [pg.K_LALT, pg.K_RALT]:
                alt = "0"
            if event.key == pg.K_SPACE:
                nn.clear_input(GRID_HEIGHT, GRID_WIDTH)
            if event.key == pg.K_t:
                train(nn, train_for)
            if event.key == pg.K_s:
                save_nn(nn)
            if event.key == pg.K_l:
                nn = load_nn()
            if event.key in PG_KP_NUMBERS:
                pressed_key = PG_KP_NUMBERS.index(event.key)
                if alt:
                    alt += str(pressed_key)
                else:
                    key = pressed_key
            if event.key in PG_NUMBERS:
                pressed_key = PG_NUMBERS.index(event.key)
                if alt:
                    alt += str(pressed_key)
                else:
                    draw(nn, pressed_key)
        elif event.type == pg.KEYUP:
            if event.key in [pg.K_LALT, pg.K_RALT]:
                train_for = int(alt) or 1
                alt = ""
            key = None
        elif event.type == pg.MOUSEBUTTONDOWN:
            mouse = True
        elif event.type == pg.MOUSEBUTTONUP:
            mouse = None
        elif event.type == pg.MOUSEWHEEL:
            if event.y>0:
                train_for *= 2
            elif train_for > 1:
                train_for //= 2

    pressing_update(frame)
    update()

pg.quit()
