from NNnumber_train import get_nn

nn = get_nn(20, 20)

import pygame as pg
pg.init()

SIDEBAR_WIDTH, WINDOW_HEIGHT = 300, 750
GAP = 4
CIRCLE_RADIUS = 25
GRID_HEIGHT = 5
GRID_WIDTH = 3

BACKGROUND = (150, 150, 150)
EMPTY_CELL = (255, 255, 255)
FILLED_CELL = (0, 0, 0)
CIRCLE = (0, 0, 0)

CELL_SIZE = WINDOW_HEIGHT//GRID_HEIGHT-GAP
WINDOW_WIDTH = (CELL_SIZE+GAP)*GRID_WIDTH+SIDEBAR_WIDTH

screen = pg.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])

def draw_grid():
    nn.
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            pg.draw.rect(screen, )

def update():
    screen.fill(BACKGROUND)
    draw_grid()
    grid = nn.answers()
    pg.draw.circle(screen, (0, 0, 255), (250, 250), 75)
    pg.display.flip()

running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    update()

pg.quit()
