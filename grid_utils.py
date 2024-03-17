from layers import NN
import pygame as pg
from common import *

def draw_cell(common: c_data, x: int, y: int, cell: float):
    left = GAP//2 + (GAP + CELL_SIZE)*x
    top = GAP//2 + (GAP + CELL_SIZE)*y
    rect = ((left, top), (CELL_SIZE, CELL_SIZE))
    color = tuple(int(empty*(1-cell)+filled*cell) for empty, filled in zip(EMPTY_CELL, FILLED_CELL))
    pg.draw.rect(common.screen, color, rect)

def draw_grid(common: c_data):
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            draw_cell(common, x, y, common.nn.get_input(x, y))

def draw_answer(common: c_data, answer: float, i: int):
    top = ANSWER_OFFSET_Y*(i+1)
    pg.draw.circle(common.screen, FILLED_CELL, (ANSWER_OFFSET_X, top), answer*CIRCLE_RADIUS)
    text = common.impact.render(f'{ANSWERS[i]}', False, FILLED_CELL)
    common.screen.blit(text, (ANSWER_OFFSET_X+ANSWER_OFFSET_Y, top-FONT_SIZE//2))

def draw_answers(common: c_data):
    answers = common.nn.answers()
    for i in range(len(ANSWERS)):
        draw_answer(common, answers[0, i], i)

PG_KP_NUMBERS = [pg.K_KP0, pg.K_KP1, pg.K_KP2, pg.K_KP3, pg.K_KP4, pg.K_KP5, pg.K_KP6, pg.K_KP7, pg.K_KP8, pg.K_KP9, pg.K_BACKSPACE][:len(ANSWERS)]
PG_NUMBERS = [pg.K_0, pg.K_1, pg.K_2, pg.K_3, pg.K_4, pg.K_5, pg.K_6, pg.K_7, pg.K_8, pg.K_9, pg.K_BACKSPACE][:len(ANSWERS)]

def to_grid(x: int, y: int):
    gx = x//(CELL_SIZE+GAP)
    gy = y//(CELL_SIZE+GAP)
    if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
        return gx, gy
    return None

def drawing(nn: NN, x: int, y: int):
    for dx, dy, add in DRAWING_NEIGHBORHOOD:
        cell = nn.get_input(x+dx, y+dy)
        if cell is not None:
            cell = min(cell + add, 1)
            nn.change_input(cell, x+dx, y+dy)

def pressing_mouse(common: c_data):
    if common.mouse is None:return
    mouse_coords = pg.mouse.get_pos()
    cell = to_grid(*mouse_coords)
    if cell is None:
        return
    x, y = cell
    if common.mouse:
        drawing(common.nn, x, y)
    else:
        common.nn.change_input(0, x, y)

def assert_sizes(nn: NN):
    assert nn.i.width == GRID_HEIGHT * GRID_WIDTH
    assert nn.o.width == len(ANSWERS)