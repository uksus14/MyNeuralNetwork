from layers import NN
import pygame as pg
from common import *

def draw_cell(common: c_data, x: int, y: int, cell: float):
    left = GAP//2 + (GAP + CELL_SIZE)*x
    top = GAP//2 + (GAP + CELL_SIZE)*y
    rect = ((left, top), (CELL_SIZE, CELL_SIZE))
    color = tuple(int(empty*(1-cell)+filled*cell) for empty, filled in zip(EMPTY_CELL, FILLED_CELL))
    pg.draw.rect(common.screen, color, rect)

def update_grid(common: c_data):
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            draw_cell(common, x, y, common.nn.get_input(x, y))

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