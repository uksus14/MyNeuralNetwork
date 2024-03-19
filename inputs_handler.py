from NNnumber_train import *
from common import c_data, get_number
from drawing_grid import to_grid, drawing


def pressing_mouse(common: c_data):
    if common.mouse is None:return
    import pygame as pg
    mouse_coords = pg.mouse.get_pos()
    cell = to_grid(*mouse_coords)
    if cell is None:
        return
    x, y = cell
    if common.mouse:
        drawing(common.nn, x, y)
    else:
        common.nn.change_input(0, x, y)

def key_down(common: c_data, key):
    import pygame as pg
    if key == pg.K_ESCAPE:
        common.running = False
    if key in [pg.K_LCTRL, pg.K_RCTRL]:
        common.ctrl = True
    if key == pg.K_SPACE:
        common.nn.clear_input()
    elif key == pg.K_t:
        if common.ctrl:
            train(common.nn, common.train_for)
        else:
            common.training_iter = training_parallel(common.nn, common.train_for)
            common.is_training = True
    elif key == pg.K_s:
        save_nn(common.nn)
    elif key == pg.K_l:
        common.nn = load_nn()
    else:
        pressed_key = get_number(key)
        if pressed_key is not None:
            load(common.nn, pressed_key)

def key_up(common: c_data, key):
    import pygame as pg
    if key in [pg.K_LCTRL, pg.K_RCTRL]:
        common.ctrl = False