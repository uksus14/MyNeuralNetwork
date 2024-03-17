from NNnumber_train import get_nn, train, save_nn, load_nn, load, training_parallel
import pygame as pg
from common import init_common
from grid_utils import *
from main_utils import *

pg.init()
pg.font.init()
common = init_common(get_nn(100))
clock = pg.time.Clock()
assert_sizes(common.nn)
messages = MessageList.create(["Welcome to the number recognizer", "Press t to train the network", "Press s to save the network", "Press l to load the network", "Press space to clear the board", "Press a number to load a sample", "Press a number while holding ctrl to change the answer", "Press escape to exit"])
def stop_training():
    global training_iter, is_training
    training_iter = None
    is_training = False
    messages.put("Training stopped")

def while_training():
    new = next(training_iter)
    if new is None:
        return stop_training()
    for message in new:
        messages.put(message)
    
def update():
    common.screen.fill(BACKGROUND)
    draw_grid(common)
    draw_answers(common)
    draw_messages(common, messages)
    text = common.small_impact.render(f"train: {common.train_for}", False, FILLED_CELL)
    common.screen.blit(text, (ANSWER_OFFSET_X+ANSWER_OFFSET_Y, int(11*ANSWER_OFFSET_Y)-FONT_SIZE//4))
    pg.display.flip()

running = True
is_training = False
training_iter = None
while running:
    clock.tick(FPS)
    if is_training:
        while_training()
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                running = False
            if event.key in [pg.K_LCTRL, pg.K_RCTRL]:
                common.ctrl = True
            if event.key == pg.K_SPACE:
                common.nn.clear_input()
            elif event.key == pg.K_t:
                if common.ctrl:
                    train(common.nn, common.train_for)
                else:
                    training_iter = training_parallel(common.nn, common.train_for)
                    is_training = True
            elif event.key == pg.K_s:
                save_nn(common.nn)
            elif event.key == pg.K_l:
                common.nn = load_nn()
            else:
                try:
                    pressed_key = PG_NUMBERS.index(event.key)
                    load(common.nn, pressed_key)
                except ValueError:
                    pass
        elif event.type == pg.KEYUP:
            if event.key in [pg.K_LCTRL, pg.K_RCTRL]:
                common.ctrl = False
        elif event.type == pg.MOUSEBUTTONDOWN:
            common.mouse = True
        elif event.type == pg.MOUSEBUTTONUP:
            common.mouse = None
        elif event.type == pg.MOUSEWHEEL:
            if event.y>0:
                common.train_for *= 2
            elif common.train_for > 1:
                common.train_for //= 2

    pressing_mouse(common)
    update()

pg.quit()
