from common import *
import pygame as pg
from NNnumber_train import get_nn
from drawing_grid import update_grid
from drawing_answers import update_answers
from drawing_messages import update_messages, MessageList
from inputs_handler import pressing_mouse, key_down, key_up

pg.init()
pg.font.init()
common = init_common(get_nn(100))
clock = pg.time.Clock()
assert_sizes(common.nn)

common.messages = MessageList.create(["Welcome to the number recognizer", "Press t to train the network", "Press s to save the network", "Press l to load the network", "Press space to clear the board", "Press a number to load a sample", "Press a number while holding ctrl to change the answer", "Press escape to exit"])

def while_training():
    new = next(common.training_iter)
    if new is None:
        common.training_iter = None
        common.is_training = False
        common.messages.put("Training stopped")
    else:
        for message in new:
            common.messages.put(message)
     
def update():
    common.screen.fill(BACKGROUND)
    update_grid(common)
    update_answers(common)
    update_messages(common, common.messages)
    text = common.small_impact.render(f"train: {common.train_for}", False, FILLED_CELL)
    common.screen.blit(text, (ANSWER_OFFSET_X+ANSWER_OFFSET_Y, int(11*ANSWER_OFFSET_Y)-FONT_SIZE//4))
    pg.display.flip()

while common.running:
    clock.tick(FPS)
    if common.is_training:
        while_training()
    for event in pg.event.get():
        if event.type == pg.QUIT:
            common.running = False
        elif event.type == pg.KEYDOWN:
            key_down(common, event.key)
        elif event.type == pg.KEYUP:
            key_up(common, event.key)
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
