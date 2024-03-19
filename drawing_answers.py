from common import *
import pygame as pg

def draw_answer(common: c_data, answer: float, i: int):
    top = ANSWER_OFFSET_Y*(i+1)
    pg.draw.circle(common.screen, FILLED_CELL, (ANSWER_OFFSET_X, top), answer*CIRCLE_RADIUS)
    text = common.impact.render(f'{ANSWERS[i]}', False, FILLED_CELL)
    common.screen.blit(text, (ANSWER_OFFSET_X+ANSWER_OFFSET_Y, top-FONT_SIZE//2))

def update_answers(common: c_data):
    answers = common.nn.answers()
    for i in range(len(ANSWERS)):
        draw_answer(common, answers[0, i], i)