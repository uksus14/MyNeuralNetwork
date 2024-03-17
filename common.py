from layers import NN
import pygame as pg

ANSWERS = list(range(10))
FPS = 60

SIDEBAR_WIDTH, WINDOW_HEIGHT = 300, 756
GAP = 4
CIRCLE_RADIUS = 25
GRID_HEIGHT = 28
GRID_WIDTH = 28
FONT_SIZE = 30

DRAWING_NEIGHBORHOOD = (            (0, -1, .4),
                        (-1, 0, .4), (0, 0, 1),  (1, 0, .4),
                                    (0, 1, .4))

BACKGROUND = (150, 150, 150)
EMPTY_CELL = (255, 255, 255)
FILLED_CELL = (0, 0, 0)
CIRCLE = (0, 0, 0)

CELL_SIZE = WINDOW_HEIGHT // GRID_HEIGHT - GAP
WINDOW_WIDTH = (CELL_SIZE + GAP) * GRID_WIDTH + SIDEBAR_WIDTH
ANSWER_OFFSET_Y = WINDOW_HEIGHT // (len(ANSWERS) + 2)
ANSWER_OFFSET_X = WINDOW_WIDTH - SIDEBAR_WIDTH + ANSWER_OFFSET_Y

class c_data:
    def __init__(self, nn: NN):
        self.nn = nn
        self.screen = pg.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])
        self.train_for = 128
        self.impact = pg.font.SysFont('impact', FONT_SIZE)
        self.small_impact = pg.font.SysFont('impact', FONT_SIZE//2)
        self.ctrl = False
        self.mouse = None
common: c_data = None
def init_common(nn: NN) -> c_data:
    global common
    if common is None:
        common = c_data(nn)
    return common