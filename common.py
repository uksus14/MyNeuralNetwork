from layers import NN


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

def get_number(key) -> int|None:
    import pygame as pg
    PG_KP_NUMBERS = [pg.K_KP0, pg.K_KP1, pg.K_KP2, pg.K_KP3, pg.K_KP4, pg.K_KP5, pg.K_KP6, pg.K_KP7, pg.K_KP8, pg.K_KP9, pg.K_BACKSPACE][:len(ANSWERS)]
    PG_NUMBERS = [pg.K_0, pg.K_1, pg.K_2, pg.K_3, pg.K_4, pg.K_5, pg.K_6, pg.K_7, pg.K_8, pg.K_9, pg.K_BACKSPACE][:len(ANSWERS)]
    if key in PG_KP_NUMBERS:
        return PG_KP_NUMBERS.index(key)
    if key in PG_NUMBERS:
        return PG_NUMBERS.index(key)
    return None


def assert_sizes(nn: NN):
    assert nn.i.width == GRID_HEIGHT * GRID_WIDTH
    assert nn.o.width == len(ANSWERS)

class c_data:
    def __init__(self, nn: NN):
        import pygame as pg

        self.nn = nn
        self.screen = pg.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])
        self.train_for = 128
        self.impact = pg.font.SysFont('impact', FONT_SIZE)
        self.small_impact = pg.font.SysFont('impact', FONT_SIZE//2)
        self.ctrl = False
        self.mouse = None
        self.running = True
        self.is_training = False
        self.training_iter = None
        self.messages = None
common: c_data = None
def init_common(nn: NN) -> c_data:
    global common
    if common is None:
        common = c_data(nn)
    return common