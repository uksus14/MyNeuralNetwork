from datetime import datetime, timedelta
from math import sqrt
from typing import Self
from grid_utils import c_data

move_time = timedelta(milliseconds=200)
appear_time = timedelta(seconds=1)
initial_time = timedelta(seconds=3)

def now() -> datetime:
    return datetime.now()

class Message:
    func = sqrt
    def __init__(self, message: str, stay: timedelta = appear_time):
        self.message = message
        self._stay = stay
        self._start = now()
        self._from_place = -1
        self._to_place = -1
        self._start_shift = now()-move_time
    def is_valid(self) -> bool:
        return self.start+self.stay<now()
    def opasity(self) -> float:
        return self.func(1-(now()-self.start)/self.stay) if self.is_valid() else 0
    @property
    def place(self) -> int:
        if self._start_shift + move_time < now():
            self._from_place = self._to_place
        return self._from_place + (self._to_place-self._from_place)*(now() - self._start_shift)/move_time
    def change_place(self, place: int):
        self._to_place = place
        self._start_shift = now()

class MessageList:
    def __init__(self):
        self.messages: list[Message] = []
    def fix_places(self):
        for i, message in enumerate(self.messages):
            message.change_place(i)
    def put(self, message: str|Message, stay: timedelta = None):
        if not isinstance(message, Message):
            if stay is None:
                message = Message(message)
            else:
                message = Message(message, stay)
        self.messages.append(message)
        self.fix_places()
    @classmethod
    def create(cls, messages: list[str], stay: timedelta = initial_time) -> Self:
        list = cls()
        list.messages = [Message(message, stay) for message in messages]
        list.fix_places()
        return list
    def filter(self):
        self.messages = [message for message in self.messages if message.is_valid()]
        self.fix_places()
    def get(self) -> list[Message]:
        return self.messages
    
def draw_messages(common: c_data, messages: MessageList):
    for message in messages.get():
        pass