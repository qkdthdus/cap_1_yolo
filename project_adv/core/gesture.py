class HandFSM:
    def __init__(self):
        self.was_closed = False

    def update(self, is_closed, is_open):
        trigger_ready = is_open and self.was_closed
        self.was_closed = is_closed
        return trigger_ready
