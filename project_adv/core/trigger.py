import time

class TriggerManager:
    def __init__(self, cooldown=3.0):
        self.cooldown = cooldown
        self.cooldown_until = 0

    def can_trigger(self):
        return time.time() > self.cooldown_until

    def fire(self):
        self.cooldown_until = time.time() + self.cooldown
