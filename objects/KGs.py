from objects.KG import KG
from config.PARISConfig import PARISConfig


class KGs:
    def __init__(self, kg1: KG, kg2: KG, paris_config=None):
        self.kg_l = kg1
        self.kg_r = kg2
        self.PARIS_config = paris_config

        self.__init()

    def __init(self):
        if self.PARIS_config is None:
            self.PARIS_config = PARISConfig()

