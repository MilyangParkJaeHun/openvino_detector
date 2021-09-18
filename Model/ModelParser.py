from abc import ABC, abstractmethod

class ModelParser(ABC):

    @abstractmethod
    def parse_output(self, model, output):
        pass

    @abstractmethod
    def get_output(self, model):
        pass