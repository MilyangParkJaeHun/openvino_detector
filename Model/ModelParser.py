"""
    ModelParser.py

    Author: Park Jaehun
    Refactoring: Park Jaehun , 2021.09.18
"""

from abc import ABC, abstractmethod

class ModelParser(ABC):
    @abstractmethod
    def get_output(self, model):
        """
        Get inference results from model output layer.
        """
        pass

    @abstractmethod
    def parse_output(self, model, output):
        """
        Parse results in form of bounding box from model's ouput.
        """
        pass