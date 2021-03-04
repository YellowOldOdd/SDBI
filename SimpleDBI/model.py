from abc import ABCMeta, abstractmethod

class Model(metaclass=ABCMeta):

    @abstractmethod
    def load(self, model_dict):
        pass
    
    @abstractmethod
    def forward(self, maxbytes=-1) :
        pass

    