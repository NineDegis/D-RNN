import torch


class ConstSingleton:
    _instance = None

    @classmethod
    def __getInstance(cls):
        return cls._instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls._instance = cls(*args, **kargs)
        cls.instance = cls.__getInstance
        return cls._instance

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise Exception("Can't rebind const(%s)" % name)
        self.__dict__[name] = value


class ConfigRNN(ConstSingleton):
    """Set Hyper-parameters of models in here.
    """

    def __init__(self):
        # [Train]
        self.LEARNING_RATE = 0.1
        self.MAX_EPOCH = 3000
        self.WEIGHT_DECAY = 0.0003
        self.CRITERION = torch.nn.CrossEntropyLoss()

        # [Model]
        self.BI_DIRECTION = True
        self.HIDDEN_SIZE = 100
        self.OUTPUT_SIZE = 2  # output is one of pos([1, 0]) and neg([0, 1]).
        self.BATCH_SIZE = 32
        self.VOCAB_SIZE = 89527  # It is only used when the `EMBED_METHOD` is "DEFAULT"

        # [Data]
        self.PAD_WORD = "<PAD>"
        self.EMBED_SIZE = 100
        self.MAX_SEQ_SIZE = 5000
        self.EMBED_METHOD = "SKIP_GRAM"
        self.SHUFFLE = True

        # [ETC]
        self.DEBUG_MODE = True
        self.LOGGING_ENABLE = False
        self.CHECKPOINT_ENABLE = False
