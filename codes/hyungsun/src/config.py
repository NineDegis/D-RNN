"""
    Model & Training configurations controller.

    Usage :
        model = CNN()
        config = Config().load(model)
        learning_rate = config["LEARNING_RATE"]
"""
import configparser

CONFIG_FILE_NAME = "config/config.ini"
DEFAULT_SECTION = "DEFAULT"


class ConfigManager(object):
    def __init__(self, model):
        self.config = configparser.ConfigParser()
        if len(self.config.read(CONFIG_FILE_NAME)) == 0:
            raise FileNotFoundError("File '{}' does not exist.".format(CONFIG_FILE_NAME))
        self.model_name = model.__class__.__name__

    def load(self):
        section = self.model_name if self.config.has_section(self.model_name) else DEFAULT_SECTION
        return self.config[section]
