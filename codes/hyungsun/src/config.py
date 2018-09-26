import configparser


class ConfigManager(object):
    """Model & Training configurations controller.

        Usage :
            model = CNN()
            config = Config().load(model)
            learning_rate = config["LEARNING_RATE"]
    """
    CONFIG_FILE_NAME = "config/config.ini"
    DEFAULT_SECTION = "DEFAULT"

    def __init__(self, section):
        self.config = configparser.ConfigParser()
        if len(self.config.read(self.CONFIG_FILE_NAME)) == 0:
            raise FileNotFoundError("File '{}' does not exist.".format(self.CONFIG_FILE_NAME))
        self.section = section

    def load(self):
        section = self.section if self.config.has_section(self.section) else self.DEFAULT_SECTION
        return self.config[section]
