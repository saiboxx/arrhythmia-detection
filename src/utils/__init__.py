from datetime import datetime
import yaml


class timer:
    """
    Utility class for a simple tracking of wall time.
    """
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print(self.name + " ...")
        self.start = datetime.now()

    def __exit__(self, type, value, traceback):
        self.stop = datetime.now()
        delta = self.stop - self.start
        seconds = delta.seconds
        minutes, seconds_of_minute = divmod(seconds, 60)
        hours, minutes_of_hour = divmod(minutes, 60)
        print(self.name + " took {:02}:{:02}:{:02}".format(int(hours), int(minutes_of_hour), int(seconds_of_minute)))


def load_config() -> dict:
    """
    Loads the config.yml file to memory and returns it as dictionary.
    :return: Dictionary containing the config.
    """
    with open('config.yml', 'r') as ymlfile:
        return yaml.load(ymlfile, Loader=yaml.FullLoader)

