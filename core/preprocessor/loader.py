from ..utils import alloc_logger
from ..config import config
import re
from bs4 import BeautifulSoup


logger = alloc_logger("Loader.log")


class Info:
    def __init__(self, ID, content, positive:bool, is_raw: bool = False):
        self.ID = ID
        self.content = content
        self.positive = positive

    def __str__(self):
        return "Info[id={:d}, content={}, positive={}]".format(self.ID, self.content, self.positive)

class Loader:
    def __init__(self, csv_file_name: str):
        self.csv_file = open(csv_file_name, 'r', encoding='utf8')
        self.csv_file.readline()        # remove header
        self.reg = re.compile("(\\d+),(.*),((positive)|(negative))")

        self.logger = alloc_logger("Loader.log", Loader)

    def __iter__(self):
        return self

    def __next__(self):
        m = None
        while m is None:
            line = self.csv_file.readline().strip()
            if not line:
                raise StopIteration
            m = self.reg.fullmatch(line)
            if m is None:
                self.logger.log_message("unmatch with line content:\t", line)

        ID = int(m.group(1))
        content = m.group(2)
        positive = m.group(3) == "positive"

        content = BeautifulSoup(content, 'html.parser').text.replace('\n\n','')

        return Info(ID, content, positive)

def walk_csv(csv_file_name) -> Loader:
    return Loader(csv_file_name)

def list_csv(csv_file_name) -> list:
    return list(walk_csv(csv_file_name))

if __name__ == "__main__":
    pass