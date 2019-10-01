#-*- coding:utf-8 -*-

import logging
import os

FORMATTER = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'

class Logger():
    def __init__(self,path,Clevel=logging.DEBUG,Flevel=logging.DEBUG):
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(FORMATTER)
        sh = logging.StreamHandler()
        sh.setLevel(Clevel)
        sh.setFormatter(fmt)
        fh = logging.FileHandler(path)
        fh.setLevel(Flevel)
        fh.setFormatter(fmt)

        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

if __name__ == '__main__':
    PATH = r'../data/demo14/demo15.log'
    logger = Logger(PATH)
    logger.debug('debug message')