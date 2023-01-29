import logging
import time
import os

__all__ = ['Logger']

LEVEL_MAP = {
    'debug':logging.DEBUG,
    'info':logging.INFO,
    'warning':logging.WARNING,
    'error':logging.ERROR,
    'critical':logging.CRITICAL
}

DEFAULT_FORMAT = "%(asctime)s - %(pathname)s [line:%(lineno)d] - %(levelname)s : %(message)s"

LOG_DIR = r"G:\python_logs"

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

def get_today():
    return time.strftime("%Y-%m-%d",time.localtime())

def get_now():
    return time.strftime("%H-%M-%S",time.localtime())

class Logger:
    def __init__(self,filename,level='debug',level_for_screen='info',fmt=DEFAULT_FORMAT):
        '''
        :param filename:
        :param level:
        :param level_for_screen:
        :param fmt:
        '''
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)

        self.logger.setLevel(LEVEL_MAP[level])

        sh = logging.StreamHandler()
        sh.setLevel((LEVEL_MAP[level_for_screen]))
        sh.setFormatter(format_str)

        _sub_dir = get_now()
        fh = logging.FileHandler(self._get_true_file_name(filename,_sub_dir),encoding='utf-8')
        sh.setLevel((LEVEL_MAP[level]))
        sh.setFormatter(format_str)

        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def _get_true_file_name(self,filename,_sub_dir):
        sub_dir = os.path.join(LOG_DIR,get_today())
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        if not os.path.exists(os.path.join(sub_dir,_sub_dir)):
            os.makedirs(os.path.join(sub_dir,_sub_dir))

        t_filename = os.path.join(sub_dir,_sub_dir,'{}'.format(filename))

        return t_filename

    def get_logger(self):
        return self.logger

if __name__ == '__main__':
    logger = Logger('test','debug').get_logger()
    logger.info('fuck')
