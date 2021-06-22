# coding: U8
import logging
import sys
import time

__all__ = ["setup_glog_stdout"]


class GlogFormatter(logging.Formatter):
    LEVEL_MAP = {
        logging.FATAL: 'F',  # FATAL is alias of CRITICAL
        logging.ERROR: 'E',
        logging.WARN: 'W',
        logging.INFO: 'I',
        logging.DEBUG: 'D'
    }

    def __init__(self):
        super(GlogFormatter, self).__init__()

    @staticmethod
    def _format_message(record):
        try:
            record_message = '%s' % (record.msg % record.args)
        except TypeError:
            record_message = record.msg
        return record_message

    def format(self, record):
        try:
            level = GlogFormatter.LEVEL_MAP[record.levelno]
        except KeyError:
            level = '?'
        date = time.localtime(record.created)
        date_usec = (record.created - int(record.created)) * 1e6
        record_message = '%c%02d%02d %02d:%02d:%02d.%06d %s %s:%d] %s' % (
            level, date.tm_mon, date.tm_mday, date.tm_hour, date.tm_min,
            date.tm_sec, date_usec,
            record.process if record.process is not None else '?????',
            record.filename,
            record.lineno,
            self._format_message(record))
        record.getMessage = lambda: record_message
        return logging.Formatter.format(self, record)


def setup_glog_stdout(logger):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(GlogFormatter())
    logger.addHandler(handler)
    return logger
