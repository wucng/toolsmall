#!/usr/bin/env python
# encoding: utf-8

import os
import datetime
import logging.config

def init(filedir:os.pardir):
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(filedir)
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    LOG_FILE = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + str(os.getpid()) + ".log"

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    LOGGING = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                'format': '%(asctime)s [%(name)s:%(lineno)d] [%(levelname)s]- %(message)s'
            },
            'standard': {
                'format': '%(asctime)s [%(threadName)s:%(thread)d] [%(name)s:%(lineno)d] [%(levelname)s]- %(message)s'
            },
        },

        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                # "stream": "ext://sys.stdout"
            },

            "default": {
                "class": "logging.handlers.RotatingFileHandler",
                # 'class': 'cloghandler.ConcurrentRotatingFileHandler',
                "level": "INFO",
                "formatter": "simple",
                "filename": os.path.join(LOG_DIR, LOG_FILE),
                'mode': 'w+',
                "maxBytes": 1024 * 1024 * 10,  # 5 MB
                "backupCount": 20,
                "encoding": "utf8"
            },
        },
        "root": {
            'handlers': ['default', 'console'],
            'level': "INFO",
            'propagate': False
        }
    }

    logging.config.dictConfig(LOGGING)


def get_logger(filedir:os.pardir,name="detection"):
    init(filedir)
    log = logging.getLogger(name)
    return log


if __name__ == "__main__":
    log = get_logger(filedir="./")

    log.info("")
    log.warn("")
    log.debug("")
    # except Exception as e:
    # log.exception(e)