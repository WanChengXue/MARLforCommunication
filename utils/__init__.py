import logging
import os
import shutil
import sys

def create_folder(fold_path):
    if os.path.exists(fold_path):
        shutil.rmtree(fold_path)
    os.makedirs(fold_path, exist_ok=True)


def setup_logger(name, log_file_path, level=logging.DEBUG):
    create_folder(log_file_path)
    log_file = log_file_path + "/log"
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s,%(msecs)d,%(levelname)s::%(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger