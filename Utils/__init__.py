import logging
import os
import pathlib
import shutil
import sys

def create_folder(folder_path, delete_origin=False):
    # 这个函数的作用是,当文件夹存在就删除,然后重新创建一个新的
    if isinstance(folder_path, str):
        folder_path = pathlib.Path(folder_path)
    if not os.path.exists(folder_path):
        # shutil.rmtree(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
    else:
        if delete_origin:
            shutil.rmtree(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)

def check_folder_exist(folder_path):
    # 这个函数是检查传入的文件夹路径中是不是空的,如果是空的就创建,不是就什么也不做
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return False
    else:
        return True


def setup_logger(name, log_file_path, level=logging.DEBUG):
    create_folder(log_file_path)
    log_file = log_file_path /"log"
    handler = logging.FileHandler(log_file,encoding="utf-8",mode="a")
    formatter = logging.Formatter("%(asctime)s,%(msecs)d,%(levelname)s,%(name)s::%(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger