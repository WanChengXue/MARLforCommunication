import pyflann
import numpy as np
import itertools
import threading
from tqdm import tqdm

class Space(object):
    _instance_lock = threading.Lock()
    _instances = None

    def __new__(cls, low, high, action_dim):
        if cls._instances is None:
            with cls._instance_lock:
                cls._instances = object.__new__(cls)
        return cls._instances


    def __init__(self, low, high, action_dim):
        # ------ 传入的low表示动作的最小值列表 -----
        self._low = low
        # ------ 传入的high表示动作的最大值列表 -----
        self._high = high
        # ------ 这个_range表示的是每一个维度的动作变化区间 ------
        self._range = self._high - self._low
        # --------- 这个_dimensions表示的是动作的维度 --------
        self._dimensions = action_dim
        # ---------- 均匀撒点，
        self.__space = init_uniform_space([0] * self._dimensions,
                                          [1] * self._dimensions)
        self._flann = pyflann.FLANN()
        self.rebuild_flann()

    def rebuild_flann(self):
        self._index = self._flann.build_index(self.__space, algorithm='kdtree')


    def search_point(self, point, k):
        # ------ 传入一个点point，以及需要找到的最近的k个点 --------
        # ------ 传入的point的维度是batch_size * action_dim --------
        assert k>1, '------ 这里的实现必须要保证k大于1 --------'
        p_in = self.import_point(point).astype('float64')
        # ----- 返回最近的k个目标的索引 ------
        search_res, _ = self._flann.nn_index(p_in, k)
        # ----- 通过动作空间直接返回k个动作 -----
        knns = self.__space[search_res]
        # ------ 使用这个查找操作，得到的knns的维度是batch_size * k_nearest * 1 -------
        p_out = self.export_point(knns)
        return p_out

    def import_point(self, point):
        # ------ 这个地方返回的是所有动作在这个动作axis上面的连续值 ------
        return (point - self._low) / self._range

    def export_point(self, point):
        return self._low + point * self._range

    def get_space(self):
        # ----- 这个地方返回动作空间 ------
        return self.__space

    def shape(self):
        # ----- 这个地方是返回动作空间的维度 -----
        return self.__space.shape

    def get_number_of_actions(self):
        # ------ 这个地方返回的是所有可能的动作的长度 -----
        return self.shape()[0]


class Discrete_space(Space):
    """
        Discrete action space with n actions (the integers in the range [0, n))
        0, 1, 2, ..., n-2, n-1
    """

    def __init__(self, n):  # n: the number of the discrete actions
        super().__init__([0], [n - 1], n)

    def export_point(self, point):
        return super().export_point(point).astype(int)


def init_uniform_space(low, high):
    # ------- 传入的low，high分别表示动作的上下限 -----
    dims = len(low)
    # ----- points表示动作空间的大小 -------
    # ----- 这里的points_in_each_axis表示映射到每一个轴上的动作数目 -----
    points_in_each_axis = 2

    axis = []
    for i in range(dims):
        axis.append(list(np.linspace(low[i], high[i], points_in_each_axis)))
        # ----- 这里是添加每一个轴上能够取的值，如果[0,1] ------
    space = []
    # ------ 这个地方通过itertools工具，做笛卡尔直积，得到所有动作构成的大列表 -------
    for _ in tqdm(itertools.product(*axis)):
        space.append(list(_))

    return np.array(space)

# def new_action_obj(low, high, dim):
#     obj = Space(low, high, dim)
#     print(id(obj))

# from multiprocessing import Process
# for i in range(4):
#         # logger_path = pathlib.Path("./config_folder") / ("process_"+ str(i))
#         # logger_name = "Process_"+ str(i)
#         # logger = setup_logger(logger_name, logger_path)
#         # p = Process(target=single_process_generate_sample,args=(logger,))
#         # p.start()
#         p = Process(target=new_action_obj, args=(0,1,30))
#         p.start()

# test = Space(0,1,30)