import zmq


def zmq_nonblocking_recv(socket):
    raw_data_list = []
    while True:
        try:
            data = socket.recv(zmq.NOBLOCK)
            raw_data_list.append(data)
        except zmq.ZMQError as e:
            break
    return raw_data_list


def zmq_nonblocking_multipart_recv(socket):
    # 这个函数大概是说这个socket会传多个数据过来,因此这里建立了一个list,把这些数据存放在这个列表里面
    raw_data_list = []
    while True:
        try:
            data = socket.recv_multipart(zmq.NOBLOCK)
            raw_data_list.append(data)
        except zmq.ZMQError as e:
            break
    return raw_data_list
