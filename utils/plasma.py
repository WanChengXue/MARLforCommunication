def generate_plasma_id(global_rank, data_server_id):
    # 这个函数是用来生成一个plasma id的,其中plasma id的长度为20, 构成格式为:
    plasma_str = str(global_rank * 100 + data_server_id + 10000) * 2 + '_policy_co'
    return bytes(plasma_str, encoding='utf-8')
    