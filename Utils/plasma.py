def generate_plasma_id(machine_index, device_index, data_server_index):
    # 这个函数是用来生成独一无二的plasma id，必须是长度为20的bytes list
    # 组合规则，根据这个dataserver的机器id，设备的索引，进程的索引三个部分构成
    plasma_id = 'plasma_id' + str(machine_index) * 3 + str(device_index) * 4 + str(data_server_index) * 4
    if len(plasma_id) > 20:
        plasma_id = plasma_id[:20]
    return bytes(plasma_id, encoding='utf-8')
    