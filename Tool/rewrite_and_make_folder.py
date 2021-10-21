import pathlib
import os
import shutil

def create_folder(folder_name):
    # create a folder, if folder exists, load model, else, create 
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    folder_name.mkdir(parents=True, exist_ok=True)
    
def rewrite_and_make_folder(args):
    if args.transformer_start:
        model_matrix_path = pathlib.Path("./Exp/Transformer_folder")
    elif args.attention_start:
        model_matrix_path = pathlib.Path("./Exp/Attention_folder")
    elif args.commNet_start:
        model_matrix_path = pathlib.Path("./Exp/CommNet_folder")
    elif args.maddpg_start:
        model_matrix_path = pathlib.Path("./Exp/Maddpg_folder")
    else:
        model_matrix_path = pathlib.Path("./Exp/Pointer_network_folder")


    if args.rank_start:
        algorithm_matrix_path = model_matrix_path / "Rank"
    elif args.edge_max_start:
        algorithm_matrix_path = model_matrix_path / "Edge_max"
    elif args.priority_start:
        algorithm_matrix_path = model_matrix_path /"Priority"
    elif args.weighted_start:
        weighted_matrix_path = model_matrix_path / "Weighted"
        args.weighted_ratio = args.weighted_ratio
        algorithm_matrix_path = weighted_matrix_path / ("weighted_ratio_" + str(args.weighted_ratio))
    elif args.PF_start:
        algorithm_matrix_path = model_matrix_path /"PF"
    else:
        algorithm_matrix_path = model_matrix_path /"Max_SE"

    # 定义parameter sharing 开关
    if args.parameter_sharing:
        Matrix_model_folder = algorithm_matrix_path /'Sharing_model' 
        Matrix_vision_folder = algorithm_matrix_path / 'Sharing_exp'
        Matrix_result_folder = algorithm_matrix_path / 'Sharing_result'
        Matrix_figure_folder = algorithm_matrix_path / 'Sharing_figure' 
    else:
        Matrix_model_folder = algorithm_matrix_path / 'Model' 
        Matrix_vision_folder = algorithm_matrix_path / 'Exp'
        Matrix_result_folder = algorithm_matrix_path / 'Result'
        Matrix_figure_folder = algorithm_matrix_path / 'Figure' 


    model_folder = Matrix_model_folder /(str(args.total_user_antennas) + '_user_' + str(args.user_velocity) + 'KM')
    vision_folder = Matrix_vision_folder / (str(args.total_user_antennas) + '_user_' + str(args.user_velocity) + 'KM')
    result_folder = Matrix_result_folder /(str(args.total_user_antennas) + '_user_' + str(args.user_velocity) + 'KM')
    figure_folder = Matrix_figure_folder / (str(args.total_user_antennas) + '_user_' + str(args.user_velocity) + 'KM')
    # 这个函数将对ArgumentParser中的一些参数进行重新写入,包括Exp,Model,result这三个文件夹

    if args.Training:
        create_folder(model_folder)
        create_folder(result_folder)
        create_folder(figure_folder)
    else:
        Matrix_vision_folder = Matrix_vision_folder /'_eval'
        vision_folder = Matrix_vision_folder /(str(args.total_user_antennas) + '_user_' + str(args.user_velocity) + 'KM')
    create_folder(vision_folder)
    args.model_folder = model_folder
    args.vision_folder = vision_folder
    args.result_folder = result_folder
    args.figure_folder = figure_folder
    return model_folder, vision_folder, result_folder, figure_folder