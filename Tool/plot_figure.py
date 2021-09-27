import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

# os.mkdir("./Comprison_figure")
def traditional_method():
    user_number = ['10_user','20_user','30_user','40_user']
    velocity = ['3KM','30KM','90KM']
    root_path = "../data_part/data/"
    # global_greedy = []
    # individual_greedy = []
    for user_index in user_number:
        for velocity_index in velocity:
            greedy_path = root_path + user_index +'/' + velocity_index + '/' + 'Greedy_result/' + 'Sum_capacity_10_10.npy'
            individual_greedy_path = root_path + user_index +'/' + velocity_index + '/' + 'Individual_greedy_result/' + 'Sum_capacity_10_10.npy'
            greedy_delay_path = root_path + user_index +'/' + velocity_index + '/' + 'Greedy_result/' + 'Sum_capacity_delay_10_10.npy'
            individual_greedy_delay_path = root_path + user_index +'/' + velocity_index + '/' + 'Individual_greedy_result/' + 'Sum_capacity_delay_10_10.npy'
            # global_greedy.append(greedy_path)
            # individual_greedy.append(individual_greedy_path)
            plt.figure()
            plt.plot(np.load(greedy_path))
            plt.plot(np.load(individual_greedy_path))
            plt.plot(np.load(greedy_delay_path))
            plt.plot(np.load(individual_greedy_delay_path))
            plt.legend(['greedy','i_greedy','delay_greedy','delay_i_greedy'])
            plt.title('user number: ' + user_index + ' velocity: ' + velocity_index)
            plt.savefig("./Comprison_figure/"+user_index+'_'+velocity_index+'.png')
            plt.close()
        

def RL_comprison():
    # user_number = ['20_user']
    # velocity = ['30KM','90KM']
    user_number = ['10_user','20_user','30_user','40_user']
    velocity = ['3KM','30KM','90KM']
    index_set = [0,1,2,3]
    count = 0
    root_path = "../data_part/data/"
    # global_greedy = []
    # individual_greedy = []
    for user_index in user_number:
        for velocity_index in velocity:
            if count in index_set:
                greedy_path = root_path + user_index +'/' + velocity_index + '/' + 'Greedy_result/' + 'Sum_capacity_10_10.npy'
                individual_greedy_path = root_path + user_index +'/' + velocity_index + '/' + 'Individual_greedy_result/' + 'Sum_capacity_10_10.npy'
                RL_path = './Result/' + user_index +'_' + velocity_index + '/' + 'RL_result.npy'
                # greedy_delay_path = root_path + user_index +'/' + velocity_index + '/' + 'Greedy_result/' + 'Sum_capacity_delay_10_10.npy'
                # individual_greedy_delay_path = root_path + user_index +'/' + velocity_index + '/' + 'Individual_greedy_result/' + 'Sum_capacity_delay_10_10.npy'
                # global_greedy.append(greedy_path)
                # individual_greedy.append(individual_greedy_path)
                plt.figure()
                plt.plot(np.load(greedy_path))
                plt.plot(np.load(individual_greedy_path))
                plt.plot(np.load(RL_path)[0,:])
                plt.legend(['greedy','i_greedy','RL_method'])
                plt.title('user number: ' + user_index + ' velocity: ' + velocity_index)
                plt.savefig("./Comprison_figure/"+user_index+'_'+velocity_index+'.png')
                plt.close()
            count += 1

def RL_delay_comprison():
    user_number = ['10_user','20_user','30_user','40_user']
    velocity = ['3KM','30KM','90KM']
    index_set = [0,1,2,3]
    count = 0
    root_path = "../data_part/data/"
    # global_greedy = []
    # individual_greedy = []
    for user_index in user_number:
        for velocity_index in velocity:
            if count in index_set:
                greedy_path = root_path + user_index +'/' + velocity_index + '/' + 'Greedy_result/' + 'Sum_capacity_10_10.npy'
                individual_greedy_path = root_path + user_index +'/' + velocity_index + '/' + 'Individual_greedy_result/' + 'Sum_capacity_10_10.npy'
                greedy_delay_path = root_path + user_index +'/' + velocity_index + '/' + 'Greedy_result/' + 'Sum_capacity_delay_10_10.npy'
                individual_greedy_delay_path = root_path + user_index +'/' + velocity_index + '/' + 'Individual_greedy_result/' + 'Sum_capacity_delay_10_10.npy'
                RL_path = './Result/' + user_index +'_' + velocity_index + '/' + 'RL_result.npy'
                RL_delay_path = './Result/' + user_index +'_' + velocity_index + '_delay/' + 'RL_result.npy'
                # greedy_delay_path = root_path + user_index +'/' + velocity_index + '/' + 'Greedy_result/' + 'Sum_capacity_delay_10_10.npy'
                # individual_greedy_delay_path = root_path + user_index +'/' + velocity_index + '/' + 'Individual_greedy_result/' + 'Sum_capacity_delay_10_10.npy'
                # global_greedy.append(greedy_path)
                # individual_greedy.append(individual_greedy_path)
                plt.figure()
                # plt.plot(np.load(greedy_path))
                # plt.plot(np.load(individual_greedy_path))
                plt.plot(np.load(greedy_delay_path))
                plt.plot(np.load(individual_greedy_delay_path))
                # plt.plot(np.load(RL_path)[0,:])
                plt.plot(np.load(RL_delay_path)[0,:])
                plt.legend(['delay_greedy','delay_i_greedy','RL_delay_method'])
                # plt.legend(['greedy','i_greedy','delay_greedy','delay_i_greedy','RL_method','RL_delay_method'])
                plt.title('user number: ' + user_index + ' velocity: ' + velocity_index)
                plt.savefig("./Comprison_figure/"+user_index+'_'+velocity_index+'.png')
                plt.close()
            count += 1
# RL_comprison()

def traditional_RL_method():
    user_number = ['10_user','20_user','30_user','40_user']
    velocity = ['3KM','30KM','90KM']
    root_path = "../data_part/data/"
    # global_greedy = []
    index_set = [0,1,2,3]
    count = 0
    # individual_greedy = []
    for user_index in user_number:
        for velocity_index in velocity:
            if count in index_set:
                greedy_path = root_path + user_index +'/' + velocity_index + '/' + 'Greedy_result/' + 'Sum_capacity_10_10.npy'
                individual_greedy_path = root_path + user_index +'/' + velocity_index + '/' + 'Individual_greedy_result/' + 'Sum_capacity_10_10.npy'
                greedy_delay_path = root_path + user_index +'/' + velocity_index + '/' + 'Greedy_result/' + 'Sum_capacity_delay_10_10.npy'
                individual_greedy_delay_path = root_path + user_index +'/' + velocity_index + '/' + 'Individual_greedy_result/' + 'Sum_capacity_delay_10_10.npy'
                # global_greedy.append(greedy_path)
                # individual_greedy.append(individual_greedy_path)
                RL_path = './Result/' + user_index +'_' + velocity_index + '/' + 'RL_result.npy'
                RL_delay_path = './Result/' + user_index +'_' + velocity_index + '_delay/' + 'RL_result.npy'
                plt.figure()
                plt.plot(np.load(greedy_path))
                plt.plot(np.load(individual_greedy_path))
                plt.plot(np.load(greedy_delay_path))
                plt.plot(np.load(individual_greedy_delay_path))
                plt.plot(np.load(RL_path).squeeze())
                plt.plot(np.load(RL_delay_path).squeeze())
                plt.legend(['greedy','i_greedy','delay_greedy','delay_i_greedy','RL','delay_RL'])
                plt.title('user number: ' + user_index + ' velocity: ' + velocity_index)
                plt.savefig("./Comprison_figure/"+user_index+'_'+velocity_index+'.png')
                plt.close()
            count += 1
# RL_delay_comprison()
# traditional_RL_method()
# RL_comprison()

def priority_PF():
    user_number = ['10_user','20_user','30_user','40_user']
    velocity = ['3KM','30KM','90KM']
    root_path = "../data_part/data/"
    count = 0 
    # index_set = [0,1,2,3,4,5,6,7,8,9,10,11]
    index_set = [6]
    for user_index in user_number:
        for velocity_index in velocity:
            if count in index_set:
                Edge_path = root_path + user_index + '/' + velocity_index + '/Greedy_PF_result/Edge_capacity_PF_10_10_0.npy'
                Sum_capacity_path = root_path + user_index + '/' + velocity_index + '/Greedy_PF_result/Sum_capacity_PF_10_10_0.npy'
                PF_sum_path = root_path + user_index + '/' + velocity_index + '/Greedy_PF_result/PF_sum_10_10_0.npy'

                Edge_path_new = root_path + user_index + '/' + velocity_index + '/Greedy_two_layer_PF_result/Edge_capacity_PF_10_10_0.npy'
                Sum_capacity_path_new = root_path + user_index + '/' + velocity_index + '/Greedy_two_layer_PF_result/Sum_capacity_PF_10_10_0.npy'
                PF_sum_path_new = root_path + user_index + '/' + velocity_index + '/Greedy_two_layer_PF_result/PF_sum_10_10_0.npy' 
                
                Edge_path_new_one = root_path + user_index + '/' + velocity_index + '/Greedy_two_layer_PF_result_new/Edge_capacity_PF_10_10_0.npy'
                Sum_capacity_path_new_one = root_path + user_index + '/' + velocity_index + '/Greedy_two_layer_PF_result_new/Sum_capacity_PF_10_10_0.npy'
                PF_sum_path_new_one = root_path + user_index + '/' + velocity_index + '/Greedy_two_layer_PF_result_new/PF_sum_10_10_0.npy' 

                Edge_individual_path = root_path + user_index + '/' + velocity_index + '/Greedy_PF_individual_result/Edge_capacity_PF_10_10_0.npy'
                Sum_capacity_individual_path = root_path + user_index + '/' + velocity_index + '/Greedy_PF_individual_result/Sum_capacity_PF_10_10_0.npy'
                PF_sum_individual_path = root_path + user_index + '/' + velocity_index + '/Greedy_PF_individual_result/PF_sum_10_10_0.npy'
                
                RL_edge_path = "./Attention_folder/PF/Result/" + user_index + '_' + velocity_index + '/RL_edge.npy'
                RL_sum_capacity_path = "./Attention_folder/PF/Result/" + user_index + '_' + velocity_index + '/RL_system_npy.npy'
                RL_pf_sum_path = "./Attention_folder/PF/Result/" + user_index + '_' + velocity_index + '/RL_pf_sum.npy'
                # RL_edge_path = "./Attention_folder/Edge_max/Result/" + user_index + '_' + velocity_index + '/RL_edge.npy'
                # RL_sum_capacity_path = "./Attention_folder/Edge_max/Result/" + user_index + '_' + velocity_index + '/RL_system_npy.npy'
                # RL_pf_sum_path = "./Attention_folder/Edge_max/Result/" + user_index + '_' + velocity_index + '/RL_pf_sum.npy'

                # 这个地方是delay部分
                delay_Edge_individual_path = root_path + user_index + '/' + velocity_index + '/Greedy_PF_individual_result_delay/Edge_capacity_PF_10_10_0.npy'
                delay_Sum_capacity_individual_path = root_path + user_index + '/' + velocity_index + '/Greedy_PF_individual_result_delay/Sum_capacity_PF_10_10_0.npy'
                delay_PF_sum_individual_path = root_path + user_index + '/' + velocity_index + '/Greedy_PF_individual_result_delay/PF_sum_10_10_0.npy'
                
                # 2个TTI的delay
                delay_2_Edge_individual_path = root_path + user_index + '/' + velocity_index + '/Greedy_PF_individual_result_delay_2/Edge_capacity_PF_10_10_0.npy'
                delay_2_Sum_capacity_individual_path = root_path + user_index + '/' + velocity_index + '/Greedy_PF_individual_result_delay_2/Sum_capacity_PF_10_10_0.npy'
                delay_2_PF_sum_individual_path = root_path + user_index + '/' + velocity_index + '/Greedy_PF_individual_result_delay_2/PF_sum_10_10_0.npy'
                # 3个TTI的delay
                delay_3_Edge_individual_path = root_path + user_index + '/' + velocity_index + '/Greedy_PF_individual_result_delay_3/Edge_capacity_PF_10_10_0.npy'
                delay_3_Sum_capacity_individual_path = root_path + user_index + '/' + velocity_index + '/Greedy_PF_individual_result_delay_3/Sum_capacity_PF_10_10_0.npy'
                delay_3_PF_sum_individual_path = root_path + user_index + '/' + velocity_index + '/Greedy_PF_individual_result_delay_3/PF_sum_10_10_0.npy'

                Edge_data = np.load(Edge_path)
                Sum_capacity_data = np.load(Sum_capacity_path).squeeze()
                PF_sum_data = np.load(PF_sum_path).squeeze()
                print("=====plan 1 ====")
                print(np.mean(Sum_capacity_data)/270)
                print(np.mean(Edge_data))
                Edge_data_new = np.load(Edge_path_new)
                Sum_capacity_data_new = np.load(Sum_capacity_path_new).squeeze()
                PF_sum_data_new = np.load(PF_sum_path_new).squeeze()
                print("====plan 2 =====")
                # print("=====plan 1 ====")
                print(np.mean(Sum_capacity_data_new)/270)
                print(np.mean(Edge_data_new))

                Edge_data_new_one = np.load(Edge_path_new_one)
                Sum_capacity_data_new_one = np.load(Sum_capacity_path_new_one).squeeze()
                PF_sum_data_new_one = np.load(PF_sum_path_new_one).squeeze()
                print("=====plan 3 ====")
                print(np.mean(Sum_capacity_data_new_one))
                print(np.mean(Edge_data_new_one))
                Edge_data_individual_data = np.load(Edge_individual_path).squeeze()
                Sum_capacity_individual_data = np.load(Sum_capacity_individual_path).squeeze()
                PF_sum_individual_data = np.load(PF_sum_individual_path).squeeze()
                print("=====plan 4 ====")
                print(np.mean(Sum_capacity_individual_data)/270)
                print(np.mean(Edge_data_individual_data))
                RL_edge_data = np.load(RL_edge_path).squeeze()
                RL_sum_capacity_data = np.load(RL_sum_capacity_path).squeeze()[0:200]
                RL_pf_sum_data = np.load(RL_pf_sum_path).squeeze()[0:200]

                print("===RL ===")
                print(np.mean(RL_sum_capacity_data)/270)
                print(np.mean(RL_edge_data))

                delay_Edge_data_individual_data = np.load(delay_Edge_individual_path).squeeze()
                delay_Sum_capacity_individual_data = np.load(delay_Sum_capacity_individual_path).squeeze()
                delay_PF_sum_individual_data = np.load(delay_PF_sum_individual_path).squeeze()                
                print("=======individual delay scenario========")
                print(np.mean(delay_Sum_capacity_individual_data)/270)
                print(np.mean(delay_Edge_data_individual_data))
                # 2个delay
                delay_2_Edge_data_individual_data = np.load(delay_2_Edge_individual_path).squeeze()
                delay_2_Sum_capacity_individual_data = np.load(delay_2_Sum_capacity_individual_path).squeeze()
                delay_2_PF_sum_individual_data = np.load(delay_2_PF_sum_individual_path).squeeze()     
                print("========individual delay 2 TTI scenario=========")
                print(np.mean(delay_2_Sum_capacity_individual_data)/270)
                print(np.mean(delay_2_Edge_data_individual_data))
                # 3个delay
                delay_3_Edge_data_individual_data = np.load(delay_3_Edge_individual_path).squeeze()
                delay_3_Sum_capacity_individual_data = np.load(delay_3_Sum_capacity_individual_path).squeeze()
                delay_3_PF_sum_individual_data = np.load(delay_3_PF_sum_individual_path).squeeze()     
                print("========individual delay 2 TTI scenario=========")
                print(np.mean(delay_3_Sum_capacity_individual_data)/270)
                print(np.mean(delay_3_Edge_data_individual_data))

                plt.figure()
                plt.plot(Edge_data)
                plt.plot(Edge_data_new)
                # plt.plot(Edge_data_new_one)
                plt.plot(RL_edge_data)
                plt.plot(Edge_data_individual_data)
                plt.plot(delay_Edge_data_individual_data)
                # plt.legend(["User greedy", "Cell greedy, user greedy", "Init cell priority based on capacity",'RL_method'])
                plt.legend(["User greedy", "Cell greedy, user greedy", 'RL_method','IG','delay_IG'])
                plt.title('Edge data curve, user number: ' + user_index + ' velocity: ' + velocity_index)
                plt.savefig("./Attention_folder/PF/Figure/Edge_user_" + user_index + '_' + velocity_index + '.png')
                plt.close()
                
                plt.figure()
                plt.plot(Sum_capacity_data)
                plt.plot(Sum_capacity_data_new)
                # plt.plot(Sum_capacity_data_new_one)
                plt.plot(RL_sum_capacity_data)
                plt.plot(Sum_capacity_individual_data)
                plt.plot(delay_Sum_capacity_individual_data)
                # plt.legend(["User greedy", "Cell greedy, user greedy", "Init cell priority based on capacity", "RL_method"])
                plt.legend(["User greedy", "Cell greedy, user greedy", 'RL_method','IG', 'delay_IG'])
                plt.title('Sum capacity curve, user number: ' + user_index + 'velocity: ' + velocity_index)
                plt.savefig("./Attention_folder/PF/Figure/Sum_capacity_" + user_index + '_' + velocity_index + '.png')
                plt.close()

                plt.figure() 
                plt.plot(PF_sum_data)
                plt.plot(PF_sum_data_new)
                # plt.plot(PF_sum_data_new_one)
                plt.plot(RL_pf_sum_data)
                plt.plot(PF_sum_individual_data)
                plt.plot(delay_PF_sum_individual_data)
                # plt.legend(["User greedy", "Cell greedy, user greedy", "Init cell priority based on capacity", "RL_method"])
                plt.legend(["User greedy", "Cell greedy, user greedy", 'RL_method','IG', 'delay_IG'])
                plt.title('PF sum curve, user number: ' + user_index + 'velocity: ' + velocity_index)
                plt.savefig("./Attention_folder/PF/Figure/PF_sum_" + user_index + '_' + velocity_index + '.png')
                plt.close()


                c1 = plt.hist(Sum_capacity_data, bins=Sum_capacity_data.shape[0], cumulative=True, histtype="step")
                # plt.plot(np.sort(Sum_capacity_data/270), c[0])
                c2 = plt.hist(Sum_capacity_data_new, bins=Sum_capacity_data_new.shape[0], cumulative=True, histtype="step")
                
                c3 = plt.hist(RL_sum_capacity_data, bins=RL_sum_capacity_data.shape[0], cumulative=True, histtype="step")
                
                c4 = plt.hist(Sum_capacity_individual_data, bins=Sum_capacity_individual_data.shape[0], cumulative=True, histtype="step")

                c5 = plt.hist(delay_Sum_capacity_individual_data, bins = delay_Sum_capacity_individual_data.shape[0], cumulative=True, histtype="step")
                c6 = plt.hist(delay_2_Sum_capacity_individual_data, bins = delay_2_Sum_capacity_individual_data.shape[0], cumulative=True, histtype="step")
                c7 = plt.hist(delay_3_Sum_capacity_individual_data, bins = delay_3_Sum_capacity_individual_data.shape[0], cumulative=True, histtype="step")
                
                plt.figure()
                plt.plot(np.sort(Sum_capacity_data/270), c1[0]/200)
                plt.plot(np.sort(Sum_capacity_data_new/270), c2[0]/200)
                plt.plot(np.sort(RL_sum_capacity_data/270), c3[0]/200)
                plt.plot(np.sort(Sum_capacity_individual_data/270), c4[0]/200)
                plt.plot(np.sort(delay_Sum_capacity_individual_data/270),c5[0]/200)
                plt.plot(np.sort(delay_2_Sum_capacity_individual_data/270),c6[0]/200)
                plt.plot(np.sort(delay_3_Sum_capacity_individual_data/270),c7[0]/200)
                plt.legend(["User greedy", "Cell greedy, user greedy", 'RL_method','IG','delay_1_IG','delay_2_IG','delay_3_IG'])
                plt.title('PF sum curve, user number: ' + user_index + 'velocity: ' + velocity_index)
                plt.savefig('./' +user_index + '_' + velocity_index + '_' + "cdf.png")
                plt.close()
            count += 1
priority_PF()