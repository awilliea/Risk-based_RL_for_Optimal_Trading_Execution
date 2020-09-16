import os 
import pickle
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from argparse import ArgumentParser

def output_comparision_between_rewards_v2(experiment_dir, logs_dir_list, name_list, mean_epoch=50, direction='buy', trend='dn'):
    profit_list = []
    profit_vol_list = []
    var_list = []
    var_pv_list = []
    test_profit_list = []
    test_profit_vol_list = []
    test_var_list = []
    test_var_pv_list = []
    
    
    for dir_name in logs_dir_list:
        print(dir_name)
        with open(os.path.join(dir_name,"Profit_{}_paper_{}_order.pkl".format(direction, trend)),'rb') as f:
            profit = pickle.load(f)[2:]
            
        with open(os.path.join(dir_name,"Profit_vol_{}_paper_{}_order.pkl".format(direction, trend)),'rb') as f:
            profit_vol = pickle.load(f)[2:]
                
        with open(os.path.join(dir_name,"Var_PV_{}_paper_{}_order.pkl".format(direction, trend)),'rb') as f:
            var_pv = pickle.load(f)[2:]
           
        with open(os.path.join(dir_name,"Var_Profit_{}_paper_{}_order.pkl".format(direction, trend)),'rb') as f:
            var = pickle.load(f)[2:]

        with open(os.path.join(dir_name,"Test_Profit_{}_paper_{}_order.pkl".format(direction, trend)),'rb') as f:
            test_profit = pickle.load(f)[2:]

       
        with open(os.path.join(dir_name,"Test_Profit_vol_{}_paper_{}_order.pkl".format(direction, trend)),'rb') as f:
            test_profit_vol = pickle.load(f)[2:]

        with open(os.path.join(dir_name,"Test_Var_PV_{}_paper_{}_order.pkl".format(direction, trend)),'rb') as f:
            test_var_pv = pickle.load(f)[2:]

        with open(os.path.join(dir_name,"Test_Var_Profit_{}_paper_{}_order.pkl".format(direction, trend)),'rb') as f:
            test_var = pickle.load(f)[2:]

     
        profit_list.append(profit)
        profit_vol_list.append(profit_vol)
        var_pv_list.append(var_pv)
        var_list.append(var)
        test_profit_list.append(test_profit)
        test_profit_vol_list.append(test_profit_vol)
        test_var_pv_list.append(test_var_pv)
        test_var_list.append(test_var)
        
    profit_list = np.array(profit_list)
    profit_vol_list = np.array(profit_vol_list)
    var_list = np.array(var_list)
    std_list = var_list**0.5
    var_pv_list = np.array(var_pv_list)
    test_profit_list = np.array(test_profit_list)
    test_profit_vol_list = np.array(test_profit_vol_list)
    test_var_list = np.array(test_var_list)
    test_std_list = test_var_list**0.5
    test_var_pv_list = np.array(test_var_pv_list)
    
    
    data = [profit_list[:,-1],test_profit_list[:,-1],profit_vol_list[:,-1],test_profit_vol_list[:,-1],\
            std_list[:,-1],test_std_list[:,-1],var_pv_list[:,-1],test_var_pv_list[:,-1]]
    dataframe = pd.DataFrame(columns=logs_dir_list,index=['Train profit','Test profit',\
                                                    'Train profit_vol','Test profit_vol',\
                                                    'Train geneal risk', 'Test general risk',\
                                                    'Train agent risk', 'Test agent risk'],\
                             data=data)
    plot_profit(experiment_dir, profit_list, name_list, mean_epoch, "train")
    plot_std(experiment_dir, std_list, name_list, mean_epoch, "train")
    plot_var(experiment_dir, var_pv_list, name_list, mean_epoch, "train")
    
    return dataframe
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
def plot_profit(experiment_dir, frontier_avg_profit_list, frontiers, mean_epoch=50, plot_type='train'):
    frontier_avg_profit_list = np.array(frontier_avg_profit_list)
    name = '{}/images/Profit_comparison_{}.png'.format(experiment_dir,plot_type)
    plt.figure(figsize=(15,8))
    plt.title(name)
    length = len(frontier_avg_profit_list[-1])
    for idx in range(len(frontiers)):
        avg_profit_list = frontier_avg_profit_list[idx]
        front_profit = [np.mean(avg_profit_list[:i+1]) for i in range(mean_epoch-1)]
        mean_profit = moving_average(avg_profit_list,mean_epoch)
        front_profit.extend(mean_profit)
        plt.plot( front_profit[:length], label='{}-Epoch Profit means on {}'.format(mean_epoch,frontiers[idx]))
    plt.ylabel('Profit')
    plt.xlabel('Episodes')
    plt.legend(loc='best')
    plt.savefig(name)
    #plt.show()
    plt.close()
    
def plot_var(experiment_dir, frontier_var_profit_vol_list, frontiers, mean_epoch=50, plot_type='train'):
    frontier_var_profit_vol_list = np.array(frontier_var_profit_vol_list)
    name = '{}/images/Agent_risk_comparison_{}.png'.format(experiment_dir,plot_type)
    plt.figure(figsize=(15,8))
    plt.title(name)
    length = len(frontier_var_profit_vol_list[-1])
    for idx in range(len(frontiers)):
        var_profit_vol_list = frontier_var_profit_vol_list[idx]
        front_vars = [np.mean(var_profit_vol_list[:i+1]) for i in range(mean_epoch-1)]
        mean_vars = moving_average(var_profit_vol_list,mean_epoch)
        front_vars.extend(mean_vars)
        plt.plot(front_vars[:length], label='{}-Epoch Agent risk means on {}'.format(mean_epoch,frontiers[idx]))
    plt.ylabel('Agent risk')
    plt.xlabel('Episodes')
    plt.legend(loc='best')
    plt.savefig(name)
    #plt.show()
    plt.close()
    
def plot_std(experiment_dir, frontier_var_profit_vol_list, frontiers, mean_epoch=50, plot_type='train'):
    frontier_var_profit_vol_list = np.array(frontier_var_profit_vol_list)
    name = '{}/images/General_risk_comparison_{}.png'.format(experiment_dir,plot_type)
    plt.figure(figsize=(15,8))
    plt.title(name)
    length = len(frontier_var_profit_vol_list[-1])
    for idx in range(len(frontiers)):
        var_profit_vol_list = frontier_var_profit_vol_list[idx]
        front_vars = [np.mean(var_profit_vol_list[:i+1]) for i in range(mean_epoch-1)]
        mean_vars = moving_average(var_profit_vol_list,mean_epoch)
        front_vars.extend(mean_vars)
        plt.plot(front_vars[:length], label='{}-Epoch General risk means on {}'.format(mean_epoch,frontiers[idx]))
    plt.ylabel('Gereral risk')
    plt.xlabel('Episodes')
    plt.legend(loc='best')
    plt.savefig(name)
    #plt.show()
    plt.close()
    


 
parser = ArgumentParser()
parser.add_argument("-s", help="side name", dest="side_name", default="buy")
parser.add_argument("-d", help="data trend", dest="trend", default="dn")
parser.add_argument('--tdir', help="The type of the experiment \
                                    1: original reward vs rewarding shaping \
                                    2: MORL \
                                    3: Risk-sensitive \
                                    4: Risk=averse", dest='dir_type', default=1, type=int)
parser.add_argument('--ver', help="Yo may want to try different random seed", dest='ver', default='v2')
parser.add_argument('--result_dir', help="Where to save the result dataframe", dest='result_dir', default='./results')

args = parser.parse_args()
direction = args.side_name
trend = args.trend
dir_type = args.dir_type
version = args.ver
dir_names_1 = ["TFA_DDQN_{}_CNN_100000_500000_400_100_[1.0, 0.0]_profit".format(version),\
                 "TFA_DDQN_{}_CNN_100000_500000_400_100_[1.0, 0.0]_profit_vol".format(version)]
dir_names_2 = ["TFA_DDQN_{}_CNN_100000_500000_400_100_[1.0, 0.0]_profit_vol".format(version),\
            "TFA_DDQN_{}_CNN_100000_500000_400_100_[0.8, 0.2]_profit_vol".format(version),\
            "TFA_DDQN_{}_CNN_100000_500000_400_100_[0.6, 0.4]_profit_vol".format(version),\
            "TFA_DDQN_{}_CNN_100000_500000_400_100_[0.4, 0.6]_profit_vol".format(version),\
            "TFA_DDQN_{}_CNN_100000_500000_400_100_[0.2, 0.8]_profit_vol".format(version),\
            "TFA_DDQN_{}_CNN_100000_500000_400_100_[0.0, 1.0]_profit_vol".format(version)]
dir_names_3 = ["TFA_DDQN_{}_CNN_100000_500000_400_100_[1.0, 0.0]_profit_vol".format(version),\
"TFA_Sensitive_DDQN_{}_0.2_CNN_100000_500000_400_100_[1.0, 0.0]".format(version),\
"TFA_Sensitive_DDQN_{}_0.4_CNN_100000_500000_400_100_[1.0, 0.0]".format(version),\
"TFA_Sensitive_DDQN_{}_0.6_CNN_100000_500000_400_100_[1.0, 0.0]".format(version),\
"TFA_Sensitive_DDQN_{}_0.8_CNN_100000_500000_400_100_[1.0, 0.0]".format(version),\
"TFA_Sensitive_DDQN{}_1.0_CNN_100000_500000_400_100_[1.0, 0.0]".format(version)]

dir_names_4 = ["TFA_DDQN_{}_CNN_100000_500000_400_100_[1.0, 0.0]_profit_vol".format(version),\
"TFA_Averse_DDQN_{}_0.8_CNN_100000_500000_400_100_[1.0, 0.0]".format(version),\
"TFA_Averse_DDQN_{}_0.6_CNN_100000_500000_400_100_[1.0, 0.0]".format(version),\
"TFA_Averse_DDQN_{}_0.4_CNN_100000_500000_400_100_[1.0, 0.0]".format(version),\
"TFA_Averse_DDQN_v{}_0.2_CNN_100000_500000_400_100_[1.0, 0.0]".format(version)]
names_1 = ["Original","Reward shaping"]
names_2 = ['[1.0, 0.0]','[0.8, 0.2]','[0.6, 0.4]','[0.4, 0.6]','[0.2, 0.8]','[0.0, 1.0]']  
names_3 = ['0.0','0.2','0.4','0.6','0.8','1.0']
names_4 = ['1.0','0.8','0.6','0.4','0.2']

if dir_type == 1:
    logs_dir_list = dir_names_1
    name_list = names_1  
elif dir_type == 2:
    logs_dir_list = dir_names_2
    name_list = names_2
elif dir_type == 3:
    logs_dir_list = dir_names_3
    name_list = names_3
elif dir_type == 4:
    logs_dir_list = dir_names_4
    name_list = names_4

if __name__ == '__main__':
    
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
        
    experiment_name = direction+'_'+trend+'_'+str(dir_type)+'_'+version
    experiment_dir = os.path.join(args.result_dir, experiment_name)
    if not os.path.isdir(experiment_dir):
        os.mkdir(experiment_dir)
        os.mkdir(os.path.join(experiment_dir, 'images'))
        
    df = output_comparision_between_rewards_v2(experiment_dir, logs_dir_list, name_list, direction=direction, trend=trend)
    df=df.drop(['Train profit_vol','Test profit_vol'])
    df.columns = name_list
    print(df.T)

    result_path = os.path.join(experiment_dir, 'dataframe.pkl')
    df.T.to_pickle(result_path)
    print("The result is saved to",result_path)