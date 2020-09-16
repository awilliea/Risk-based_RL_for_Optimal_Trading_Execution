from enviroment.action_space import ActionSpace
from enviroment.action_state import ActionState
from enviroment.orderbook import Orderbook, OrderbookState, OrderbookEntry
from enviroment.order_side import OrderSide
from enviroment.order_type import OrderType
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections
from datetime import datetime
import pickle
import os

# force_execution=True means that if times out, use market order to achieve our goals
def evaluateReturns(T,I,actionSpace,levels=range(-100, 101), crossval=10, force_execution=True, trade_log=False):
    t = T[-1]
    i = I[-1]
    ys = [] # levels*crossval
    ys2 = []
    variance_list = []
    variance_pv_list = []
    profit_vol_list = []
    for level in levels:
        profit = []
        profit_vol = []
        profit2 = []
        volatility = []
        
        a = level # 0.1?
        for _ in range(crossval):
            action = actionSpace.createAction(a, ActionState(t, i), force_execution=force_execution)
            volatility.append(actionSpace.volatility)
            refBefore = action.getReferencePrice() # best ask price
            if trade_log:
                print("\nLEVEL: " + str(level))
                print("-----------")
                print("Reference price: " + str(refBefore) + "("+str(action.getOrderbookState().getTimestamp())+")")
            action.run(actionSpace.orderbook)
            refAfter = action.getOrderbookState().getTradePrice()
            paid = action.getAvgPrice()
            if trade_log:
                print("Order: " + str(action.getOrder()))
                print("Trades:")
                print(action.getTrades())
            if paid == 0.0:
                assert force_execution == False
                continue
            elif action.getOrder().getSide() == OrderSide.BUY:
                profit.append(refBefore - paid)
                profit2.append(refAfter - paid)
            else:
                profit.append(paid - refBefore)
                profit2.append(paid - refAfter)
                
        profit = np.array(profit)
        volatility = np.array(volatility)
        profit_vol = profit/volatility
        variance_pv = np.var(profit_vol)
        variance = np.var(profit)
        ys.append(profit)
        ys2.append(profit2)
        profit_vol_list.append(profit_vol)
        variance_list.append(variance)
        variance_pv_list.append(variance_pv)
    x = levels
    
    return (x, ys, ys2, profit_vol_list, variance_list, variance_pv_list)

# force_execution=True means that if times out, use market order to achieve our goals
def evaluateReturns_market(T,I,actionSpace,levels=range(-100, 101), crossval=10, force_execution=True, \
                           trade_log=False, filter_outliers=True):
    t = T[-1]
    i = I[-1]
    profit = [] # levels*crossval
    profit2 = []
    volatility = []
    
    a = None # for market orders
    for _ in range(crossval):
        action = actionSpace.createAction(a, ActionState(t, i), force_execution=force_execution)
        volatility.append(actionSpace.volatility)
        refBefore = action.getReferencePrice() # best ask price
        if trade_log:
            print("\nLEVEL: " + str(level))
            print("-----------")
            print("Reference price: " + str(refBefore) + "("+str(action.getOrderbookState().getTimestamp())+")")
        action.run(actionSpace.orderbook)
        refAfter = action.getOrderbookState().getTradePrice()
        paid = action.getAvgPrice()
        if trade_log:
            print("Order: " + str(action.getOrder()))
            print("Trades:")
            print(action.getTrades())
        if paid == 0.0:
            assert force_execution == False
            continue
        elif action.getOrder().getSide() == OrderSide.BUY:
            profit.append(refBefore - paid)
            profit2.append(refAfter - paid)
        else:
            profit.append(paid - refBefore)
            profit2.append(paid - refAfter)
    profit = np.array(profit)
    volatility = np.array(volatility)
    profit_vol = profit/volatility
    variance_pv = np.var(profit_vol)
    variance = np.var(profit)

    if filter_outliers:
        y = [np.mean(reject_outliers(np.array(profit)))]
        y2 = [np.mean(reject_outliers(np.array(profit2)))]
        y3 = [np.mean(reject_outliers(np.array(profit_vol)))]
    else:
        y = [np.mean(np.array(profit)) ]
        y2 = [np.mean(np.array(profit2))]
        y3 = [np.mean(np.array(profit_vol))]
        
    return y,y3,variance,variance_pv

def reject_outliers(data, m=1.5):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def priceReturnCurve(T,I,actionspace,enable_after_exec_return=True, levels=range(-100, 101), crossval=10, force_execution=True, 
                     filter_outliers=False, trade_log=False, file_name = 'test.png'):
    (x, ys, ys2, profit_vol_list, variance_list, variance_pv_list) = evaluateReturns(T,I,actionspace,levels, crossval, force_execution, trade_log)
    if filter_outliers:
        y = [np.mean(reject_outliers(np.array(x))) for x in ys]
        y2 = [np.mean(reject_outliers(np.array(x))) for x in ys2]
        y3 = [np.mean(reject_outliers(np.array(profit_vol_list)))]
    else:
        y = [np.mean(np.array(x)) for x in ys]
        y2 = [np.mean(np.array(x)) for x in ys2]
        y3 = [np.mean(np.array(profit_vol_list))]
    
    return y, y3, variance_list, variance_pv_list

def get_baseline_profits(order_type, orderbook, dataset='Train'):
    all_T = [[0,100]]
    I = [1.0]
    sides = [OrderSide.BUY, OrderSide.SELL]
    profits = []
    profits_vol = []
    variance_list = []
    variance_pv_list = []
    for T in all_T:
        for side in sides:
            if side == OrderSide.BUY:
                filename = 'Time_'+str(T[-1])+'_Side_BUY'
            else:
                filename = 'Time_'+str(T[-1])+'_Side_SELL'

            actionSpace = ActionSpace(orderbook, side, T, I)

            filename = dataset+'_'+filename
            if order_type == 'limit':
                y, y_vol,variance, variance_pv = priceReturnCurve(T,I,actionSpace,crossval=100, force_execution=True, filter_outliers=False,enable_after_exec_return=False,
                                file_name = './baseline/'+filename+'.png')
                profit_name = 'baseline_limit_profits.pkl'
            else:
                y, y_vol,variance, variance_pv = evaluateReturns_market(T,I,actionSpace,crossval=100, filter_outliers=False)
                profit_name = 'baseline_market_profits.pkl'
            profits.append(y)
            profits_vol.append(y_vol)
            variance_list.append(variance)
            variance_pv_list.append(variance_pv)
            
    profit_name = dataset + '_' + profit_name

        
    return profits, profits_vol, variance_list, variance_pv_list

def analysis_profits(profits):
    buy_maxs = []
    sell_maxs = []
    all_T = [[0,100]]

    for i,ps in enumerate(profits):
        buy_profits = []
        sell_profits = []
        t = int(i/2)

        if i%2 == 0:
            buy_profits.append(ps)
            buy_max = max(ps)
            buy_maxs.append(buy_max)

            print('Time',all_T[t],'buy_max',buy_max)
        else:
            sell_profits.append(ps)
            sell_max = max(ps)
            sell_maxs.append(sell_max)
            print('Time',all_T[t],'sell_max',sell_max)

    buy_maxs = np.array(buy_maxs)
    sell_maxs = np.array(sell_maxs)
    final_buy_max = max(buy_maxs)
    final_sell_max = max(sell_maxs)
    
    print('Total buy_max',final_buy_max,'Time',all_T[np.argmax(buy_maxs)])
    print('Total sell_max',final_sell_max,'Time',all_T[np.argmax(sell_maxs)])
    
    return [final_buy_max, final_sell_max, final_buy_max+final_sell_max]

def get_baselines(order_type, data, data_type='Train'):
    profits, profit_vols, variance, variance_pv = get_baseline_profits(order_type, data, data_type)
    order_profit_baselines = analysis_profits(profits)
    order_profit_vol_baselines = analysis_profits(profit_vols)
    
    if order_type == 'limit':
        order_var_baselines = analysis_profits(variance)
        order_var_pv_baselines = analysis_profits(variance_pv)
    else:
        order_var_baselines = analysis_profits(np.reshape(variance,(-1,1)))
        order_var_pv_baselines = analysis_profits(np.reshape(variance_pv,(-1,1)))
    baselines = order_profit_baselines[:-1]+order_profit_vol_baselines[:-1]\
+[x**0.5 for x in order_var_baselines[:-1]]+order_var_pv_baselines[:-1]
    
    return baselines

def get_all_baselines(order_type, train_data, test_data):
    train_baselines = get_baselines(order_type, train_data, data_type='Train')
    test_baselines = get_baselines(order_type, test_data, data_type='Test')
    baselines = pd.DataFrame(columns=['E[{} order on train data]'.format(order_type),\
                                      'E[{} order on test data]'.format(order_type)], 
                         index=['buy_profit','sell_profit',\
                                 'buy_profit_vol','sell_profit_vol',\
                                 'buy_general_risk','sell_general_risk',\
                                 'buy_agent_risk','sell_agent_risk'
                                 ])
    baselines.iloc[:,0] = train_baselines
    baselines.iloc[:,1] = test_baselines
    
    return baselines

if __name__ == '__main__':
    
    if not os.path.isdir('results'):
        os.mkdir('results')
    orderbook_dn = Orderbook()
    orderbook_dn.loadFromEvents('data/events/ob-train.tsv')
    orderbook_dn_test = Orderbook()
    orderbook_dn_test.loadFromEvents('data/events/ob_dnTrend_test.tsv')
    
    orderbook_up = Orderbook()
    orderbook_up.loadFromEvents('data/events/ob_upTrend_train.tsv')
    orderbook_up_test = Orderbook()
    orderbook_up_test.loadFromEvents('data/events/ob_upTrend_test.tsv')

    limit_order_baselines_up = get_all_baselines('limit', orderbook_up, orderbook_up_test)
    market_order_baselines_up = get_all_baselines('market', orderbook_up, orderbook_up_test)
    baselines_up = pd.concat([market_order_baselines_up, limit_order_baselines_up],axis=1)
    baselines_up_dropped_pv = baselines_up.drop(['buy_profit_vol','sell_profit_vol'])
    
    limit_order_baselines_dn = get_all_baselines('limit', orderbook_dn, orderbook_dn_test)
    market_order_baselines_dn = get_all_baselines('market', orderbook_dn, orderbook_dn_test)
    baselines_dn = pd.concat([market_order_baselines_dn, limit_order_baselines_dn],axis=1)
    baselines_dn_dropped_pv = baselines_dn.drop(['buy_profit_vol','sell_profit_vol'])
    
    print("Baselines for downtrend dataset")
    print(baselines_dn_dropped_pv)
    print("Baselines for uptrend dataset")
    print(baselines_up_dropped_pv)
    
    baselines_dn_dropped_pv.to_pickle("./results/baselines_downtrend.pkl")
    baselines_up_dropped_pv.to_pickle("./results/baselines_uptrend.pkl")
    