import logging
import copy
import random
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from enviroment.action import Action
from enviroment.action_state import ActionState
from enviroment.order import Order
from enviroment.order_type import OrderType
from enviroment.order_side import OrderSide
from enviroment.feature_type import FeatureType

class ExecutionEnv(gym.Env):

    def __init__(self):
        self.orderbookIndex = None
        self.actionState = None
        self.execution = None
        self.volatility = None
        self.max_price = None
        self.min_price = None
        self.reward_type = None
        self.episode = 0
        self.preference = [1.0, 0.0]
        self.last_inventory = 1.0
        self.configure()

    def _generate_Sequence(self, min, max, step):
        """ Generate sequence (that unlike xrange supports float)

        max: defines the sequence maximum
        step: defines the interval
        """
        i = min
        I = []
        while i <= max:
            I.append(i)
            i = i + step
        return I

    def configure(self,
                  orderbook=None,
                  side=OrderSide.SELL,
                  levels=(-100, 100, 1),
                  T=(0, 100, 10),
                  I=(0, 1, 0.1),
                  lookback=30,
                  bookSize=40,
                  featureType=FeatureType.ORDERS,
                  #featureType=FeatureType.TRADES,
                  callbacks = [],
                  reward_type = 'profit',
                  preference=[1.0,0.0],
                  market_action=False):
        self.orderbook = orderbook
        self.side = side
        self.market_action = market_action
        self.preference = preference
        self.reward_type = reward_type
        if self.market_action:
            self.levels = self._generate_Sequence(min=levels[0], max=levels[1], step=levels[2])
            self.levels.extend([None])
            self.action_space = spaces.Discrete(len(self.levels))
        else:
            self.levels = self._generate_Sequence(min=levels[0], max=levels[1], step=levels[2])
            self.action_space = spaces.Discrete(len(self.levels))
        self.T = self._generate_Sequence(min=T[0], max=T[1], step=T[2])
        self.I = self._generate_Sequence(min=I[0], max=I[1], step=I[2])
        self.lookback = lookback # results in (bid|size, ask|size) -> 4*5
        self.bookSize = bookSize
        self.featureType = featureType
        if self.featureType == FeatureType.ORDERS:
            self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(2*self.lookback+1, self.bookSize, 2))
        else:
            self.observation_space = spaces.Box(low=0.0, high=100.0, shape=(self.lookback+1, 1, 4))
        self.callbacks = callbacks
        self.episodeActions = []

    def setOrderbook(self, orderbook):
        self.orderbook = orderbook

    def setSide(self, side):
        self.side = side

    def setLevels(self, min, max, step):
        if self.market_action:
            self.levels = self._generate_Sequence(min=levels[0], max=levels[1], step=levels[2])
            self.levels.extend([None])
            self.action_space = spaces.Discrete(len(self.levels))
        else:
            self.levels = self._generate_Sequence(min=min, max=max, step=step)
            self.action_space = spaces.Discrete(len(self.levels))

    def setT(self, min, max, step):
        self.T = self._generate_Sequence(min=min, max=max, step=step)

    def setI(self, min, max, step):
        self.I = self._generate_Sequence(min=min, max=max, step=step)

    def setLookback(self, lookback):
        self.lookback = lookback
        if self.bookSize is not None:
            self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(2*self.lookback, self.bookSize, 2))

    def setBookSize(self, bookSize):
        self.bookSize = bookSize
        if self.lookback is not None:
            self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(2*self.lookback, self.bookSize, 2))



    def _determine_next_inventory(self, execution):
        qty_remaining = execution.getQtyNotExecuted()
        # TODO: Working with floats requires such an ugly threshold
        if qty_remaining > 0.0000001:
            # Approximate next closest inventory given remaining and I
            i_next = min([0.0] + self.I, key=lambda x: abs(x - qty_remaining))
            logging.info('Qty remain: ' + str(qty_remaining)
                         + ' -> inventory: ' + str(qty_remaining)
                         + ' -> next i: ' + str(i_next))
        else:
            i_next = 0.0

        logging.info('Next inventory for execution: ' + str(i_next))
        return i_next

    def _determine_next_time(self, t):
        if t > 0:
            t_next = self.T[self.T.index(t) - 1]
        else:
            t_next = t

        logging.info('Next timestep for execution: ' + str(t_next))
        return t_next

    def _determine_runtime(self, t):
        if t != 0:
            T_index = self.T.index(t)
            runtime = self.T[T_index] - self.T[T_index - 1]
        else:
            runtime = t
        return runtime

    def _get_random_orderbook_state(self):
        return self.orderbook.getRandomState(runtime=max(self.T), min_head=self.lookback)

    def _create_execution(self, a):
        runtime = self._determine_runtime(self.actionState.getT())
        orderbookState = self.orderbook.getState(self.orderbookIndex)

        if runtime <= 0.0 or a is None:
            price = None
            ot = OrderType.MARKET
        else:
            price = orderbookState.getPriceAtLevel(self.side, a)
            ot = OrderType.LIMIT

        order = Order(
            orderType=ot,
            orderSide=self.side,
            cty=self.actionState.getI(),
            price=price
        )
        execution = Action(a=a, runtime=runtime)
        execution.setState(self.actionState)
        execution.setOrder(order)
        execution.setOrderbookState(orderbookState)
        execution.setOrderbookIndex(self.orderbookIndex)
#         execution.setReferencePrice(orderbookState.getBestAsk())
        if self.side == OrderSide.BUY:
            execution.setReferencePrice(orderbookState.getBestAsk())
        else:
            execution.setReferencePrice(orderbookState.getBestBid())
        return execution

    def _update_execution(self, execution, a):
        runtime = self._determine_runtime(self.actionState.getT())
        orderbookState = self.orderbook.getState(self.orderbookIndex)

        if runtime <= 0.0 or a is None:
            price = None
            ot = OrderType.MARKET
        else:
            price = execution.getOrderbookState().getPriceAtLevel(self.side, a)
            ot = OrderType.LIMIT

        order = Order(
            orderType=ot,
            orderSide=self.side,
            cty=self.actionState.getI(),
            price=price
        )
        execution.setRuntime(runtime)
        execution.setState(self.actionState)
        execution.setOrder(order)
        execution.setOrderbookState(orderbookState)
        execution.setOrderbookIndex(self.orderbookIndex)
        return execution

    def _makeFeature(self, orderbookIndex, qty):
        if self.featureType == FeatureType.ORDERS:
            return self.orderbook.getBidAskFeatures(
                state_index=orderbookIndex,
                lookback=self.lookback,
                qty=self.I[-1],#i_next+0.0001,
                normalize=True,
                price=True,
                size=True,
                levels = self.bookSize
            )
        else:
            state = self.orderbook.getState(orderbookIndex)
            return self.orderbook.getHistTradesFeature(
                ts=state.getUnixTimestamp(),
                lookback=self.lookback,
                normalize=True,
                norm_size=qty,
                norm_price=state.getBidAskMid()
            )

    def step(self, action):
        self.episode += 1
        action = self.levels[action]
        #self.episodeActions.append(action)
        if self.execution is None:
            self.execution = self._create_execution(action)
            self.execution.volatility = self.volatility
            self.execution.max_price = self.max_price
            self.execution.min_price = self.min_price
            #print('Created execution ref price',self.execution.referencePrice)
        else:
            self.execution = self._update_execution(self.execution, action)
            #print('Updated execution ref price',self.execution.referencePrice)
        logging.info(
            'Created/Updated execution.' +
            '\nAction: ' + str(action) + ' (' + str(self.execution.getOrder().getType()) + ')' +
            '\nt: ' + str(self.actionState.getT()) +
            '\nruntime: ' + str(self.execution.getRuntime()) +
            '\ni: ' + str(self.actionState.getI())
        )
        self.execution, counterTrades = self.execution.run(self.orderbook)

        i_next = self._determine_next_inventory(self.execution)
        t_next = self._determine_next_time(self.execution.getState().getT())
        
        if self.last_inventory != 0.0:
            inventory_diff_ratio = (self.last_inventory - i_next)/self.last_inventory # 0~1
        else:
            inventory_diff_ratio = 0.0
        self.last_inventory = i_next
        
        # construct negative reward
        if inventory_diff_ratio == 0.0:
            var_reward = -1.0
        else:
            var_reward = inventory_diff_ratio
        
        # Consider the volatility
        #if inventory_diff_ratio != 0.0:
        #    var_reward = inventory_diff_ratio*self.volatility
        #else:
        #    var_reward = -1.0*self.volatility
        
        feature = self._makeFeature(orderbookIndex=self.execution.getOrderbookIndex(), qty=i_next)
        state_next = ActionState(t_next, i_next, {self.featureType.value: feature})
        done = self.execution.isFilled() or state_next.getI() == 0
        if done:
            reward = self.execution.getReward(self.reward_type)
            volumeRatio = 1.0
            if self.callbacks is not []:
                for cb in self.callbacks:
                    cb.on_episode_end(self.episode, {'episode_reward': reward, 'episode_actions': self.episodeActions})
            self.episodeActions = []
        else:
            reward, volumeRatio = self.execution.calculateRewardWeighted(counterTrades, self.I[-1], self.reward_type)

        logging.info(
            'Run execution.' +
            '\nTrades: ' + str(len(counterTrades)) +
            '\nReward: ' + str(reward) + ' (Ratio: ' + str(volumeRatio) + ')' +
            '\nDone: ' + str(done)
        )
        self.orderbookIndex = self.execution.getOrderbookIndex()
        self.actionState = state_next
        
        total_reward = reward*self.preference[0] + var_reward*self.preference[1]
        
        return state_next.toArray(), total_reward, done, self.volatility

    def reset(self):
        return self._reset(t=self.T[-1], i=self.I[-1])

    def _reset(self, t, i):
        orderbookState, orderbookIndex = self._get_random_orderbook_state()
        feature = self._makeFeature(orderbookIndex=orderbookIndex, qty=i)
        state = ActionState(t, i, {self.featureType.value: feature}) #np.array([[t, i]])
        self.execution = None
        self.orderbookIndex = orderbookIndex
        self.actionState = state
        self.set_volatility(orderbookIndex, orderbookState)
        self.set_max_min_profit(orderbookIndex, orderbookState)
        self.last_inventory = 1.0
        return state.toArray()
    
    def set_volatility(self, orderbookindex, orderbookstate):
        i=1
        done = False
        mid_price_list = []
        while not done and i+orderbookindex < len(self.orderbook.states):
            mid_price_list.append(self.orderbook.states[orderbookindex+i].getBidAskMid()) 
            if (self.orderbook.states[orderbookindex+i].timestamp - orderbookstate.timestamp).seconds >= self.T[-1]:
                #print(i)
                done = True
            i += 1
        mid_price_list = np.array(mid_price_list)
        std = np.std(mid_price_list)
        self.volatility = std
        
    def set_max_min_profit(self, orderbookindex, orderbookstate):
        i=1
        done = False
        mid_price_list = []
        while not done and i+orderbookindex < len(self.orderbook.states):
            mid_price_list.append(self.orderbook.states[orderbookindex+i].getBidAskMid()) 
            if (self.orderbook.states[orderbookindex+i].timestamp - orderbookstate.timestamp).seconds >= self.T[-1]:
                #print(i)
                done = True
            i += 1
        self.max_price = max(mid_price_list)
        self.min_price = min(mid_price_list)
        
    def render(self, mode='human', close=False):
        pass

    def seed(self, seed):
        pass
