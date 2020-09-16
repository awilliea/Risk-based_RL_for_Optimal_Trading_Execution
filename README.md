# Risk-based Reward Shaping Reinforcement Learning for Optimal Trading Execution
This is a research extened from Juchli(2018) https://github.com/mjuchli/ctc-executioner

## Enviroment
You can create an virtual enviroment with requirement.txt.
e.g. 
```
conda create --name <env_name> python==3.7.6
source activate <env_name>
```
However, we revised some part of the tf-agents packages but have not done the pull-request. You need to overwrite the tf-agents packages with ours (in this github directory)
```
pip install -r requirements.txt
cp -r tf_agents ~/anaconda3/envs/<env_name>/lib/python3.7/site-packages/
```

## Usage

Get the results of the baselines
```
python get_baseines.py
```

Training the normal RL agent by CNN for buying signals on the downtrend dataset
```
python TFA_DDQN_Risk.py -s buy -d dn -m CNN -r profit
```

Training the reward-shaping RL agent by CNN for buying signals on the downtrend dataset
```
python TFA_DDQN_Risk.py -s buy -d dn -m CNN -r profit_vol
```

Training the MORL agent by CNN for buying signals on the downtrend dataset given preference = [0.8,0.2] (0.8 for profit and 0.2 for risk, which is a strategy for a risk-seeker)
```
python TFA_DDQN_Risk.py -s buy -d dn -m CNN -r profit_vol -p [0.8,0.2]
```

Training the risk-sensitive agent by CNN for buying signals on the downtrend dataset given beta = 0.2 (which is a strategy for a risk-seeker)
```
python TFA_DDQN_Risk.py -s buy -d dn -m CNN -r profit_vol -b 0.2
```

Training the risk-averse agent by CNN for buying signals on the downtrend dataset given lambda = 0.8 (which is a strategy for a risk-seeker)
```
python TFA_DDQN_Risk.py -s buy -d dn -m CNN -r profit_vol -l 0.2
```
