import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

class StockTradingEnv(gym.Env):

    def __init__(self, 
                 ticker_list: list, 
                 start_date: str, 
                 end_date: str, 
                 initial_balance: float = 10000.0, 
                 max_shares_per_trade: int = 10):

        super(StockTradingEnv, self).__init__()

        # Get infos
        self.ticker_list = ticker_list
        self.n_assets = len(ticker_list) # N
        self.initial_balance = initial_balance
        self.max_shares_per_trade = max_shares_per_trade
        
        # Load data
        self.df = self._load_data(start_date, end_date)
        self.dates = self.df.index.unique().sort_values()
        self.max_steps = len(self.dates) - 1

        # Action space
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.n_assets,), 
            dtype=np.float32
        )

        # Observation Space 
        self.n_features = 5  # (Open, High, Low, Close, Volume)
        
        total_obs_dim = 1 + self.n_assets + (self.n_assets * self.n_features)

        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_obs_dim,), 
            dtype=np.float32
        )

        self.balance = None
        self.shares_held = None
        self.current_step = 0
        self.portfolio_value = None

    def _load_data(self, start, end):
        data = yf.download(
            self.ticker_list, 
            start=start, 
            end=end, 
            auto_adjust=False, 
            actions=False, 
            group_by='ticker'
        )

        data = data.ffill().bfill()

        if self.n_assets == 1:
            if isinstance(data.columns, pd.MultiIndex):
                ticker_name = self.ticker_list[0]
                data = data[ticker_name]

        data = data.dropna()
        data = data.sort_index()

        if data.empty:
            raise ValueError
            
        return data

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.n_assets, dtype=int)
        self.current_step = 0
        self.portfolio_value = self.initial_balance
        
        state = self._get_observation()
        info = {}
        
        return state, info

    def step(self, action):
        prev_portfolio_value = self.portfolio_value

        current_prices = []
        for ticker in self.ticker_list:
            if self.n_assets == 1:
                price = self.df.iloc[self.current_step]['Close']
            else:
                price = self.df[ticker].iloc[self.current_step]['Close']
            current_prices.append(price)
        
        current_prices = np.array(current_prices)
        raw_shares_to_trade = (action * self.max_shares_per_trade).astype(int)

        # Sell
        for i in range(self.n_assets):
            if raw_shares_to_trade[i] < 0:
                shares_to_sell = abs(raw_shares_to_trade[i])

                actual_sell = min(shares_to_sell, self.shares_held[i])
                
                if actual_sell > 0:
                    gain = actual_sell * current_prices[i]
                    self.balance += gain
                    self.shares_held[i] -= actual_sell

        # Buy
        for i in range(self.n_assets):
            if raw_shares_to_trade[i] > 0:
                shares_to_buy = raw_shares_to_trade[i]
                
                cost = shares_to_buy * current_prices[i]
                
                if self.balance >= cost:
                    actual_buy = shares_to_buy
                else:
                    actual_buy = int(self.balance // current_prices[i])
                
                if actual_buy > 0:
                    cost = actual_buy * current_prices[i]
                    self.balance -= cost
                    self.shares_held[i] += actual_buy

        current_asset_value = np.sum(self.shares_held * current_prices)
        self.portfolio_value = self.balance + current_asset_value

        self.current_step += 1
        
        done = self.current_step >= self.max_steps
        truncated = False
        
        if not done:
            next_state = self._get_observation()
        else:
            self.current_step -= 1
            next_state = self._get_observation()
            self.current_step += 1
        
        # Reward calculation
        step_reward = self.portfolio_value - prev_portfolio_value
        reward = step_reward / self.initial_balance * 100 
        
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'step_reward': step_reward
        }
        
        return next_state, reward, done, truncated, info

    def _get_observation(self):
        market_data = []
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume']

        if self.n_assets == 1:
            data_slice = self.df.iloc[self.current_step][features]
            market_data = data_slice.values.astype(np.float32)
        else:
            temp_data = []
            for ticker in self.ticker_list:
                ticker_series = self.df[ticker].iloc[self.current_step][features]
                temp_data.append(ticker_series.values)
            market_data = np.concatenate(temp_data).astype(np.float32)

        obs = np.concatenate((
            np.array([self.balance], dtype=np.float32),  # (1)
            self.shares_held.astype(np.float32),         # (N)
            market_data                                  # (N * 5)
        ))
        
        return obs.astype(np.float32)