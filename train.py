import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from a2c_algorithm import A2CAgent
from ppo_algorithm import PPOAgent
from env import StockTradingEnv 

def make_env(ticker_list, start_date, end_date):
    env = StockTradingEnv(ticker_list=ticker_list, start_date=start_date, end_date=end_date)
    return env

def train_agent(algo_class, env, total_timesteps=20000):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = algo_class(state_dim, action_dim)
    
    current_step = 0
    episode = 0
    
    print(f"Train {algo_class.__name__}")

    while current_step < total_timesteps:
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_transition((state, action, reward, next_state, done, log_prob))
            state = next_state
            episode_reward += reward
            current_step += 1
            
            if current_step >= total_timesteps:
                break
        
        agent.update()
        episode += 1
        
        print(f"Episode: {episode}, Steps: {current_step}/{total_timesteps}, Reward: {episode_reward:.2f}")
            
    return agent

def evaluate_agent(model, env, env_name="Test"):
    obs, _ = env.reset()
    done = False
    portfolio_values = []
    
    while not done:
        action, _ = model.select_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        portfolio_values.append(info['portfolio_value'])
        
    final_value = portfolio_values[-1]

    initial = env.initial_balance
    profit_pct = ((final_value - initial) / initial) * 100
    
    print(f"[{env_name}] Final Value: ${final_value:,.2f} ({profit_pct:.2f}%)")
    return portfolio_values, final_value

def run_scenario(n_tickers, train_tickers, test_scenarios, train_period, test_period):
    print(f"\nScenario N={n_tickers} Start")
    
    env_train = make_env(train_tickers, *train_period)
    
    print(f"Training PPO on {train_tickers}")
    model_ppo = train_agent(PPOAgent, env_train, total_timesteps=20000)
    
    print(f"Training A2C on {train_tickers}")
    model_a2c = train_agent(A2CAgent, env_train, total_timesteps=20000)
    
    results = {'PPO': {}, 'A2C': {}}
    
    for scenario_name, test_tickers in test_scenarios.items():
        print(f"\nTesting: {scenario_name} (Tickers: {test_tickers})")
        
        env_test = make_env(test_tickers, *test_period)
        
        p_vals_ppo, f_val_ppo = evaluate_agent(model_ppo, env_test, f"PPO-{scenario_name}")
        results['PPO'][scenario_name] = p_vals_ppo
        
        env_test.reset()
        
        p_vals_a2c, f_val_a2c = evaluate_agent(model_a2c, env_test, f"A2C-{scenario_name}")
        results['A2C'][scenario_name] = p_vals_a2c

    return results

if __name__ == "__main__":
    TRAIN_PERIOD = ("2018-01-01", "2021-12-31")
    TEST_PERIOD = ("2022-01-01", "2023-12-31")

    # Case 1: N=1
    n1_scenarios = {
        "Seen": ["AAPL"],
        "Unseen_MSFT": ["MSFT"],
        "Unseen_GOOGL": ["GOOGL"],
        "Unseen_AMZN": ["AMZN"]
    }
    
    results_n1 = run_scenario(
        n_tickers=1, 
        train_tickers=["AAPL"], 
        test_scenarios=n1_scenarios,
        train_period=TRAIN_PERIOD,
        test_period=TEST_PERIOD
    )

    # Case 2: N>1
    n4_train_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    n4_scenarios = {
        "Seen": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "Mixed": ["AAPL", "MSFT", "TSLA", "NVDA"], # 2 Seen + 2 Unseen
        "New": ["TSLA", "NVDA", "META", "NFLX"]    # 4 Unseen
    }

    results_n4 = run_scenario(
        n_tickers=4,
        train_tickers=n4_train_tickers,
        test_scenarios=n4_scenarios,
        train_period=TRAIN_PERIOD,
        test_period=TEST_PERIOD
    )

    def plot_results(results, title_suffix):
        plt.figure(figsize=(12, 6))
        for algo, scenarios in results.items():
            for scenario_name, values in scenarios.items():
                plt.plot(values, label=f"{algo} - {scenario_name}")
        
        plt.title(f"Portfolio Value Over Time ({title_suffix})")
        plt.xlabel("Trading Days")
        plt.ylabel("Value ($)")
        plt.legend()
        plt.grid(True)
        plt.show()

    plot_results(results_n1, "N=1 Case")
    plot_results(results_n4, "N=4 Case")
    
    print("\nDone!")