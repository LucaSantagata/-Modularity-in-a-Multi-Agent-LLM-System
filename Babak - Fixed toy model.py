import pandas as pd
import numpy as np
import random
import re
import time
import os
import requests
import pickle
from tqdm import tqdm

# --- CONFIGURATION ---
API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
MODEL = "mistral-small-latest"

N_TRIALS = 10       
N_AGENTS = 3
N_ARMS = 20
N_ROUNDS = 30
STD_DEV = 5.0
MOCK_MODE = False   

# --- PATHS ---
OUTPUT_FILE = "final_static_10trials.csv"

# --- HELPER FUNCTIONS ---
def call_mistral(system_prompt, user_prompt):
    if MOCK_MODE: return "MOCK_RESPONSE"
    
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL, 
        "messages": [
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ]
    }
    
    for attempt in range(5):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json()['choices'][0]['message']['content']
            if isinstance(content, list): 
                return "".join([c.get('text', '') for c in content if isinstance(c, dict) and c.get('type') == 'text']).strip()
            return str(content).strip()
        except Exception as e:
            wait = 5 * (2 ** attempt)
            print(f"API Error (Attempt {attempt+1}): {e}. Waiting {wait}s...")
            time.sleep(wait)
    return "ERROR"

# --- AGENT CLASS (NO COMMUNICATION) ---
class SoloAgent:
    def __init__(self, agent_id, total_rounds):
        self.agent_id = f"Agent_{agent_id}"
        self.history = []

    def get_system_prompt(self):
        return f"You are {self.agent_id}. You have {N_ROUNDS} rounds to pull arms and maximize reward. Rewards range 0-100."

    def _format_history(self):
        if not self.history: return "No history yet."
        return "\n".join([f"Round {r['round']}: Pulled Arm {r['arm']}, Reward: {r['payoff']:.2f}" for r in self.history])

    def make_choice(self, assigned_arms):
        # Prompt focuses purely on personal history
        user_prompt = (
            f"My History:\n{self._format_history()}\n\n"
            f"Assigned Options: {assigned_arms}\n"
            f"Task: Pick one arm from the assigned options to maximize reward. "
            "Return ONLY the integer of the arm."
        )
        
        resp = call_mistral(self.get_system_prompt(), user_prompt)
        
        if MOCK_MODE: return random.choice(assigned_arms)
        
        nums = re.findall(r'\d+', resp)
        if nums:
            choice = int(nums[0])
            if choice in assigned_arms: return choice
            
        return random.choice(assigned_arms)

    def update_history(self, r, arm, payoff):
        self.history.append({'round': r, 'arm': arm, 'payoff': payoff})

# --- MANAGER (STATIC ASSIGNMENT) ---
class StaticManager:
    def __init__(self, num_arms, num_agents):
        self.num_arms = num_arms
        self.num_agents = num_agents
        self.all_arms = list(range(num_arms))
        
        # --- KEY CHANGE: Generate assignment ONCE during initialization ---
        deck = self.all_arms[:]
        random.shuffle(deck)
        self.static_assignment = {f"Agent_{i}": [] for i in range(self.num_agents)}
        
        for i, arm in enumerate(deck):
            target = i % self.num_agents
            self.static_assignment[f"Agent_{target}"].append(arm)

    def assign_arms(self):
        # Return the exact same dictionary every time
        return self.static_assignment

# --- MAIN EXPERIMENT LOOP ---
if __name__ == "__main__":
    
    all_trials_data = []
    
    print(f"Starting Static Experiment: {N_TRIALS} Trials, {N_AGENTS} Agents, {N_ROUNDS} Rounds.")
    print("Communication: DISABLED")
    print("Assignments: FIXED per Trial")

    for trial in range(N_TRIALS):
        print(f"\n--- Starting Trial {trial + 1}/{N_TRIALS} ---")
        
        # 1. Reset Environment for this trial
        trial_means = np.linspace(0, 100, N_ARMS)
        np.random.shuffle(trial_means)
        
        def get_reward(arm_idx):
            mean = trial_means[arm_idx]
            return round(np.clip(np.random.normal(loc=mean, scale=STD_DEV), 0.0, 100.0), 2)
        
        # 2. Reset Agents & Manager
        # The Manager will create the one-time assignment here
        agents = [SoloAgent(i, N_ROUNDS) for i in range(N_AGENTS)]
        manager = StaticManager(N_ARMS, N_AGENTS)
        
        # 3. Run Rounds
        for t in tqdm(range(N_ROUNDS), desc=f"Trial {trial+1}"):
            # This returns the SAME arms every round for this trial
            assignments = manager.assign_arms()
            
            # Phase 2: Action
            for agent in agents:
                my_arms = assignments[agent.agent_id]
                choice = agent.make_choice(my_arms)
                payoff = get_reward(choice)
                
                agent.update_history(t, choice, payoff)
                
                # Record Data
                record = {
                    "Trial_ID": trial,
                    "Round": t,
                    "Agent": agent.agent_id,
                    "Assigned_Arms": list(my_arms),
                    "Chosen_Arm": choice,
                    "Payoff": payoff
                }
                all_trials_data.append(record)
        
        # Save intermediate progress
        pd.DataFrame(all_trials_data).to_csv(OUTPUT_FILE, index=False)

    print("\nAll 10 Trials Complete!")
    print(f"Data saved to {OUTPUT_FILE}")