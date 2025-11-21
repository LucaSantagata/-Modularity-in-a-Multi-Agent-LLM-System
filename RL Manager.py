import pandas as pd
import numpy as np
import random
import re
import time
import os
import pickle
import sys
from tqdm import tqdm
from mistralai import Mistral
from scipy.special import softmax

# --- CONFIGURATION ---
API_KEY = "YOUR_ACTUAL_API_KEY_HERE" 
MODEL = "mistral-small-latest"

# Experiment Parameters
N_AGENTS = 30
N_ARMS = 50
N_ROUNDS = 20
MOCK_MODE = False

# RL Parameters
RL_LEARNING_RATE = 0.5
RL_TEMPERATURE = 2.0

# --- CLUSTER PATHS ---
BASE_DIR = os.getcwd() 
CHECKPOINT_DF = os.path.join(BASE_DIR, "checkpoint_experiment_df.pkl")
CHECKPOINT_LOG = os.path.join(BASE_DIR, "checkpoint_message_log.pkl")
CHECKPOINT_WEIGHTS = os.path.join(BASE_DIR, "checkpoint_weights_log.pkl") # <--- NEW FILE

# Initialize Client
if not MOCK_MODE:
    client = Mistral(api_key=API_KEY)
else:
    client = None

# --- 1. HELPER FUNCTIONS ---
def get_empty_structure():
    return {'assigned_arms': [], 'chosen_arm': [], 'payoff': []}

def initialize_experiment_log(num_agents, num_iterations):
    data = []
    for _ in range(num_iterations):
        row_data = {f"Agent_{i}": get_empty_structure() for i in range(num_agents)}
        row_data['Total_Payoff'] = 0.0
        data.append(row_data)
    return pd.DataFrame(data)

def initialize_message_log():
    return []

def initialize_weights_log():
    return []

# --- 2. CLUSTER-SAFE CHECKPOINTING ---
def save_checkpoint(df, msg_log, weights_log):
    """
    Saves DF, Messages, AND the Weight Matrix History.
    Atomic writes to prevent corruption.
    """
    # Define Temps
    tmp_df = CHECKPOINT_DF + ".tmp"
    tmp_msg = CHECKPOINT_LOG + ".tmp"
    tmp_w = CHECKPOINT_WEIGHTS + ".tmp"
    
    # Write Temps
    df.to_pickle(tmp_df)
    with open(tmp_msg, "wb") as f: pickle.dump(msg_log, f)
    with open(tmp_w, "wb") as f: pickle.dump(weights_log, f)
        
    # Atomic Rename
    os.replace(tmp_df, CHECKPOINT_DF)
    os.replace(tmp_msg, CHECKPOINT_LOG)
    os.replace(tmp_w, CHECKPOINT_WEIGHTS)

def load_checkpoint(n_agents, n_rounds):
    if os.path.exists(CHECKPOINT_DF) and os.path.exists(CHECKPOINT_LOG) and os.path.exists(CHECKPOINT_WEIGHTS):
        print(f"Found checkpoint. Loading...")
        df = pd.read_pickle(CHECKPOINT_DF)
        with open(CHECKPOINT_LOG, "rb") as f: msg_log = pickle.load(f)
        with open(CHECKPOINT_WEIGHTS, "rb") as f: weights_log = pickle.load(f)
            
        start_round = 0
        for i in range(n_rounds):
            if df.at[i, "Agent_0"]['chosen_arm']: 
                start_round = i + 1
            else:
                break
        print(f"Resuming experiment from Round {start_round}...")
        return start_round, df, msg_log, weights_log
    else:
        print("No checkpoint found. Starting fresh experiment.")
        df = initialize_experiment_log(n_agents, n_rounds)
        msg_log = initialize_message_log()
        weights_log = initialize_weights_log()
        return 0, df, msg_log, weights_log

# --- 3. ENVIRONMENT & API ---
np.random.seed(42) 

# Generate Evenly Spaced Means (0-100) & Shuffle
_sorted_means = np.linspace(0, 100, N_ARMS)
TRUE_ARM_MEANS = _sorted_means.copy()
np.random.shuffle(TRUE_ARM_MEANS)

print(f"Environment Initialized (0-100). Best Arm Mean: {np.max(TRUE_ARM_MEANS):.2f}")

def get_arm_reward(arm_index):
    mean = TRUE_ARM_MEANS[arm_index]
    # Scale=5.0 for 0-100 range
    reward = np.clip(np.random.normal(loc=mean, scale=5.0), 0.0, 100.0)
    return round(reward, 2)

def call_mistral(system_prompt, user_prompt):
    if MOCK_MODE: return "MOCK_RESPONSE"
    max_retries = 5
    base_wait = 5
    
    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            content = response.choices[0].message.content
            if isinstance(content, list):
                return "".join([c.text for c in content if c.type == 'text']).strip()
            return content
        except Exception as e:
            err = str(e).lower()
            wait = base_wait * (2 ** attempt)
            print(f"\n[API Warning] Attempt {attempt+1} failed: {e}")
            if "429" in err or "rate limit" in err:
                print(f"Rate Limit Hit. Sleeping {wait}s...")
            else:
                print(f"Retrying in {wait}s...")
            time.sleep(wait)
    return "ERROR"

# --- 4. CLASSES ---
class Agent:
    def __init__(self, agent_id, total_agents, total_arms, total_rounds):
        self.agent_id = f"Agent_{agent_id}"
        self.total_agents = total_agents
        self.total_arms = total_arms
        self.total_rounds = total_rounds
        self.history = []
        self.inbox = []
        
    def get_system_prompt(self):
        return (f"You are {self.agent_id}, in a {self.total_agents}-agent bandit game. "
                f"Rewards range from 0 to 100. Goal: Maximize team reward.")

    def _format_history(self):
        if not self.history: return "No history yet."
        # Returns FULL history
        return "\n".join([f"Round {r['round']}: Pulled Arm {r['arm']}, Reward: {r['payoff']:.2f}" for r in self.history])

    def generate_message(self, current_round, assigned_arms, assignment_map):
        hist_str = self._format_history()
        user_prompt = (
            f"--- ROUND {current_round} ---\n"
            f"My Assigned Arms: {assigned_arms}\n"
            f"All Agents' Assignments: {assignment_map}\n"
            f"My History:\n{hist_str}\n\n"
            "Task: Based on history and map, message ONE agent. "
            "Tell them something useful. Keep the message **short**.\n"
            "Format: 'TO: Agent_X | MSG: <content>'"
        )
        resp = call_mistral(self.get_system_prompt(), user_prompt)
        
        if MOCK_MODE: return f"Agent_{random.randint(0, self.total_agents-1)}", "Short msg"
        
        match = re.search(r"TO:\s*(Agent_\d+).*?MSG:\s*(.*)", resp, re.DOTALL | re.IGNORECASE)
        if match: return match.group(1).strip(), match.group(2).strip()
        return None, None

    def receive_message(self, sender, content):
        self.inbox.append(f"From {sender}: {content}")

    def make_choice(self, assigned_arms):
        inbox_txt = "\n".join(self.inbox) if self.inbox else "No messages."
        user_prompt = (f"Msgs:\n{inbox_txt}\n\nPick one arm from {assigned_arms}. Return ONLY the integer.")
        resp = call_mistral(self.get_system_prompt(), user_prompt)
        
        if MOCK_MODE: return random.choice(assigned_arms)
        
        nums = re.findall(r'\d+', resp)
        if nums:
            choice = int(nums[0])
            if choice in assigned_arms: return choice
        return random.choice(assigned_arms)

    def update_history(self, r, arm, payoff):
        self.history.append({'round': r, 'arm': arm, 'payoff': payoff})

class RLManager:
    def __init__(self, num_arms, num_agents, lr, temp):
        self.num_arms = num_arms
        self.num_agents = num_agents
        self.lr = lr
        self.temp = temp
        # Matrix: Rows=Agents, Cols=Arms
        self.weights = np.zeros((num_agents, num_arms))

    def assign_arms(self):
        assignment = {f"Agent_{i}": [] for i in range(self.num_agents)}
        
        # 1. Softmax Assignment
        probs = softmax(self.weights / self.temp, axis=0)
        agent_indices = list(range(self.num_agents))
        temp_counts = np.zeros(self.num_agents, dtype=int)
        
        for arm_idx in range(self.num_arms):
            col_probs = probs[:, arm_idx]
            chosen = np.random.choice(agent_indices, p=col_probs)
            assignment[f"Agent_{chosen}"].append(arm_idx)
            temp_counts[chosen] += 1
            
        # 2. Safety Net (No empty agents)
        empty_agents = [i for i, c in enumerate(temp_counts) if c == 0]
        if empty_agents:
            wealthy = [i for i, c in enumerate(temp_counts) if c > 1]
            random.shuffle(empty_agents)
            
            for poor in empty_agents:
                if not wealthy: break
                donor = wealthy[0]
                
                steal = assignment[f"Agent_{donor}"].pop()
                temp_counts[donor] -= 1
                
                assignment[f"Agent_{poor}"].append(steal)
                temp_counts[poor] += 1
                
                if temp_counts[donor] <= 1: wealthy.pop(0)
                
        return assignment

    def update_weights(self, agent_id_str, arm_idx, reward):
        idx = int(agent_id_str.split("_")[1])
        norm_reward = reward / 100.0
        # Gradient Ascent
        self.weights[idx, arm_idx] += self.lr * norm_reward

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load with Weights Log
    start_round, experiment_df, message_log, weights_log = load_checkpoint(N_AGENTS, N_ROUNDS)
    
    # 2. Setup with RL
    agents = [Agent(i, N_AGENTS, N_ARMS, N_ROUNDS) for i in range(N_AGENTS)]
    manager = RLManager(N_ARMS, N_AGENTS, learning_rate=RL_LEARNING_RATE, temperature=RL_TEMPERATURE)
    
    # 3. Restore Agent Memory
    if start_round > 0:
        print("Restoring Memory...")
        for r in range(start_round):
            for agent in agents:
                rec = experiment_df.at[r, agent.agent_id]
                if rec['chosen_arm']:
                    agent.update_history(r, rec['chosen_arm'][0], rec['payoff'][0])
        
        # Restore Manager Weights (Crucial for RL consistency)
        if weights_log:
            print("Restoring Manager Weights...")
            # Set weights to the state at the end of the last completed round
            manager.weights = weights_log[-1].copy()
            
    print(f"Starting RL Simulation. Alpha={RL_LEARNING_RATE}, Temp={RL_TEMPERATURE}")
    
    # 4. Loop
    for t in tqdm(range(start_round, N_ROUNDS), desc="RL Progress"):
        
        assignments = manager.assign_arms()
        
        # Phase 1: Message
        msgs_this_round = []
        for agent in agents:
            tid, content = agent.generate_message(t, assignments[agent.agent_id], assignments)
            if tid and tid in assignments and tid != agent.agent_id:
                rec = {'iteration': t, 'sender': agent.agent_id, 'receiver': tid, 'message': content}
                msgs_this_round.append(rec)
                message_log.append(rec)
        
        # Deliver
        agent_map = {a.agent_id: a for a in agents}
        for m in msgs_this_round: agent_map[m['receiver']].receive_message(m['sender'], m['message'])
            
        # Phase 2: Action
        round_payoffs = []
        for agent in agents:
            my_arms = assignments[agent.agent_id]
            choice = agent.make_choice(my_arms)
            payoff = get_arm_reward(choice)
            
            agent.update_history(t, choice, payoff)
            
            # Update Manager
            manager.update_weights(agent.agent_id, choice, payoff)
            
            experiment_df.at[t, agent.agent_id] = {
                'assigned_arms': list(my_arms),
                'chosen_arm': [choice],
                'payoff': [payoff]
            }
            round_payoffs.append(payoff)
            
        # Stats
        experiment_df.at[t, 'Total_Payoff'] = np.average(round_payoffs)
        
        # SNAPSHOT THE MATRIX
        weights_log.append(manager.weights.copy())
        
        # Atomic Save
        save_checkpoint(experiment_df, message_log, weights_log)
        
    print("Done! Exporting...")
    experiment_df.to_csv("final_rl_results.csv")
    pd.DataFrame(message_log).to_csv("final_rl_messages.csv")
    
    # Save final weights history for visualization
    with open("final_weights_history.pkl", "wb") as f:
        pickle.dump(weights_log, f)
        
    print("All files saved.")