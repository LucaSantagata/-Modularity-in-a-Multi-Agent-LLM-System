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

# --- CONFIGURATION ---
# PASTE YOUR API KEY HERE
API_KEY = "YOUR_ACTUAL_API_KEY_HERE" 
MODEL = "mistral-small-latest"

# Experiment Parameters
N_AGENTS = 30
N_ARMS = 50
N_ROUNDS = 20
ARMS_PER_AGENT = 5

# Set to True if you want to test without spending money (Random answers)
MOCK_MODE = False

# --- CLUSTER PATHS ---
# Uses current directory to ensure files are found on the cluster node
BASE_DIR = os.getcwd() 
CHECKPOINT_DF = os.path.join(BASE_DIR, "checkpoint_experiment_df.pkl")
CHECKPOINT_LOG = os.path.join(BASE_DIR, "checkpoint_message_log.pkl")

# Initialize Mistral Client
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

# --- 2. CLUSTER-SAFE CHECKPOINTING ---
def save_checkpoint(df, msg_log):
    """
    Atomic Save: Writes to .tmp first, then renames. 
    Prevents data corruption if cluster kills job mid-write.
    """
    tmp_df_path = CHECKPOINT_DF + ".tmp"
    tmp_log_path = CHECKPOINT_LOG + ".tmp"
    
    df.to_pickle(tmp_df_path)
    with open(tmp_log_path, "wb") as f:
        pickle.dump(msg_log, f)
        
    os.replace(tmp_df_path, CHECKPOINT_DF)
    os.replace(tmp_log_path, CHECKPOINT_LOG)

def load_checkpoint(n_agents, n_rounds):
    if os.path.exists(CHECKPOINT_DF) and os.path.exists(CHECKPOINT_LOG):
        print(f"Found checkpoint. Loading...")
        df = pd.read_pickle(CHECKPOINT_DF)
        with open(CHECKPOINT_LOG, "rb") as f:
            msg_log = pickle.load(f)
            
        start_round = 0
        for i in range(n_rounds):
            # If Agent_0 has made a choice, that round is done.
            if df.at[i, "Agent_0"]['chosen_arm']: 
                start_round = i + 1
            else:
                break
        print(f"Resuming experiment from Round {start_round}...")
        return start_round, df, msg_log
    else:
        print("No checkpoint found. Starting fresh experiment.")
        df = initialize_experiment_log(n_agents, n_rounds)
        msg_log = initialize_message_log()
        return 0, df, msg_log

# --- 3. ENVIRONMENT & API ---
np.random.seed(42) 

# Generate Evenly Spaced Means from 0 to 100
_sorted_means = np.linspace(0, 100, N_ARMS)
# Copy and Shuffle so Arm IDs don't correlate with Reward
TRUE_ARM_MEANS = _sorted_means.copy()
np.random.shuffle(TRUE_ARM_MEANS)

print(f"Environment Initialized (0-100). Best Arm Mean: {np.max(TRUE_ARM_MEANS):.2f}")

def get_arm_reward(arm_index):
    """
    Returns reward [0, 100].
    Noise scale=5.0 to match the larger range (equivalent to 0.05 on 0-1 scale).
    """
    mean = TRUE_ARM_MEANS[arm_index]
    reward = np.random.normal(loc=mean, scale=5.0)
    reward = np.clip(reward, 0.0, 100.0)
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
            
            # Handle reasoning models that return chunks
            if isinstance(content, list):
                return "".join([c.text for c in content if c.type == 'text']).strip()
            return content
            
        except Exception as e:
            error_msg = str(e).lower()
            wait_time = base_wait * (2 ** attempt)
            
            print(f"\n[API Warning] Attempt {attempt+1} failed: {e}")
            if "429" in error_msg or "rate limit" in error_msg:
                print(f"Rate Limit Hit. Sleeping {wait_time}s...")
            else:
                print(f"Retrying in {wait_time}s...")
            time.sleep(wait_time)
            
    return "ERROR"

# --- 4. AGENT & MANAGER CLASSES ---
class Agent:
    def __init__(self, agent_id, total_agents, total_arms, total_rounds):
        self.agent_id = f"Agent_{agent_id}"
        self.total_agents = total_agents
        self.total_arms = total_arms
        self.total_rounds = total_rounds
        self.history = []
        self.inbox = []
        
    def get_system_prompt(self):
        return (
            f"You are {self.agent_id}, in a {self.total_agents}-agent bandit game. "
            f"There are {self.total_arms} arms. Rewards range from 0 to 100. "
            "Goal: Maximize team reward."
        )

    def _format_history(self):
        # FULL HISTORY (No slicing)
        if not self.history: return "No history yet."
        return "\n".join([f"Round {r['round']}: Pulled Arm {r['arm']}, Reward: {r['payoff']:.2f}" for r in self.history])

    # --- PHASE 1: MSG ---
    def generate_message(self, current_round, assigned_arms, assignment_map):
        hist_str = self._format_history()
        
        user_prompt = (
            f"--- ROUND {current_round} ---\n"
            f"My Assigned Arms: {assigned_arms}\n"
            f"All Agents' Assignments: {assignment_map}\n"
            f"My History:\n{hist_str}\n\n"
            "Task: Based on your history and the map, choose one agent to message. "
            "Tell them something useful. Keep the message **short**.\n"
            "Response Format: 'TO: Agent_X | MSG: <content>'"
        )
        
        response_text = call_mistral(self.get_system_prompt(), user_prompt)
        
        if MOCK_MODE:
            target = f"Agent_{random.randint(0, self.total_agents-1)}"
            return target, "Short mock message"
        
        match = re.search(r"TO:\s*(Agent_\d+).*?MSG:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None, None

    def receive_message(self, sender_id, content):
        self.inbox.append(f"From {sender_id}: {content}")

    # --- PHASE 2: ACTION ---
    def make_choice(self, assigned_arms):
        inbox_text = "\n".join(self.inbox) if self.inbox else "No messages."
        
        user_prompt = (
            f"Messages received:\n{inbox_text}\n\n"
            f"Considering history and advice, pick one arm from {assigned_arms}. "
            "Return ONLY the integer."
        )
        
        response_text = call_mistral(self.get_system_prompt(), user_prompt)
        
        if MOCK_MODE: return random.choice(assigned_arms)
        
        numbers = re.findall(r'\d+', response_text)
        if numbers:
            choice = int(numbers[0])
            if choice in assigned_arms: return choice
            
        return random.choice(assigned_arms)

    def update_history(self, round_num, arm, payoff):
        self.history.append({'round': round_num, 'arm': arm, 'payoff': payoff})

class RandomManager:
    def __init__(self, num_arms, num_agents, arms_per_agent):
        self.num_arms = num_arms
        self.num_agents = num_agents
        self.apa = arms_per_agent
        self.all_arms = list(range(num_arms))

    def assign_arms(self):
        assignment = {f"Agent_{i}": [] for i in range(self.num_agents)}
        
        # Double Shuffle to remove bias
        deck = self.all_arms[:]
        random.shuffle(deck)
        agent_order = list(range(self.num_agents))
        random.shuffle(agent_order)
        
        # 1. Coverage
        for i, arm in enumerate(deck):
            target_idx = agent_order[i % self.num_agents]
            assignment[f"Agent_{target_idx}"].append(arm)
            
        # 2. Top Up
        for i in range(self.num_agents):
            key = f"Agent_{i}"
            needed = self.apa - len(assignment[key])
            if needed > 0:
                avail = [x for x in self.all_arms if x not in assignment[key]]
                if avail:
                    assignment[key].extend(random.sample(avail, min(len(avail), needed)))
        return assignment

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    # Load State
    start_round, experiment_df, message_log = load_checkpoint(N_AGENTS, N_ROUNDS)
    
    # Setup
    agents = [Agent(i, N_AGENTS, N_ARMS, N_ROUNDS) for i in range(N_AGENTS)]
    manager = RandomManager(N_ARMS, N_AGENTS, ARMS_PER_AGENT)
    
    # Restore Memory
    if start_round > 0:
        print("Restoring Agent Memories...")
        for r in range(start_round):
            for agent in agents:
                rec = experiment_df.at[r, agent.agent_id]
                if rec['chosen_arm']:
                    agent.update_history(r, rec['chosen_arm'][0], rec['payoff'][0])
    
    print(f"Starting Simulation: {N_AGENTS} Agents, {N_ROUNDS} Rounds. Range [0,100].")
    
    # Main Loop
    for t in tqdm(range(start_round, N_ROUNDS), desc="Experiment Progress"):
        
        # A. Assign
        assignments = manager.assign_arms()
        
        # B. Message
        msgs_this_round = []
        for agent in agents:
            my_arms = assignments[agent.agent_id]
            tid, content = agent.generate_message(t, my_arms, assignments)
            
            if tid and tid in assignments and tid != agent.agent_id:
                record = {
                    'iteration': t, 
                    'sender': agent.agent_id, 
                    'receiver': tid, 
                    'message': content
                }
                msgs_this_round.append(record)
                message_log.append(record)
        
        # C. Deliver
        agent_map = {a.agent_id: a for a in agents}
        for m in msgs_this_round:
            agent_map[m['receiver']].receive_message(m['sender'], m['message'])
            
        # D. Action
        round_payoffs = []
        for agent in agents:
            my_arms = assignments[agent.agent_id]
            choice = agent.make_choice(my_arms)
            payoff = get_arm_reward(choice)
            
            agent.update_history(t, choice, payoff)
            
            experiment_df.at[t, agent.agent_id] = {
                'assigned_arms': list(my_arms),
                'chosen_arm': [choice],
                'payoff': [payoff]
            }
            round_payoffs.append(payoff)
            
        # E. Stats & Save
        total_score = np.average(round_payoffs)
        experiment_df.at[t, 'Total_Payoff'] = total_score
        
        save_checkpoint(experiment_df, message_log)
        
    print("\nExperiment Complete!")
    experiment_df.to_csv("final_experiment_results.csv")
    pd.DataFrame(message_log).to_csv("final_message_log.csv")
    print("Final Results Saved.")