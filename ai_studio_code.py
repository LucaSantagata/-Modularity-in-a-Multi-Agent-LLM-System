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
API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
MODEL = "mistral-small-latest"

# Experiment Parameters
N_AGENTS = 30
N_ARMS = 50
N_ROUNDS = 20
ARMS_PER_AGENT = 5
MOCK_MODE = False

# --- CLUSTER PATHS ---
# On a cluster, it is safer to use absolute paths or the current directory
# This ensures you don't lose files if the job runs in a temp folder.
BASE_DIR = os.getcwd() 
CHECKPOINT_DF = os.path.join(BASE_DIR, "checkpoint_experiment_df.pkl")
CHECKPOINT_LOG = os.path.join(BASE_DIR, "checkpoint_message_log.pkl")

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

# --- 2. CLUSTER-SAFE CHECKPOINTING (ATOMIC WRITES) ---
def save_checkpoint(df, msg_log):
    """
    Saves data to a .tmp file first, then renames it.
    This prevents data corruption if the cluster kills the job mid-write.
    """
    # 1. Define Temp Filenames
    tmp_df_path = CHECKPOINT_DF + ".tmp"
    tmp_log_path = CHECKPOINT_LOG + ".tmp"
    
    # 2. Write to Temp Files
    df.to_pickle(tmp_df_path)
    with open(tmp_log_path, "wb") as f:
        pickle.dump(msg_log, f)
        
    # 3. Atomic Rename (The Safe Switch)
    # This replaces the old checkpoint with the new one instantly
    os.replace(tmp_df_path, CHECKPOINT_DF)
    os.replace(tmp_log_path, CHECKPOINT_LOG)

def load_checkpoint(n_agents, n_rounds):
    if os.path.exists(CHECKPOINT_DF) and os.path.exists(CHECKPOINT_LOG):
        print(f"Found checkpoint at {CHECKPOINT_DF}. Loading...")
        df = pd.read_pickle(CHECKPOINT_DF)
        with open(CHECKPOINT_LOG, "rb") as f:
            msg_log = pickle.load(f)
            
        start_round = 0
        for i in range(n_rounds):
            # Check Agent_0 to see if round is complete
            if df.at[i, "Agent_0"]['chosen_arm']: 
                start_round = i + 1
            else:
                break
        print(f"Resuming from Round {start_round}...")
        return start_round, df, msg_log
    else:
        print("No checkpoint found. Starting fresh.")
        df = initialize_experiment_log(n_agents, n_rounds)
        msg_log = initialize_message_log()
        return 0, df, msg_log

# --- 3. ENVIRONMENT & API ---
np.random.seed(42) 
TRUE_ARM_MEANS = np.clip(np.random.normal(loc=0.5, scale=0.15, size=N_ARMS), 0.0, 1.0)

def get_arm_reward(arm_index):
    mean = TRUE_ARM_MEANS[arm_index]
    return round(np.random.normal(loc=mean, scale=0.05), 4)

def call_mistral(system_prompt, user_prompt):
    if MOCK_MODE: return "MOCK_RESPONSE"
    max_retries = 5
    base_wait = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model=MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            content = response.choices[0].message.content
            if isinstance(content, list):
                # Handle reasoning models
                return "".join([c.text for c in content if c.type == 'text']).strip()
            return content
        except Exception as e:
            err = str(e).lower()
            wait = base_wait * (2 ** attempt)
            print(f"[Warning] API Error (Attempt {attempt+1}): {e}. Waiting {wait}s...")
            time.sleep(wait)
    return "ERROR"

# --- 4. CLASSES (Condensed for file) ---
class Agent:
    def __init__(self, agent_id, total_agents, total_arms, total_rounds):
        self.agent_id = f"Agent_{agent_id}"
        self.total_agents = total_agents
        self.total_arms = total_arms
        self.total_rounds = total_rounds
        self.history = []
        self.inbox = []
        
    def get_system_prompt(self):
        return f"You are {self.agent_id}, in a {self.total_agents}-agent bandit game. Goal: Maximize team reward."

    def _format_history(self):
        # FULL HISTORY ENABLED
        if not self.history: return "No history yet."
        return "\n".join([f"Round {r['round']}: Arm {r['arm']}, Reward {r['payoff']:.4f}" for r in self.history])

    def generate_message(self, current_round, assigned_arms, assignment_map):
        user_prompt = (f"--- ROUND {current_round} ---\nArms: {assigned_arms}\nMap: {assignment_map}\n"
                       f"History:\n{self._format_history()}\n\nTask: Message one agent.\n"
                       "Format: 'TO: Agent_X | MSG: <content>'")
        resp = call_mistral(self.get_system_prompt(), user_prompt)
        
        if MOCK_MODE: return f"Agent_{random.randint(0, self.total_agents-1)}", "Hello"
        
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

class RandomManager:
    def __init__(self, num_arms, num_agents, arms_per_agent):
        self.num_arms, self.num_agents, self.apa = num_arms, num_agents, arms_per_agent
        self.all_arms = list(range(num_arms))
        
    def assign_arms(self):
        assignment = {f"Agent_{i}": [] for i in range(self.num_agents)}
        deck, agents = self.all_arms[:], list(range(self.num_agents))
        random.shuffle(deck)
        random.shuffle(agents)
        
        for i, arm in enumerate(deck):
            assignment[f"Agent_{agents[i % self.num_agents]}"].append(arm)
            
        for i in range(self.num_agents):
            key = f"Agent_{i}"
            needed = self.apa - len(assignment[key])
            if needed > 0:
                avail = [x for x in self.all_arms if x not in assignment[key]]
                assignment[key].extend(random.sample(avail, min(len(avail), needed)))
        return assignment

# --- 5. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Load State
    start_round, experiment_df, message_log = load_checkpoint(N_AGENTS, N_ROUNDS)
    
    # 2. Setup Objects
    agents = [Agent(i, N_AGENTS, N_ARMS, N_ROUNDS) for i in range(N_AGENTS)]
    manager = RandomManager(N_ARMS, N_AGENTS, ARMS_PER_AGENT)
    
    # 3. Restore Memory
    if start_round > 0:
        print("Restoring memory...")
        for r in range(start_round):
            for agent in agents:
                rec = experiment_df.at[r, agent.agent_id]
                if rec['chosen_arm']:
                    agent.update_history(r, rec['chosen_arm'][0], rec['payoff'][0])
                    
    print(f"Running from Round {start_round}...")
    
    # 4. Loop
    for t in tqdm(range(start_round, N_ROUNDS)):
        # Manager
        assignments = manager.assign_arms()
        
        # Phase 1
        msgs = []
        for agent in agents:
            tid, content = agent.generate_message(t, assignments[agent.agent_id], assignments)
            if tid and tid in assignments and tid != agent.agent_id:
                rec = {'iteration': t, 'sender': agent.agent_id, 'receiver': tid, 'message': content}
                msgs.append(rec)
                message_log.append(rec)
        
        # Deliver
        agent_map = {a.agent_id: a for a in agents}
        for m in msgs: agent_map[m['receiver']].receive_message(m['sender'], m['message'])
            
        # Phase 2
        rewards = []
        for agent in agents:
            my_arms = assignments[agent.agent_id]
            choice = agent.make_choice(my_arms)
            payoff = get_arm_reward(choice)
            
            agent.update_history(t, choice, payoff)
            experiment_df.at[t, agent.agent_id] = {
                'assigned_arms': list(my_arms), 'chosen_arm': [choice], 'payoff': [payoff]
            }
            rewards.append(payoff)
            
        experiment_df.at[t, 'Total_Payoff'] = np.average(rewards)
        
        # SAVE ATOMICALLY
        save_checkpoint(experiment_df, message_log)

    print("Done! Exporting CSVs...")
    experiment_df.to_csv("final_results.csv")
    pd.DataFrame(message_log).to_csv("final_messages.csv")