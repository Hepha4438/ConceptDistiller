import os
import sys
import re
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import threading
import queue
import multiprocessing
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO, DQN
from train_ppo_concept import ConceptPPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from datetime import datetime

from config import get_dqn_config, get_ppo_config, get_ppo_concept_config, ENV_DIFFICULTY


def get_next_model_number(save_dir, prefix="minigrid"):
    """
    Find the next available model number (000-999) in the save directory.
    Scans for existing files matching pattern: {prefix}_XXX.zip
    Returns the next number after the highest found.
    """
    if not os.path.exists(save_dir):
        return 0
    
    # Find all files matching the pattern
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{3}})\.zip")
    existing_numbers = []
    
    for filename in os.listdir(save_dir):
        match = pattern.match(filename)
        if match:
            existing_numbers.append(int(match.group(1)))
    
    # Return next number (0 if none found)
    if not existing_numbers:
        return 0
    
    return max(existing_numbers) + 1


class ProgressCallback(BaseCallback):
    """Callback to send training progress to UI queue"""
    def __init__(self, progress_queue, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.progress_queue = progress_queue
        self.total_timesteps = total_timesteps
        self.last_update = 0
        
    def _on_step(self) -> bool:
        # Update every 100 steps to avoid overwhelming the queue
        if self.num_timesteps - self.last_update >= 100:
            info = {
                'timesteps': self.num_timesteps,
                'total_timesteps': self.total_timesteps,
                'progress': (self.num_timesteps / self.total_timesteps) * 100
            }
            
            # Get episode info if available
            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                info['ep_len_mean'] = ep_info.get('l', 0)
                info['ep_rew_mean'] = ep_info.get('r', 0)
            
            self.progress_queue.put(('progress', info))
            self.last_update = self.num_timesteps
        
        return True


class StopTrainingCallback(BaseCallback):
    """Callback to stop training when flag is set"""
    def __init__(self, stop_flag, verbose=0):
        super().__init__(verbose)
        self.stop_flag = stop_flag  # Should be a function that returns True to stop
        
    def _on_step(self) -> bool:
        # Return False to stop training, True to continue
        return not self.stop_flag()


def get_stop_flag(ui_instance):
    """Helper function to check if training should stop"""
    return not ui_instance.is_training


def run_env_human_mode(model_path, env_id, algorithm, num_episodes, result_queue):
    """Run environment in human mode in a separate process"""
    try:
        import gymnasium as gym
        from minigrid.wrappers import ImgObsWrapper
        from stable_baselines3 import PPO, DQN
        from train_ppo_concept import ConceptPPO
        import numpy as np
        from datetime import datetime
        
        # Create environment
        env = gym.make(env_id, render_mode="human")
        env = ImgObsWrapper(env)
        
        # Load model
        if algorithm.upper() == "PPO":
            model = PPO.load(model_path, env=env)
        elif algorithm.upper() == "DQN":
            model = DQN.load(model_path, env=env)
        elif algorithm.upper() == "PPO_CONCEPT":
            model = ConceptPPO.load(model_path, env=env)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Test the model
        episode_rewards = []
        episode_lengths = []
        successes = 0
        
        start_time = datetime.now()
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                
                # Safety check to prevent infinite loops
                if steps > 1000:
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Check success (if reward > 0 typically means success in MiniGrid)
            if total_reward > 0:
                successes += 1
        
        end_time = datetime.now()
        test_duration = (end_time - start_time).total_seconds()
        
        env.close()
        
        # Calculate statistics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_steps = np.mean(episode_lengths)
        std_steps = np.std(episode_lengths)
        success_rate = (successes / num_episodes) * 100
        
        result = {
            'env_id': env_id,
            'algorithm': algorithm,
            'episodes': num_episodes,
            'mode': 'human',
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_steps': avg_steps,
            'std_steps': std_steps,
            'success_rate': success_rate,
            'test_time': test_duration,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
        
        result_queue.put(result)
        
    except Exception as e:
        result_queue.put({'error': str(e), 'env_id': env_id, 'algorithm': algorithm})

class ModelTesterUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MiniGrid RL Training & Testing UI")
        self.root.geometry("1400x900")
        
        # Data storage
        self.models = {}
        self.test_configs = {}
        self.results = []
        self.result_queue = queue.Queue()
        self.training_queue = queue.Queue()
        
        # Training state
        self.is_training = False
        
        # Create UI
        self.create_ui()
        self.load_models()
        
        # Start queue checkers
        self.check_queue()
        self.check_training_queue()
    
    def create_ui(self):
        """Create the main UI layout"""
        # Main notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Training Tab (NEW)
        self.training_frame = ttk.Frame(notebook)
        notebook.add(self.training_frame, text="🎯 Training")
        self.create_training_tab()
        
        # Optuna Tuning Tab (Position 2)
        self.optuna_frame = ttk.Frame(notebook)
        notebook.add(self.optuna_frame, text="🎛️ Optuna Tuning")
        self.create_optuna_tab()
        
        # Model Selection Tab
        self.model_frame = ttk.Frame(notebook)
        notebook.add(self.model_frame, text="📂 Model Selection")
        self.create_model_selection_tab()
        
        # Configuration Tab
        self.config_frame = ttk.Frame(notebook)
        notebook.add(self.config_frame, text="⚙️ Test Configuration")
        self.create_config_tab()
        
        # Results Tab
        self.results_frame = ttk.Frame(notebook)
        notebook.add(self.results_frame, text="📊 Test Results")
        self.create_results_tab()
        
        # GradCAM Tab
        self.gradcam_frame = ttk.Frame(notebook)
        notebook.add(self.gradcam_frame, text="🔍 GradCAM Analysis")
        self.create_gradcam_tab()
    
    def create_training_tab(self):
        """Create the training interface"""
        # Main container
        main_container = ttk.Frame(self.training_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel: Configuration
        config_panel = ttk.LabelFrame(main_container, text="Training Configuration", padding=10)
        config_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Environment Selection
        env_frame = ttk.LabelFrame(config_panel, text="Environment", padding=5)
        env_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(env_frame, text="Select Environment:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.train_env_var = tk.StringVar(value="MiniGrid-Empty-5x5-v0")
        
        # Load environment options from config.py instead of hardcoding
        from config import ENV_DIFFICULTY
        # Sort by difficulty: easy → medium → hard → extreme
        easy_envs = sorted([env for env, diff in ENV_DIFFICULTY.items() if diff == "easy"])
        medium_envs = sorted([env for env, diff in ENV_DIFFICULTY.items() if diff == "medium"])
        hard_envs = sorted([env for env, diff in ENV_DIFFICULTY.items() if diff == "hard"])
        extreme_envs = sorted([env for env, diff in ENV_DIFFICULTY.items() if diff == "extreme"])
        env_options = easy_envs + medium_envs + hard_envs + extreme_envs
        
        env_combo = ttk.Combobox(env_frame, textvariable=self.train_env_var, values=env_options, width=30)
        env_combo.grid(row=0, column=1, padx=5, pady=2)
        env_combo.bind('<<ComboboxSelected>>', self.on_env_changed)
        
        # Algorithm Selection
        algo_frame = ttk.LabelFrame(config_panel, text="Algorithm", padding=5)
        algo_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(algo_frame, text="Select Algorithm:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.train_algo_var = tk.StringVar(value="PPO")
        algo_combo = ttk.Combobox(algo_frame, textvariable=self.train_algo_var, values=["PPO", "DQN", "PPO_CONCEPT"], width=15)
        algo_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        algo_combo.bind('<<ComboboxSelected>>', self.on_algo_changed)
        
        ttk.Label(algo_frame, text="Device:").grid(row=0, column=2, sticky=tk.W, padx=(20, 5), pady=2)
        self.train_device_var = tk.StringVar(value="mps")
        device_combo = ttk.Combobox(algo_frame, textvariable=self.train_device_var, 
                                    values=["cpu", "cuda", "mps"], width=10)
        device_combo.grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)
        
        # Hyperparameters Section
        hyperparam_frame = ttk.LabelFrame(config_panel, text="Hyperparameters", padding=5)
        hyperparam_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create scrollable frame for hyperparameters
        canvas = tk.Canvas(hyperparam_frame, height=300)
        scrollbar = ttk.Scrollbar(hyperparam_frame, orient="vertical", command=canvas.yview)
        self.hyperparam_frame = ttk.Frame(canvas)
        
        self.hyperparam_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.hyperparam_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Dictionary to store hyperparameter widgets
        self.hyperparam_vars = {}
        
        # Load initial hyperparameters
        self.load_hyperparameters()
        
        # Training Control Buttons
        button_frame = ttk.Frame(config_panel)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.train_button = ttk.Button(button_frame, text="▶️ Start Training", 
                                       command=self.start_training, style="Accent.TButton")
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="⏹️ Stop Training", 
                                     command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="🔄 Reload from config.py", 
                  command=self.load_hyperparameters).pack(side=tk.LEFT, padx=5)
        
        # Right panel: Progress and Logs
        progress_panel = ttk.LabelFrame(main_container, text="Training Progress", padding=10)
        progress_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Progress Information
        info_frame = ttk.Frame(progress_panel)
        info_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(info_frame, text="Status:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.train_status_var = tk.StringVar(value="Ready")
        ttk.Label(info_frame, textvariable=self.train_status_var, foreground="blue").grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)
        
        ttk.Label(info_frame, text="Progress:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.train_progress_var = tk.StringVar(value="0 / 0 steps (0.0%)")
        ttk.Label(info_frame, textvariable=self.train_progress_var).grid(row=1, column=1, sticky=tk.W, padx=10, pady=2)
        
        # Progress Bar (TQDM style)
        progress_bar_frame = ttk.Frame(progress_panel)
        progress_bar_frame.pack(fill=tk.X, pady=5)
        
        self.train_progress_bar = ttk.Progressbar(progress_bar_frame, mode='determinate', length=400)
        self.train_progress_bar.pack(fill=tk.X)
        
        self.train_progress_percent_var = tk.StringVar(value="0.0%")
        ttk.Label(progress_bar_frame, textvariable=self.train_progress_percent_var, 
                 font=('Arial', 10, 'bold')).pack(pady=2)
        
        # Episode Statistics
        stats_frame = ttk.LabelFrame(progress_panel, text="Episode Statistics", padding=5)
        stats_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(stats_frame, text="Episode Length (mean):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.ep_len_var = tk.StringVar(value="N/A")
        ttk.Label(stats_frame, textvariable=self.ep_len_var, font=('Arial', 10, 'bold'), 
                 foreground="green").grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)
        
        ttk.Label(stats_frame, text="Episode Reward (mean):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.ep_rew_var = tk.StringVar(value="N/A")
        ttk.Label(stats_frame, textvariable=self.ep_rew_var, font=('Arial', 10, 'bold'), 
                 foreground="green").grid(row=1, column=1, sticky=tk.W, padx=10, pady=2)
        
        # Training Log
        log_frame = ttk.LabelFrame(progress_panel, text="Training Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.train_log = scrolledtext.ScrolledText(log_frame, height=15, width=50, wrap=tk.WORD)
        self.train_log.pack(fill=tk.BOTH, expand=True)
    
    def create_model_selection_tab(self):
        """Create the model selection interface"""
        # Title
        ttk.Label(self.model_frame, text="Available Models", 
                 font=("Arial", 14, "bold")).pack(pady=10)
        
        # Model tree view with checkbox column
        columns = ("Select", "Environment", "Algorithm", "Model Path", "Status")
        self.model_tree = ttk.Treeview(self.model_frame, columns=columns, show="tree headings")
        
        self.model_tree.heading("Select", text="☑️")
        self.model_tree.column("Select", width=50, anchor=tk.CENTER)
        
        for col in columns[1:]:
            self.model_tree.heading(col, text=col)
            if col == "Model Path":
                self.model_tree.column(col, width=300)
            elif col == "Status":
                self.model_tree.column(col, width=80)
            else:
                self.model_tree.column(col, width=180)
        
        # Bind selection event to update checkbox display
        self.model_tree.bind('<<TreeviewSelect>>', self.on_tree_select)
        
        # Bind click event on tree to handle checkbox clicks
        self.model_tree.bind('<Button-1>', self.on_tree_click)
        
        # Scrollbar for tree
        tree_scroll = ttk.Scrollbar(self.model_frame, orient="vertical", command=self.model_tree.yview)
        self.model_tree.configure(yscrollcommand=tree_scroll.set)
        
        # Pack tree and scrollbar
        tree_frame = ttk.Frame(self.model_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.model_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons
        button_frame = ttk.Frame(self.model_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="🔄 Refresh Models", 
                  command=self.load_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="✅ Select All", 
                  command=self.select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="❌ Deselect All", 
                  command=self.deselect_all).pack(side=tk.LEFT, padx=5)
    
    def on_tree_click(self, event):
        """Handle click on tree to toggle checkbox selection"""
        region = self.model_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        
        column = self.model_tree.identify_column(event.x)
        item = self.model_tree.identify_row(event.y)
        
        # Check if click is on checkbox column (column #1 which is "Select")
        if column == "#1" and item:
            self.toggle_item_selection(item)
            # Prevent default selection behavior
            return "break"
    
    def toggle_item_selection(self, item):
        """Toggle selection state of an item (model or environment)"""
        tags = self.model_tree.item(item, 'tags')
        
        if 'model' in tags:
            # Toggle single model selection
            current_selection = list(self.model_tree.selection())
            if item in current_selection:
                # Deselect
                current_selection.remove(item)
            else:
                # Select
                current_selection.append(item)
            self.model_tree.selection_set(current_selection)
        else:
            # This is an environment node - toggle all its children
            children = self.get_all_model_children(item)
            if children:
                current_selection = set(self.model_tree.selection())
                
                # Check if all children are selected
                all_selected = all(child in current_selection for child in children)
                
                if all_selected:
                    # Deselect all children
                    for child in children:
                        current_selection.discard(child)
                else:
                    # Select all children
                    current_selection.update(children)
                
                self.model_tree.selection_set(list(current_selection))
    
    def get_all_model_children(self, item):
        """Recursively get all model items under a parent item"""
        models = []
        for child in self.model_tree.get_children(item):
            tags = self.model_tree.item(child, 'tags')
            if 'model' in tags:
                models.append(child)
            else:
                # Recursively check children (for nested structures)
                models.extend(self.get_all_model_children(child))
        return models
    
    def on_tree_select(self, event):
        """Update checkbox display when items are selected"""
        selected_items = self.model_tree.selection()
        
        # Update all items to show/hide checkbox
        for item in self.model_tree.get_children():
            self.update_checkbox_recursive(item, selected_items)
    
    def update_checkbox_recursive(self, item, selected_items):
        """Recursively update checkbox display for tree items"""
        tags = self.model_tree.item(item, 'tags')
        values = list(self.model_tree.item(item, 'values'))
        
        if 'model' in tags:
            # This is a model item
            if item in selected_items:
                values[0] = "✅"  # Selected
            else:
                values[0] = "☐"  # Unselected
            self.model_tree.item(item, values=values)
        else:
            # This is an environment/folder node
            children = self.get_all_model_children(item)
            if children:
                selected_children = [c for c in children if c in selected_items]
                if len(selected_children) == 0:
                    values[0] = "☐"  # None selected
                elif len(selected_children) == len(children):
                    values[0] = "✅"  # All selected
                else:
                    values[0] = "◐"  # Partially selected
                self.model_tree.item(item, values=values)
        
        # Recursively update children
        for child in self.model_tree.get_children(item):
            self.update_checkbox_recursive(child, selected_items)
    
    def load_hyperparameters(self):
        """Load default hyperparameters based on selected env and algo"""
        env_id = self.train_env_var.get()
        algo = self.train_algo_var.get()
        
        # Clear existing hyperparameter widgets
        for widget in self.hyperparam_frame.winfo_children():
            widget.destroy()
        
        self.hyperparam_vars.clear()
        
        # Get config based on algo
        if algo == "DQN":
            config = get_dqn_config(env_id)
        elif algo == "PPO":
            config = get_ppo_config(env_id)
        elif algo == "PPO_CONCEPT":
            config = get_ppo_concept_config(env_id)
        else:
            config = get_ppo_config(env_id)  # fallback
        
        # Create widgets for each hyperparameter
        row = 0
        for param_name, param_value in config.items():
            # Label
            label = ttk.Label(self.hyperparam_frame, text=f"{param_name}:")
            label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
            
            # Entry
            if isinstance(param_value, bool):
                var = tk.BooleanVar(value=param_value)
                widget = ttk.Checkbutton(self.hyperparam_frame, variable=var)
            elif isinstance(param_value, int):
                var = tk.IntVar(value=param_value)
                widget = ttk.Entry(self.hyperparam_frame, textvariable=var, width=20)
            elif isinstance(param_value, float):
                var = tk.StringVar(value=str(param_value))
                widget = ttk.Entry(self.hyperparam_frame, textvariable=var, width=20)
            else:
                var = tk.StringVar(value=str(param_value))
                widget = ttk.Entry(self.hyperparam_frame, textvariable=var, width=20)
            
            widget.grid(row=row, column=1, sticky=tk.W, padx=5, pady=3)
            self.hyperparam_vars[param_name] = var
            
            row += 1
        
        # Add device and seed (not in config)
        ttk.Label(self.hyperparam_frame, text="seed:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
        seed_var = tk.IntVar(value=42)
        ttk.Entry(self.hyperparam_frame, textvariable=seed_var, width=20).grid(row=row, column=1, sticky=tk.W, padx=5, pady=3)
        self.hyperparam_vars['seed'] = seed_var
        
    def on_env_changed(self, event):
        """Handle environment selection change"""
        self.load_hyperparameters()
    
    def on_algo_changed(self, event):
        """Handle algorithm selection change"""
        self.load_hyperparameters()
    
    def start_training(self):
        """Start training in a separate thread"""
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already in progress!")
            return
        
        # Get configuration
        env_id = self.train_env_var.get()
        algo = self.train_algo_var.get()
        device = self.train_device_var.get()
        
        # Get hyperparameters
        hyperparams = {}
        for param_name, var in self.hyperparam_vars.items():
            try:
                if isinstance(var, tk.IntVar):
                    hyperparams[param_name] = var.get()
                elif isinstance(var, tk.BooleanVar):
                    hyperparams[param_name] = var.get()
                elif isinstance(var, tk.StringVar):
                    val_str = var.get()
                    # Try to convert to appropriate type
                    if '.' in val_str or 'e-' in val_str.lower():
                        hyperparams[param_name] = float(val_str)
                    else:
                        try:
                            hyperparams[param_name] = int(val_str)
                        except:
                            hyperparams[param_name] = val_str
            except Exception as e:
                messagebox.showerror("Error", f"Invalid value for {param_name}: {str(e)}")
                return
        
        hyperparams['device'] = device
        hyperparams['env_id'] = env_id
        
        # Confirm
        total_timesteps = hyperparams.get('total_timesteps', 100000)
        msg = f"Start training {algo} on {env_id}?\n\n"
        msg += f"Total timesteps: {total_timesteps:,}\n"
        msg += f"Device: {device}\n"
        msg += f"\nThis may take several minutes..."
        
        if not messagebox.askyesno("Confirm Training", msg):
            return
        
        # Update UI state
        self.is_training = True
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.train_status_var.set(f"Training {algo} on {env_id}...")
        self.train_progress_bar['value'] = 0
        self.train_progress_var.set(f"0 / {total_timesteps} steps (0.0%)")
        self.train_progress_percent_var.set("0.0%")
        self.ep_len_var.set("N/A")
        self.ep_rew_var.set("N/A")
        self.train_log.delete(1.0, tk.END)
        self.train_log.insert(tk.END, f"Starting training at {datetime.now().strftime('%H:%M:%S')}\n")
        self.train_log.insert(tk.END, f"Environment: {env_id}\n")
        self.train_log.insert(tk.END, f"Algorithm: {algo}\n")
        self.train_log.insert(tk.END, f"Device: {device}\n")
        self.train_log.insert(tk.END, f"Total timesteps: {total_timesteps:,}\n")
        self.train_log.insert(tk.END, "-" * 50 + "\n")
        
        # Reset last logged percent for new training session
        self._last_logged_percent = 0
        
        # Start training thread
        training_thread = threading.Thread(
            target=self.run_training,
            args=(algo, hyperparams),
            daemon=True
        )
        training_thread.start()
    
    def stop_training(self):
        """Stop ongoing training"""
        if messagebox.askyesno("Confirm", "Stop training?"):
            self.is_training = False
            self.train_status_var.set("Stopping...")
    
    def run_training(self, algo, hyperparams):
        """Run training with progress callback for UI updates"""
        try:
            import glob
            from datetime import datetime
            
            env_id = hyperparams.pop('env_id')
            seed = hyperparams.pop('seed', 42)
            total_timesteps = hyperparams.pop('total_timesteps')
            device = hyperparams.pop('device', 'mps')
            
            if algo == "DQN":
                from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
                from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
                from train_dqn import make_env
                from minigrid_features_extractor import MinigridFeaturesExtractor
                
                # Create environment
                train_env = DummyVecEnv([make_env(env_id, seed=seed)])
                train_env = VecMonitor(train_env)
                
                eval_env = DummyVecEnv([make_env(env_id, seed=seed+100)])
                eval_env = VecMonitor(eval_env)
                
                save_dir = f"models/{env_id}/dqn"
                os.makedirs(save_dir, exist_ok=True)
                
                # Checkpoint directory (cleared each training run)
                checkpoint_dir = f"{save_dir}/last_train"
                if os.path.exists(checkpoint_dir):
                    import shutil
                    shutil.rmtree(checkpoint_dir)
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Get next model number for best model naming
                model_number = get_next_model_number(save_dir, prefix="dqn_minigrid")
                model_name = f"dqn_minigrid_{model_number:03d}"
                best_model_path = f"{save_dir}/{model_name}"
                
                # Create tensorboard log directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                tensorboard_log = f"minigrid_tensorboard/{env_id}/dqn"
                run_name = f"{model_name}_{timestamp}"
                os.makedirs(tensorboard_log, exist_ok=True)
                
                policy_kwargs = dict(
                    features_extractor_class=MinigridFeaturesExtractor,
                    features_extractor_kwargs=dict(features_dim=128),
                    net_arch=[256, 256],
                )
                
                # Create model
                model = DQN(
                    "CnnPolicy",
                    train_env,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=tensorboard_log,
                    seed=seed,
                    device=device,
                    **hyperparams
                )
                
                # Create callbacks
                checkpoint_callback = CheckpointCallback(
                    save_freq=10000,
                    save_path=checkpoint_dir,  # Save to last_train/
                    name_prefix="dqn_checkpoint",
                    save_replay_buffer=True,
                    save_vecnormalize=True,
                )
                
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=checkpoint_dir,  # Save to last_train/ during training
                    log_path=checkpoint_dir,  # evaluations.npz also in last_train/
                    eval_freq=5000,
                    n_eval_episodes=10,
                    deterministic=True,
                    render=False,
                )
                
                progress_callback = ProgressCallback(self.training_queue, total_timesteps)
                stop_callback = StopTrainingCallback(lambda: not self.is_training)
                
                from stable_baselines3.common.callbacks import CallbackList
                callback = CallbackList([checkpoint_callback, eval_callback, progress_callback, stop_callback])
                
                # Train with callback
                try:
                    model.learn(
                        total_timesteps=total_timesteps,
                        callback=callback,
                        tb_log_name=run_name,
                        progress_bar=False
                    )
                except KeyboardInterrupt:
                    # Handle manual stop
                    pass
                
                # Check if training was stopped early
                if not self.is_training:
                    self.training_queue.put(('stopped', {
                        'algo': algo,
                        'env_id': env_id
                    }))
                    train_env.close()
                    eval_env.close()
                    return
                
                # Copy best_model.zip from last_train/ to final location with numbered name
                best_model_in_last_train = f"{checkpoint_dir}/best_model.zip"
                if os.path.exists(best_model_in_last_train):
                    import shutil
                    shutil.copy2(best_model_in_last_train, f"{best_model_path}.zip")
                    final_save_path = best_model_path
                else:
                    final_save_path = best_model_path
                
                train_env.close()
                eval_env.close()
                
                self.training_queue.put(('complete', {
                    'algo': algo,
                    'env_id': env_id,
                    'save_path': final_save_path + '.zip'
                }))
                    
            elif algo == "PPO":  # PPO
                from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
                from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
                from train_ppo import make_env
                from minigrid_features_extractor import MinigridFeaturesExtractor
                
                n_envs = hyperparams.pop('n_envs', 4)
                
                # Create environments
                train_env = DummyVecEnv([make_env(env_id, seed=seed+i) for i in range(n_envs)])
                train_env = VecMonitor(train_env)
                
                eval_env = DummyVecEnv([make_env(env_id, seed=seed+100)])
                eval_env = VecMonitor(eval_env)
                
                save_dir = f"models/{env_id}/ppo"
                os.makedirs(save_dir, exist_ok=True)
                
                # Checkpoint directory (cleared each training run)
                checkpoint_dir = f"{save_dir}/last_train"
                if os.path.exists(checkpoint_dir):
                    import shutil
                    shutil.rmtree(checkpoint_dir)
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Get next model number for best model naming
                model_number = get_next_model_number(save_dir, prefix="ppo_minigrid")
                model_name = f"ppo_minigrid_{model_number:03d}"
                best_model_path = f"{save_dir}/{model_name}"
                
                # Create tensorboard log directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                tensorboard_log = f"minigrid_tensorboard/{env_id}/ppo"
                run_name = f"{model_name}_{timestamp}"
                os.makedirs(tensorboard_log, exist_ok=True)
                
                policy_kwargs = dict(
                    features_extractor_class=MinigridFeaturesExtractor,
                    features_extractor_kwargs=dict(features_dim=128),
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),
                )
                
                # Create model
                model = PPO(
                    "CnnPolicy",
                    train_env,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=tensorboard_log,
                    seed=seed,
                    device=device,
                    **hyperparams
                )
                
                # Create callbacks
                checkpoint_callback = CheckpointCallback(
                    save_freq=max(10000 // n_envs, 1),
                    save_path=checkpoint_dir,  # Save to last_train/
                    name_prefix="ppo_checkpoint",
                    save_replay_buffer=False,
                    save_vecnormalize=True,
                )
                
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=checkpoint_dir,  # Save to last_train/ during training
                    log_path=checkpoint_dir,  # evaluations.npz also in last_train/
                    eval_freq=max(5000 // n_envs, 1),
                    n_eval_episodes=10,
                    deterministic=True,
                    render=False,
                )
                
                progress_callback = ProgressCallback(self.training_queue, total_timesteps)
                stop_callback = StopTrainingCallback(lambda: not self.is_training)
                
                from stable_baselines3.common.callbacks import CallbackList
                callback = CallbackList([checkpoint_callback, eval_callback, progress_callback, stop_callback])
                
                # Train with callback
                try:
                    model.learn(
                        total_timesteps=total_timesteps,
                        callback=callback,
                        tb_log_name=run_name,
                        progress_bar=False
                    )
                except KeyboardInterrupt:
                    # Handle manual stop
                    pass
                
                # Check if training was stopped early
                if not self.is_training:
                    self.training_queue.put(('stopped', {
                        'algo': algo,
                        'env_id': env_id
                    }))
                    train_env.close()
                    eval_env.close()
                    return
                
                # Copy best_model.zip from last_train/ to final location with numbered name
                best_model_in_last_train = f"{checkpoint_dir}/best_model.zip"
                if os.path.exists(best_model_in_last_train):
                    import shutil
                    shutil.copy2(best_model_in_last_train, f"{best_model_path}.zip")
                    final_save_path = best_model_path
                else:
                    final_save_path = best_model_path
                
                train_env.close()
                eval_env.close()
                
                self.training_queue.put(('complete', {
                    'algo': algo,
                    'env_id': env_id,
                    'save_path': final_save_path + '.zip'
                }))
            
            elif algo == "PPO_CONCEPT":  # PPO with Concept Learning
                from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
                from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
                from train_ppo_concept import make_env, ConceptLoggingCallback
                from minigrid_features_extractor import MinigridFeaturesExtractor
                
                n_envs = hyperparams.pop('n_envs', 4)
                n_concepts = hyperparams.pop('n_concepts', 4)
                lambda_1 = hyperparams.pop('lambda_1', 0.01)
                lambda_2 = hyperparams.pop('lambda_2', 0.002)
                lambda_3 = hyperparams.pop('lambda_3', 0.0002)
                
                # Create environments
                train_env = DummyVecEnv([make_env(env_id, seed=seed+i) for i in range(n_envs)])
                train_env = VecMonitor(train_env)
                
                eval_env = DummyVecEnv([make_env(env_id, seed=seed+100)])
                eval_env = VecMonitor(eval_env)
                
                save_dir = f"models/{env_id}/ppo_concept"
                os.makedirs(save_dir, exist_ok=True)
                
                # Checkpoint directory (cleared each training run)
                checkpoint_dir = f"{save_dir}/last_train"
                if os.path.exists(checkpoint_dir):
                    import shutil
                    shutil.rmtree(checkpoint_dir)
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Get next model number for best model naming
                model_number = get_next_model_number(save_dir, prefix="ppo_concept_minigrid")
                model_name = f"ppo_concept_minigrid_{model_number:03d}"
                best_model_path = f"{save_dir}/{model_name}"
                
                # Create tensorboard log directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                tensorboard_log = f"minigrid_tensorboard/{env_id}/ppo_concept"
                run_name = f"{model_name}_{timestamp}"
                os.makedirs(tensorboard_log, exist_ok=True)
                
                policy_kwargs = dict(
                    features_extractor_class=MinigridFeaturesExtractor,
                    features_extractor_kwargs=dict(features_dim=128, concept_distilling=True, n_concepts=n_concepts),
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),
                )
                
                # Create model
                model = ConceptPPO(
                    "CnnPolicy",
                    train_env,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=tensorboard_log,
                    seed=seed,
                    device=device,
                    lambda_1=lambda_1,
                    lambda_2=lambda_2,
                    lambda_3=lambda_3,
                    **hyperparams
                )
                
                # Create callbacks
                concept_logging_cb = ConceptLoggingCallback()  # ✅ Log concept_detail metrics
                
                checkpoint_callback = CheckpointCallback(
                    save_freq=max(10000 // n_envs, 1),
                    save_path=checkpoint_dir,  # Save to last_train/
                    name_prefix="ppo_concept_checkpoint",
                    save_replay_buffer=False,
                    save_vecnormalize=True,
                )
                
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=checkpoint_dir,  # Save to last_train/ during training
                    log_path=checkpoint_dir,  # evaluations.npz also in last_train/
                    eval_freq=max(5000 // n_envs, 1),
                    n_eval_episodes=10,
                    deterministic=True,
                    render=False,
                )
                
                progress_callback = ProgressCallback(self.training_queue, total_timesteps)
                stop_callback = StopTrainingCallback(lambda: not self.is_training)
                
                from stable_baselines3.common.callbacks import CallbackList
                callback = CallbackList([concept_logging_cb, checkpoint_callback, eval_callback, progress_callback, stop_callback])
                
                # Train with callback
                try:
                    model.learn(
                        total_timesteps=total_timesteps,
                        callback=callback,
                        tb_log_name=run_name,
                        progress_bar=False
                    )
                except KeyboardInterrupt:
                    # Handle manual stop
                    pass
                
                # Check if training was stopped early
                if not self.is_training:
                    self.training_queue.put(('stopped', {
                        'algo': algo,
                        'env_id': env_id
                    }))
                    train_env.close()
                    eval_env.close()
                    return
                
                # Copy best_model.zip from last_train/ to final location with numbered name
                best_model_in_last_train = f"{checkpoint_dir}/best_model.zip"
                if os.path.exists(best_model_in_last_train):
                    import shutil
                    shutil.copy2(best_model_in_last_train, f"{best_model_path}.zip")
                    final_save_path = best_model_path
                else:
                    final_save_path = best_model_path
                
                train_env.close()
                eval_env.close()
                
                self.training_queue.put(('complete', {
                    'algo': algo,
                    'env_id': env_id,
                    'save_path': final_save_path + '.zip'
                }))
                
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.training_queue.put(('error', error_msg))
    
    def check_training_queue(self):
        """Check training queue for updates"""
        try:
            while True:
                msg_type, data = self.training_queue.get_nowait()
                
                if msg_type == 'progress':
                    # Update progress bar
                    progress = data['progress']
                    self.train_progress_bar['value'] = progress
                    self.train_progress_percent_var.set(f"{progress:.1f}%")
                    
                    timesteps = data['timesteps']
                    total = data['total_timesteps']
                    self.train_progress_var.set(f"{timesteps:,} / {total:,} steps ({progress:.1f}%)")
                    
                    # Update episode stats if available
                    if 'ep_len_mean' in data:
                        self.ep_len_var.set(f"{data['ep_len_mean']:.1f}")
                    if 'ep_rew_mean' in data:
                        self.ep_rew_var.set(f"{data['ep_rew_mean']:.3f}")
                    
                    # Log to text widget (only at integer percentages: 1%, 2%, 3%, ... 100%)
                    # Check if we're at a new integer percentage and haven't logged it yet
                    current_percent = int(progress)
                    if not hasattr(self, '_last_logged_percent'):
                        self._last_logged_percent = -1
                    
                    if current_percent > self._last_logged_percent and current_percent > 0:
                        self._last_logged_percent = current_percent
                        self.train_log.insert(tk.END, f"Progress: {current_percent}% - ")
                        if 'ep_rew_mean' in data:
                            self.train_log.insert(tk.END, f"Reward: {data['ep_rew_mean']:.3f}\n")
                        else:
                            self.train_log.insert(tk.END, "\n")
                        self.train_log.see(tk.END)
                
                elif msg_type == 'complete':
                    self.is_training = False
                    self.train_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    self.train_status_var.set("Training completed!")
                    
                    self.train_log.insert(tk.END, "\n" + "=" * 50 + "\n")
                    self.train_log.insert(tk.END, f"Training completed at {datetime.now().strftime('%H:%M:%S')}\n")
                    self.train_log.insert(tk.END, f"Model saved to: {data['save_path']}\n")
                    self.train_log.see(tk.END)
                    
                    messagebox.showinfo("Success", 
                        f"Training completed!\n\nModel saved to:\n{data['save_path']}")
                    
                    # Refresh models list
                    self.load_models()
                
                elif msg_type == 'error':
                    self.is_training = False
                    self.train_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    self.train_status_var.set("Error occurred")
                    
                    self.train_log.insert(tk.END, f"\nERROR: {data}\n")
                    self.train_log.see(tk.END)
                    
                    messagebox.showerror("Training Error", f"An error occurred:\n{data}")
                
                elif msg_type == 'stopped':
                    self.is_training = False
                    self.train_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    self.train_status_var.set("Training stopped!")
                    
                    self.train_log.insert(tk.END, "\n" + "=" * 50 + "\n")
                    self.train_log.insert(tk.END, f"Training stopped at {datetime.now().strftime('%H:%M:%S')}\n")
                    self.train_log.insert(tk.END, f"Training was stopped early for {data['env_id']} - {data['algo']}\n")
                    self.train_log.see(tk.END)
                    
                    messagebox.showinfo("Training Stopped", 
                        f"Training stopped!\n\nEnvironment: {data['env_id']}\nAlgorithm: {data['algo']}\n\nAny checkpoints saved in last_train/ folder.")
        
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_training_queue)
    
    def create_config_tab(self):
        """Create the test configuration interface"""
        # Configuration for each environment
        main_frame = ttk.Frame(self.config_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side: Environment configurations
        config_frame = ttk.LabelFrame(main_frame, text="Test Configurations")
        config_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Scrollable frame for configurations
        canvas = tk.Canvas(config_frame)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right side: Control buttons and progress
        control_frame = ttk.LabelFrame(main_frame, text="Test Control")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        ttk.Button(control_frame, text="Update Configurations", 
                  command=self.update_configurations).pack(pady=5, fill=tk.X)
        ttk.Button(control_frame, text="Run Selected Tests", 
                  command=self.run_tests).pack(pady=5, fill=tk.X)
        
        # Progress bar
        ttk.Label(control_frame, text="Progress:").pack(pady=(20, 5))
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.progress_var).pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(control_frame, mode='determinate')
        self.progress_bar.pack(pady=5, fill=tk.X)
        
        # Status text
        ttk.Label(control_frame, text="Status:").pack(pady=(20, 5))
        self.status_text = scrolledtext.ScrolledText(control_frame, height=10, width=40)
        self.status_text.pack(pady=5, fill=tk.BOTH, expand=True)
    
    def create_results_tab(self):
        """Create the results display interface"""
        # Results table
        columns = ("Environment", "Algorithm", "Model", "Episodes", "Mode", "Avg Reward", 
                  "Std Reward", "Avg Steps", "Std Steps", "Success Rate", "Test Time", "Video")
        
        self.results_tree = ttk.Treeview(self.results_frame, columns=columns, show="headings")
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            if col == "Model":
                self.results_tree.column(col, width=200)  # Wider for model names
            elif col == "Video":
                self.results_tree.column(col, width=80)   # Narrower for video button
            else:
                self.results_tree.column(col, width=120)
        
        # Bind double-click event to handle video opening
        self.results_tree.bind('<Double-1>', self.on_result_double_click)
        
        # Scrollbars
        results_scroll_y = ttk.Scrollbar(self.results_frame, orient="vertical", 
                                        command=self.results_tree.yview)
        results_scroll_x = ttk.Scrollbar(self.results_frame, orient="horizontal", 
                                        command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=results_scroll_y.set, 
                                   xscrollcommand=results_scroll_x.set)
        
        # Pack results tree
        self.results_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        results_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        results_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Export button
        button_frame = ttk.Frame(self.results_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Export to CSV", 
                  command=self.export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Results", 
                  command=self.clear_results).pack(side=tk.LEFT, padx=5)
        
        # Help text
        help_frame = ttk.Frame(self.results_frame)
        help_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        help_label = ttk.Label(help_frame, 
                              text="💡 Tip: Double-click on a row with '📹 Available' to watch the video!",
                              font=('Arial', 9), foreground='blue')
        help_label.pack(side=tk.LEFT)
    
    def load_models(self):
        """Scan the models directory and populate the tree"""
        self.models.clear()
        
        # Clear existing items
        for item in self.model_tree.get_children():
            self.model_tree.delete(item)
        
        models_dir = "models"
        if not os.path.exists(models_dir):
            messagebox.showwarning("Warning", "Models directory not found!")
            return
        
        # Scan for models
        for env_id in os.listdir(models_dir):
            env_path = os.path.join(models_dir, env_id)
            if not os.path.isdir(env_path):
                continue
                
            env_node = self.model_tree.insert("", "end", text=env_id, 
                                            values=("☐", env_id, "", "", "Environment"))
            
            for algorithm in os.listdir(env_path):
                algo_path = os.path.join(env_path, algorithm)
                if not os.path.isdir(algo_path):
                    continue
                
                # Find model files
                for model_file in os.listdir(algo_path):
                    if model_file.endswith('.zip'):
                        model_path = os.path.join(algo_path, model_file)
                        model_key = f"{env_id}_{algorithm}_{model_file}"
                        
                        self.models[model_key] = {
                            'env_id': env_id,
                            'algorithm': algorithm,
                            'path': model_path,
                            'file': model_file
                        }
                        
                        # Add to tree with checkbox column (☐ = unselected)
                        self.model_tree.insert(env_node, "end", 
                                             text=model_file,
                                             values=("☐", env_id, algorithm, model_path, "✓ Available"),
                                             tags=('model',))
        
        # Expand all nodes
        self.expand_all_nodes()
        self.update_configurations()
    
    def expand_all_nodes(self):
        """Expand all nodes in the tree"""
        def expand_node(node):
            self.model_tree.item(node, open=True)
            for child in self.model_tree.get_children(node):
                expand_node(child)
        
        for child in self.model_tree.get_children():
            expand_node(child)
    
    def select_all(self):
        """Select all model items"""
        for item in self.model_tree.get_children():
            self.select_recursively(item)
    
    def select_recursively(self, item):
        """Recursively select items"""
        if 'model' in self.model_tree.item(item, 'tags'):
            self.model_tree.selection_add(item)
        for child in self.model_tree.get_children(item):
            self.select_recursively(child)
    
    def deselect_all(self):
        """Deselect all items"""
        self.model_tree.selection_remove(self.model_tree.selection())
    
    def update_configurations(self):
        """Update configuration options based on available environments"""
        # Clear existing configurations
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.test_configs.clear()
        
        # Get unique environments
        environments = set()
        for model_key, model_info in self.models.items():
            environments.add(model_info['env_id'])
        
        # Create configuration for each environment
        for i, env_id in enumerate(sorted(environments)):
            self.create_env_config(env_id, i)
    
    def create_env_config(self, env_id, row):
        """Create configuration widgets for an environment"""
        frame = ttk.LabelFrame(self.scrollable_frame, text=f"Environment: {env_id}")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Episodes
        episodes_frame = ttk.Frame(frame)
        episodes_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(episodes_frame, text="Episodes:").pack(side=tk.LEFT)
        episodes_var = tk.IntVar(value=100)
        episodes_spinbox = tk.Spinbox(episodes_frame, from_=1, to=100, 
                                     textvariable=episodes_var, width=10)
        episodes_spinbox.pack(side=tk.RIGHT)
        
        # Render mode
        mode_frame = ttk.Frame(frame)
        mode_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(mode_frame, text="Render Mode:").pack(side=tk.LEFT)
        mode_var = tk.StringVar(value="rgb_array")
        mode_combo = ttk.Combobox(mode_frame, textvariable=mode_var, 
                                 values=["rgb_array", "human"], width=12)
        mode_combo.pack(side=tk.RIGHT)
        
        self.test_configs[env_id] = {
            'episodes': episodes_var,
            'mode': mode_var
        }
    
    def run_tests(self):
        """Run tests for selected models"""
        selected_items = self.model_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select models to test!")
            return
        
        # Get selected models
        test_models = []
        for item in selected_items:
            if 'model' in self.model_tree.item(item, 'tags'):
                values = self.model_tree.item(item, 'values')
                # Skip checkbox column (index 0)
                env_id, algorithm, model_path = values[1], values[2], values[3]
                
                # Extract model filename from path
                model_file = os.path.basename(model_path)
                
                if env_id in self.test_configs:
                    config = self.test_configs[env_id]
                    test_models.append({
                        'env_id': env_id,
                        'algorithm': algorithm,
                        'model_path': model_path,
                        'model_file': model_file,  # Add model filename
                        'episodes': config['episodes'].get(),
                        'mode': config['mode'].get()
                    })
        
        if not test_models:
            messagebox.showwarning("Warning", "No valid models selected!")
            return
        
        # Start testing in a separate thread
        self.progress_bar['maximum'] = len(test_models)
        self.progress_bar['value'] = 0
        self.progress_var.set(f"Testing 0/{len(test_models)} models...")
        
        thread = threading.Thread(target=self.run_tests_thread, args=(test_models,))
        thread.daemon = True
        thread.start()
    
    def run_tests_thread(self, test_models):
        """Run tests in a separate thread"""
        import sys
        for i, model_config in enumerate(test_models):
            try:
                status_msg = f"Testing {model_config['env_id']} - {model_config['algorithm']}..."
                print(f"[DEBUG] {status_msg}")
                sys.stdout.flush()
                self.result_queue.put(('status', status_msg))
                
                result = self.test_single_model(model_config)
                
                print(f"[DEBUG] Putting result to queue: {result}")
                sys.stdout.flush()
                self.result_queue.put(('result', result))
                self.result_queue.put(('progress', i + 1))
                
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                print(f"[ERROR] Exception in run_tests_thread: {error_msg}")
                sys.stdout.flush()
                
                error_result = {
                    'env_id': model_config['env_id'],
                    'algorithm': model_config['algorithm'],
                    'error': str(e)
                }
                self.result_queue.put(('error', error_result))
                self.result_queue.put(('progress', i + 1))
        
        print("[DEBUG] All tests complete, sending complete message")
        sys.stdout.flush()
        self.result_queue.put(('complete', None))
    
    def test_single_model(self, model_config):
        """
        Test a single model using test_agent.py
        This ensures CLI/UI sync and automatic video recording
        """
        model_path = model_config['model_path']
        env_id = model_config['env_id']
        algorithm = model_config['algorithm']
        num_episodes = model_config['episodes']
        render_mode = model_config['mode']
        model_file = model_config.get('model_file', os.path.basename(model_path))  # Get model filename
        
        try:
            # Import test_agent function from test_agent.py
            # This ensures we use the SAME testing logic as CLI
            from test_agent import test_agent
            
            print(f"[DEBUG] Testing model: {model_path}")
            print(f"[DEBUG] Environment: {env_id}, Algorithm: {algorithm}")
            
            # Call test_agent with video recording enabled
            result = test_agent(
                model_path=model_path,
                env_id=env_id,
                algorithm=algorithm,
                num_episodes=num_episodes,
                render=(render_mode == "human"),
                deterministic=True,
                max_steps=1000,
                save_video=True  # Enable video recording - saves to ./video/env_id/algo/model/
            )
            
            print(f"[DEBUG] test_agent returned: {result}")
            
            # test_agent returns: {mean_reward, std_reward, best_reward, best_episode, 
            #                      mean_length, std_length, success_rate, episode_rewards, episode_lengths}
            
            # Format for UI display
            formatted_result = {
                'env_id': env_id,
                'algorithm': algorithm,
                'model_file': model_file,  # Add model filename
                'episodes': num_episodes,
                'mode': render_mode,
                'avg_reward': result['mean_reward'],
                'std_reward': result['std_reward'],
                'avg_steps': result['mean_length'],
                'std_steps': result['std_length'],
                'success_rate': result['success_rate'],
                'test_time': 0.0,  # test_agent doesn't track this currently
                'episode_rewards': result['episode_rewards'],
                'episode_lengths': result['episode_lengths']
            }
            
            print(f"[DEBUG] Formatted result: {formatted_result}")
            return formatted_result
            
        except Exception as e:
            import traceback
            error_msg = f"Test failed for {model_path}: {str(e)}\n{traceback.format_exc()}"
            print(f"[ERROR] {error_msg}")
            raise Exception(error_msg)
    
    def check_queue(self):
        """Check for messages from the testing thread"""
        import sys
        try:
            while True:
                message_type, data = self.result_queue.get_nowait()
                print(f"[DEBUG check_queue] Received: {message_type}, data keys: {data.keys() if isinstance(data, dict) else type(data)}")
                sys.stdout.flush()
                
                if message_type == 'status':
                    self.status_text.insert(tk.END, data + "\n")
                    self.status_text.see(tk.END)
                
                elif message_type == 'result':
                    print(f"[DEBUG] Adding result to table: {data}")
                    sys.stdout.flush()
                    self.add_result_to_table(data)
                    self.results.append(data)
                    print(f"[DEBUG] Result added successfully")
                    sys.stdout.flush()
                
                elif message_type == 'error':
                    error_msg = f"Error testing {data['env_id']} - {data['algorithm']}: {data['error']}\n"
                    self.status_text.insert(tk.END, error_msg)
                    self.status_text.see(tk.END)
                
                elif message_type == 'progress':
                    self.progress_bar['value'] = data
                    total = self.progress_bar['maximum']
                    self.progress_var.set(f"Testing {data}/{int(total)} models...")
                
                elif message_type == 'complete':
                    self.progress_var.set("Testing completed!")
                    self.status_text.insert(tk.END, "All tests completed!\n")
                    self.status_text.see(tk.END)
        
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_queue)
    
    def add_result_to_table(self, result):
        """Add a test result to the results table"""
        # Check if video exists for this model
        env_id = result['env_id']
        algorithm = result['algorithm']
        model_file = result.get('model_file', 'N/A')
        
        video_status = "📹 Available" if self.check_video_exists(env_id, algorithm, model_file) else "❌ Not Available"
        
        values = (
            env_id,
            algorithm,
            model_file,  # Add model filename
            result['episodes'],
            result['mode'],
            f"{result['avg_reward']:.3f}",
            f"{result['std_reward']:.3f}",
            f"{result['avg_steps']:.1f}",
            f"{result['std_steps']:.1f}",
            f"{result['success_rate']:.1f}%",
            f"{result['test_time']:.1f}s",
            video_status  # Add video status
        )
        
        self.results_tree.insert("", "end", values=values)
    
    def export_results(self):
        """Export results to CSV"""
        if not self.results:
            messagebox.showinfo("Info", "No results to export!")
            return
        
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save results as..."
        )
        
        if filename:
            df = pd.DataFrame(self.results)
            df.to_csv(filename, index=False)
            messagebox.showinfo("Success", f"Results exported to {filename}")
    
    def clear_results(self):
        """Clear all results"""
        if messagebox.askyesno("Confirm", "Clear all results?"):
            self.results.clear()
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            self.status_text.delete(1.0, tk.END)

    def on_result_double_click(self, event):
        """Handle double-click on results table to open video"""
        selection = self.results_tree.selection()
        if not selection:
            return
            
        item = selection[0]
        values = self.results_tree.item(item, 'values')
        if len(values) < 12:  # Ensure we have all columns including Video
            return
            
        env_id = values[0]
        algorithm = values[1]
        model_name = values[2]
        video_status = values[11]  # Video column
        
        if "📹 Available" in video_status:
            self.open_video(env_id, algorithm, model_name)
        else:
            messagebox.showinfo("No Video", f"No video available for {model_name}\n\nRun a test with this model to generate a video.")
    
    def open_video(self, env_id, algorithm, model_name):
        """Open video file using system default video player"""
        import subprocess
        import platform
        
        # Construct video path based on test_agent.py naming convention
        model_dir = model_name.replace('.zip', '')
        video_dir = f"./video/{env_id}/{algorithm.lower()}/{model_dir}"
        
        # Find video file in directory
        import glob
        video_files = glob.glob(f"{video_dir}/*.mp4")
        
        if not video_files:
            messagebox.showerror("Video Not Found", 
                f"No video files found in: {video_dir}")
            return
            
        video_path = video_files[0]  # Take the first (should be only one)
        
        try:
            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.run(["open", video_path])
            elif system == "Windows":
                subprocess.run(["start", video_path], shell=True)
            else:  # Linux
                subprocess.run(["xdg-open", video_path])
                
            print(f"Opening video: {video_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open video: {str(e)}")
    
    def check_video_exists(self, env_id, algorithm, model_name):
        """Check if video file exists for given model"""
        model_dir = model_name.replace('.zip', '')
        video_dir = f"./video/{env_id}/{algorithm.lower()}/{model_dir}"
        
        import glob
        import os
        
        if not os.path.exists(video_dir):
            return False
            
        video_files = glob.glob(f"{video_dir}/*.mp4")
        return len(video_files) > 0

    def create_gradcam_tab(self):
        """Create the GradCAM analysis interface"""
        main_container = ttk.Frame(self.gradcam_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel: Model selection and configuration
        left_panel = ttk.LabelFrame(main_container, text="GradCAM Configuration", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Model Selection
        model_frame = ttk.LabelFrame(left_panel, text="Select PPO_CONCEPT Model", padding=5)
        model_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Treeview for models
        tree_frame = ttk.Frame(model_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.gradcam_tree = ttk.Treeview(tree_frame, columns=("Environment", "Model"), show="tree headings", height=15)
        self.gradcam_tree.heading("Environment", text="Environment")
        self.gradcam_tree.heading("Model", text="Model")
        self.gradcam_tree.column("#0", width=30)
        self.gradcam_tree.column("Environment", width=250)
        self.gradcam_tree.column("Model", width=250)
        
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.gradcam_tree.yview)
        self.gradcam_tree.configure(yscrollcommand=scrollbar.set)
        
        self.gradcam_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons
        btn_frame = ttk.Frame(model_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="🔄 Refresh Models", command=self.load_gradcam_models).pack(side=tk.LEFT, padx=5)
        
        # Parameters
        params_frame = ttk.LabelFrame(left_panel, text="Analysis Parameters", padding=5)
        params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(params_frame, text="Episodes to test:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.gradcam_episodes_var = tk.IntVar(value=10)
        episodes_spinbox = tk.Spinbox(params_frame, from_=1, to=100, textvariable=self.gradcam_episodes_var, width=10)
        episodes_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)
        
        ttk.Label(params_frame, text="Device:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.gradcam_device_var = tk.StringVar(value="cpu")
        device_combo = ttk.Combobox(params_frame, textvariable=self.gradcam_device_var, 
                                    values=["cpu", "cuda", "mps"], width=10, state="readonly")
        device_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)
        
        # FPS for video
        ttk.Label(params_frame, text="Video FPS:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.gradcam_fps_var = tk.IntVar(value=6)
        fps_spinbox = tk.Spinbox(params_frame, from_=1, to=30, textvariable=self.gradcam_fps_var, width=10)
        fps_spinbox.grid(row=2, column=1, sticky=tk.W, padx=5, pady=3)
        
        # Run button
        run_frame = ttk.Frame(left_panel)
        run_frame.pack(fill=tk.X, pady=10)
        
        self.gradcam_run_btn = ttk.Button(run_frame, text="▶️ Run GradCAM Analysis", 
                                         command=self.run_gradcam_analysis)
        self.gradcam_run_btn.pack(fill=tk.X, pady=5)
        
        # Status
        status_frame = ttk.LabelFrame(left_panel, text="Status", padding=5)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.gradcam_status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.gradcam_status_var, font=('Arial', 10)).pack(pady=5)
        
        self.gradcam_progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.gradcam_progress.pack(fill=tk.X, padx=5, pady=5)
        
        # Result buttons
        result_btn_frame = ttk.Frame(status_frame)
        result_btn_frame.pack(fill=tk.X, pady=5)
        
        self.gradcam_open_dir_btn = ttk.Button(result_btn_frame, text="📁 Open Output Directory",
                                               command=self.open_gradcam_directory, state=tk.DISABLED)
        self.gradcam_open_dir_btn.pack(fill=tk.X, pady=2)
        
        self.gradcam_open_video_btn = ttk.Button(result_btn_frame, text="🎬 Open Video",
                                                 command=self.open_gradcam_video, state=tk.DISABLED)
        self.gradcam_open_video_btn.pack(fill=tk.X, pady=2)
        
        # Store last result paths
        self.last_gradcam_dir = None
        self.last_gradcam_video = None
        
        # Right panel: Output log
        right_panel = ttk.LabelFrame(main_container, text="Analysis Log", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.gradcam_log = scrolledtext.ScrolledText(right_panel, height=30, width=60, wrap=tk.WORD)
        self.gradcam_log.pack(fill=tk.BOTH, expand=True)
        
        # Load models
        self.load_gradcam_models()
    
    def load_gradcam_models(self):
        """Load PPO_CONCEPT models from models directory"""
        # Clear existing
        for item in self.gradcam_tree.get_children():
            self.gradcam_tree.delete(item)
        
        models_dir = "models"
        if not os.path.exists(models_dir):
            self.gradcam_log.insert(tk.END, "⚠️ Models directory not found!\n")
            return
        
        count = 0
        for env_id in sorted(os.listdir(models_dir)):
            env_path = os.path.join(models_dir, env_id)
            if not os.path.isdir(env_path):
                continue
            
            ppo_concept_path = os.path.join(env_path, "ppo_concept")
            if not os.path.exists(ppo_concept_path):
                continue
            
            # Create environment node
            env_node = self.gradcam_tree.insert("", "end", text="", values=(env_id, ""))
            
            # Add models
            for model_file in sorted(os.listdir(ppo_concept_path)):
                if model_file.endswith('.zip') and not model_file.startswith('.'):
                    model_path = os.path.join(ppo_concept_path, model_file)
                    self.gradcam_tree.insert(env_node, "end", text="", 
                                           values=("", model_file),
                                           tags=('model',))
                    count += 1
        
        self.gradcam_log.insert(tk.END, f"✓ Loaded {count} PPO_CONCEPT models\n")
        self.gradcam_log.see(tk.END)
    
    def run_gradcam_analysis(self):
        """Run GradCAM analysis on selected model"""
        selection = self.gradcam_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a model to analyze!")
            return
        
        # Get selected model
        item = selection[0]
        values = self.gradcam_tree.item(item, 'values')
        
        # Check if it's a model (not environment node)
        if 'model' not in self.gradcam_tree.item(item, 'tags'):
            messagebox.showwarning("Invalid Selection", "Please select a model (not an environment folder)!")
            return
        
        # Get environment from parent
        parent = self.gradcam_tree.parent(item)
        parent_values = self.gradcam_tree.item(parent, 'values')
        env_id = parent_values[0]
        model_file = values[1]
        
        model_path = f"models/{env_id}/ppo_concept/{model_file}"
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found:\n{model_path}")
            return
        
        # Get parameters
        episodes = self.gradcam_episodes_var.get()
        device = self.gradcam_device_var.get()
        fps = self.gradcam_fps_var.get()
        
        # Output directory
        model_name = model_file.replace('.zip', '')
        out_dir = f"gradcam_out/{env_id}/ppo_concept/{model_name}"
        
        # Log start
        self.gradcam_log.insert(tk.END, f"\n{'='*60}\n")
        self.gradcam_log.insert(tk.END, f"Starting GradCAM Analysis\n")
        self.gradcam_log.insert(tk.END, f"{'='*60}\n")
        self.gradcam_log.insert(tk.END, f"Environment: {env_id}\n")
        self.gradcam_log.insert(tk.END, f"Model: {model_file}\n")
        self.gradcam_log.insert(tk.END, f"Episodes: {episodes}\n")
        self.gradcam_log.insert(tk.END, f"Device: {device}\n")
        self.gradcam_log.insert(tk.END, f"Output: {out_dir}\n")
        self.gradcam_log.insert(tk.END, f"{'='*60}\n")
        self.gradcam_log.see(tk.END)
        
        # Disable button and start progress
        self.gradcam_run_btn.config(state=tk.DISABLED)
        self.gradcam_progress.start(10)
        self.gradcam_status_var.set("Running analysis...")
        
        # Run in thread
        thread = threading.Thread(target=self._run_gradcam_thread, 
                                 args=(model_path, env_id, episodes, device, out_dir, fps))
        thread.daemon = True
        thread.start()
    
    def _run_gradcam_thread(self, model_path, env_id, episodes, device, out_dir, fps):
        """Run GradCAM analysis in background thread"""
        try:
            from test_agent_gradcam import run_and_collect_best_episode, generate_gradcam_for_best
            
            # Run episodes and collect best
            self.gradcam_log.insert(tk.END, "\n🎮 Running episodes to find best...\n")
            self.gradcam_log.see(tk.END)
            
            model, best_obs, frames, best_reward, _ = run_and_collect_best_episode(
                model_path=model_path,
                env_id=env_id,
                algorithm="PPO_CONCEPT",
                num_episodes=episodes,
                deterministic=True,
                device=device,
                out_dir=out_dir,
                max_steps=1000
            )
            
            self.gradcam_log.insert(tk.END, f"✓ Best episode reward: {best_reward}\n")
            self.gradcam_log.insert(tk.END, f"✓ Total frames: {len(best_obs)}\n")
            self.gradcam_log.see(tk.END)
            
            # Generate GradCAM
            self.gradcam_log.insert(tk.END, "\n🔍 Generating GradCAM visualizations...\n")
            self.gradcam_log.see(tk.END)
            
            result_dir = generate_gradcam_for_best(
                model=model,
                best_obs=best_obs,
                frames=frames,
                out_dir=out_dir,
                device=device,
                fps=fps
            )
            
            self.gradcam_log.insert(tk.END, f"\n{'='*60}\n")
            self.gradcam_log.insert(tk.END, f"✅ Analysis Complete!\n")
            self.gradcam_log.insert(tk.END, f"{'='*60}\n")
            self.gradcam_log.insert(tk.END, f"Results saved to:\n{result_dir}\n")
            self.gradcam_log.insert(tk.END, f"{'='*60}\n\n")
            self.gradcam_log.see(tk.END)
            
            # Success
            self.root.after(0, self._gradcam_complete, result_dir)
            
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.gradcam_log.insert(tk.END, f"\n❌ ERROR:\n{error_msg}\n")
            self.gradcam_log.see(tk.END)
            self.root.after(0, self._gradcam_error, str(e))
    
    def _gradcam_complete(self, result_dir):
        """Handle GradCAM completion"""
        self.gradcam_progress.stop()
        self.gradcam_run_btn.config(state=tk.NORMAL)
        self.gradcam_status_var.set("✅ Analysis complete!")
        
        # Store paths
        self.last_gradcam_dir = result_dir
        
        # Find video file
        import glob
        video_files = glob.glob(f"{result_dir}/*.mp4")
        if video_files:
            self.last_gradcam_video = video_files[0]
            self.gradcam_open_video_btn.config(state=tk.NORMAL)
        else:
            self.last_gradcam_video = None
            self.gradcam_open_video_btn.config(state=tk.DISABLED)
        
        # Enable directory button
        self.gradcam_open_dir_btn.config(state=tk.NORMAL)
        
        # Ask to open (now with choice)
        response = messagebox.askquestion("Complete", 
            f"GradCAM analysis complete!\n\nWhat would you like to open?",
            icon='info',
            type=messagebox.YESNOCANCEL)
        
        if response == 'yes':  # Open directory
            self.open_gradcam_directory()
        elif response == 'no':  # Open video
            self.open_gradcam_video()
    
    def _gradcam_error(self, error_msg):
        """Handle GradCAM error"""
        self.gradcam_progress.stop()
        self.gradcam_run_btn.config(state=tk.NORMAL)
        self.gradcam_status_var.set("❌ Error occurred")
        messagebox.showerror("Analysis Error", f"An error occurred:\n\n{error_msg}")
    
    def open_gradcam_directory(self):
        """Open the GradCAM output directory"""
        if not self.last_gradcam_dir or not os.path.exists(self.last_gradcam_dir):
            messagebox.showwarning("No Directory", "No output directory available.\nPlease run an analysis first.")
            return
        
        import subprocess
        import platform
        
        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", self.last_gradcam_dir])
            elif platform.system() == "Windows":
                subprocess.run(["explorer", self.last_gradcam_dir])
            else:  # Linux
                subprocess.run(["xdg-open", self.last_gradcam_dir])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open directory:\n{str(e)}")
    
    def open_gradcam_video(self):
        """Open the GradCAM video file"""
        if not self.last_gradcam_video or not os.path.exists(self.last_gradcam_video):
            messagebox.showwarning("No Video", "No video file available.\nPlease run an analysis first.")
            return
        
        import subprocess
        import platform
        
        try:
            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.run(["open", self.last_gradcam_video])
            elif system == "Windows":
                subprocess.run(["start", self.last_gradcam_video], shell=True)
            else:  # Linux
                subprocess.run(["xdg-open", self.last_gradcam_video])
            
            self.gradcam_log.insert(tk.END, f"\n📹 Opened video: {os.path.basename(self.last_gradcam_video)}\n")
            self.gradcam_log.see(tk.END)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open video:\n{str(e)}")

    def create_optuna_tab(self):
        """Create the Optuna hyperparameter tuning interface"""
        main_container = ttk.Frame(self.optuna_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel: Configuration
        left_panel = ttk.LabelFrame(main_container, text="Optuna Configuration", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Environment selection
        env_frame = ttk.LabelFrame(left_panel, text="Environment", padding=5)
        env_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(env_frame, text="Select Environment:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.optuna_env_var = tk.StringVar(value="MiniGrid-Empty-5x5-v0")
        
        from config import ENV_DIFFICULTY
        easy_envs = sorted([env for env, diff in ENV_DIFFICULTY.items() if diff == "easy"])
        medium_envs = sorted([env for env, diff in ENV_DIFFICULTY.items() if diff == "medium"])
        hard_envs = sorted([env for env, diff in ENV_DIFFICULTY.items() if diff == "hard"])
        extreme_envs = sorted([env for env, diff in ENV_DIFFICULTY.items() if diff == "extreme"])
        all_envs = easy_envs + medium_envs + hard_envs + extreme_envs
        
        env_combo = ttk.Combobox(env_frame, textvariable=self.optuna_env_var, values=all_envs, width=35)
        env_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        env_combo.bind('<<ComboboxSelected>>', self.on_optuna_env_changed)
        
        # Tuning parameters
        params_frame = ttk.LabelFrame(left_panel, text="Tuning Parameters", padding=5)
        params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(params_frame, text="Number of trials:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.optuna_trials_var = tk.IntVar(value=20)
        trials_spinbox = tk.Spinbox(params_frame, from_=5, to=200, textvariable=self.optuna_trials_var, width=10)
        trials_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)
        
        ttk.Label(params_frame, text="Timesteps per trial:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.optuna_timesteps_var = tk.IntVar(value=30000)
        timesteps_spinbox = tk.Spinbox(params_frame, from_=10000, to=200000, increment=10000,
                                       textvariable=self.optuna_timesteps_var, width=10)
        timesteps_spinbox.grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)
        
        ttk.Label(params_frame, text="Parallel envs:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.optuna_n_envs_var = tk.IntVar(value=4)
        n_envs_spinbox = tk.Spinbox(params_frame, from_=1, to=16, textvariable=self.optuna_n_envs_var, width=10)
        n_envs_spinbox.grid(row=2, column=1, sticky=tk.W, padx=5, pady=3)
        
        ttk.Label(params_frame, text="Startup trials:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=3)
        self.optuna_startup_trials_var = tk.IntVar(value=5)
        startup_spinbox = tk.Spinbox(params_frame, from_=0, to=50, textvariable=self.optuna_startup_trials_var, width=10)
        startup_spinbox.grid(row=3, column=1, sticky=tk.W, padx=5, pady=3)
        ttk.Label(params_frame, text="(random sampling)", font=('Arial', 8), foreground='gray').grid(row=3, column=2, sticky=tk.W, padx=2, pady=3)
        
        ttk.Label(params_frame, text="Device:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=3)
        self.optuna_device_var = tk.StringVar(value="cpu")
        device_combo = ttk.Combobox(params_frame, textvariable=self.optuna_device_var,
                                    values=["cpu", "cuda", "mps"], width=10, state="readonly")
        device_combo.grid(row=4, column=1, sticky=tk.W, padx=5, pady=3)
        
        # Button to load defaults from config
        ttk.Button(params_frame, text="🔄 Load Defaults from Config", 
                  command=self.load_optuna_defaults).grid(row=5, column=0, columnspan=3, pady=5)
        
        # Final training option
        final_frame = ttk.LabelFrame(left_panel, text="Final Training (Optional)", padding=5)
        final_frame.pack(fill=tk.X, pady=5)
        
        self.optuna_train_final_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(final_frame, text="Train final model with best params", 
                       variable=self.optuna_train_final_var).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=3)
        
        ttk.Label(final_frame, text="Final timesteps:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.optuna_final_timesteps_var = tk.IntVar(value=200000)
        final_timesteps_spinbox = tk.Spinbox(final_frame, from_=50000, to=1000000, increment=50000,
                                             textvariable=self.optuna_final_timesteps_var, width=10)
        final_timesteps_spinbox.grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)
        
        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.optuna_run_btn = ttk.Button(control_frame, text="▶️ Start Optuna Tuning",
                                         command=self.run_optuna_tuning)
        self.optuna_run_btn.pack(fill=tk.X, pady=5)
        
        self.optuna_stop_btn = ttk.Button(control_frame, text="⏹️ Stop Tuning",
                                          command=self.stop_optuna_tuning, state=tk.DISABLED)
        self.optuna_stop_btn.pack(fill=tk.X, pady=5)
        
        # Status
        status_frame = ttk.LabelFrame(left_panel, text="Status", padding=5)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.optuna_status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.optuna_status_var, font=('Arial', 10)).pack(pady=5)
        
        self.optuna_progress = ttk.Progressbar(status_frame, mode='determinate')
        self.optuna_progress.pack(fill=tk.X, padx=5, pady=5)
        
        # Right panel: Log and results
        right_panel = ttk.LabelFrame(main_container, text="Tuning Log & Results", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.optuna_log = scrolledtext.ScrolledText(right_panel, height=35, width=70, wrap=tk.WORD)
        self.optuna_log.pack(fill=tk.BOTH, expand=True)
        
        # Initialize Optuna state
        self.optuna_is_running = False
        self.optuna_stop_requested = False
        
        # Load initial defaults
        self.load_optuna_defaults()
    
    def load_optuna_defaults(self):
        """Load tuning parameters from config based on selected environment"""
        from config import get_optuna_tuning_config, get_ppo_concept_config
        
        env_id = self.optuna_env_var.get()
        
        # Get tuning config
        tuning_config = get_optuna_tuning_config(env_id)
        self.optuna_trials_var.set(tuning_config['n_trials'])
        self.optuna_timesteps_var.set(tuning_config['timesteps_per_trial'])
        self.optuna_n_envs_var.set(tuning_config['n_envs'])
        self.optuna_startup_trials_var.set(tuning_config.get('n_startup_trials', 5))
        
        # Get final timesteps from PPO_CONCEPT config
        ppo_concept_config = get_ppo_concept_config(env_id)
        self.optuna_final_timesteps_var.set(ppo_concept_config['total_timesteps'])
        
        self.optuna_log.insert(tk.END, f"✓ Loaded defaults for {env_id}\n")
        self.optuna_log.insert(tk.END, f"  Trials: {tuning_config['n_trials']}\n")
        self.optuna_log.insert(tk.END, f"  Startup trials: {tuning_config.get('n_startup_trials', 5)} (random sampling)\n")
        self.optuna_log.insert(tk.END, f"  Timesteps/trial: {tuning_config['timesteps_per_trial']:,}\n")
        self.optuna_log.insert(tk.END, f"  Parallel envs: {tuning_config['n_envs']}\n")
        self.optuna_log.insert(tk.END, f"  Final timesteps: {ppo_concept_config['total_timesteps']:,}\n\n")
        self.optuna_log.see(tk.END)
    
    def on_optuna_env_changed(self, event):
        """Handle environment selection change in Optuna tab"""
        self.load_optuna_defaults()
    
    def run_optuna_tuning(self):
        """Run Optuna hyperparameter tuning"""
        if self.optuna_is_running:
            messagebox.showwarning("Tuning Running", "Optuna tuning is already running!")
            return
        
        # Get parameters
        env_id = self.optuna_env_var.get()
        n_trials = self.optuna_trials_var.get()
        timesteps = self.optuna_timesteps_var.get()
        n_envs = self.optuna_n_envs_var.get()
        device = self.optuna_device_var.get()
        train_final = self.optuna_train_final_var.get()
        final_timesteps = self.optuna_final_timesteps_var.get()
        
        # Validation
        if n_trials < 5:
            messagebox.showwarning("Invalid Input", "Number of trials must be at least 5!")
            return
        
        # Log start
        from datetime import datetime
        self.optuna_log.delete(1.0, tk.END)
        self.optuna_log.insert(tk.END, f"{'='*60}\n")
        self.optuna_log.insert(tk.END, f"Optuna Tuning Started at {datetime.now().strftime('%H:%M:%S')}\n")
        self.optuna_log.insert(tk.END, f"{'='*60}\n")
        self.optuna_log.insert(tk.END, f"Environment: {env_id}\n")
        self.optuna_log.insert(tk.END, f"Trials: {n_trials}\n")
        self.optuna_log.insert(tk.END, f"Timesteps/trial: {timesteps:,}\n")
        self.optuna_log.insert(tk.END, f"Parallel envs: {n_envs}\n")
        self.optuna_log.insert(tk.END, f"Device: {device}\n")
        if train_final:
            self.optuna_log.insert(tk.END, f"Final training: Yes ({final_timesteps:,} timesteps)\n")
        else:
            self.optuna_log.insert(tk.END, f"Final training: No\n")
        self.optuna_log.insert(tk.END, f"{'='*60}\n\n")
        self.optuna_log.insert(tk.END, f"🎯 Optimization Objective:\n")
        self.optuna_log.insert(tk.END, f"   Score = 1.0*reward - 0.05*L_ortho - 0.02*L_spar - 0.01*L_l1\n")
        self.optuna_log.insert(tk.END, f"   (Maximize reward while minimizing concept losses)\n\n")
        self.optuna_log.insert(tk.END, f"{'='*60}\n\n")
        self.optuna_log.see(tk.END)
        
        # Update UI state
        self.optuna_is_running = True
        self.optuna_stop_requested = False
        self.optuna_run_btn.config(state=tk.DISABLED)
        self.optuna_stop_btn.config(state=tk.NORMAL)
        self.optuna_status_var.set("Running tuning...")
        self.optuna_progress['maximum'] = n_trials
        self.optuna_progress['value'] = 0
        
        # Run in thread
        thread = threading.Thread(target=self._run_optuna_thread,
                                 args=(env_id, n_trials, timesteps, n_envs, device, train_final, final_timesteps))
        thread.daemon = True
        thread.start()
    
    def _run_optuna_thread(self, env_id, n_trials, timesteps, n_envs, device, train_final, final_timesteps):
        """Run Optuna tuning in background thread"""
        try:
            from tune_ppo_concept_optuna import optimize_hyperparameters, train_with_best_params, set_ui_log_callback
            
            # Set callback for real-time logging
            def log_callback(message):
                self.root.after(0, lambda: self._append_optuna_log(message))

            set_ui_log_callback(log_callback)
            
            try:
                # Run optimization
                study = optimize_hyperparameters(
                    env_id=env_id,
                    n_trials=n_trials,
                    total_timesteps=timesteps,
                    n_envs=n_envs,
                    seed=42,
                    device=device,
                    storage=None
                )
                
                # Show best params (already logged by optimize_hyperparameters)
                # Just update progress
                self.root.after(0, lambda: self.optuna_progress.config(value=n_trials))
                
                # Optional final training
                if train_final:
                    log_callback(f"\n🎯 Training final model with best params...\n")

                    train_with_best_params(
                        study=study,
                        env_id=env_id,
                        total_timesteps=final_timesteps,
                        n_envs=n_envs,
                        seed=42,
                        device=device
                    )
                    
                    log_callback(f"\n✅ Final model training complete!\n\n")
                
            finally:
                set_ui_log_callback(None)  # Clear callback
            
            # Success
            self.root.after(0, self._optuna_complete)
            
        except Exception as e:
            import traceback
            error_msg = f"\n❌ ERROR:\n{str(e)}\n{traceback.format_exc()}\n"
            self.root.after(0, lambda: self._append_optuna_log(error_msg))
            self.root.after(0, self._optuna_error, str(e))

    def _append_optuna_log(self, message):
        """Append message to optuna log (called from main thread)"""
        self.optuna_log.insert(tk.END, message)
        self.optuna_log.see(tk.END)
    
    def stop_optuna_tuning(self):
        """Request to stop Optuna tuning"""
        if not self.optuna_is_running:
            return
        
        self.optuna_stop_requested = True
        self.optuna_status_var.set("Stopping...")
        self.optuna_log.insert(tk.END, f"\n⚠️  Stop requested by user. Finishing current trial...\n")
        self.optuna_log.see(tk.END)
        messagebox.showinfo("Stop Requested", "Tuning will stop after the current trial completes.")
    
    def _optuna_complete(self):
        """Handle Optuna completion"""
        self.optuna_is_running = False
        self.optuna_run_btn.config(state=tk.NORMAL)
        self.optuna_stop_btn.config(state=tk.DISABLED)
        self.optuna_status_var.set("✅ Tuning complete!")
        self.optuna_progress['value'] = self.optuna_progress['maximum']
        
        messagebox.showinfo("Complete", "Optuna tuning complete!\nCheck log for best parameters.")
    
    def _optuna_error(self, error_msg):
        """Handle Optuna error"""
        self.optuna_is_running = False
        self.optuna_run_btn.config(state=tk.NORMAL)
        self.optuna_stop_btn.config(state=tk.DISABLED)
        self.optuna_status_var.set("❌ Error occurred")
        
        messagebox.showerror("Tuning Error", f"An error occurred:\n\n{error_msg}")

def main():
    # Set multiprocessing start method to avoid issues on macOS
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # Already set
    
    root = tk.Tk()
    app = ModelTesterUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
