import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import subprocess
import sys
import os
from PIL import Image, ImageTk

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


class UPRYTApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("UPRYT - Complete Posture Analysis System")
        self.root.geometry("900x750")
        self.root.resizable(True, True)
        self.process = None
        self.is_running = False
        
        self._setup_styles()
        self._create_header()
        self._create_mode_selection()
        self._create_options()
        self._create_control_buttons()
        self._create_log_area()
        self._create_footer()
        
        # Initialize mode-dependent UI state
        self._on_mode_changed()
        
        self._check_models()
    
    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"))
        self.style.configure("Section.TLabel", font=("Segoe UI", 12, "bold"))
        self.style.configure("Action.TButton", font=("Segoe UI", 11), padding=10)
        self.style.configure("Status.TLabel", font=("Segoe UI", 10))
        self.style.configure("Info.TLabel", font=("Segoe UI", 9))
    
    def _create_header(self):
        header_frame = ttk.Frame(self.root, padding=15)
        header_frame.pack(fill="x")
        
        logo_path = os.path.join(PROJECT_DIR, "logo", "upryt_white.png")
        if os.path.exists(logo_path):
            try:
                logo_img = Image.open(logo_path)
                logo_img = logo_img.resize((50, 50), Image.Resampling.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(logo_img)
                logo_label = tk.Label(header_frame, image=self.logo_photo, bg="#f0f0f0")
                logo_label.pack(side="left", padx=(0, 15))
            except Exception:
                pass
        
        title_label = ttk.Label(header_frame, text="UPRYT", style="Header.TLabel")
        title_label.pack(side="left")
        
        subtitle = ttk.Label(header_frame, text="Complete Posture & Wellness Analysis", font=("Segoe UI", 10))
        subtitle.pack(side="left", padx=(10, 0))
        
        separator = ttk.Separator(self.root, orient="horizontal")
        separator.pack(fill="x", padx=15, pady=5)
    
    def _create_mode_selection(self):
        mode_frame = ttk.LabelFrame(self.root, text="Select Mode", padding=15)
        mode_frame.pack(fill="x", padx=20, pady=10)
        
        # Default to train mode
        self.mode_var = tk.StringVar(value="train")
        
        modes = [
            ("combined", "Combined Mode (Recommended)", "Full system: Posture + Attention + Hands + Dashboard", "🔄"),
            ("realtime", "Real-Time Posture Only", "Monitor posture with RL feedback", "🧍"),
            ("hand", "Hand Tracking Only", "Track typing posture and hand position", "✋"),
            ("attention", "Attention Tracking Only", "Monitor face, gaze, and focus", "👀"),
            ("train", "Train RL Agent", "Train reinforcement learning model", "🧠"),
            ("compare", "Compare Algorithms", "Benchmark PPO vs DQN vs Rule-Based", "📊"),
        ]
        
        for mode, title, desc, icon in modes:
            rb = ttk.Radiobutton(mode_frame, text=f"{icon} {title}", variable=self.mode_var, 
                                value=mode, command=self._on_mode_changed)
            rb.pack(anchor="w", pady=(5, 0))
            
            desc_label = ttk.Label(mode_frame, text=f"     {desc}", font=("Segoe UI", 9))
            desc_label.pack(anchor="w")
    
    def _create_options(self):
        self.options_notebook = ttk.Notebook(self.root)
        self.options_notebook.pack(fill="x", padx=20, pady=10)
        
        self._create_realtime_tab()   # Tab 0
        self._create_training_tab()   # Tab 1
        self._create_features_tab()   # Tab 2
        
        # Initially select Training tab (safer default)
        self.options_notebook.select(1)
    
    def _create_realtime_tab(self):
        realtime_frame = ttk.Frame(self.options_notebook, padding=15)
        self.options_notebook.add(realtime_frame, text=" Real-Time Options ")
        
        row = 0
        
        algo_frame = ttk.Frame(realtime_frame)
        algo_frame.pack(fill="x", pady=5)
        ttk.Label(algo_frame, text="Algorithm:").pack(side="left")
        self.algorithm_var = tk.StringVar(value="auto")
        self.algorithm_combo = ttk.Combobox(algo_frame, textvariable=self.algorithm_var,
                                            values=["ppo", "dqn", "rule", "auto"], state="readonly", width=12)
        self.algorithm_combo.pack(side="left", padx=(10, 20))
        
        camera_frame = ttk.Frame(realtime_frame)
        camera_frame.pack(fill="x", pady=5)
        ttk.Label(camera_frame, text="Camera:").pack(side="left")
        self.camera_var = tk.StringVar(value="0")
        self.camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_var,
                                         values=["0", "1", "2"], state="readonly", width=5)
        self.camera_combo.pack(side="left", padx=(10, 0))
        
        self.audio_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(realtime_frame, text="Audio Alerts", 
                       variable=self.audio_var).pack(anchor="w", pady=2)
        
        self.skip_calibration_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(realtime_frame, text="Skip Calibration (use defaults)", 
                       variable=self.skip_calibration_var).pack(anchor="w", pady=2)
    
    def _create_training_tab(self):
        training_frame = ttk.Frame(self.options_notebook, padding=15)
        self.options_notebook.add(training_frame, text=" Training Options ")
        
        algo_frame = ttk.Frame(training_frame)
        algo_frame.pack(fill="x", pady=5)
        ttk.Label(algo_frame, text="Algorithm:").pack(side="left")
        self.train_algorithm_var = tk.StringVar(value="ppo")
        self.train_algorithm_combo = ttk.Combobox(algo_frame, textvariable=self.train_algorithm_var,
                                                  values=["ppo", "dqn"], state="readonly", width=12)
        self.train_algorithm_combo.pack(side="left", padx=(10, 20))
        
        episodes_frame = ttk.Frame(training_frame)
        episodes_frame.pack(fill="x", pady=5)
        ttk.Label(episodes_frame, text="Episodes:").pack(side="left")
        self.episodes_var = tk.StringVar(value="500")
        episodes_spin = ttk.Spinbox(episodes_frame, from_=10, to=10000,
                                    textvariable=self.episodes_var, width=8)
        episodes_spin.pack(side="left", padx=(10, 20))
        
        users_frame = ttk.Frame(training_frame)
        users_frame.pack(fill="x", pady=5)
        ttk.Label(users_frame, text="Simulated Users:").pack(side="left")
        self.users_var = tk.StringVar(value="5")
        users_spin = ttk.Spinbox(users_frame, from_=1, to=20,
                                  textvariable=self.users_var, width=8)
        users_spin.pack(side="left", padx=(10, 0))
        
        self.enhanced_training_var = tk.BooleanVar(value=True)
        enhanced_check = ttk.Checkbutton(training_frame, text="Enhanced Training (curriculum + domain randomization)", 
                                        variable=self.enhanced_training_var)
        enhanced_check.pack(anchor="w", pady=5)
        
        ttk.Label(training_frame, text="Enhanced training uses curriculum learning and\ndomain randomization for more robust agents.",
                 font=("Segoe UI", 8), foreground="gray").pack(anchor="w")
    
    def _create_features_tab(self):
        features_frame = ttk.Frame(self.options_notebook, padding=15)
        self.options_notebook.add(features_frame, text=" Additional Features ")
        
        tracking_frame = ttk.LabelFrame(features_frame, text="Tracking Features", padding=10)
        tracking_frame.pack(fill="x", pady=10)
        
        self.attention_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tracking_frame, text="Attention Tracking (face/gaze)", 
                       variable=self.attention_var).pack(anchor="w", pady=2)
        
        self.hands_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tracking_frame, text="Hand Tracking (typing posture)", 
                       variable=self.hands_var).pack(anchor="w", pady=2)
        
        self.dashboard_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tracking_frame, text="Dashboard Export (session data)", 
                       variable=self.dashboard_var).pack(anchor="w", pady=2)
        
        learning_frame = ttk.LabelFrame(features_frame, text="Learning Features", padding=10)
        learning_frame.pack(fill="x", pady=10)
        
        self.online_learning_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(learning_frame, text="Online Learning (learn from sessions)", 
                       variable=self.online_learning_var).pack(anchor="w", pady=2)
        
        self.multi_camera_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(learning_frame, text="Multi-Camera Support", 
                       variable=self.multi_camera_var).pack(anchor="w", pady=2)
        
        info_text = ("Combined Mode includes all tracking features by default.\n"
                    "Use individual modes to run specific trackers standalone.")
        ttk.Label(features_frame, text=info_text, font=("Segoe UI", 8), 
                 foreground="gray").pack(pady=10)
    
    def _on_mode_changed(self):
        mode = self.mode_var.get()
        
        if mode == "realtime":
            self.algorithm_combo["values"] = ["ppo", "dqn", "rule", "auto"]
            self.algorithm_var.set("auto")
            self.options_notebook.tab(0, state="normal")
            self.options_notebook.tab(1, state="disabled")
            self.options_notebook.tab(2, state="normal")
            self.options_notebook.select(0)
        elif mode == "combined":
            self.algorithm_combo["values"] = ["ppo", "dqn", "rule", "auto"]
            self.algorithm_var.set("auto")
            self.options_notebook.tab(0, state="normal")
            self.options_notebook.tab(1, state="disabled")
            self.options_notebook.tab(2, state="normal")
            self.options_notebook.select(0)
        elif mode == "hand":
            self.algorithm_combo["values"] = []
            self.options_notebook.tab(0, state="disabled")
            self.options_notebook.tab(1, state="disabled")
            self.options_notebook.tab(2, state="disabled")
        elif mode == "attention":
            self.algorithm_combo["values"] = []
            self.options_notebook.tab(0, state="disabled")
            self.options_notebook.tab(1, state="disabled")
            self.options_notebook.tab(2, state="disabled")
        elif mode == "train":
            self.algorithm_combo["values"] = []
            self.options_notebook.tab(0, state="disabled")
            self.options_notebook.tab(1, state="normal")
            self.options_notebook.tab(2, state="disabled")
            self.options_notebook.select(1)
            self._log("NOTE: Training runs in background (no camera window)", "info")
        elif mode == "compare":
            self.algorithm_combo["values"] = []
            self.options_notebook.tab(0, state="disabled")
            self.options_notebook.tab(1, state="disabled")
            self.options_notebook.tab(2, state="disabled")
        
        self._log(f"Mode changed to: {mode}", "info")
    
    def _create_control_buttons(self):
        button_frame = ttk.Frame(self.root, padding=15)
        button_frame.pack(fill="x")
        
        self.start_button = ttk.Button(button_frame, text="▶ Start", style="Action.TButton",
                                       command=self._start_operation)
        self.start_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="■ Stop", style="Action.TButton",
                                      command=self._stop_operation, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        
        self.clear_button = ttk.Button(button_frame, text="Clear Log",
                                       command=self._clear_log)
        self.clear_button.pack(side="right", padx=5)
    
    def _create_log_area(self):
        log_frame = ttk.LabelFrame(self.root, text="Output Log", padding=10)
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap="word", height=10,
                                                   font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)
        
        self.log_text.tag_config("info", foreground="blue")
        self.log_text.tag_config("success", foreground="green")
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("warning", foreground="orange")
        self.log_text.tag_config("header", foreground="purple", font=("Consolas", 9, "bold"))
    
    def _create_footer(self):
        footer_frame = ttk.Frame(self.root, padding=10)
        footer_frame.pack(fill="x")
        
        self.status_label = ttk.Label(footer_frame, text="Ready", style="Status.TLabel")
        self.status_label.pack(side="left")
        
        self.model_status_label = ttk.Label(footer_frame, text="", style="Status.TLabel")
        self.model_status_label.pack(side="right")
    
    def _check_models(self):
        model_dir = os.path.join(PROJECT_DIR, "models")
        ppo_exists = os.path.exists(os.path.join(model_dir, "ppo_final.pth"))
        dqn_exists = os.path.exists(os.path.join(model_dir, "dqn_final.pth"))
        
        if ppo_exists and dqn_exists:
            status = "✓ PPO and DQN models loaded"
            self.model_status_label.config(text=status, foreground="green")
        elif ppo_exists:
            status = "⚠ PPO model found (train DQN for comparison)"
            self.model_status_label.config(text=status, foreground="orange")
        elif dqn_exists:
            status = "⚠ DQN model found (train PPO for comparison)"
            self.model_status_label.config(text=status, foreground="orange")
        else:
            status = "✗ No trained models - run Training first"
            self.model_status_label.config(text=status, foreground="red")
    
    def _log(self, message, tag=None):
        self.log_text.insert("end", message + "\n", tag)
        self.log_text.see("end")
        self.root.update_idletasks()
    
    def _clear_log(self):
        self.log_text.delete(1.0, "end")
    
    def _update_status(self, message, color="black"):
        self.status_label.config(text=message, foreground=color)
    
    def _start_operation(self):
        mode = self.mode_var.get()
        
        self.is_running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        cmd = [sys.executable, os.path.join(PROJECT_DIR, "main.py"), f"--mode={mode}"]
        
        if mode == "realtime" or mode == "combined":
            cmd.append(f"--algorithm={self.algorithm_var.get()}")
            cmd.append(f"--camera={self.camera_var.get()}")
            
            if self.audio_var.get():
                cmd.append("--audio")
            if self.skip_calibration_var.get():
                cmd.append("--skip-calibration")
            if self.attention_var.get() and mode == "realtime":
                cmd.append("--attention")
            if self.hands_var.get() and mode == "realtime":
                cmd.append("--hands")
            if self.dashboard_var.get() and mode == "realtime":
                cmd.append("--dashboard")
            if self.online_learning_var.get():
                cmd.append("--online-learning")
            if self.multi_camera_var.get():
                cmd.append("--multi-camera")
            
            flags = []
            if self.audio_var.get(): flags.append("Audio")
            if self.skip_calibration_var.get(): flags.append("Skip Cal")
            if self.attention_var.get(): flags.append("Attention")
            if self.hands_var.get(): flags.append("Hands")
            if self.dashboard_var.get(): flags.append("Dashboard")
            if self.online_learning_var.get(): flags.append("Online Learning")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            self._log(f"{'='*60}", "header")
            self._log(f"Starting {mode.upper()} Mode ({self.algorithm_var.get().upper()}){flag_str}...", "info")
            self._log(f"{'='*60}", "header")
        elif mode == "train":
            cmd.append(f"--algorithm={self.train_algorithm_var.get()}")
            cmd.append(f"--episodes={self.episodes_var.get()}")
            cmd.append(f"--users={self.users_var.get()}")
            if self.enhanced_training_var.get():
                cmd.append("--enhanced-training")
            self._log(f"{'='*60}", "header")
            self._log(f"Training {self.train_algorithm_var.get().upper()} agent for {self.episodes_var.get()} episodes...", "info")
            if self.enhanced_training_var.get():
                self._log("Using Enhanced Training (curriculum + domain randomization)", "info")
            self._log(f"{'='*60}", "header")
        elif mode == "compare":
            self._log(f"{'='*60}", "header")
            self._log("Running Algorithm Comparison...", "info")
            self._log(f"{'='*60}", "header")
        elif mode == "hand":
            cmd = [sys.executable, os.path.join(PROJECT_DIR, "main.py"), "--mode=hand"]
            cmd.append(f"--camera={self.camera_var.get()}")
            self._log(f"{'='*60}", "header")
            self._log("Starting Hand Tracking Mode...", "info")
            self._log(f"{'='*60}", "header")
        elif mode == "attention":
            cmd = [sys.executable, os.path.join(PROJECT_DIR, "main.py"), "--mode=attention"]
            cmd.append(f"--camera={self.camera_var.get()}")
            self._log(f"{'='*60}", "header")
            self._log("Starting Attention Tracking Mode...", "info")
            self._log(f"{'='*60}", "header")
        
        self._update_status("Running...", "blue")
        
        thread = threading.Thread(target=self._run_process, args=(cmd,))
        thread.daemon = True
        thread.start()
    
    def _run_process(self, cmd):
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=PROJECT_DIR
            )
            
            for line in self.process.stdout:
                if not self.is_running:
                    break
                line = line.rstrip()
                if "CALIBRATION" in line or "SESSION" in line or "="*30 in line:
                    self._log(line, "header")
                elif "ERROR" in line or "error" in line:
                    self._log(line, "error")
                elif "WARNING" in line or "warning" in line:
                    self._log(line, "warning")
                elif "complete" in line.lower() or "success" in line.lower():
                    self._log(line, "success")
                else:
                    self._log(line)
            
            self.process.wait()
            
        except Exception as e:
            self._log(f"Error: {str(e)}", "error")
        
        finally:
            self.root.after(0, self._process_finished)
    
    def _process_finished(self):
        self.is_running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        
        if self.process and self.process.returncode == 0:
            self._log("\n" + "="*60, "header")
            self._log("Operation completed successfully!", "success")
            self._log("="*60, "header")
            self._update_status("Completed", "green")
            self._check_models()
        elif self.process and self.process.returncode is not None:
            self._log("\nOperation stopped or failed.", "warning")
            self._update_status("Stopped", "orange")
        else:
            self._log("\nProcess terminated.", "warning")
            self._update_status("Terminated", "orange")
    
    def _stop_operation(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except:
                self.process.kill()
        self.is_running = False
        self._log("Process stopped by user.", "warning")


def main():
    root = tk.Tk()
    app = UPRYTApplication(root)
    root.mainloop()


if __name__ == "__main__":
    main()
