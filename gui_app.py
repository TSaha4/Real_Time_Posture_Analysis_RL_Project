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
        self.root.title("UPRYT - Posture Analysis System")
        self.root.geometry("800x650")
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
        
        self._check_models()
    
    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"))
        self.style.configure("Section.TLabel", font=("Segoe UI", 12, "bold"))
        self.style.configure("Action.TButton", font=("Segoe UI", 11), padding=10)
        self.style.configure("Status.TLabel", font=("Segoe UI", 10))
    
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
        
        subtitle = ttk.Label(header_frame, text="Real-Time Posture Analysis with RL", font=("Segoe UI", 10))
        subtitle.pack(side="left", padx=(10, 0))
        
        separator = ttk.Separator(self.root, orient="horizontal")
        separator.pack(fill="x", padx=15, pady=5)
    
    def _create_mode_selection(self):
        mode_frame = ttk.LabelFrame(self.root, text="Select Mode", padding=15)
        mode_frame.pack(fill="x", padx=20, pady=10)
        
        self.mode_var = tk.StringVar(value="realtime")
        
        modes = [
            ("realtime", "Real-Time Monitoring", "Monitor your posture using webcam"),
            ("train", "Train RL Agent", "Train the reinforcement learning model"),
            ("compare", "Compare Algorithms", "Compare PPO vs DQN vs Rule-Based"),
        ]
        
        for mode, title, desc in modes:
            rb = ttk.Radiobutton(mode_frame, text=title, variable=self.mode_var, 
                                value=mode, command=self._on_mode_changed)
            rb.pack(anchor="w", pady=(5, 0))
            
            desc_label = ttk.Label(mode_frame, text=f"   {desc}", font=("Segoe UI", 9))
            desc_label.pack(anchor="w")
    
    def _create_options(self):
        self.options_frame = ttk.LabelFrame(self.root, text="Options", padding=15)
        self.options_frame.pack(fill="x", padx=20, pady=10)
        
        self.algorithm_label = ttk.Label(self.options_frame, text="Algorithm:")
        self.algorithm_label.pack(side="left")
        
        self.algorithm_var = tk.StringVar(value="ppo")
        self.algorithm_combo = ttk.Combobox(self.options_frame, textvariable=self.algorithm_var,
                                            values=["ppo", "dqn", "rule"], state="readonly", width=10)
        self.algorithm_combo.pack(side="left", padx=(10, 30))
        
        self.episodes_label = ttk.Label(self.options_frame, text="Episodes:")
        self.episodes_label.pack(side="left")
        
        self.episodes_var = tk.StringVar(value="500")
        self.episodes_spin = ttk.Spinbox(self.options_frame, from_=10, to=10000,
                                          textvariable=self.episodes_var, width=10)
        self.episodes_spin.pack(side="left", padx=(10, 0))
        
        self.extra_options_frame = ttk.Frame(self.root, padding=(20, 0, 20, 10))
        self.extra_options_frame.pack(fill="x")
        
        self.audio_var = tk.BooleanVar(value=False)
        self.audio_check = ttk.Checkbutton(self.extra_options_frame, text="Audio Alerts", 
                                           variable=self.audio_var)
        self.audio_check.pack(side="left", padx=10)
        
        self.multi_camera_var = tk.BooleanVar(value=False)
        self.multi_camera_check = ttk.Checkbutton(self.extra_options_frame, text="Multi-Camera",
                                                   variable=self.multi_camera_var)
        self.multi_camera_check.pack(side="left", padx=10)
        
        self.online_learning_var = tk.BooleanVar(value=False)
        self.online_learning_check = ttk.Checkbutton(self.extra_options_frame, text="Online Learning",
                                                      variable=self.online_learning_var)
        self.online_learning_check.pack(side="left", padx=10)
        
        self._on_mode_changed()
    
    def _on_mode_changed(self):
        mode = self.mode_var.get()
        
        if mode == "realtime":
            self.algorithm_combo["values"] = ["ppo", "dqn", "rule"]
            self.algorithm_var.set("ppo")
            self.algorithm_label.config(state="normal")
            self.algorithm_combo.config(state="readonly")
            self.episodes_label.config(state="disabled")
            self.episodes_spin.config(state="disabled")
            self.audio_check.config(state="normal")
            self.multi_camera_check.config(state="normal")
            self.online_learning_check.config(state="normal")
        elif mode == "train":
            self.algorithm_combo["values"] = ["ppo", "dqn"]
            self.algorithm_var.set("ppo")
            self.algorithm_label.config(state="normal")
            self.algorithm_combo.config(state="readonly")
            self.episodes_label.config(state="normal")
            self.episodes_spin.config(state="normal")
            self.audio_check.config(state="disabled")
            self.multi_camera_check.config(state="disabled")
            self.online_learning_check.config(state="disabled")
        elif mode == "compare":
            self.algorithm_label.config(state="disabled")
            self.algorithm_combo.config(state="disabled")
            self.episodes_label.config(state="disabled")
            self.episodes_spin.config(state="disabled")
            self.audio_check.config(state="disabled")
            self.multi_camera_check.config(state="disabled")
            self.online_learning_check.config(state="disabled")
    
    def _create_control_buttons(self):
        button_frame = ttk.Frame(self.root, padding=15)
        button_frame.pack(fill="x")
        
        self.start_button = ttk.Button(button_frame, text="Start", style="Action.TButton",
                                       command=self._start_operation)
        self.start_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", style="Action.TButton",
                                      command=self._stop_operation, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        
        self.clear_button = ttk.Button(button_frame, text="Clear Log",
                                       command=self._clear_log)
        self.clear_button.pack(side="right", padx=5)
    
    def _create_log_area(self):
        log_frame = ttk.LabelFrame(self.root, text="Output Log", padding=10)
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap="word", height=12,
                                                   font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)
        
        self.log_text.tag_config("info", foreground="blue")
        self.log_text.tag_config("success", foreground="green")
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("warning", foreground="orange")
    
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
            status = "PPO and DQN models loaded"
            self.model_status_label.config(text=status, foreground="green")
        elif ppo_exists:
            status = "PPO model found (train DQN for comparison)"
            self.model_status_label.config(text=status, foreground="orange")
        elif dqn_exists:
            status = "DQN model found (train PPO for comparison)"
            self.model_status_label.config(text=status, foreground="orange")
        else:
            status = "No trained models - run Training first"
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
        algorithm = self.algorithm_var.get()
        episodes = self.episodes_var.get()
        enable_audio = self.audio_var.get()
        enable_multicamera = self.multi_camera_var.get()
        enable_online_learning = self.online_learning_var.get()
        
        self.is_running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        cmd = [sys.executable, os.path.join(PROJECT_DIR, "main.py"), f"--mode={mode}"]
        
        if mode == "realtime":
            cmd.append(f"--algorithm={algorithm}")
            if enable_audio:
                cmd.append("--audio")
            if enable_multicamera:
                cmd.append("--multi-camera")
            if enable_online_learning:
                cmd.append("--online-learning")
            flags = []
            if enable_audio: flags.append("Audio")
            if enable_multicamera: flags.append("Multi-cam")
            if enable_online_learning: flags.append("Online Learning")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            self._log(f"Starting Real-Time Monitoring ({algorithm.upper()}){flag_str}...", "info")
        elif mode == "train":
            cmd.append(f"--algorithm={algorithm}")
            cmd.append(f"--episodes={episodes}")
            self._log(f"Training {algorithm.upper()} agent for {episodes} episodes...", "info")
        elif mode == "compare":
            self._log("Running Algorithm Comparison...", "info")
        
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
                self._log(line.rstrip())
            
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
            self._log("\nOperation completed successfully!", "success")
            self._update_status("Completed", "green")
            self._check_models()
        else:
            self._log("\nOperation stopped or failed.", "warning")
            self._update_status("Stopped", "orange")
    
    def _stop_operation(self):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.is_running = False


def main():
    root = tk.Tk()
    app = UPRYTApplication(root)
    root.mainloop()


if __name__ == "__main__":
    main()
