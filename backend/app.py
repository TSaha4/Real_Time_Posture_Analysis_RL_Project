import atexit
import json
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from config import config
from posture_module import PostureLabel, encode_label
from model.camera_rl_utils import (
    build_state,
    detect_face_info,
    is_bad_state,
    is_face_in_circle,
    reward_for_transition,
)

QUESTION_COUNT = 3
FREEZE_SECONDS = 1.5
FRAME_SLEEP = 0.05
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
CALIBRATION_BUFFER_SIZE = 15

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTION_PATH = PROJECT_ROOT / "frontend" / "src" / "data" / "hrQuestions.json"


class InterviewEngine:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = True
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_available = False
        self.latest_jpeg = self._encode_frame(self._placeholder_frame("Starting camera service..."))
        self.questions_bank = self._load_questions()
        self.agent_runtime_error: Optional[str] = None
        self.rl_agents = self._load_rl_agents()
        self.algorithm = self._select_initial_algorithm()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.reset_session_state(keep_calibration=False)
        self.thread.start()

    def _load_questions(self) -> List[Dict[str, Any]]:
        if QUESTION_PATH.exists():
            return json.loads(QUESTION_PATH.read_text(encoding="utf-8"))
        return [
            {"id": 1, "category": "Introduction", "question": "Tell me about yourself."},
            {"id": 2, "category": "Motivation", "question": "Why do you want this role?"},
            {"id": 3, "category": "Behavioral", "question": "Describe a difficult challenge you solved."},
            {"id": 4, "category": "Teamwork", "question": "How do you handle conflict in a team?"},
            {"id": 5, "category": "Growth", "question": "What skill are you improving right now?"},
        ]

    def _load_rl_agents(self) -> Dict[str, Any]:
        agents: Dict[str, Any] = {}
        try:
            from rl_agent import DQNAgent
            from rl_ppo_agent import PPOPPOAgent
        except (ModuleNotFoundError, OSError, ImportError) as exc:
            self.agent_runtime_error = f"RL agents unavailable: {exc}"
            return agents

        ppo_path = PROJECT_ROOT / config.system.model_dir / "ppo_final.pth"
        if ppo_path.exists():
            ppo_agent = PPOPPOAgent(state_size=config.rl.state_size, action_size=config.rl.action_size)
            if ppo_agent.load(str(ppo_path)):
                agents["ppo"] = ppo_agent

        dqn_path = PROJECT_ROOT / config.system.model_dir / "dqn_final.pth"
        if dqn_path.exists():
            dqn_agent = DQNAgent(state_size=config.rl.state_size, action_size=config.rl.action_size)
            if dqn_agent.load(str(dqn_path)):
                agents["dqn"] = dqn_agent

        if not agents and self.agent_runtime_error is None:
            self.agent_runtime_error = "No trained PPO/DQN model could be loaded."
        return agents

    def _select_initial_algorithm(self) -> str:
        if "ppo" in self.rl_agents:
            return "ppo"
        if "dqn" in self.rl_agents:
            return "dqn"
        return "heuristic"

    def reset_session_state(self, keep_calibration: bool = False) -> None:
        with getattr(self, "lock", threading.Lock()):
            if keep_calibration and getattr(self, "baseline", None):
                baseline = self.baseline
                snapshot = self.calibration_snapshot
            else:
                baseline = None
                snapshot = None

            self.mode = "posture" if baseline else "idle"
            self.baseline = baseline
            self.calibration_snapshot = snapshot
            self.calibration_frozen = False
            self.calibration_freeze_until = 0.0
            self.calibration_ready = False
            self.calibration_buffer = deque(maxlen=CALIBRATION_BUFFER_SIZE)

            self.face_inside_ratio = 0.0
            self.face_width = 0.0
            self.face_height = 0.0
            self.head_angle = 0.0
            self.eye_dir = 0.0
            self.eye_ratio = 0.0
            self.eye_distance = 0.0

            self.state: Optional[tuple[str, str, str]] = None
            self.action = "no_feedback"
            self.reward = 0.0
            self.is_bad = False
            self.is_cheating = False
            self.trust_score = 100.0 if baseline else 0.0
            self.identified_by = f"Haar face + {self.algorithm.upper() if self.algorithm != 'heuristic' else 'heuristic'} policy"
            self.suggestion = "Open calibration to create a posture baseline."

            self.last_state: Optional[tuple[str, str, str]] = None
            self.latest_info: Optional[Dict[str, Any]] = None
            self.current_label = PostureLabel.UNKNOWN
            self.current_score = 0.0
            self._rl_consecutive_alerts = 0
            self._rl_alerts_this_episode = 0
            self._rl_corrections_this_episode = 0
            self._rl_correction_streak = 0
            self._rl_max_streak = 0
            self._rl_user_fatigue = 0.0
            self._rl_motivation = 0.7
            self._rl_frustration = 0.0
            self.last_alert_action = 0
            self.last_alert_time = 0.0
            self.last_score_before_alert = 0.0

            self.exam_started = False
            self.questions: List[Dict[str, Any]] = []
            self.current_question_index = 0
            self.question_results: List[Optional[Dict[str, Any]]] = [None] * QUESTION_COUNT
            self.answering = False
            self.answer_started_at: Optional[float] = None
            self.answer_samples: List[Dict[str, Any]] = []

    def _open_camera(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            return
        self.cap = cv2.VideoCapture(0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.camera_available = True
        else:
            self.camera_available = False

    def _capture_loop(self) -> None:
        while self.running:
            self._open_camera()
            if not self.camera_available or self.cap is None:
                frame = self._placeholder_frame("Webcam not available. Connect a camera to enable posture monitoring.")
                with self.lock:
                    self.latest_jpeg = self._encode_frame(frame)
                    self.state = None
                    self.is_bad = True
                    self.is_cheating = True
                    self.suggestion = "Camera unavailable. Connect a webcam, then recalibrate."
                time.sleep(0.2)
                continue

            ok, frame = self.cap.read()
            if not ok:
                self.camera_available = False
                time.sleep(0.1)
                continue

            processed = self._process_frame(frame)
            with self.lock:
                self.latest_jpeg = self._encode_frame(processed)
            time.sleep(FRAME_SLEEP)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        info = detect_face_info(gray)
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 4

        with self.lock:
            self.latest_info = info
            self.face_inside_ratio = 0.0
            if info is not None:
                self.face_width = float(info.get("face_w", 0.0))
                self.face_height = float(info.get("face_h", 0.0))
                self.head_angle = float(info.get("head_angle", 0.0))
                self.eye_dir = float(info.get("eye_dir", 0.0))
                self.eye_ratio = float(info.get("eye_ratio") or 0.0)
                self.eye_distance = float(info.get("eye_dist", 0.0))
                _, self.face_inside_ratio = is_face_in_circle(info, center, radius)
            else:
                self.face_width = 0.0
                self.face_height = 0.0
                self.head_angle = 0.0
                self.eye_dir = 0.0
                self.eye_ratio = 0.0
                self.eye_distance = 0.0

            if self.mode in {"calibrating", "calibration_freeze"}:
                self._update_calibration_state(info, center, radius)
            elif self.baseline is not None:
                self._update_posture_state(info)
            else:
                self.state = None
                self.action = "no_action"
                self.reward = 0.0
                self.is_bad = False
                self.is_cheating = False

            suggestion = self.suggestion
            trust_score = self.trust_score
            state = self.state
            action = self.action
            reward = self.reward
            mode = self.mode
            frozen = self.calibration_frozen
            freeze_remaining = max(0.0, self.calibration_freeze_until - time.time())

        annotated = frame.copy()
        if mode in {"calibrating", "calibration_freeze"}:
            circle_color = (0, 255, 0) if self.calibration_ready else (0, 165, 255)
            cv2.circle(annotated, center, radius, circle_color, 2)
            cv2.putText(annotated, f"Calibration area: {int(self.face_inside_ratio * 100)}%", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, circle_color, 2)
            cv2.putText(annotated, "Align face inside circle and keep still", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            if frozen:
                cv2.putText(annotated, f"Saving reference... {freeze_remaining:.1f}s", (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 120), 2)
        else:
            color = (50, 205, 50) if not self.is_bad else (80, 80, 255)
            cv2.putText(annotated, f"State: {state or 'no_face'}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(annotated, f"Action: {action}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(annotated, f"Reward: {reward:.1f}", (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(annotated, f"Trust: {trust_score:.0f}%", (20, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(annotated, suggestion[:60], (20, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return annotated

    def _update_calibration_state(self, info: Optional[Dict[str, Any]], center: tuple[int, int], radius: int) -> None:
        now = time.time()
        ready = False
        if info is not None:
            ready, ratio = is_face_in_circle(info, center, radius)
            self.face_inside_ratio = ratio
            if ready:
                self.calibration_buffer.append(self._snapshot_info(info))
            else:
                self.calibration_buffer.clear()
        else:
            self.calibration_buffer.clear()
        self.calibration_ready = bool(self._to_python_value(ready))

        if self.calibration_frozen:
            self.mode = "calibration_freeze"
            if now >= self.calibration_freeze_until:
                self.calibration_frozen = False
                self.mode = "posture"
                self.trust_score = 100.0
                self.suggestion = "Calibration complete. You can now start the interview."
            else:
                self.suggestion = "Holding still to save your calibration reference."
            return

        if info is None:
            self.suggestion = "Face not detected. Center your face inside the oval."
        elif ready:
            self.suggestion = "Reference ready. Capture calibration to lock your baseline."
        else:
            self.suggestion = "Move closer to the center until most of your face is inside the oval."

    def _update_posture_state(self, info: Optional[Dict[str, Any]]) -> None:
        current_state = build_state(info, self.baseline) if self.baseline else None
        posture_label, posture_score = self._derive_posture_metrics(current_state)
        action = self._decide_action(current_state, posture_label, posture_score)
        reward = reward_for_transition(self.last_state, current_state) if self.last_state is not None else 0.0
        self.last_state = current_state

        self.state = current_state
        self.current_label = posture_label
        self.current_score = posture_score
        self.action = action
        self.reward = float(reward)
        self.is_bad = is_bad_state(current_state)
        self.is_cheating = current_state is None or current_state[2] == "away"
        self.suggestion = self._suggestion_for(current_state, action)
        self._update_trust()

        if self.answering:
            self.answer_samples.append(
                {
                    "ts": time.time(),
                    "is_bad": self.is_bad,
                    "is_cheating": self.is_cheating,
                    "reward": self.reward,
                    "score": self.current_score,
                }
            )

    def _update_trust(self) -> None:
        if self.mode == "posture":
            target = 100.0 if not self.is_bad else 92.0
        elif self.mode == "exam":
            target = 96.0 if not self.is_bad else 84.0
        elif self.mode == "exam_question":
            penalty = 0.8 if self.is_bad else -0.15
            if self.is_cheating:
                penalty += 1.2
            self.trust_score = max(0.0, min(100.0, self.trust_score - penalty))
            return
        else:
            return
        self.trust_score = max(0.0, min(100.0, (self.trust_score * 0.7) + (target * 0.3)))

    def _derive_posture_metrics(self, state: Optional[tuple[str, str, str]]) -> tuple[PostureLabel, float]:
        if state is None:
            return PostureLabel.UNKNOWN, 0.0

        position, head, gaze = state
        penalties = 0.0
        if position != "centered":
            penalties += 0.35
        if head != "straight":
            penalties += 0.35
        if gaze != "looking":
            penalties += 0.30

        score = max(0.0, min(1.0, 1.0 - penalties))
        if gaze != "looking":
            label = PostureLabel.FORWARD_HEAD
        elif position != "centered":
            label = PostureLabel.LEANING
        elif head != "straight":
            label = PostureLabel.SLOUCHING
        else:
            label = PostureLabel.GOOD
        return label, score

    def _build_agent_state(self, posture_label: PostureLabel, posture_score: float) -> np.ndarray:
        if self.last_alert_action > 0:
            self._rl_consecutive_alerts += 1
            self._rl_alerts_this_episode += 1
        else:
            self._rl_consecutive_alerts = 0

        if self.last_alert_time > 0:
            if posture_label == PostureLabel.GOOD or posture_score > self.last_score_before_alert + 0.15:
                self._rl_corrections_this_episode += 1
                self._rl_correction_streak += 1
                self._rl_max_streak = max(self._rl_max_streak, self._rl_correction_streak)
            else:
                self._rl_correction_streak = 0

        self._rl_user_fatigue = min(1.0, self._rl_alerts_this_episode / 20.0)
        correction_rate = (
            self._rl_corrections_this_episode / self._rl_alerts_this_episode
            if self._rl_alerts_this_episode > 0 else 0.0
        )

        is_good = 1.0 if posture_label == PostureLabel.GOOD else 0.0
        badness = 1.0 - posture_score

        return np.array([
            float(encode_label(posture_label)),
            posture_score,
            self._rl_consecutive_alerts / 5.0,
            self._rl_user_fatigue,
            self._rl_motivation,
            self._rl_frustration,
            is_good,
            badness,
            self._rl_alerts_this_episode / 20.0,
            correction_rate,
            self._rl_correction_streak / 10.0,
            self._rl_max_streak / 20.0,
            0.7,
            0.8,
            0.3,
            0.0,
            0.0,
            0.5,
        ], dtype=np.float32)

    def _decide_action(self, state: Optional[tuple[str, str, str]], posture_label: PostureLabel, posture_score: float) -> str:
        agent = self.rl_agents.get(self.algorithm)
        if agent is None:
            return self._heuristic_action_name(state)

        action_state = self._build_agent_state(posture_label, posture_score)
        action_idx = int(agent.get_action(action_state, training=False))
        action_name = agent.get_action_name(action_idx)

        self.last_alert_action = action_idx
        if action_idx > 0:
            self.last_alert_time = time.time()
            self.last_score_before_alert = posture_score

        if self._rl_alerts_this_episode > 15:
            self._rl_alerts_this_episode = 0
            self._rl_corrections_this_episode = 0
            self._rl_max_streak = 0

        return action_name

    def _heuristic_action_name(self, state: Optional[tuple[str, str, str]]) -> str:
        if state is None:
            return "strong_alert"
        position, head, gaze = state
        if gaze != "looking":
            return "strong_alert"
        if head != "straight" or position != "centered":
            return "subtle_alert"
        return "no_feedback"

    def _suggestion_for(self, state: Optional[tuple[str, str, str]], action: str) -> str:
        if state is None:
            return "Face not visible. Return to the frame and look at the screen."
        position, head, gaze = state
        prefix = ""
        if action == "strong_alert":
            prefix = "Immediate correction needed. "
        elif action == "subtle_alert":
            prefix = "Small posture correction recommended. "

        if gaze != "looking":
            return prefix + "Look at the screen and maintain eye contact with the interviewer."
        if head != "straight":
            return prefix + "Straighten your head and keep your chin level."
        if position != "centered":
            return prefix + "Re-center your face and reduce side-to-side movement."
        if action == "no_feedback":
            return "Posture looks good. Keep answering naturally."
        return "Posture is stable. Maintain the same position."

    def _placeholder_frame(self, message: str) -> np.ndarray:
        frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (CAMERA_WIDTH, CAMERA_HEIGHT), (18, 24, 35), -1)
        cv2.putText(frame, "UPRYT Interview Monitor", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (88, 166, 255), 3)
        cv2.putText(frame, message, (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
        cv2.putText(frame, "Backend is running but a webcam frame is not available.", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return frame

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        ok, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes() if ok else b""

    def _to_python_value(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: self._to_python_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_python_value(item) for item in value]
        return value

    def _safe_metric_dict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {key: self._to_python_value(value) for key, value in payload.items()}

    def _snapshot_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        return self._safe_metric_dict({
            "face_x": info.get("face_x", 0.0),
            "face_y": info.get("face_y", 0.0),
            "head_angle": info.get("head_angle", 0.0),
            "eye_dir": info.get("eye_dir", 0.0),
            "eye_ratio": info.get("eye_ratio") or 0.0,
            "face_w": info.get("face_w", 0.0),
            "face_h": info.get("face_h", 0.0),
            "eye_dist": info.get("eye_dist") or 0.0,
        })

    def _average_calibration_buffer(self) -> Optional[Dict[str, Any]]:
        if not self.calibration_buffer:
            return None

        keys = ["face_x", "face_y", "head_angle", "eye_dir", "eye_ratio", "face_w", "face_h", "eye_dist"]
        averaged: Dict[str, Any] = {}
        for key in keys:
            values = [float(item.get(key, 0.0)) for item in self.calibration_buffer]
            averaged[key] = sum(values) / max(1, len(values))
        return self._safe_metric_dict(averaged)

    def set_algorithm(self, algorithm: str) -> Dict[str, Any]:
        algo = algorithm.lower()
        if algo not in {"ppo", "dqn"}:
            raise HTTPException(status_code=400, detail="Algorithm must be either 'ppo' or 'dqn'.")
        if algo not in self.rl_agents:
            raise HTTPException(status_code=400, detail=f"{algo.upper()} model is not available in the models directory.")

        with self.lock:
            self.algorithm = algo
            self.identified_by = f"Haar face + {self.algorithm.upper()} policy"
            self.last_alert_action = 0
            self.last_alert_time = 0.0
            self.last_score_before_alert = 0.0
        return {"algorithm": self.algorithm, "available_algorithms": sorted(self.rl_agents.keys())}

    def capture_reference(self) -> Dict[str, Any]:
        with self.lock:
            if self.mode not in {"calibrating", "calibration_freeze"}:
                raise HTTPException(status_code=400, detail="Open calibration before capturing a reference.")
            if self.latest_info is None or not self.calibration_ready:
                raise HTTPException(status_code=400, detail="Align your face inside the calibration oval first.")
            averaged_info = self._average_calibration_buffer()
            if averaged_info is None:
                raise HTTPException(status_code=400, detail="Hold steady inside the oval for a moment before capturing.")

            self.baseline = self._safe_metric_dict({
                "face_x": averaged_info["face_x"],
                "face_y": averaged_info["face_y"],
                "head_angle": averaged_info["head_angle"],
                "eye_dir": averaged_info["eye_dir"],
                "eye_ratio": averaged_info["eye_ratio"],
                "face_w": averaged_info["face_w"],
                "face_h": averaged_info["face_h"],
            })
            self.calibration_snapshot = self._safe_metric_dict({
                "face_w": averaged_info["face_w"],
                "face_h": averaged_info["face_h"],
                "head_angle": averaged_info["head_angle"],
                "eye_dir": averaged_info["eye_dir"],
                "eye_ratio": averaged_info["eye_ratio"],
                "eye_dist": averaged_info["eye_dist"],
            })
            self.calibration_frozen = True
            self.calibration_freeze_until = time.time() + FREEZE_SECONDS
            self.mode = "calibration_freeze"
            self.trust_score = 100.0
            self.last_state = None
            return {"message": "Calibration reference captured.", "snapshot": self.calibration_snapshot}

    def start_calibration(self) -> Dict[str, str]:
        with self.lock:
            self.exam_started = False
            self.answering = False
            self.questions = []
            self.question_results = [None] * QUESTION_COUNT
            self.current_question_index = 0
            self.mode = "calibrating"
            self.calibration_frozen = False
            self.calibration_freeze_until = 0.0
            self.calibration_ready = False
            self.calibration_buffer.clear()
            self.suggestion = "Align your face inside the oval to prepare calibration."
        return {"message": "Calibration mode started."}

    def start_exam(self) -> Dict[str, Any]:
        with self.lock:
            if self.baseline is None or self.calibration_snapshot is None:
                raise HTTPException(status_code=400, detail="Calibration must be completed before the exam can start.")
            if not self.exam_started:
                self.questions = random.sample(self.questions_bank, k=min(QUESTION_COUNT, len(self.questions_bank)))
                self.question_results = [None] * QUESTION_COUNT
                self.current_question_index = 0
                self.answering = False
                self.answer_samples = []
                self.exam_started = True
            self.mode = "exam"
            self.suggestion = "Interview ready. Start the answer when you are prepared."
            return {
                "message": "Exam ready.",
                "questions": self.questions,
                "current_question_index": self.current_question_index,
                "results": self.question_results,
            }

    def begin_answer(self) -> Dict[str, Any]:
        with self.lock:
            if not self.exam_started:
                raise HTTPException(status_code=400, detail="Start the exam after calibration first.")
            if self.current_question_index >= QUESTION_COUNT:
                raise HTTPException(status_code=400, detail="All interview questions have already been completed.")
            self.answering = True
            self.answer_started_at = time.time()
            self.answer_samples = []
            self.mode = "exam_question"
            self.suggestion = "Answer naturally while keeping steady eye contact."
            return {"message": "Answer recording started.", "question_index": self.current_question_index}

    def end_question(self) -> Dict[str, Any]:
        with self.lock:
            if not self.exam_started:
                raise HTTPException(status_code=400, detail="No active exam session.")
            if not self.answering or self.answer_started_at is None:
                raise HTTPException(status_code=400, detail="Start the answer before ending the question.")

            elapsed = max(1, int(time.time() - self.answer_started_at))
            samples = list(self.answer_samples)
            self.answering = False
            self.answer_started_at = None
            self.answer_samples = []
            self.mode = "exam"

            total = max(1, len(samples))
            bad_frames = sum(1 for sample in samples if sample["is_bad"])
            away_frames = sum(1 for sample in samples if sample["is_cheating"])
            posture_quality = max(0.0, 100.0 - ((bad_frames / total) * 100.0))
            duration_ratio = min(1.0, elapsed / 180.0)
            duration_score = duration_ratio * 100.0

            # Keep posture quality important, but cap short answers heavily.
            # A 1-second answer should remain near zero even with perfect posture.
            weighted_base = (posture_quality * 0.75) + (duration_score * 0.25)
            duration_multiplier = max(0.05, duration_ratio)
            final_score_raw = weighted_base * duration_multiplier
            score = round(max(0.0, min(100.0, final_score_raw)))
            label = self._score_label(score)

            errors: List[Dict[str, Any]] = []
            if bad_frames:
                errors.append(
                    {
                        "key": "posture_quality",
                        "description": "Posture drifted during the answer",
                        "percent_frames": round((bad_frames / total) * 100),
                    }
                )
            if away_frames:
                errors.append(
                    {
                        "key": "looking_away",
                        "description": "Eye contact with the screen was lost",
                        "percent_frames": round((away_frames / total) * 100),
                    }
                )
            if elapsed < 60:
                errors.append(
                    {
                        "key": "answer_length",
                        "description": "Answer was shorter than one minute",
                        "percent_frames": round((1 - (elapsed / 60.0)) * 100),
                    }
                )

            result = {
                "question_index": self.current_question_index,
                "elapsed_time": elapsed,
                "posture_quality": round(posture_quality, 1),
                "duration_score": round(duration_score, 1),
                "base_score": round(weighted_base, 1),
                "time_penalty": max(0, round(weighted_base - score)),
                "score": score,
                "label": label,
                "errors": errors,
            }
            self.question_results[self.current_question_index] = result
            self.suggestion = "Question scored. Review the result and move to the next question."
            return result

    def next_question(self) -> Dict[str, Any]:
        with self.lock:
            if not self.exam_started:
                raise HTTPException(status_code=400, detail="No active exam session.")
            if self.question_results[self.current_question_index] is None:
                raise HTTPException(status_code=400, detail="Finish the current question before moving on.")
            if self.current_question_index >= QUESTION_COUNT - 1:
                raise HTTPException(status_code=400, detail="You are already on the final question.")
            self.current_question_index += 1
            self.mode = "exam"
            self.suggestion = "Next question loaded. Start when you are ready."
            return {"current_question_index": self.current_question_index}

    def redo_question(self) -> Dict[str, Any]:
        with self.lock:
            if not self.exam_started:
                raise HTTPException(status_code=400, detail="No active exam session.")
            self.question_results[self.current_question_index] = None
            self.answering = False
            self.answer_started_at = None
            self.answer_samples = []
            self.mode = "exam"
            self.suggestion = "Question reset. You can record this answer again."
            return {"message": "Question reset.", "current_question_index": self.current_question_index}

    def reset_exam(self) -> Dict[str, Any]:
        with self.lock:
            if self.baseline is None:
                raise HTTPException(status_code=400, detail="Calibration is required before starting a new exam.")
            self.exam_started = False
            self.questions = []
            self.question_results = [None] * QUESTION_COUNT
            self.current_question_index = 0
            self.answering = False
            self.answer_started_at = None
            self.answer_samples = []
            self.mode = "posture"
            self.suggestion = "Exam reset. Start a new interview round when ready."
            return {"message": "Exam reset."}

    def end_exam(self) -> Dict[str, Any]:
        with self.lock:
            completed = [result for result in self.question_results if result is not None]
            if len(completed) != QUESTION_COUNT:
                raise HTTPException(status_code=400, detail="Complete all 3 interview questions before ending the exam.")

            final_score = round(sum(item["score"] for item in completed) / len(completed))
            posture_avg = round(sum(item["posture_quality"] for item in completed) / len(completed), 1)
            duration_avg = round(sum(item["duration_score"] for item in completed) / len(completed), 1)
            total_elapsed = sum(item["elapsed_time"] for item in completed)
            summary = {
                "score": final_score,
                "label": self._score_label(final_score),
                "posture_average": posture_avg,
                "duration_average": duration_avg,
                "elapsed_time": total_elapsed,
                "questions_answered": len(completed),
                "question_breakdown": completed,
            }
            self.mode = "posture"
            self.suggestion = "Interview complete. Review your final score."
            return summary

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            payload = {
                "state": list(self.state) if self.state else None,
                "action": self.action,
                "reward": self.reward,
                "is_bad": self.is_bad,
                "is_cheating": self.is_cheating,
                "trust_score": round(self.trust_score, 1),
                "identified_by": self.identified_by,
                "mode": self.mode,
                "calibration_ready": self.calibration_ready,
                "face_inside_ratio": round(self.face_inside_ratio, 3),
                "face_width": round(self.face_width, 1),
                "face_height": round(self.face_height, 1),
                "head_angle": round(self.head_angle, 3),
                "eye_dir": round(self.eye_dir, 3),
                "eye_ratio": round(self.eye_ratio, 3),
                "eye_distance": round(self.eye_distance, 3),
                "calibration_snapshot": self.calibration_snapshot,
                "calibration_frozen": self.calibration_frozen,
                "calibration_freeze_remaining": round(max(0.0, self.calibration_freeze_until - time.time()), 2),
                "suggestion": self.suggestion,
                "connected": True,
                "camera_available": self.camera_available,
                "exam_started": self.exam_started,
                "answering": self.answering,
                "current_question_index": self.current_question_index,
                "algorithm": self.algorithm,
                "available_algorithms": sorted(self.rl_agents.keys()),
                "posture_label": self.current_label.value,
                "posture_score": round(self.current_score * 100, 1),
                "agent_runtime_error": self.agent_runtime_error,
            }
            return self._to_python_value(payload)

    def frame_stream(self):
        while True:
            with self.lock:
                frame = self.latest_jpeg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(FRAME_SLEEP)

    def shutdown(self) -> None:
        self.running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def _score_label(self, score: int) -> str:
        if score >= 85:
            return "Excellent"
        if score >= 70:
            return "Good"
        if score >= 55:
            return "Needs Improvement"
        return "Poor"


engine = InterviewEngine()
atexit.register(engine.shutdown)

app = FastAPI(title="UPRYT Mock Interview API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/state")
def get_state() -> Dict[str, Any]:
    return engine.get_state()


@app.get("/algorithm")
def get_algorithm() -> Dict[str, Any]:
    return {"algorithm": engine.algorithm, "available_algorithms": sorted(engine.rl_agents.keys())}


@app.get("/set_algorithm")
def set_algorithm(algorithm: str = Query(...)) -> Dict[str, Any]:
    return engine.set_algorithm(algorithm)


@app.get("/calibrate")
def calibrate() -> Dict[str, str]:
    return engine.start_calibration()


@app.get("/capture_reference")
def capture_reference() -> Dict[str, Any]:
    return engine.capture_reference()


@app.get("/start_exam")
def start_exam() -> Dict[str, Any]:
    return engine.start_exam()


@app.get("/begin_answer")
def begin_answer() -> Dict[str, Any]:
    return engine.begin_answer()


@app.get("/end_question")
def end_question() -> Dict[str, Any]:
    return engine.end_question()


@app.get("/next_question")
def next_question() -> Dict[str, Any]:
    return engine.next_question()


@app.get("/redo_question")
def redo_question() -> Dict[str, Any]:
    return engine.redo_question()


@app.get("/reset_exam")
def reset_exam() -> Dict[str, Any]:
    return engine.reset_exam()


@app.get("/end_exam")
def end_exam() -> Dict[str, Any]:
    return engine.end_exam()


@app.get("/stop_session")
def stop_session(keep_calibration: bool = Query(False)) -> Dict[str, str]:
    engine.reset_session_state(keep_calibration=keep_calibration)
    return {"message": "Session stopped."}


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    return StreamingResponse(engine.frame_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "camera_available": engine.camera_available,
        "algorithm": engine.algorithm,
        "available_algorithms": sorted(engine.rl_agents.keys()),
        "agent_runtime_error": engine.agent_runtime_error,
    })
