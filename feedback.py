import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
from config import config


class FeedbackLevel(Enum):
    NONE = 0
    GOOD = 1
    WARNING = 2
    BAD = 3


@dataclass
class FeedbackMessage:
    text: str
    level: FeedbackLevel
    sound: bool = False
    popup: bool = False


class VisualFeedback:
    def __init__(self):
        self.window_name = config.feedback.window_name
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = config.feedback.font_scale
        self.thickness = 2
        self.good_color = config.feedback.posture_good_color
        self.bad_color = config.feedback.posture_bad_color
        self.warning_color = config.feedback.posture_warning_color
        self.info_panel_height = 180
        self.info_panel_width = 350
        self.info_panel_margin = 15
        self.logo = self._load_logo()

    def _load_logo(self):
        import os
        script_dir = os.path.dirname(os.path.realpath(__file__))
        logo_path = os.path.join(script_dir, "logo", "upryt_white.png")
        if not os.path.exists(logo_path):
            logo_path = os.path.join(script_dir, "..", "logo", "upryt_white.png")
        logo_path = os.path.normpath(logo_path)
        if os.path.exists(logo_path):
            try:
                import cv2
                logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
                if logo is not None:
                    if logo.shape[2] == 4:
                        logo = cv2.cvtColor(logo, cv2.COLOR_BGRA2BGR)
                    return cv2.resize(logo, (100, 30))
            except Exception as e:
                print(f"Logo load error: {e}")
                pass
        return None

    def draw_overlay(self, frame: np.ndarray, posture_label: str, posture_score: float,
                    action_taken: str, metrics: Dict, suggestion: str = "") -> np.ndarray:
        output = frame.copy()
        panel = self._create_info_panel(frame.shape, posture_label, posture_score, action_taken, metrics, suggestion)
        output = self._blend_panel(output, panel)
        return output

    def _create_info_panel(self, frame_shape: Tuple, posture_label: str, posture_score: float,
                          action_taken: str, metrics: Dict, suggestion: str = "") -> np.ndarray:
        h, w = frame_shape[:2]
        panel_w = self.info_panel_width
        panel_h = self.info_panel_height
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        
        cv2.rectangle(panel, (0, 0), (panel_w, panel_h), (30, 30, 30), -1)
        cv2.rectangle(panel, (0, 0), (panel_w, panel_h), (100, 100, 100), 2)
        
        score_color = self._get_score_color(posture_score)
        
        cv2.putText(panel, f"Posture: {posture_label.upper()}", (10, 28),
                   self.font, 0.6, score_color, 2)
        bar_width = 150
        bar_height = 16
        bar_x = 10
        bar_y = 40
        self._draw_score_bar(panel, posture_score, bar_x, bar_y, bar_width, bar_height)
        cv2.putText(panel, f"{posture_score:.0%}", (bar_x + bar_width + 8, bar_y + 13),
                   self.font, 0.45, (255, 255, 255), 1)
        
        cv2.putText(panel, f"Action: {action_taken}", (10, 70),
                   self.font, 0.5, (200, 200, 200), 1)
        
        y_offset = 95
        if suggestion:
            suggestion_text = suggestion[:35] + "..." if len(suggestion) > 35 else suggestion
            cv2.putText(panel, suggestion_text, (10, y_offset),
                       self.font, 0.45, (255, 200, 100), 1)
            y_offset += 22
        
        for key, value in list(metrics.items())[:3]:
            text = f"{key}: {value}"
            cv2.putText(panel, text, (10, y_offset), self.font, 0.4, (160, 160, 160), 1)
            y_offset += 18
        
        if self.logo is not None:
            logo_x = panel_w - self.logo.shape[1] - 10
            logo_y = panel_h - self.logo.shape[1] - 10
            panel[logo_y:logo_y+self.logo.shape[0], logo_x:logo_x+self.logo.shape[1]] = self.logo
        
        return panel

    def _draw_score_bar(self, panel: np.ndarray, score: float, x: int, y: int, width: int, height: int):
        cv2.rectangle(panel, (x, y), (x + width, y + height), (50, 50, 50), -1)
        fill_width = int(width * score)
        if fill_width > 0:
            cv2.rectangle(panel, (x, y), (x + fill_width, y + height), self._get_score_color(score), -1)
        cv2.rectangle(panel, (x, y), (x + width, y + height), (255, 255, 255), 1)
        for i in range(1, 4):
            tick_x = x + int(width * i / 4)
            cv2.line(panel, (tick_x, y), (tick_x, y + height), (100, 100, 100), 1)

    def _get_score_color(self, score: float) -> Tuple[int, int, int]:
        if score >= 0.7:
            return self.good_color
        elif score >= 0.4:
            return self.warning_color
        return self.bad_color

    def _blend_panel(self, frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
        panel_alpha = 0.9
        h, w = frame.shape[:2]
        panel_h, panel_w = panel.shape[:2]
        
        margin = 15
        x = w - panel_w - margin
        y = margin
        
        roi = frame[y:y+panel_h, x:x+panel_w]
        blended = cv2.addWeighted(roi, 1 - panel_alpha, panel, panel_alpha, 0)
        frame[y:y+panel_h, x:x+panel_w] = blended
        return frame

    def draw_skeleton_colored(self, frame: np.ndarray, keypoints: Dict, score: float) -> np.ndarray:
        color = self._get_score_color(score)
        output = frame.copy()
        output = self._draw_skeleton_lines(output, keypoints, color)
        output = self._draw_joints(output, keypoints, color)
        return output

    def _draw_skeleton_lines(self, frame: np.ndarray, keypoints: Dict, color: Tuple[int, int, int]) -> np.ndarray:
        connections = [
            ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
            ("nose", "left_shoulder"), ("nose", "right_shoulder"),
        ]
        for joint1, joint2 in connections:
            if joint1 in keypoints and joint2 in keypoints:
                pt1 = (int(keypoints[joint1]["x"]), int(keypoints[joint1]["y"]))
                pt2 = (int(keypoints[joint2]["x"]), int(keypoints[joint2]["y"]))
                cv2.line(frame, pt1, pt2, color, 3)
        return frame

    def _draw_joints(self, frame: np.ndarray, keypoints: Dict, color: Tuple[int, int, int]) -> np.ndarray:
        key_joints = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        for joint in key_joints:
            if joint in keypoints:
                pt = (int(keypoints[joint]["x"]), int(keypoints[joint]["y"]))
                cv2.circle(frame, pt, 8, color, -1)
                cv2.circle(frame, pt, 8, (255, 255, 255), 2)
        return frame


class AlertSystem:
    def __init__(self):
        self.last_alert_time = 0.0
        self.cooldown = config.system.cooldown_period
        self.message_queue: List[FeedbackMessage] = []

    def should_alert(self, current_time: float) -> bool:
        return current_time - self.last_alert_time >= self.cooldown

    def send_alert(self, level: FeedbackLevel, current_time: float) -> Optional[FeedbackMessage]:
        if not self.should_alert(current_time):
            return None
        message = self._create_message(level)
        self.last_alert_time = current_time
        if message.sound:
            self._play_sound(level)
        if message.popup:
            self._show_popup(message)
        return message

    def _create_message(self, level: FeedbackLevel) -> FeedbackMessage:
        messages = {
            FeedbackLevel.GOOD: FeedbackMessage(text="Great posture! Keep it up!", level=level, sound=False, popup=False),
            FeedbackLevel.WARNING: FeedbackMessage(text="Consider straightening up", level=level, sound=False, popup=False),
            FeedbackLevel.BAD: FeedbackMessage(text="Fix your posture! Sit straight!", level=level, sound=config.feedback.sound_enabled, popup=config.feedback.popup_enabled),
        }
        return messages.get(level, FeedbackMessage(text="", level=FeedbackLevel.NONE))

    def _play_sound(self, level: FeedbackLevel):
        try:
            import winsound
            frequency = 800 if level == FeedbackLevel.WARNING else 1200
            winsound.Beep(frequency, 200)
        except:
            pass

    def _show_popup(self, message: FeedbackMessage):
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showwarning("UPRYT - Posture Alert", message.text)
            root.destroy()
        except:
            pass

    def reset_cooldown(self):
        self.last_alert_time = 0.0


class FeedbackManager:
    def __init__(self):
        self.visual = VisualFeedback()
        self.alerts = AlertSystem()

    def process_frame(self, frame: np.ndarray, posture_label: str, posture_score: float,
                     action_taken: str, metrics: Dict, current_time: float,
                     should_alert: bool = False, alert_level: FeedbackLevel = FeedbackLevel.NONE) -> np.ndarray:
        output = frame.copy()
        output = self.visual.draw_skeleton_colored(output, {}, posture_score)
        if should_alert:
            self.alerts.send_alert(alert_level, current_time)
        output = self.visual.draw_overlay(output, posture_label, posture_score, action_taken, metrics)
        return output


def create_feedback_message(posture_label: str, action: int) -> str:
    if action == 0:
        return ""
    action_messages = {
        1: {"good": "Good posture!", "slouching": "Sit a bit straighter", "forward_head": "Pull your head back", "leaning": "Sit upright"},
        2: {"good": "Excellent! Keep it up!", "slouching": "Straighten up now!", "forward_head": "Fix forward head posture!", "leaning": "Stop leaning! Sit straight!"},
    }
    return action_messages.get(action, {}).get(posture_label, "Fix your posture!")


if __name__ == "__main__":
    feedback = VisualFeedback()
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    metrics = {"Total Alerts": 10, "Corrections": 7, "Avg Score": 0.75, "Reward": 42.5}
    output = feedback.draw_overlay(test_frame, "good", 0.85, "no_feedback", metrics)
    output = feedback.draw_skeleton_colored(output, {}, 0.85)
    print("Feedback system test complete")
