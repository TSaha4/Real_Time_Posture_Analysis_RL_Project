import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import time
from dataclasses import dataclass

from attention_tracker import (
    AttentionTracker, AttentionState, FaceDetector, 
    GazeMetrics, CombinedPostureAttentionAnalyzer
)
from hand_tracker import (
    HandTracker, TypingPosture, HandMetrics,
    CombinedPostureAnalyzer as HandCombinedAnalyzer
)


@dataclass
class CombinedMetrics:
    posture_score: float = 0.0
    attention_score: float = 1.0
    hand_score: float = 1.0
    
    posture_label: str = "unknown"
    attention_state: str = "unknown"
    hand_state: str = "unknown"
    
    typing_intensity: float = 0.0
    is_typing: bool = False
    is_attending: bool = True
    is_away: bool = False
    
    combined_score: float = 0.0
    focus_percentage: float = 100.0
    good_typing_percentage: float = 100.0


class UnifiedAnalyzer:
    def __init__(self, enable_attention: bool = True, enable_hands: bool = True):
        self.enable_attention = enable_attention
        self.enable_hands = enable_hands
        
        self.attention_tracker = AttentionTracker() if enable_attention else None
        self.hand_tracker = HandTracker(use_holistic=True) if enable_hands else None
        
        self.last_update_time = time.time()
        self.frame_count = 0
        self._hand_skip_frames = 0
        self._last_hands_data = None
        
        self.session_metrics = {
            "total_frames": 0,
            "posture_scores": [],
            "attention_scores": [],
            "hand_scores": [],
            "combined_scores": [],
        }
    
    def analyze(self, frame: np.ndarray, posture_score: float, 
                posture_label: str) -> CombinedMetrics:
        metrics = CombinedMetrics()
        metrics.posture_score = posture_score
        metrics.posture_label = posture_label
        
        if self.enable_attention and self.attention_tracker:
            attention_metrics = self.attention_tracker.update(frame)
            metrics.attention_score = attention_metrics.attention_score
            metrics.attention_state = attention_metrics.state.value
            metrics.is_attending = self.attention_tracker.is_user_attending()
            metrics.is_away = self.attention_tracker.is_user_away()
            
            attention_data = self.attention_tracker.get_session_metrics()
            metrics.focus_percentage = attention_data.get("focus_percentage", 100.0)
        
        if self.enable_hands and self.hand_tracker:
            self._hand_skip_frames += 1
            if self._hand_skip_frames >= 5 or self._last_hands_data is None:
                self._hand_skip_frames = 0
                self._last_hands_data = self.hand_tracker.detect_hands(frame)
            
            hands_data = self._last_hands_data
            hand_metrics = self.hand_tracker.update_metrics(hands_data)
            
            if hands_data:
                frame = self.hand_tracker.draw_hands(frame, hands_data)
            
            metrics.hand_score = self.hand_tracker.get_typing_posture_score()
            metrics.hand_state = hand_metrics.posture_state.value
            metrics.typing_intensity = self.hand_tracker.get_typing_intensity()
            metrics.is_typing = self.hand_tracker.is_typing()
            
            hand_session = self.hand_tracker.get_session_metrics()
            metrics.good_typing_percentage = hand_session.get("good_typing_percentage", 100.0)
        
        metrics.combined_score = self._compute_combined_score(metrics)
        
        self._update_session_metrics(metrics)
        
        return metrics
    
    def _compute_combined_score(self, metrics: CombinedMetrics) -> float:
        base_score = metrics.posture_score
        
        if self.enable_attention and self.attention_tracker:
            attention_factor = self.attention_tracker.get_posture_attention_factor()
            base_score = base_score * 0.7 + (metrics.posture_score * attention_factor) * 0.3
        
        if self.enable_hands and self.hand_tracker:
            if metrics.hand_state == "unknown":
                hand_factor = 1.0
            else:
                hand_factor = 0.85 + metrics.hand_score * 0.15
            base_score *= hand_factor
        
        return max(0.0, min(1.0, base_score))
    
    def _update_session_metrics(self, metrics: CombinedMetrics):
        self.session_metrics["total_frames"] += 1
        self.session_metrics["posture_scores"].append(metrics.posture_score)
        self.session_metrics["attention_scores"].append(metrics.attention_score)
        self.session_metrics["hand_scores"].append(metrics.hand_score)
        self.session_metrics["combined_scores"].append(metrics.combined_score)
        
        max_history = 10000
        list_keys = ["posture_scores", "attention_scores", "hand_scores", "combined_scores"]
        for key in list_keys:
            if len(self.session_metrics[key]) > max_history:
                self.session_metrics[key] = self.session_metrics[key][-max_history:]
    
    def draw_attention_overlay(self, frame: np.ndarray) -> np.ndarray:
        if not self.enable_attention or not self.attention_tracker:
            return frame
        
        output = frame.copy()
        state = self.attention_tracker.current_state
        
        if state == AttentionState.FOCUSED:
            color = (0, 255, 0)
            text = "FOCUSED"
        elif state == AttentionState.DISTRACTED:
            color = (0, 255, 255)
            text = "DISTRACTED"
        elif state == AttentionState.AWAY:
            color = (0, 0, 255)
            text = "AWAY"
        else:
            color = (128, 128, 128)
            text = "UNKNOWN"
        
        cv2.rectangle(output, (5, 5), (180, 80), (30, 30, 30), -1)
        cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        attention = self.attention_tracker.get_attention_score()
        cv2.putText(output, f"Att: {attention:.0%}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return output
    
    def get_session_summary(self) -> Dict:
        summary = {
            "total_frames": self.session_metrics["total_frames"],
        }
        
        if self.session_metrics["posture_scores"]:
            summary["avg_posture_score"] = np.mean(self.session_metrics["posture_scores"])
        if self.session_metrics["attention_scores"]:
            summary["avg_attention_score"] = np.mean(self.session_metrics["attention_scores"])
        if self.session_metrics["hand_scores"]:
            summary["avg_hand_score"] = np.mean(self.session_metrics["hand_scores"])
        if self.session_metrics["combined_scores"]:
            summary["avg_combined_score"] = np.mean(self.session_metrics["combined_scores"])
        
        if self.enable_attention and self.attention_tracker:
            summary["attention_metrics"] = self.attention_tracker.get_session_metrics()
        
        if self.enable_hands and self.hand_tracker:
            summary["hand_metrics"] = self.hand_tracker.get_session_metrics()
        
        return summary
    
    def reset(self):
        if self.attention_tracker:
            self.attention_tracker.reset()
        if self.hand_tracker:
            self.hand_tracker.reset()
        self.session_metrics = {
            "total_frames": 0,
            "posture_scores": [],
            "attention_scores": [],
            "hand_scores": [],
            "combined_scores": [],
        }


def create_unified_analyzer(enable_attention: bool = True, 
                           enable_hands: bool = True) -> UnifiedAnalyzer:
    return UnifiedAnalyzer(enable_attention, enable_hands)
