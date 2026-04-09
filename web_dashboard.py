import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime


class DashboardExporter:
    """Stub dashboard exporter that logs data for potential web dashboard integration"""
    
    def __init__(self):
        self.session_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self.start_time: Optional[float] = None
        self.frames: List[Dict] = []
        self.alerts: List[Dict] = []
        self.session_stats: Dict[str, Any] = {
            "total_frames": 0,
            "total_alerts": 0,
            "avg_posture_score": 0.0,
            "good_posture_ratio": 0.0,
            "attention_score_avg": 1.0,
        }
        self.score_history: List[float] = []
        self.label_history: List[str] = []
    
    def start_session(self, session_id: str, user_id: str = "default_user"):
        """Start a new session"""
        self.session_id = session_id
        self.user_id = user_id
        self.start_time = time.time()
        self.frames = []
        self.alerts = []
        self.score_history = []
        self.label_history = []
        print(f"Dashboard: Session started - {session_id}")
    
    def record_frame(self, posture_score: float, posture_label: str, attention_score: float = 1.0):
        """Record a frame's posture and attention data"""
        if self.start_time is None:
            return
        
        frame_data = {
            "timestamp": time.time() - self.start_time,
            "posture_score": posture_score,
            "posture_label": posture_label,
            "attention_score": attention_score,
        }
        
        self.frames.append(frame_data)
        self.score_history.append(posture_score)
        self.label_history.append(posture_label)
        
        # Update running stats
        self.session_stats["total_frames"] = len(self.frames)
        if self.score_history:
            self.session_stats["avg_posture_score"] = sum(self.score_history) / len(self.score_history)
        if self.label_history:
            good_count = sum(1 for l in self.label_history if l == "good")
            self.session_stats["good_posture_ratio"] = good_count / len(self.label_history)
    
    def record_alert(self, alert_triggered: bool):
        """Record an alert event"""
        if self.start_time is None or not alert_triggered:
            return
        
        alert_data = {
            "timestamp": time.time() - self.start_time,
            "type": "posture_alert",
        }
        
        self.alerts.append(alert_data)
        self.session_stats["total_alerts"] = len(self.alerts)
    
    def finalize_session_stats(self):
        """Finalize session statistics"""
        if self.start_time:
            self.session_stats["session_duration"] = time.time() - self.start_time
    
    def end_session(self) -> Dict:
        """End the current session and return summary"""
        self.finalize_session_stats()
        
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "stats": self.session_stats,
            "frame_count": len(self.frames),
            "alert_count": len(self.alerts),
        }
    
    def export_for_web(self) -> Dict:
        """Export data in web-friendly format"""
        return {
            "session": {
                "id": self.session_id,
                "user": self.user_id,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            },
            "stats": self.session_stats,
            "recent_frames": self.frames[-100:] if len(self.frames) > 100 else self.frames,
            "recent_alerts": self.alerts[-20:] if len(self.alerts) > 20 else self.alerts,
            "posture_timeline": self.score_history[-300:] if len(self.score_history) > 300 else self.score_history,
        }


def create_dashboard_exporter() -> DashboardExporter:
    """Factory function to create dashboard exporter"""
    return DashboardExporter()


if __name__ == "__main__":
    # Test the dashboard exporter
    exporter = create_dashboard_exporter()
    exporter.start_session("test_session", "test_user")
    
    # Simulate some frames
    for i in range(50):
        score = 0.5 + (i % 20) / 40.0  # Oscillating score
        label = "good" if score > 0.6 else "bad"
        exporter.record_frame(score, label, 0.9)
        
        if i % 10 == 0:
            exporter.record_alert(True)
    
    exporter.end_session()
    print("Dashboard exporter test complete")
    print(json.dumps(exporter.export_for_web(), indent=2))