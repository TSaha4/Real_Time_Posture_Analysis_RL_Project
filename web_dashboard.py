import json
import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np


@dataclass
class SessionSummary:
    session_id: str
    user_id: str
    start_time: str
    end_time: str
    duration_seconds: float
    avg_posture_score: float
    max_posture_score: float
    min_posture_score: float
    good_posture_percentage: float
    total_alerts: int
    successful_corrections: int
    correction_rate: float
    posture_distribution: Dict[str, float] = field(default_factory=dict)
    hourly_breakdown: Dict[str, float] = field(default_factory=dict)
    attention_metrics: Dict[str, float] = field(default_factory=dict)
    achievement_progress: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionSummary":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class DashboardDataExporter:
    def __init__(self, data_dir: str = "data/dashboard"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.current_session: Optional[SessionSummary] = None
        self.sessions_cache: List[SessionSummary] = []
        self._load_existing_data()

    def _load_existing_data(self):
        sessions_file = os.path.join(self.data_dir, "sessions.json")
        if os.path.exists(sessions_file):
            try:
                with open(sessions_file, 'r') as f:
                    data = json.load(f)
                    self.sessions_cache = [SessionSummary.from_dict(s) for s in data]
            except Exception:
                self.sessions_cache = []

    def _save_data(self):
        sessions_file = os.path.join(self.data_dir, "sessions.json")
        try:
            with open(sessions_file, 'w') as f:
                json.dump([s.to_dict() for s in self.sessions_cache[-100:]], f, indent=2)
        except Exception as e:
            print(f"Error saving dashboard data: {e}")

    def start_session(self, session_id: str, user_id: str) -> SessionSummary:
        self.current_session = SessionSummary(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now().isoformat(),
            end_time="",
            duration_seconds=0.0,
            avg_posture_score=0.0,
            max_posture_score=0.0,
            min_posture_score=1.0,
            good_posture_percentage=0.0,
            total_alerts=0,
            successful_corrections=0,
            correction_rate=0.0,
        )
        return self.current_session

    def end_session(self) -> Optional[SessionSummary]:
        if not self.current_session:
            return None
        
        self.current_session.end_time = datetime.now().isoformat()
        
        self.sessions_cache.append(self.current_session)
        if len(self.sessions_cache) > 100:
            self.sessions_cache.pop(0)
        
        self._save_data()
        
        session = self.current_session
        self.current_session = None
        return session

    def record_frame(self, posture_score: float, posture_label: str, 
                     attention_score: float = 1.0, hour: int = None):
        if not self.current_session:
            return
        
        if hour is None:
            hour = datetime.now().hour
        
        hour_key = f"hour_{hour}"
        if hour_key not in self.current_session.hourly_breakdown:
            self.current_session.hourly_breakdown[hour_key] = []
        
        self.current_session.hourly_breakdown[hour_key].append(posture_score)
        
        self.current_session.max_posture_score = max(
            self.current_session.max_posture_score, posture_score)
        self.current_session.min_posture_score = min(
            self.current_session.min_posture_score, posture_score)
        
        if posture_label not in self.current_session.posture_distribution:
            self.current_session.posture_distribution[posture_label] = 0
        self.current_session.posture_distribution[posture_label] += 1
        
        if "attention" not in self.current_session.attention_metrics:
            self.current_session.attention_metrics = {
                "avg_score": 0.0,
                "samples": 0,
                "total_score": 0.0,
            }
        self.current_session.attention_metrics["total_score"] += attention_score
        self.current_session.attention_metrics["samples"] += 1
        self.current_session.attention_metrics["avg_score"] = (
            self.current_session.attention_metrics["total_score"] / 
            max(1, self.current_session.attention_metrics["samples"])
        )

    def record_alert(self, correction_made: bool):
        if not self.current_session:
            return
        
        self.current_session.total_alerts += 1
        if correction_made:
            self.current_session.successful_corrections += 1
        
        total = max(1, self.current_session.total_alerts)
        self.current_session.correction_rate = (
            self.current_session.successful_corrections / total)

    def finalize_session_stats(self):
        if not self.current_session:
            return
        
        scores = []
        for hour_scores in self.current_session.hourly_breakdown.values():
            scores.extend(hour_scores)
        
        if scores:
            self.current_session.avg_posture_score = np.mean(scores)
            
            good_count = sum(1 for s in scores if s >= 0.7)
            self.current_session.good_posture_percentage = (good_count / len(scores)) * 100
        
        start = datetime.fromisoformat(self.current_session.start_time)
        end = datetime.fromisoformat(self.current_session.end_time)
        self.current_session.duration_seconds = (end - start).total_seconds()
        
        total = sum(self.current_session.posture_distribution.values())
        if total > 0:
            self.current_session.posture_distribution = {
                k: (v / total) * 100 
                for k, v in self.current_session.posture_distribution.items()
            }

    def get_weekly_summary(self, user_id: str = None) -> Dict:
        today = datetime.now().date()
        week_ago = today - timedelta(days=7)
        
        week_sessions = [
            s for s in self.sessions_cache
            if datetime.fromisoformat(s.start_time).date() >= week_ago
            and (user_id is None or s.user_id == user_id)
        ]
        
        if not week_sessions:
            return {
                "total_sessions": 0,
                "total_time_minutes": 0,
                "avg_posture_score": 0,
                "improvement_trend": "no_data",
                "daily_breakdown": [],
            }
        
        daily_data = defaultdict(lambda: {
            "sessions": 0, "time_minutes": 0, "avg_score": [], "corrections": 0
        })
        
        for session in week_sessions:
            day = datetime.fromisoformat(session.start_time).strftime("%Y-%m-%d")
            daily_data[day]["sessions"] += 1
            daily_data[day]["time_minutes"] += session.duration_seconds / 60
            daily_data[day]["avg_score"].append(session.avg_posture_score)
            daily_data[day]["corrections"] += session.successful_corrections
        
        daily_breakdown = []
        for i in range(7):
            date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            day_data = daily_data.get(date, {"sessions": 0, "time_minutes": 0, "avg_score": [], "corrections": 0})
            daily_breakdown.append({
                "date": date,
                "sessions": day_data["sessions"],
                "time_minutes": day_data["time_minutes"],
                "avg_score": np.mean(day_data["avg_score"]) if day_data["avg_score"] else 0,
                "corrections": day_data["corrections"],
            })
        
        total_time = sum(s.duration_seconds for s in week_sessions) / 60
        all_scores = [s.avg_posture_score for s in week_sessions]
        
        trend = "stable"
        if len(daily_breakdown) >= 3:
            recent = np.mean([d["avg_score"] for d in daily_breakdown[:3] if d["avg_score"] > 0])
            older = np.mean([d["avg_score"] for d in daily_breakdown[3:] if d["avg_score"] > 0])
            if recent > older + 5:
                trend = "improving"
            elif recent < older - 5:
                trend = "declining"
        
        return {
            "total_sessions": len(week_sessions),
            "total_time_minutes": total_time,
            "avg_posture_score": np.mean(all_scores) if all_scores else 0,
            "improvement_trend": trend,
            "daily_breakdown": daily_breakdown,
            "best_day": max(daily_breakdown, key=lambda x: x["avg_score"])["date"] if daily_breakdown else None,
            "total_corrections": sum(s.successful_corrections for s in week_sessions),
        }

    def get_achievements_status(self, user_id: str = None) -> Dict:
        user_sessions = [
            s for s in self.sessions_cache
            if user_id is None or s.user_id == user_id
        ]
        
        if not user_sessions:
            return {"achievements": [], "progress": {}}
        
        achievements = {
            "first_session": len(user_sessions) >= 1,
            "week_warrior": self._has_sessions_on_days(user_sessions, 7),
            "posture_pro": sum(s.successful_corrections for s in user_sessions) >= 50,
            "perfect_sessions": sum(1 for s in user_sessions if s.good_posture_percentage >= 90) >= 1,
            "consistent": self._has_sessions_on_days(user_sessions, 5),
        }
        
        total_time = sum(s.duration_seconds for s in user_sessions) / 3600
        achievements["time_invested"] = total_time >= 10
        
        progress = {
            "sessions": (len(user_sessions), 10),
            "corrections": (sum(s.successful_corrections for s in user_sessions), 50),
            "perfect_sessions": (sum(1 for s in user_sessions if s.good_posture_percentage >= 90), 5),
        }
        
        return {
            "achievements": achievements,
            "progress": progress,
            "total_achievements": sum(achievements.values()),
            "total_achievementable": len(achievements),
        }

    def _has_sessions_on_days(self, sessions: List[SessionSummary], min_days: int) -> bool:
        days_with_sessions = set()
        for session in sessions:
            day = datetime.fromisoformat(session.start_time).date()
            days_with_sessions.add(day)
        return len(days_with_sessions) >= min_days

    def export_for_web(self, user_id: str = None) -> Dict:
        weekly = self.get_weekly_summary(user_id)
        achievements = self.get_achievements_status(user_id)
        
        user_sessions = [
            s.to_dict() for s in self.sessions_cache
            if user_id is None or s.user_id == user_id
        ][-10:]
        
        for session in user_sessions:
            session["duration_minutes"] = session.pop("duration_seconds", 0) / 60
            session["start_time"] = datetime.fromisoformat(session["start_time"]).strftime("%Y-%m-%d %H:%M")
            session["end_time"] = datetime.fromisoformat(session["end_time"]).strftime("%Y-%m-%d %H:%M") if session["end_time"] else "ongoing"
        
        return {
            "generated_at": datetime.now().isoformat(),
            "weekly_summary": weekly,
            "achievements": achievements,
            "recent_sessions": user_sessions,
            "current_streak": self._calculate_streak(user_id),
            "next_goals": self._get_next_goals(user_id),
        }

    def _calculate_streak(self, user_id: str = None) -> int:
        user_sessions = [
            s for s in self.sessions_cache
            if user_id is None or s.user_id == user_id
        ]
        
        if not user_sessions:
            return 0
        
        days_with_sessions = sorted(set(
            datetime.fromisoformat(s.start_time).date() 
            for s in user_sessions
        ), reverse=True)
        
        if not days_with_sessions:
            return 0
        
        if (datetime.now().date() - days_with_sessions[0]).days > 1:
            return 0
        
        streak = 1
        for i in range(1, len(days_with_sessions)):
            if (days_with_sessions[i-1] - days_with_sessions[i]).days == 1:
                streak += 1
            else:
                break
        
        return streak

    def _get_next_goals(self, user_id: str = None) -> List[Dict]:
        user_sessions = [
            s for s in self.sessions_cache
            if user_id is None or s.user_id == user_id
        ]
        
        goals = []
        
        if len(user_sessions) < 10:
            goals.append({
                "type": "sessions",
                "current": len(user_sessions),
                "target": 10,
                "description": f"Complete {10 - len(user_sessions)} more sessions",
            })
        
        total_corrections = sum(s.successful_corrections for s in user_sessions)
        if total_corrections < 50:
            goals.append({
                "type": "corrections",
                "current": total_corrections,
                "target": 50,
                "description": f"Make {50 - total_corrections} more corrections",
            })
        
        streak = self._calculate_streak(user_id)
        if streak < 7:
            goals.append({
                "type": "streak",
                "current": streak,
                "target": 7,
                "description": f"Maintain a {7 - streak}-day streak",
            })
        
        return goals


class WebDashboardAPI:
    def __init__(self, data_exporter: DashboardDataExporter = None):
        self.data_exporter = data_exporter or DashboardDataExporter()

    def get_dashboard_data(self, user_id: str = None) -> str:
        data = self.data_exporter.export_for_web(user_id)
        return json.dumps(data, indent=2)

    def get_weekly_report(self, user_id: str = None) -> str:
        data = self.data_exporter.get_weekly_summary(user_id)
        return json.dumps(data, indent=2)

    def get_achievements(self, user_id: str = None) -> str:
        data = self.data_exporter.get_achievements_status(user_id)
        return json.dumps(data, indent=2)

    def get_session_history(self, user_id: str = None, limit: int = 20) -> str:
        sessions = [
            s.to_dict() for s in self.data_exporter.sessions_cache
            if user_id is None or s.user_id == user_id
        ][-limit:]
        return json.dumps(sessions, indent=2)


def create_dashboard_exporter(data_dir: str = "data/dashboard") -> DashboardDataExporter:
    return DashboardDataExporter(data_dir)


def create_web_api(exporter: DashboardDataExporter = None) -> WebDashboardAPI:
    return WebDashboardAPI(exporter)


if __name__ == "__main__":
    print("Testing Web Dashboard Data Structure...")
    
    exporter = create_dashboard_exporter()
    
    print("Creating test session...")
    session = exporter.start_session("session_001", "user_test")
    
    for i in range(100):
        posture_score = 0.6 + np.random.uniform(-0.2, 0.3)
        posture_label = np.random.choice(["good", "slouching", "forward_head", "leaning"],
                                         p=[0.5, 0.2, 0.2, 0.1])
        attention = 0.7 + np.random.uniform(-0.3, 0.3)
        hour = 10 + i // 10
        
        exporter.record_frame(posture_score, posture_label, attention, hour)
        
        if i % 10 == 0:
            exporter.record_alert(np.random.choice([True, False], p=[0.7, 0.3]))
    
    session = exporter.end_session()
    exporter.finalize_session_stats()
    
    print("\nWeekly Summary:")
    print(json.dumps(exporter.get_weekly_summary(), indent=2))
    
    print("\nAchievements:")
    print(json.dumps(exporter.get_achievements_status(), indent=2))
    
    print("\nDashboard Export:")
    print(json.dumps(exporter.export_for_web(), indent=2)[:1000] + "...")
