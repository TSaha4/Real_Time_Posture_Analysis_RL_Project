import numpy as np
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib


@dataclass
class UserBaseline:
    neck_angle: float = 90.0
    shoulder_diff: float = 0.0
    spine_inclination: float = 0.0
    forward_head_y: float = 0.0
    shoulder_alignment: float = 0.0
    calibration_time: str = ""
    num_samples: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UserBaseline":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def compute_deviation(self, current_features: Dict[str, float]) -> float:
        deviations = []
        if "neck_angle" in current_features:
            deviations.append(abs(current_features["neck_angle"] - self.neck_angle))
        if "shoulder_diff" in current_features:
            deviations.append(abs(current_features["shoulder_diff"] - self.shoulder_diff))
        if "spine_inclination" in current_features:
            deviations.append(abs(current_features["spine_inclination"] - self.spine_inclination))
        if "forward_head_y" in current_features:
            deviations.append(abs(current_features["forward_head_y"] - self.forward_head_y))
        return np.mean(deviations) if deviations else 0.0


@dataclass
class UserProfile:
    user_id: str
    display_name: str = "User"
    created_at: str = ""
    last_session: str = ""
    baseline: UserBaseline = field(default_factory=UserBaseline)
    preferences: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    achievements: List[str] = field(default_factory=list)
    streak_data: Dict[str, Any] = field(default_factory=dict)
    sensitivity_settings: Dict[str, float] = field(default_factory=lambda: {
        "threshold_multiplier": 1.0,
        "alert_interval_modifier": 1.0,
        "fatigue_sensitivity": 0.5,
    })

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "last_session": self.last_session,
            "baseline": self.baseline.to_dict() if isinstance(self.baseline, UserBaseline) else self.baseline,
            "preferences": self.preferences,
            "statistics": self.statistics,
            "achievements": self.achievements,
            "streak_data": self.streak_data,
            "sensitivity_settings": self.sensitivity_settings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        baseline_data = data.get("baseline", {})
        baseline = UserBaseline.from_dict(baseline_data) if baseline_data else UserBaseline()
        return cls(
            user_id=data.get("user_id", ""),
            display_name=data.get("display_name", "User"),
            created_at=data.get("created_at", ""),
            last_session=data.get("last_session", ""),
            baseline=baseline,
            preferences=data.get("preferences", {}),
            statistics=data.get("statistics", {}),
            achievements=data.get("achievements", []),
            streak_data=data.get("streak_data", {}),
            sensitivity_settings=data.get("sensitivity_settings", {}),
        )


class UserProfileManager:
    def __init__(self, profiles_dir: str = "data/profiles"):
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)
        self.current_profile: Optional[UserProfile] = None
        self.all_profiles: Dict[str, UserProfile] = {}

    def create_profile(self, user_id: str, display_name: str = "User") -> UserProfile:
        profile = UserProfile(
            user_id=user_id,
            display_name=display_name,
            created_at=datetime.now().isoformat(),
            last_session=datetime.now().isoformat(),
            baseline=UserBaseline(calibration_time=datetime.now().isoformat()),
            streak_data={
                "current_streak": 0,
                "longest_streak": 0,
                "last_session_date": "",
                "sessions_today": 0,
            },
        )
        self.current_profile = profile
        self.save_profile(profile)
        return profile

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        if user_id in self.all_profiles:
            return self.all_profiles[user_id]
        
        filepath = self._get_profile_path(user_id)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                profile = UserProfile.from_dict(data)
                self.all_profiles[user_id] = profile
                return profile
            except Exception as e:
                print(f"Error loading profile: {e}")
        return None

    def save_profile(self, profile: UserProfile) -> bool:
        try:
            profile.last_session = datetime.now().isoformat()
            filepath = self._get_profile_path(profile.user_id)
            with open(filepath, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
            self.all_profiles[profile.user_id] = profile
            return True
        except Exception as e:
            print(f"Error saving profile: {e}")
            return False

    def update_baseline(self, user_id: str, features: Dict[str, float], num_samples: int = 30) -> bool:
        profile = self.get_profile(user_id)
        if not profile:
            profile = self.create_profile(user_id)
        
        baseline = UserBaseline(
            neck_angle=features.get("neck_angle", 90.0),
            shoulder_diff=features.get("shoulder_diff", 0.0),
            spine_inclination=features.get("spine_inclination", 0.0),
            forward_head_y=features.get("forward_head_y", 0.0),
            shoulder_alignment=features.get("shoulder_alignment", 0.0),
            calibration_time=datetime.now().isoformat(),
            num_samples=num_samples,
        )
        profile.baseline = baseline
        return self.save_profile(profile)

    def list_profiles(self) -> List[Dict[str, str]]:
        profiles = []
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('.json'):
                user_id = filename[:-5]
                filepath = os.path.join(self.profiles_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    profiles.append({
                        "user_id": user_id,
                        "display_name": data.get("display_name", "User"),
                        "last_session": data.get("last_session", "Never"),
                    })
                except Exception:
                    continue
        return profiles

    def delete_profile(self, user_id: str) -> bool:
        filepath = self._get_profile_path(user_id)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                if user_id in self.all_profiles:
                    del self.all_profiles[user_id]
                return True
            except Exception:
                return False
        return False

    def _get_profile_path(self, user_id: str) -> str:
        return os.path.join(self.profiles_dir, f"{user_id}.json")

    def generate_user_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{timestamp}_{np.random.randint(1000, 9999)}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]


class GamificationSystem:
    ACHIEVEMENTS = {
        "first_session": {
            "name": "First Steps",
            "description": "Complete your first session",
            "icon": "🎯",
            "threshold": 1,
        },
        "streak_3": {
            "name": "Getting Started",
            "description": "Maintain a 3-day streak",
            "icon": "🔥",
            "threshold": 3,
        },
        "streak_7": {
            "name": "Week Warrior",
            "description": "Maintain a 7-day streak",
            "icon": "💪",
            "threshold": 7,
        },
        "streak_30": {
            "name": "Monthly Master",
            "description": "Maintain a 30-day streak",
            "icon": "🏆",
            "threshold": 30,
        },
        "corrections_50": {
            "name": "Posture Pro",
            "description": "Achieve 50 successful corrections",
            "icon": "⭐",
            "threshold": 50,
        },
        "corrections_200": {
            "name": "Posture Expert",
            "description": "Achieve 200 successful corrections",
            "icon": "🌟",
            "threshold": 200,
        },
        "perfect_session": {
            "name": "Perfect Session",
            "description": "Complete a session with 90%+ good posture",
            "icon": "✨",
            "threshold": 90,
        },
        "no_alerts_5": {
            "name": "Self-Corrector",
            "description": "Maintain good posture for 5 minutes without alerts",
            "icon": "🧘",
            "threshold": 300,
        },
        "awareness_100": {
            "name": "Posture Aware",
            "description": "Log 100 sessions",
            "icon": "📊",
            "threshold": 100,
        },
        "improvement_10": {
            "name": "Rising Star",
            "description": "Improve your average posture score by 10%",
            "icon": "📈",
            "threshold": 10,
        },
    }

    def __init__(self, profile_manager: UserProfileManager = None):
        self.profile_manager = profile_manager or UserProfileManager()
        self.session_stats = {
            "corrections_made": 0,
            "alerts_sent": 0,
            "good_posture_time": 0,
            "total_time": 0,
            "sessions_without_alerts": 0,
            "perfect_session_flag": False,
        }
        self.current_session_start = None
        self.gamification_enabled = True

    def start_session(self):
        self.current_session_start = datetime.now()
        self.session_stats = {
            "corrections_made": 0,
            "alerts_sent": 0,
            "good_posture_time": 0,
            "total_time": 0,
            "sessions_without_alerts": 0,
            "perfect_session_flag": False,
        }

    def record_correction(self):
        self.session_stats["corrections_made"] += 1

    def record_alert(self):
        self.session_stats["alerts_sent"] += 1

    def record_posture_score(self, score: float, duration_seconds: float):
        self.session_stats["total_time"] += duration_seconds
        if score >= 0.7:
            self.session_stats["good_posture_time"] += duration_seconds
        if self.session_stats["total_time"] > 0:
            good_pct = (self.session_stats["good_posture_time"] / self.session_stats["total_time"]) * 100
            if good_pct >= 90:
                self.session_stats["perfect_session_flag"] = True

    def end_session(self, user_id: str) -> Dict[str, Any]:
        if not self.profile_manager or not user_id:
            return {"new_achievements": [], "stats": self.session_stats}

        profile = self.profile_manager.get_profile(user_id)
        if not profile:
            profile = self.profile_manager.create_profile(user_id)

        self._update_streak(profile)
        self._update_statistics(profile)
        new_achievements = self._check_achievements(profile)
        profile.achievements = list(set(profile.achievements + new_achievements))
        self.profile_manager.save_profile(profile)

        return {
            "new_achievements": new_achievements,
            "stats": self.session_stats,
            "current_streak": profile.streak_data.get("current_streak", 0),
            "total_achievements": len(profile.achievements),
        }

    def _update_streak(self, profile: UserProfile):
        today = datetime.now().date()
        last_session = profile.streak_data.get("last_session_date", "")
        
        if last_session:
            last_date = datetime.fromisoformat(last_session).date()
            days_diff = (today - last_date).days
            
            if days_diff == 1:
                profile.streak_data["current_streak"] += 1
            elif days_diff > 1:
                profile.streak_data["current_streak"] = 1
        else:
            profile.streak_data["current_streak"] = 1
        
        profile.streak_data["longest_streak"] = max(
            profile.streak_data.get("longest_streak", 0),
            profile.streak_data["current_streak"]
        )
        profile.streak_data["last_session_date"] = datetime.now().isoformat()
        
        sessions_today = profile.streak_data.get("sessions_today", 0)
        profile.streak_data["sessions_today"] = sessions_today + 1

    def _update_statistics(self, profile: UserProfile):
        stats = profile.statistics
        
        stats["total_sessions"] = stats.get("total_sessions", 0) + 1
        stats["total_corrections"] = stats.get("total_corrections", 0) + self.session_stats["corrections_made"]
        stats["total_alerts"] = stats.get("total_alerts", 0) + self.session_stats["alerts_sent"]
        
        total_time = stats.get("total_time", 0) + self.session_stats["total_time"]
        good_time = stats.get("good_posture_time", 0) + self.session_stats["good_posture_time"]
        
        stats["total_time"] = total_time
        stats["good_posture_time"] = good_time
        
        if total_time > 0:
            stats["avg_posture_score"] = (good_time / total_time) * 100
        
        if self.session_stats["perfect_session_flag"]:
            stats["perfect_sessions"] = stats.get("perfect_sessions", 0) + 1

    def _check_achievements(self, profile: UserProfile) -> List[str]:
        new_achievements = []
        stats = profile.statistics
        
        if stats.get("total_sessions", 0) >= self.ACHIEVEMENTS["first_session"]["threshold"]:
            if "first_session" not in profile.achievements:
                new_achievements.append("first_session")
        
        streak = profile.streak_data.get("current_streak", 0)
        if streak >= 3 and "streak_3" not in profile.achievements:
            new_achievements.append("streak_3")
        if streak >= 7 and "streak_7" not in profile.achievements:
            new_achievements.append("streak_7")
        if streak >= 30 and "streak_30" not in profile.achievements:
            new_achievements.append("streak_30")
        
        corrections = stats.get("total_corrections", 0)
        if corrections >= 50 and "corrections_50" not in profile.achievements:
            new_achievements.append("corrections_50")
        if corrections >= 200 and "corrections_200" not in profile.achievements:
            new_achievements.append("corrections_200")
        
        if stats.get("perfect_sessions", 0) >= 1 and "perfect_session" not in profile.achievements:
            new_achievements.append("perfect_session")
        
        if stats.get("total_sessions", 0) >= 100 and "awareness_100" not in profile.achievements:
            new_achievements.append("awareness_100")
        
        return new_achievements

    def get_achievement_info(self, achievement_id: str) -> Optional[Dict]:
        return self.ACHIEVEMENTS.get(achievement_id)

    def get_all_achievements(self) -> List[Dict]:
        return list(self.ACHIEVEMENTS.values())


class TrendAnalyzer:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.session_history: List[Dict] = []
        self.daily_stats: Dict[str, Dict] = defaultdict(lambda: {
            "sessions": 0,
            "avg_score": 0.0,
            "total_corrections": 0,
            "total_alerts": 0,
            "total_time": 0,
            "good_time": 0,
        })

    def add_session(self, session_data: Dict):
        self.session_history.append(session_data)
        
        date = datetime.now().strftime("%Y-%m-%d")
        self.daily_stats[date]["sessions"] += 1
        self.daily_stats[date]["total_corrections"] += session_data.get("corrections", 0)
        self.daily_stats[date]["total_alerts"] += session_data.get("alerts", 0)
        self.daily_stats[date]["total_time"] += session_data.get("duration", 0)
        self.daily_stats[date]["good_time"] += session_data.get("good_time", 0)
        
        if self.daily_stats[date]["sessions"] > 0:
            self.daily_stats[date]["avg_score"] = (
                self.daily_stats[date]["good_time"] / max(1, self.daily_stats[date]["total_time"])
            ) * 100

    def get_weekly_trend(self) -> Dict[str, Any]:
        today = datetime.now().date()
        week_stats = []
        
        for i in range(7):
            date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            if date in self.daily_stats:
                week_stats.append({
                    "date": date,
                    "sessions": self.daily_stats[date]["sessions"],
                    "avg_score": self.daily_stats[date]["avg_score"],
                    "corrections": self.daily_stats[date]["total_corrections"],
                    "time_minutes": self.daily_stats[date]["total_time"] / 60,
                })
            else:
                week_stats.append({
                    "date": date,
                    "sessions": 0,
                    "avg_score": 0,
                    "corrections": 0,
                    "time_minutes": 0,
                })
        
        return {
            "daily_breakdown": week_stats,
            "summary": {
                "total_sessions": sum(d["sessions"] for d in week_stats),
                "avg_score": np.mean([d["avg_score"] for d in week_stats if d["sessions"] > 0]) if any(d["sessions"] > 0 for d in week_stats) else 0,
                "total_corrections": sum(d["corrections"] for d in week_stats),
                "total_time_minutes": sum(d["time_minutes"] for d in week_stats),
            }
        }

    def get_trend_direction(self) -> str:
        if len(self.session_history) < 10:
            return "insufficient_data"
        
        recent_scores = [s.get("avg_score", 50) for s in self.session_history[-10:]]
        
        if len(recent_scores) >= 5:
            early_avg = np.mean(recent_scores[:5])
            late_avg = np.mean(recent_scores[-5:])
            
            if late_avg - early_avg > 5:
                return "improving"
            elif early_avg - late_avg > 5:
                return "declining"
        
        return "stable"

    def get_insights(self) -> List[str]:
        insights = []
        trend = self.get_trend_direction()
        
        if trend == "improving":
            insights.append("Your posture has been improving! Keep up the good work.")
        elif trend == "declining":
            insights.append("Your posture scores have been declining. Try taking more breaks.")
        else:
            insights.append("Your posture is staying consistent.")
        
        recent = self.session_history[-7:] if len(self.session_history) >= 7 else self.session_history
        if recent:
            avg_score = np.mean([s.get("avg_score", 0) for s in recent])
            if avg_score < 50:
                insights.append("Consider setting up your workstation for better ergonomics.")
            elif avg_score > 80:
                insights.append("Excellent posture awareness! You're a posture champion.")
        
        return insights

    def export_data(self, filepath: str) -> bool:
        try:
            data = {
                "session_history": self.session_history,
                "daily_stats": dict(self.daily_stats),
            }
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False


class AdaptiveSensitivity:
    def __init__(self, base_threshold: float = 1.0):
        self.base_threshold = base_threshold
        self.current_threshold = base_threshold
        self.adjustment_history: List[Dict] = []
        self.performance_scores: List[float] = []

    def update_performance(self, alert_success_rate: float, user_complaint_level: float = 0):
        score = alert_success_rate - (user_complaint_level * 0.3)
        self.performance_scores.append(score)
        
        if len(self.performance_scores) > 20:
            self.performance_scores.pop(0)
        
        self._adjust_threshold()

    def _adjust_threshold(self):
        if len(self.performance_scores) < 10:
            return
        
        recent_avg = np.mean(self.performance_scores[-10:])
        overall_avg = np.mean(self.performance_scores)
        
        if recent_avg < overall_avg - 0.1:
            self.current_threshold *= 1.1
            self.adjustment_history.append({
                "timestamp": datetime.now().isoformat(),
                "reason": "performance_drop",
                "change": 1.1,
            })
        elif recent_avg > overall_avg + 0.1:
            self.current_threshold *= 0.95
            self.adjustment_history.append({
                "timestamp": datetime.now().isoformat(),
                "reason": "performance_improvement",
                "change": 0.95,
            })
        
        self.current_threshold = max(0.5, min(2.0, self.current_threshold))

    def get_threshold(self) -> float:
        return self.current_threshold

    def reset(self):
        self.current_threshold = self.base_threshold
        self.performance_scores = []
        self.adjustment_history = []


def create_user_profile_manager(profiles_dir: str = "data/profiles") -> UserProfileManager:
    return UserProfileManager(profiles_dir)


def create_gamification_system(profile_manager: UserProfileManager = None) -> GamificationSystem:
    return GamificationSystem(profile_manager)


if __name__ == "__main__":
    print("Testing User Profile & Gamification System...")
    
    profile_manager = UserProfileManager()
    
    profile = profile_manager.create_profile("test_user", "Test User")
    print(f"Created profile: {profile.user_id}")
    
    profile_manager.update_baseline("test_user", {
        "neck_angle": 88.5,
        "shoulder_diff": 3.2,
        "spine_inclination": -1.5,
        "forward_head_y": 12.3,
    }, num_samples=30)
    
    gamification = GamificationSystem(profile_manager)
    gamification.start_session()
    
    for _ in range(10):
        gamification.record_correction()
        gamification.record_alert()
        gamification.record_posture_score(0.75, 30.0)
    
    result = gamification.end_session("test_user")
    print(f"Session result: {result}")
    
    print("\nGamification achievements:")
    for ach in gamification.get_all_achievements():
        print(f"  {ach['icon']} {ach['name']}: {ach['description']}")
    
    print("\nTesting Trend Analysis...")
    analyzer = TrendAnalyzer()
    for i in range(10):
        analyzer.add_session({
            "date": datetime.now().isoformat(),
            "avg_score": 60 + i * 2 + np.random.randint(-5, 5),
            "corrections": np.random.randint(5, 20),
            "alerts": np.random.randint(10, 30),
            "duration": np.random.randint(600, 3600),
            "good_time": np.random.randint(300, 1800),
        })
    
    trend = analyzer.get_trend_direction()
    insights = analyzer.get_insights()
    print(f"Trend: {trend}")
    for insight in insights:
        print(f"  - {insight}")
