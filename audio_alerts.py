import os
import time
import threading
import random
from typing import Optional, List, Dict, Callable
from enum import Enum
import numpy as np


class AlertSound(Enum):
    SUBTLE = "subtle"
    STRONG = "strong"
    GOOD = "good"
    FATIGUE = "fatigue"
    WARNING = "warning"
    ENCOURAGEMENT = "encouragement"


class AlertPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class AudioPattern:
    def __init__(self, frequencies: List[int], durations: List[int], gaps: List[float] = None):
        self.frequencies = frequencies
        self.durations = durations
        self.gaps = gaps or [0.05] * len(frequencies)
        
    def __len__(self):
        return len(self.frequencies)
    
    def total_duration(self) -> float:
        return sum(self.durations) / 1000.0 + sum(self.gaps)


class AlertPatternLibrary:
    SUBTLE_PATTERNS = [
        AudioPattern([500], [100], [0.1]),
        AudioPattern([550], [120], [0.08]),
        AudioPattern([480], [90], [0.12]),
        AudioPattern([520, 580], [80, 60], [0.1, 0.05]),
        AudioPattern([450], [110], [0.15]),
    ]
    
    STRONG_PATTERNS = [
        AudioPattern([800, 1000], [150, 100], [0.05, 0.08]),
        AudioPattern([1000, 800, 1000], [100, 50, 100], [0.05, 0.03, 0.05]),
        AudioPattern([900, 1100], [120, 80], [0.03, 0.1]),
        AudioPattern([1200], [200], [0.15]),
        AudioPattern([850, 850, 1100], [80, 80, 120], [0.05, 0.05, 0.1]),
    ]
    
    GOOD_PATTERNS = [
        AudioPattern([523, 659, 784], [100, 100, 150], [0.1, 0.1, 0.1]),
        AudioPattern([659, 784], [120, 180], [0.08, 0.15]),
        AudioPattern([880, 1047], [100, 150], [0.1, 0.12]),
        AudioPattern([784, 988], [150, 100], [0.1, 0.1]),
    ]
    
    FATIGUE_PATTERNS = [
        AudioPattern([400, 350, 400, 350], [150, 100, 150, 100], [0.1] * 4),
        AudioPattern([300, 400, 300], [200, 150, 200], [0.15] * 3),
    ]
    
    WARNING_PATTERNS = [
        AudioPattern([600, 600], [80, 80], [0.1, 0.15]),
        AudioPattern([700], [150], [0.2]),
        AudioPattern([550, 650, 550], [60, 60, 60], [0.08] * 3),
    ]
    
    ENCOURAGEMENT_PATTERNS = [
        AudioPattern([523, 659], [80, 120], [0.15, 0.15]),
        AudioPattern([784], [100], [0.1]),
        AudioPattern([659, 523], [100, 80], [0.1, 0.1]),
    ]
    
    @classmethod
    def get_pattern(cls, sound_type: AlertSound) -> AudioPattern:
        if sound_type == AlertSound.SUBTLE:
            return random.choice(cls.SUBTLE_PATTERNS)
        elif sound_type == AlertSound.STRONG:
            return random.choice(cls.STRONG_PATTERNS)
        elif sound_type == AlertSound.GOOD:
            return random.choice(cls.GOOD_PATTERNS)
        elif sound_type == AlertSound.FATIGUE:
            return random.choice(cls.FATIGUE_PATTERNS)
        elif sound_type == AlertSound.WARNING:
            return random.choice(cls.WARNING_PATTERNS)
        elif sound_type == AlertSound.ENCOURAGEMENT:
            return random.choice(cls.ENCOURAGEMENT_PATTERNS)
        return cls.SUBTLE_PATTERNS[0]


class AudioAlertSystem:
    def __init__(self, enabled: bool = True, volume: float = 0.7, 
                 adaptive_volume: bool = True, pattern_variety: bool = True):
        self.enabled = enabled
        self.volume = volume
        self.adaptive_volume = adaptive_volume
        self.pattern_variety = pattern_variety
        self.last_alert_time = 0.0
        self.cooldown = 3.0
        self.alert_history: List[Dict] = []
        self.fatigue_warning_played = False
        self.consecutive_same_sound = 0
        self.last_sound_type = None
        self.base_volume = volume
        self.alert_count = 0
        self.session_start = time.time()
        self.custom_callback: Optional[Callable] = None
        self.intensity_level = 1.0
        self.alert_patterns = AlertPatternLibrary()
        
    def play_alert(self, sound_type: AlertSound, priority: AlertPriority = AlertPriority.NORMAL,
                   intensity: float = 1.0) -> bool:
        if not self.enabled:
            return False
        
        current_time = time.time()
        
        cooldown = self.cooldown
        if priority == AlertPriority.HIGH:
            cooldown *= 0.7
        elif priority == AlertPriority.URGENT:
            cooldown *= 0.5
        elif priority == AlertPriority.LOW:
            cooldown *= 1.5
        
        if current_time - self.last_alert_time < cooldown:
            return False
        
        self.last_alert_time = current_time
        self.alert_count += 1
        
        alert_info = {
            "type": sound_type,
            "priority": priority,
            "timestamp": current_time,
            "intensity": intensity,
        }
        self.alert_history.append(alert_info)
        
        self._cleanup_history()
        self._update_adaptive_settings()
        
        if self._should_play_fatigue_warning():
            self._play_fatigue_warning()
            self.fatigue_warning_played = True
            return True
        
        if sound_type == self.last_sound_type and self.pattern_variety:
            self.consecutive_same_sound += 1
        else:
            self.consecutive_same_sound = 0
        self.last_sound_type = sound_type
        
        thread = threading.Thread(target=self._play_sound_async, 
                                  args=(sound_type, priority, intensity))
        thread.daemon = True
        thread.start()
        return True
    
    def _cleanup_history(self):
        current_time = time.time()
        cutoff = current_time - 120
        self.alert_history = [a for a in self.alert_history if a["timestamp"] > cutoff]
    
    def _update_adaptive_settings(self):
        if not self.adaptive_volume:
            return
        
        if len(self.alert_history) >= 5:
            recent_alerts = [a for a in self.alert_history 
                           if time.time() - a["timestamp"] < 60]
            
            alert_rate = len(recent_alerts)
            
            if alert_rate > 10:
                self.volume = self.base_volume * 0.7
                self.intensity_level = 0.8
            elif alert_rate > 5:
                self.volume = self.base_volume * 0.85
                self.intensity_level = 0.9
            else:
                self.volume = self.base_volume
                self.intensity_level = 1.0
    
    def _should_play_fatigue_warning(self) -> bool:
        current_time = time.time()
        recent = [t for t in self.alert_history if current_time - t["timestamp"] < 30]
        
        if len(recent) >= 8 and not self.fatigue_warning_played:
            recent_types = [t["type"] for t in recent[-5:]]
            if len(set(recent_types)) <= 2:
                return True
        return False
    
    def _play_sound_async(self, sound_type: AlertSound, priority: AlertPriority, 
                         intensity: float):
        try:
            if os.name == "nt":
                self._play_windows(sound_type, priority, intensity)
            else:
                self._play_linux(sound_type, priority, intensity)
            
            if self.custom_callback:
                self.custom_callback(sound_type, priority)
        except Exception:
            pass
    
    def _play_windows(self, sound_type: AlertSound, priority: AlertPriority, 
                      intensity: float):
        import winsound
        
        pattern = self.alert_patterns.get_pattern(sound_type)
        
        if self.consecutive_same_sound > 2 and self.pattern_variety:
            pattern = self.alert_patterns.get_pattern(
                random.choice([AlertSound.SUBTLE, AlertSound.STRONG])
            )
        
        effective_volume = self.volume * intensity * self.intensity_level
        
        for i, (freq, duration_ms) in enumerate(zip(pattern.frequencies, pattern.durations)):
            adjusted_freq = int(freq * (1 + (intensity - 1) * 0.1))
            adjusted_duration = int(duration_ms * effective_volume)
            adjusted_duration = max(30, min(500, adjusted_duration))
            
            gap = pattern.gaps[i] if i < len(pattern.gaps) else 0.05
            gap = gap * (2 - effective_volume)
            
            try:
                winsound.Beep(adjusted_freq, adjusted_duration)
                if gap > 0:
                    time.sleep(gap)
            except Exception:
                pass
    
    def _play_linux(self, sound_type: AlertSound, priority: AlertPriority, 
                    intensity: float):
        try:
            import subprocess
            
            pattern = self.alert_patterns.get_pattern(sound_type)
            freq = pattern.frequencies[0]
            duration = pattern.durations[0] / 1000.0
            
            effective_volume = self.volume * intensity * self.intensity_level
            
            for i, (f, d) in enumerate(zip(pattern.frequencies, pattern.durations)):
                adjusted_d = d / 1000.0 * effective_volume
                subprocess.run(
                    ["play", "-n", "-q", "sine", str(int(f * (1 + (intensity - 1) * 0.1))), 
                     "-d", str(adjusted_d)],
                    capture_output=True
                )
                if i < len(pattern.gaps):
                    time.sleep(pattern.gaps[i])
        except Exception:
            pass
    
    def _play_fatigue_warning(self):
        try:
            if os.name == "nt":
                import winsound
                for i in range(3):
                    winsound.Beep(400, 200)
                    time.sleep(0.15)
                time.sleep(0.3)
                winsound.Beep(500, 100)
        except:
            pass
    
    def play_melody(self, melody_type: str = "success"):
        if not self.enabled:
            return
        
        if melody_type == "success":
            pattern = AlertPatternLibrary.GOOD_PATTERNS[0]
        elif melody_type == "encouragement":
            pattern = AlertPatternLibrary.ENCOURAGEMENT_PATTERNS[0]
        else:
            return
        
        def play():
            if os.name == "nt":
                import winsound
                for freq, dur in zip(pattern.frequencies, pattern.durations):
                    winsound.Beep(freq, dur)
                    time.sleep(0.1)
        
        thread = threading.Thread(target=play)
        thread.daemon = True
        thread.start()
    
    def get_alert_frequency(self, window_seconds: int = 60) -> float:
        current_time = time.time()
        recent = [t for t in self.alert_history if current_time - t["timestamp"] < window_seconds]
        return len(recent) / (window_seconds / 60)
    
    def is_fatigued(self, threshold: float = 10) -> bool:
        return self.get_alert_frequency(60) >= threshold
    
    def get_alert_stats(self) -> Dict:
        if not self.alert_history:
            return {
                "total_alerts": 0,
                "alert_rate_per_min": 0.0,
                "session_duration_min": 0.0,
                "avg_priority": 0,
            }
        
        current_time = time.time()
        session_duration = (current_time - self.session_start) / 60
        
        priorities = [a["priority"].value for a in self.alert_history]
        
        return {
            "total_alerts": len(self.alert_history),
            "alert_rate_per_min": len(self.alert_history) / max(0.1, session_duration),
            "session_duration_min": session_duration,
            "avg_priority": np.mean(priorities) if priorities else 0,
            "recent_rate_30s": self.get_alert_frequency(30),
            "recent_rate_60s": self.get_alert_frequency(60),
        }
    
    def set_volume(self, volume: float):
        self.volume = max(0.0, min(1.0, volume))
        self.base_volume = self.volume
    
    def set_cooldown(self, cooldown: float):
        self.cooldown = max(0.5, cooldown)
    
    def set_custom_callback(self, callback: Callable):
        self.custom_callback = callback
    
    def reset(self):
        self.alert_history = []
        self.fatigue_warning_played = False
        self.last_alert_time = 0.0
        self.alert_count = 0
        self.session_start = time.time()
        self.consecutive_same_sound = 0
        self.last_sound_type = None
        self.volume = self.base_volume
        self.intensity_level = 1.0


class VoiceAlertSystem:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.last_announcement_time = 0.0
        self.announcement_cooldown = 10.0
        
    def announce(self, message: str) -> bool:
        if not self.enabled:
            return False
        
        current_time = time.time()
        if current_time - self.last_announcement_time < self.announcement_cooldown:
            return False
        
        self.last_announcement_time = current_time
        
        thread = threading.Thread(target=self._speak_async, args=(message,))
        thread.daemon = True
        thread.start()
        return True
    
    def _speak_async(self, message: str):
        try:
            if os.name == "nt":
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(message)
                engine.runAndWait()
            else:
                import subprocess
                subprocess.run(["espeak", message], capture_output=True)
        except:
            pass
    
    def announce_posture_fix(self, issue: str):
        messages = {
            "slouching": "Please sit up straight. Roll your shoulders back.",
            "forward_head": "Move your head back. Your ears should align with your shoulders.",
            "leaning": "Sit evenly. Balance your weight on both hips.",
        }
        message = messages.get(issue, "Please fix your posture.")
        self.announce(message)


def create_audio_alert(sound_type: str = "subtle") -> Optional[AlertSound]:
    sound_map = {
        "subtle": AlertSound.SUBTLE,
        "strong": AlertSound.STRONG,
        "good": AlertSound.GOOD,
        "fatigue": AlertSound.FATIGUE,
    }
    return sound_map.get(sound_type.lower())


if __name__ == "__main__":
    audio = AudioAlertSystem(enabled=True)
    print("Testing audio alerts...")
    audio.play_alert(AlertSound.SUBTLE)
    time.sleep(1)
    audio.play_alert(AlertSound.STRONG)
    time.sleep(1)
    audio.play_alert(AlertSound.GOOD)
    print("Alert frequency:", audio.get_alert_frequency())
