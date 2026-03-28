import os
import time
import threading
from typing import Optional
from enum import Enum


class AlertSound(Enum):
    SUBTLE = "subtle"
    STRONG = "strong"
    GOOD = "good"
    FATIGUE = "fatigue"


class AudioAlertSystem:
    def __init__(self, enabled: bool = True, volume: float = 0.7):
        self.enabled = enabled
        self.volume = volume
        self.last_alert_time = 0.0
        self.cooldown = 3.0
        self.alert_history = []
        self.fatigue_warning_played = False
        
    def play_alert(self, sound_type: AlertSound) -> bool:
        if not self.enabled:
            return False
        
        current_time = time.time()
        if current_time - self.last_alert_time < self.cooldown:
            return False
        
        self.last_alert_time = current_time
        self.alert_history.append(current_time)
        
        self._cleanup_history()
        
        if len(self.alert_history) >= 10:
            recent = [t for t in self.alert_history if current_time - t < 30]
            if len(recent) >= 10 and not self.fatigue_warning_played:
                self._play_fatigue_warning()
                self.fatigue_warning_played = True
                return True
        
        thread = threading.Thread(target=self._play_sound_async, args=(sound_type,))
        thread.daemon = True
        thread.start()
        return True
    
    def _cleanup_history(self):
        current_time = time.time()
        self.alert_history = [t for t in self.alert_history if current_time - t < 60]
    
    def _play_sound_async(self, sound_type: AlertSound):
        try:
            if os.name == "nt":
                self._play_windows(sound_type)
            else:
                self._play_linux(sound_type)
        except Exception:
            pass
    
    def _play_windows(self, sound_type: AlertSound):
        import winsound
        
        sounds = {
            AlertSound.SUBTLE: (600, 150),
            AlertSound.STRONG: (1000, 300),
            AlertSound.GOOD: (800, 200),
            AlertSound.FATIGUE: (500, 500),
        }
        
        freq, duration = sounds.get(sound_type, (800, 200))
        
        if sound_type == AlertSound.GOOD:
            winsound.Beep(freq, duration)
            time.sleep(0.1)
            winsound.Beep(freq + 200, duration)
        elif sound_type == AlertSound.FATIGUE:
            for _ in range(3):
                winsound.Beep(freq, 150)
                time.sleep(0.1)
        else:
            winsound.Beep(freq, duration)
    
    def _play_linux(self, sound_type: AlertSound):
        try:
            import subprocess
            
            freq_map = {
                AlertSound.SUBTLE: 600,
                AlertSound.STRONG: 1000,
                AlertSound.GOOD: 800,
                AlertSound.FATIGUE: 500,
            }
            
            freq = freq_map.get(sound_type, 800)
            duration = 0.2 if sound_type != AlertSound.FATIGUE else 0.5
            
            subprocess.run(
                ["play", "-n", "-q", "sine", str(freq), "-d", str(duration)],
                capture_output=True
            )
        except:
            pass
    
    def _play_fatigue_warning(self):
        try:
            if os.name == "nt":
                import winsound
                for i in range(3):
                    winsound.Beep(400, 200)
                    time.sleep(0.15)
        except:
            pass
    
    def get_alert_frequency(self, window_seconds: int = 60) -> float:
        current_time = time.time()
        recent = [t for t in self.alert_history if current_time - t < window_seconds]
        return len(recent) / (window_seconds / 60)
    
    def is_fatigued(self, threshold: float = 10) -> bool:
        return self.get_alert_frequency(60) >= threshold
    
    def reset(self):
        self.alert_history = []
        self.fatigue_warning_played = False
        self.last_alert_time = 0.0


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
