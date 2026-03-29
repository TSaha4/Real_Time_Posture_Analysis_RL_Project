"""
UPRYT Mobile - Real-Time Posture Analysis App
Kivy-based mobile application for Android
"""

__version__ = "1.0.0"

import os
import time
import numpy as np
from functools import partial

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.switch import Switch
from kivy.uix.slider import Slider
from kivy.uix.spinner import Spinner
from kivy.uix.checkbox import CheckBox
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.textinput import TextInput
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, ObjectProperty
from kivy.clock import Clock
from kivy.graphics import Rectangle, Color, Ellipse
from kivy.core.image import Image as CoreImage
from kivy.core.window import Window
from kivy.logger import Logger

# Camera imports - handled gracefully
try:
    import cv2
    import mediapipe as mp
    CAMERA_AVAILABLE = True
except ImportError as e:
    CAMERA_AVAILABLE = False
    Logger.warning(f"Camera/ML libraries not available: {e}")


class PostureMetrics:
    """Holds current posture metrics"""
    posture_score = 0.0
    attention_score = 1.0
    hand_score = 1.0
    posture_label = "UNKNOWN"
    attention_state = "UNKNOWN"
    hand_state = "UNKNOWN"
    is_typing = False
    fps = 0.0
    combined_score = 0.0


class CameraWidget(BoxLayout):
    """Custom camera display widget"""
    texture = ObjectProperty(None)
    frame_count = 0
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.posture_metrics = PostureMetrics()
        self.running = False
        self.cap = None
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        
    def start_camera(self):
        """Initialize camera"""
        if not CAMERA_AVAILABLE:
            return False
            
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1)
            
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.running = True
                Clock.schedule_interval(self.update_frame, 1/30)
                return True
        except Exception as e:
            Logger.error(f"Camera error: {e}")
        return False
    
    def stop_camera(self):
        """Stop camera"""
        self.running = False
        Clock.unschedule(self.update_frame)
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
    
    def update_frame(self, dt):
        """Update camera frame"""
        if not self.running or not self.cap:
            return
            
        try:
            ret, frame = self.cap.read()
            if ret:
                # Process frame (placeholder for posture detection)
                self._process_frame(frame)
                
                # Convert frame for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 0)  # Flip for display
                
                # Create texture
                buf = frame.tobytes()
                texture = CoreImage.blit_buffer(
                    buf, colorfmt='rgb', bufferfmt='ubyte',
                    size=(frame.shape[1], frame.shape[0])
                ).texture
                
                self.texture = texture
                
                # Calculate FPS
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = current_time
                    self.posture_metrics.fps = self.current_fps
                    
        except Exception as e:
            Logger.error(f"Frame update error: {e}")
    
    def _process_frame(self, frame):
        """Process frame for posture detection"""
        # This is where you'd integrate the posture detection
        # For now, simulate posture metrics
        pass


class MainScreen(Screen):
    """Main application screen"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera_widget = None
        self.detector = None
        self.is_monitoring = False
        self.posture_metrics = PostureMetrics()
    
    def on_enter(self):
        """Called when entering screen"""
        pass
    
    def on_leave(self):
        """Called when leaving screen"""
        self.stop_monitoring()
    
    def toggle_monitoring(self):
        """Toggle posture monitoring"""
        if self.is_monitoring:
            self.stop_monitoring()
        else:
            self.start_monitoring()
    
    def start_monitoring(self):
        """Start posture monitoring"""
        if not CAMERA_AVAILABLE:
            self.show_error("Camera not available. Install on device.")
            return
            
        self.is_monitoring = True
        self.ids.start_button.text = "Stop Monitoring"
        self.ids.status_label.text = "Monitoring..."
        
        # Initialize detector
        if self.detector is None:
            from posture_detector import PostureDetector
            self.detector = PostureDetector()
        
        # Start camera
        if self.camera_widget:
            self.camera_widget.start_camera()
            
            # Start posture processing
            Clock.schedule_interval(self.process_posture, 1/15)
    
    def stop_monitoring(self):
        """Stop posture monitoring"""
        self.is_monitoring = False
        self.ids.start_button.text = "Start Monitoring"
        self.ids.status_label.text = "Ready"
        
        if self.camera_widget:
            self.camera_widget.stop_camera()
        
        Clock.unschedule(self.process_posture)
    
    def process_posture(self, dt):
        """Process current frame for posture"""
        if not self.is_monitoring or not self.camera_widget or not self.detector:
            return
            
        frame = self.get_current_frame()
        if frame is not None:
            metrics = self.detector.analyze(frame)
            self.update_metrics(metrics)
    
    def get_current_frame(self):
        """Get current camera frame"""
        if self.camera_widget and self.camera_widget.cap:
            ret, frame = self.camera_widget.cap.read()
            if ret:
                return frame
        return None
    
    def update_metrics(self, metrics):
        """Update displayed metrics"""
        if not metrics:
            return
            
        self.posture_metrics = metrics
        
        # Update UI
        score = int(metrics.posture_score * 100)
        self.ids.score_label.text = f"{score}%"
        self.ids.posture_label.text = metrics.posture_label
        self.ids.attention_label.text = f"Attention: {metrics.attention_state}"
        self.ids.fps_label.text = f"FPS: {metrics.fps:.0f}"
        
        # Update score bar
        self.ids.score_bar.value = score
        
        # Update status color
        if metrics.posture_score >= 0.7:
            self.ids.status_indicator.canvas.before.children[0].rgb = (0.2, 0.8, 0.2)
        elif metrics.posture_score >= 0.4:
            self.ids.status_indicator.canvas.before.children[0].rgb = (0.9, 0.9, 0.2)
        else:
            self.ids.status_indicator.canvas.before.children[0].rgb = (0.9, 0.2, 0.2)
    
    def show_error(self, message):
        """Show error popup"""
        popup = Popup(
            title='Error',
            content=Label(text=message),
            size_hint=(0.8, 0.3)
        )
        popup.open()
    
    def play_alert(self):
        """Play audio alert"""
        try:
            from kivy.core.audio import SoundLoader
            # Alert sound would go here
            pass
        except:
            pass


class SettingsScreen(Screen):
    """Settings screen"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def on_enter(self):
        """Load settings when entering"""
        self.load_settings()
    
    def load_settings(self):
        """Load settings from storage"""
        try:
            from kivy.storage.jsonstore import JsonStore
            store = JsonStore('settings.json')
            
            if store.exists('sensitivity'):
                self.ids.sensitivity_slider.value = store.get('sensitivity')['value']
            if store.exists('audio_enabled'):
                self.ids.audio_switch.active = store.get('audio_enabled')['value']
            if store.exists('attention_enabled'):
                self.ids.attention_switch.active = store.get('attention_enabled')['value']
        except:
            pass
    
    def save_settings(self):
        """Save settings to storage"""
        try:
            from kivy.storage.jsonstore import JsonStore
            store = JsonStore('settings.json')
            
            store.put('sensitivity', value=self.ids.sensitivity_slider.value)
            store.put('audio_enabled', value=self.ids.audio_switch.active)
            store.put('attention_enabled', value=self.ids.attention_switch.active)
        except:
            pass
    
    def reset_to_defaults(self):
        """Reset settings to defaults"""
        self.ids.sensitivity_slider.value = 50
        self.ids.audio_switch.active = True
        self.ids.attention_switch.active = True
        self.save_settings()


class StatsScreen(Screen):
    """Statistics screen"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session_data = {
            'total_time': 0,
            'good_posture_time': 0,
            'alerts_sent': 0,
            'corrections': 0,
            'sessions': 0
        }
    
    def on_enter(self):
        """Load stats when entering"""
        self.load_stats()
    
    def load_stats(self):
        """Load and display stats"""
        try:
            from kivy.storage.jsonstore import JsonStore
            store = JsonStore('stats.json')
            
            if store.exists('data'):
                self.session_data = store.get('data')['value']
            
            # Update UI
            self.ids.total_time_label.text = f"{self.session_data['total_time']:.0f} min"
            self.ids.sessions_label.text = f"{self.session_data['sessions']}"
            self.ids.alerts_label.text = f"{self.session_data['alerts_sent']}"
            
            if self.session_data['total_time'] > 0:
                good_pct = (self.session_data['good_posture_time'] / 
                          (self.session_data['total_time'] * 60)) * 100
                self.ids.good_posture_label.text = f"{good_pct:.0f}%"
            else:
                self.ids.good_posture_label.text = "0%"
                
        except:
            pass
    
    def reset_stats(self):
        """Reset all statistics"""
        self.session_data = {
            'total_time': 0,
            'good_posture_time': 0,
            'alerts_sent': 0,
            'corrections': 0,
            'sessions': 0
        }
        try:
            from kivy.storage.jsonstore import JsonStore
            store = JsonStore('stats.json')
            store.put('data', value=self.session_data)
        except:
            pass
        self.load_stats()


class UPRYTApp(App):
    """Main UPRYT Application"""
    
    def build(self):
        """Build the application"""
        # Create screen manager
        sm = ScreenManager()
        
        # Add screens
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(SettingsScreen(name='settings'))
        sm.add_widget(StatsScreen(name='stats'))
        
        return sm
    
    def on_start(self):
        """Called when app starts"""
        Logger.info("UPRYT App started")
    
    def on_pause(self):
        """Called when app is paused"""
        return True  # Allow pausing
    
    def on_resume(self):
        """Called when app resumes"""
        pass


if __name__ == '__main__':
    UPRYTApp().run()
