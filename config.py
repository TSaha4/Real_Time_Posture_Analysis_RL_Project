import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PoseConfig:
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1
    smooth: bool = True
    enable_segmentation: bool = False


@dataclass
class PostureConfig:
    neck_angle_threshold: float = 20.0
    shoulder_diff_threshold: float = 20.0
    spine_inclination_threshold: float = 15.0
    forward_head_threshold: float = 25.0
    score_weights: dict = field(default_factory=lambda: {
        "neck_angle": 0.3,
        "shoulder_align": 0.25,
        "spine_inclination": 0.25,
        "symmetry": 0.2
    })


@dataclass
class RLConfig:
    state_size: int = 18
    action_size: int = 3
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    learning_rate: float = 0.001  # Start higher, will decay
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 128  # Increased from 64 for stability
    memory_size: int = 20000  # Increased from 10000
    target_update_freq: int = 10
    learning_freq: int = 4
    gradient_clip: float = 1.0
    lr_decay: float = 0.5  # Decay factor
    lr_decay_interval: int = 500  # Episodes between decay


@dataclass
class SystemConfig:
    decision_interval: float = 2.5
    calibration_frames: int = 20
    calibration_timeout: float = 30.0
    calibration_min_frames: int = 10
    cooldown_period: float = 3.0
    log_dir: str = "logs"
    model_dir: str = "models"
    save_interval: int = 100


@dataclass
class RewardConfig:
    posture_improve: float = 10.0
    alert_ignored: float = -5.0
    alert_fatigue: float = -3.0
    sustained_good: float = 0.5  # Reduced from 2.0 to prevent inflated rewards
    no_face_detected: float = -2.0
    posture_worsen: float = -5.0


@dataclass
class FeedbackConfig:
    window_name: str = "UPRYT - Posture Analysis"
    posture_good_color: Tuple[int, int, int] = (0, 255, 0)
    posture_bad_color: Tuple[int, int, int] = (0, 0, 255)
    posture_warning_color: Tuple[int, int, int] = (0, 255, 255)
    font_scale: float = 0.7
    sound_enabled: bool = False
    popup_enabled: bool = False


@dataclass
class AudioConfig:
    enabled: bool = False
    volume: float = 0.7
    cooldown: float = 3.0
    voice_enabled: bool = False
    fatigue_threshold: int = 10


@dataclass
class MultiCameraConfig:
    enabled: bool = False
    camera_ids: List[int] = field(default_factory=lambda: [0, 1])
    fusion_method: str = "weighted"


@dataclass
class OnlineLearningConfig:
    enabled: bool = False
    update_frequency: int = 100
    batch_size: int = 32
    learning_rate: float = 0.0001
    store_experience: bool = True
    min_experiences: int = 32
    max_buffer_size: int = 5000
    auto_adjust_lr: bool = True
    save_interval: int = 300


@dataclass
class PoseModelConfig:
    model_type: str = "pose_landmarker"
    use_holistic: bool = False
    use_movenet: bool = False
    enable_face: bool = True
    enable_hands: bool = False
    enable_attention_tracking: bool = False
    enable_hand_tracking: bool = False


@dataclass
class GamificationConfig:
    enabled: bool = True
    achievements_enabled: bool = True
    streaks_enabled: bool = True
    profiles_dir: str = "data/profiles"


@dataclass
class TrainingEnhancementsConfig:
    enable_curriculum: bool = True
    enable_domain_randomization: bool = True
    curriculum_stages: int = 3
    episodes_per_stage: int = 150
    randomization_strength: float = 0.3


@dataclass
class AudioEnhancementConfig:
    adaptive_volume: bool = True
    pattern_variety: bool = True
    cooldown_base: float = 3.0
    cooldown_low_priority: float = 4.5
    cooldown_high_priority: float = 2.0


class Config:
    pose = PoseConfig()
    posture = PostureConfig()
    rl = RLConfig()
    system = SystemConfig()
    reward = RewardConfig()
    feedback = FeedbackConfig()
    audio = AudioConfig()
    multi_camera = MultiCameraConfig()
    online_learning = OnlineLearningConfig()
    pose_model = PoseModelConfig()
    gamification = GamificationConfig()
    training = TrainingEnhancementsConfig()
    audio_enhancement = AudioEnhancementConfig()

    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.system.log_dir, exist_ok=True)
        os.makedirs(cls.system.model_dir, exist_ok=True)
        os.makedirs(cls.gamification.profiles_dir, exist_ok=True)


config = Config()
