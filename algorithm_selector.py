import time
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import deque


@dataclass
class AlgorithmMetrics:
    alerts_sent: int = 0
    corrections_made: int = 0
    ignored_alerts: int = 0
    posture_scores: List[float] = field(default_factory=list)
    last_score: float = 0.8
    evaluation_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    @property
    def correction_rate(self) -> float:
        if self.alerts_sent == 0:
            return 0.0
        return self.corrections_made / max(1, self.alerts_sent)
    
    @property
    def avg_posture_score(self) -> float:
        if not self.posture_scores:
            return 0.0
        return np.mean(self.posture_scores[-50:])
    
    @property
    def alert_fatigue(self) -> float:
        if self.alerts_sent == 0:
            return 0.0
        return self.ignored_alerts / self.alerts_sent
    
    def reset_evaluation_window(self):
        self.evaluation_history.append({
            "correction_rate": self.correction_rate,
            "avg_score": self.avg_posture_score,
            "alerts": self.alerts_sent,
        })
        self.alerts_sent = 0
        self.corrections_made = 0
        self.ignored_alerts = 0


@dataclass
class PendingAlert:
    timestamp: float
    action: int
    score_before: float
    checked: bool = False


class AlgorithmSelector:
    """
    Automatically selects the best performing algorithm based on correction rate.
    
    Evaluation window: 30 seconds (configurable)
    Primary metric: Correction rate (% of alerts that lead to posture improvement)
    Secondary metric: Average posture score
    
    Also considers switching to 'rule' (no alerts) when user has consistently good posture.
    Minimum 5 alerts required before switching.
    """
    
    def __init__(
        self,
        evaluation_interval: float = 30.0,
        switch_threshold: float = 0.10,  # 10% improvement required to switch
        min_alerts_before_switch: int = 5,
        good_posture_threshold: float = 0.80,
        sustained_good_duration: float = 60.0
    ):
        self.evaluation_interval = evaluation_interval
        self.switch_threshold = switch_threshold
        self.min_alerts_before_switch = min_alerts_before_switch
        self.good_posture_threshold = good_posture_threshold
        self.sustained_good_duration = sustained_good_duration
        
        # Algorithm performance tracking
        self.algorithms: Dict[str, AlgorithmMetrics] = {
            "ppo": AlgorithmMetrics(),
            "dqn": AlgorithmMetrics(),
            "rule": AlgorithmMetrics()
        }
        
        # Current state
        self.current_algorithm = "ppo"
        self.last_evaluation_time = time.time()
        self.pending_alerts: List[PendingAlert] = []
        
        # Good posture tracking for rule switching
        self.good_posture_start_time: Optional[float] = None
        self.consecutive_good_frames = 0
        
        # History for logging
        self.switch_history: List[Dict] = []
        
    def get_active_algorithm(self) -> str:
        """Returns the currently active algorithm"""
        return self.current_algorithm
    
    def record_alert(self, action: int, score_before: float):
        """Record when an alert is sent"""
        if action == 0:  # No feedback
            return
            
        algo_metrics = self.algorithms[self.current_algorithm]
        algo_metrics.alerts_sent += 1
        
        # Track pending alert to check for correction later
        self.pending_alerts.append(PendingAlert(
            timestamp=time.time(),
            action=action,
            score_before=score_before
        ))
    
    def record_posture_change(self, new_score: float):
        """Check if recent alerts led to corrections"""
        current_algo = self.algorithms[self.current_algorithm]
        current_algo.last_score = new_score
        current_algo.posture_scores.append(new_score)
        
        # Check pending alerts for corrections
        correction_made = False
        for pending in self.pending_alerts:
            if not pending.checked and time.time() - pending.timestamp <= 10.0:
                # Check if score improved significantly after alert
                improvement = new_score - pending.score_before
                if improvement > 0.05:  # 5% improvement = correction
                    current_algo.corrections_made += 1
                    correction_made = True
                else:
                    current_algo.ignored_alerts += 1
                pending.checked = True
        
        # Track good posture for rule switching
        if new_score >= self.good_posture_threshold:
            self.consecutive_good_frames += 1
            if self.consecutive_good_frames == 1:
                self.good_posture_start_time = time.time()
        else:
            self.consecutive_good_frames = 0
            self.good_posture_start_time = None
    
    def should_switch(self) -> bool:
        """Evaluate if algorithm switch is needed"""
        current_time = time.time()
        
        # Check if enough time has passed since last evaluation
        if current_time - self.last_evaluation_time < self.evaluation_interval:
            return False
        
        self.last_evaluation_time = current_time
        
        # Check for sustained good posture (switch to rule)
        if self._should_switch_to_rule():
            return True
        
        # Check if another algorithm is performing better
        return self._should_switch_based_on_performance()
    
    def _should_switch_to_rule(self) -> bool:
        """Check if should switch to rule (no alerts) due to sustained good posture"""
        if self.current_algorithm == "rule":
            return False
            
        if self.good_posture_start_time is None:
            return False
            
        sustained_time = time.time() - self.good_posture_start_time
        if sustained_time >= self.sustained_good_duration:
            avg_score = self.algorithms[self.current_algorithm].avg_posture_score
            if avg_score >= self.good_posture_threshold:
                return True
        return False
    
    def _should_switch_based_on_performance(self) -> bool:
        """Check if another algorithm is performing significantly better"""
        current_metrics = self.algorithms[self.current_algorithm]
        
        # Need minimum alerts before we can make a decision
        if current_metrics.alerts_sent < self.min_alerts_before_switch:
            return False
        
        # Check other algorithms
        best_algo = self.current_algorithm
        best_rate = current_metrics.correction_rate
        
        for algo_name, metrics in self.algorithms.items():
            if algo_name == self.current_algorithm:
                continue
                
            # Need minimum alerts for this algorithm too
            if metrics.alerts_sent < self.min_alerts_before_switch:
                continue
                
            # Check if this algorithm is significantly better
            rate_diff = metrics.correction_rate - best_rate
            if rate_diff > self.switch_threshold:
                best_algo = algo_name
                best_rate = metrics.correction_rate
        
        if best_algo != self.current_algorithm:
            self._record_switch(best_algo, best_rate, current_metrics.correction_rate)
            return True
        
        return False
    
    def _record_switch(self, new_algo: str, new_rate: float, old_rate: float):
        """Record algorithm switch in history"""
        self.switch_history.append({
            "timestamp": time.time(),
            "from": self.current_algorithm,
            "to": new_algo,
            "old_correction_rate": old_rate,
            "new_correction_rate": new_rate
        })
    
    def switch_algorithm(self) -> str:
        """Perform the switch and return new algorithm"""
        if self._should_switch_to_rule() and self.current_algorithm != "rule":
            self.current_algorithm = "rule"
            print(f"[AlgorithmSelector] Sustained good posture detected - switching to Rule (no alerts)")
        elif self._should_switch_based_on_performance():
            # Find best algorithm
            best_algo = self.current_algorithm
            best_rate = self.algorithms[self.current_algorithm].correction_rate
            
            for algo_name, metrics in self.algorithms.items():
                if algo_name != "rule" and metrics.correction_rate > best_rate:
                    best_algo = algo_name
                    best_rate = metrics.correction_rate
            
            if best_algo != self.current_algorithm:
                old_rate = self.algorithms[self.current_algorithm].correction_rate
                self.current_algorithm = best_algo
                print(f"[AlgorithmSelector] Performance-based switch: {best_algo} (rate: {best_rate:.2%} vs {old_rate:.2%})")
        
        # Reset evaluation window metrics
        for metrics in self.algorithms.values():
            metrics.reset_evaluation_window()
        
        return self.current_algorithm
    
    def get_stats(self) -> Dict:
        """Get current statistics for all algorithms"""
        return {
            "current_algorithm": self.current_algorithm,
            "algorithms": {
                algo: {
                    "correction_rate": f"{m.correction_rate:.2%}",
                    "avg_score": f"{m.avg_posture_score:.2f}",
                    "alerts_sent": m.alerts_sent,
                    "corrections": m.corrections_made,
                    "ignored": m.ignored_alerts,
                }
                for algo, m in self.algorithms.items()
            },
            "switch_history": self.switch_history[-5:] if self.switch_history else [],
            "good_posture_frames": self.consecutive_good_frames,
            "time_since_last_eval": time.time() - self.last_evaluation_time,
        }
    
    def reset(self):
        """Reset all metrics for a new session"""
        for metrics in self.algorithms.values():
            metrics.alerts_sent = 0
            metrics.corrections_made = 0
            metrics.ignored_alerts = 0
            metrics.posture_scores = []
            metrics.evaluation_history.clear()
        self.pending_alerts.clear()
        self.good_posture_start_time = None
        self.consecutive_good_frames = 0
        self.last_evaluation_time = time.time()
        self.switch_history.clear()


def create_algorithm_selector(
    evaluation_interval: float = 30.0,
    switch_threshold: float = 0.10,
    min_alerts: int = 5
) -> AlgorithmSelector:
    """Factory function to create AlgorithmSelector with default settings"""
    return AlgorithmSelector(
        evaluation_interval=evaluation_interval,
        switch_threshold=switch_threshold,
        min_alerts_before_switch=min_alerts
    )


if __name__ == "__main__":
    # Test the selector
    selector = AlgorithmSelector(evaluation_interval=5.0, min_alerts_before_switch=3)
    
    print("Testing AlgorithmSelector...")
    
    # Simulate some alerts and corrections
    for i in range(10):
        selector.record_alert(1, 0.6)  # Alert sent
        time.sleep(0.1)
        selector.record_posture_change(0.6 + (0.05 if i % 2 == 0 else -0.02))
    
    print(f"Current algorithm: {selector.get_active_algorithm()}")
    print(f"Stats: {selector.get_stats()}")
    
    # Check if should switch
    if selector.should_switch():
        selector.switch_algorithm()
    
    print(f"After potential switch: {selector.get_active_algorithm()}")
    print("Test complete!")