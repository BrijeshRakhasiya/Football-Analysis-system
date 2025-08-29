import cv2
import numpy as np
from typing import Dict, Tuple, List

class EventDetector:
    def __init__(self):
        self.event_colors = {
            'pass': (0, 255, 0),        # Green
            'shot': (255, 0, 0),        # Blue
            'free_kick': (255, 165, 0),  # Orange
            'corner_kick': (255, 255, 0), # Yellow
            'loss_of_possession': (0, 0, 255)  # Red
        }
        self.last_possession = None
        self.corner_zones = {
            'top_left': (0, 0, 15, 15),      # Increased corner zone
            'bottom_left': (0, 85, 15, 100),  # Increased corner zone
            'top_right': (85, 0, 100, 15),    # Increased corner zone
            'bottom_right': (85, 85, 100, 100) # Increased corner zone
        }
        self.last_event_frame = 0
        self.event_cooldown = 5  # Reduced cooldown between events
        
    def detect_pass(self, tracks: Dict, frame_num: int) -> bool:
        if frame_num < 3:  # Reduced window size
            return False
            
        ball_track = tracks['ball']
        current_pos = ball_track[frame_num][1].get('transformed_position')
        prev_pos = ball_track[frame_num - 3][1].get('transformed_position')
        
        if current_pos is None or prev_pos is None:
            return False
            
        # Calculate velocity and direction
        velocity = np.array(current_pos) - np.array(prev_pos)
        speed = np.linalg.norm(velocity)
        
        # Check if ball is moving between players
        current_possession = None
        prev_possession = None
        
        for player_id, track in tracks['players'][frame_num].items():
            if track.get('has_ball', False):
                current_possession = player_id
                break
                
        for player_id, track in tracks['players'][frame_num - 3].items():
            if track.get('has_ball', False):
                prev_possession = player_id
                break
        
        # Pass conditions:
        # 1. Ball speed is in pass range
        # 2. Possession changed between players
        # 3. Players are from same team
        if 0.5 < speed < 4 and current_possession != prev_possession:  # Adjusted thresholds
            if current_possession is not None and prev_possession is not None:
                current_team = tracks['players'][frame_num][current_possession].get('team')
                prev_team = tracks['players'][frame_num - 3][prev_possession].get('team')
                return current_team == prev_team
        
        return False

    def detect_shot(self, tracks: Dict, frame_num: int) -> bool:
        if frame_num < 3:  # Reduced window size
            return False
            
        ball_track = tracks['ball']
        current_pos = ball_track[frame_num][1].get('transformed_position')
        prev_pos = ball_track[frame_num - 3][1].get('transformed_position')
        
        if current_pos is None or prev_pos is None:
            return False
            
        velocity = np.array(current_pos) - np.array(prev_pos)
        speed = np.linalg.norm(velocity)
        
        # Check if ball is moving towards goal
        if speed > 3:  # Lowered threshold
            # Define goal positions (adjust these based on your coordinate system)
            left_goal = np.array([0, 50])   # Left goal center
            right_goal = np.array([100, 50]) # Right goal center
            
            # Calculate direction to both goals
            direction = velocity / speed
            to_left_goal = np.array(current_pos) - left_goal
            to_right_goal = np.array(current_pos) - right_goal
            
            # Check if ball is moving towards either goal
            angle_left = np.abs(np.dot(direction, to_left_goal) / np.linalg.norm(to_left_goal))
            angle_right = np.abs(np.dot(direction, to_right_goal) / np.linalg.norm(to_right_goal))
            
            return angle_left > 0.7 or angle_right > 0.7  # Angle threshold for shot detection
        
        return False

    def is_in_corner(self, position: Tuple[float, float]) -> bool:
        x, y = position
        for zone in self.corner_zones.values():
            if zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3]:
                return True
        return False

    def classify_event(self, tracks: Dict, frame_num: int) -> str:
        # Check event cooldown
        if frame_num - self.last_event_frame < self.event_cooldown:
            return "no_event"

        ball_pos = tracks['ball'][frame_num][1].get('transformed_position')
        if ball_pos is None:
            return "no_event"

        # 1. Corner Kick Detection
        if self.is_in_corner(ball_pos):
            self.last_event_frame = frame_num
            return "corner_kick"

        # 2. Free Kick Detection
        ball_stationary = True
        if frame_num > 10:
            prev_pos = tracks['ball'][frame_num-10][1].get('transformed_position')
            if prev_pos is not None:
                ball_stationary = np.linalg.norm(np.array(ball_pos) - np.array(prev_pos)) < 0.5

        if ball_stationary and frame_num - self.last_event_frame > 30:  # Additional cooldown for free kicks
            self.last_event_frame = frame_num
            return "free_kick"

        # 3. Shot Detection
        if self.detect_shot(tracks, frame_num):
            self.last_event_frame = frame_num
            return "shot"

        # 4. Pass Detection
        if self.detect_pass(tracks, frame_num):
            self.last_event_frame = frame_num
            return "pass"

        # 5. Loss of Possession Detection
        current_possession = None
        for player_id, track in tracks['players'][frame_num].items():
            if track.get('has_ball', False):
                current_possession = track.get('team')
                break

        if (self.last_possession is not None and 
            current_possession is not None and 
            current_possession != self.last_possession):
            self.last_possession = current_possession
            self.last_event_frame = frame_num
            return "loss_of_possession"

        self.last_possession = current_possession
        return "no_event"

    def draw_event_overlay(self, frame: np.ndarray, event_type: str, frame_num: int) -> np.ndarray:
        if event_type == "no_event":
            return frame

        overlay = frame.copy()
        color = self.event_colors.get(event_type, (255, 255, 255))

        # Draw event box
        cv2.rectangle(overlay, (30, 30), (400, 100), (0, 0, 0), -1)
        
        # Draw event text
        event_text = f"EVENT: {event_type.replace('_', ' ').upper()}"
        cv2.putText(overlay, event_text, (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Add frame number
        cv2.putText(overlay, f"Frame: {frame_num}", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Blend overlay with original frame
        alpha = 0.7
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)