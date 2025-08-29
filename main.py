import cv2 
import numpy as np 
from utils import read_video , save_video 
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from event_classification.event_detector import EventDetector

def main() :
    # Read Video 
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # Initialize Tracker 
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(video_frames , read_from_stub=True , stub_path='stubs/track_stubs.pkl')

    # Get Object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)
    
    # Initialize event detector
    event_detector = EventDetector()
    
    # Process events for each frame
    events_log = []
    for frame_num in range(len(video_frames)):
        event_type = event_detector.classify_event(tracks, frame_num)
        if event_type != "no_event":
            print(f"Frame {frame_num}: Detected {event_type}")
        events_log.append(event_type)
        
        # Draw event overlay
        if event_type != "no_event":
            output_video_frames[frame_num] = event_detector.draw_event_overlay(
                output_video_frames[frame_num],
                event_type,
                frame_num
            )
    
    # Print event statistics
    print("\nEvent Statistics:")
    event_types = ['pass', 'shot', 'free_kick', 'corner_kick', 'loss_of_possession']
    for event_type in event_types:
        count = events_log.count(event_type)
        print(f"{event_type.replace('_', ' ').title()}: {count}")

    # Draw object Tracks 
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # Save Video 
    save_video(output_video_frames , "output_videos/output_video.avi")

if __name__ == "__main__" :
    main()