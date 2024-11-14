import cv2
import argparse
from utils import *
import mediapipe as mp
from body_part_angle import BodyPartAngle
from types_of_exercise import TypeOfExercise
from exercise_handler import *

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# 운동 종류 리스트
pose_name_list = ['exit', 'sit-up', 'squat', 'walk', 'push-up', 'pull-up']
cap = cv2.VideoCapture(0)  # webcam

cap.set(3, 800)  # width
cap.set(4, 480)  # height
max_counter = 10

# setup mediapipe
with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    is_selected_exercise_name = False # 운동 자세 이름을 선택했는지 확인 False = 선택이 안됨. True = 선택 완료.
    counter = 0  # movement of exercise
    status = True  # state of move

    while cap.isOpened():
        ret, frame = cap.read()
        # result_screen = np.zeros((250, 400, 3), np.uint8)
        frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 모드)
        frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
        # recolor frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        # make detection
        results = pose.process(frame)
        # recolor back to BGR
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not is_selected_exercise_name:
            exercise_name = get_exercise_name(frame, results, mp_drawing, pose_name_list)

            if exercise_name == None:
                continue
            elif exercise_name != None:
                is_selected_exercise_name = True

        try:
            landmarks = results.pose_landmarks.landmark
            counter, status = TypeOfExercise(landmarks).calculate_exercise(
                exercise_name, counter, status)
        except:
            pass

        frame = score_table(exercise_name, frame, counter, status, max_counter)

        # render detections (for landmarks)
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255),
                                   thickness=2,
                                   circle_radius=4),
            mp_drawing.DrawingSpec(color=(174, 139, 45),
                                   thickness=2,
                                   circle_radius=4),
        )

        cv2.imshow('Video', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        elif exercise_name == 'exit':
            break
        elif counter == max_counter:
            break

    cap.release()
    cv2.destroyAllWindows()
