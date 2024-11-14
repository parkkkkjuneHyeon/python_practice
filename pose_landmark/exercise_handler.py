import cv2
import numpy as np
import mediapipe as mp
from utils import *


def get_exercise_name(frame, result_landmarks, mp_drawing, pose_name_list):
    
    # *** 차후에 핸드폰 화면에 맞춰야함. ***
    # 사각형 영역 정의 (오른쪽에서 왼쪽으로 나열)
    frame_width = 800  # 캡처할 프레임의 너비 (화면의 크기를 기준으로 조정)
    box_width = 110    # 각 운동을 위한 박스의 너비
    box_height = 80   # 각 운동을 위한 박스의 높이
    padding = 10      # 각 박스 사이의 간격

    # 오른쪽 끝에서부터 왼쪽으로 배치하는 사각형 박스들
    exercise_boxes = {}

    for i, exercise in enumerate(pose_name_list):
        x1 = frame_width - (i + 1) * (box_width + padding)
        y1 = 100
        x2 = x1 + box_width
        y2 = y1 + box_height
        exercise_boxes[exercise] = (x1, y1, x2, y2)

    # 선택된 운동 이름
    selected_exercise = None  

    # 사각형 영역 및 운동 목록 표시
    for exercise, (x1, y1, x2, y2) in exercise_boxes.items():
        color = (0, 255, 255) if exercise == selected_exercise else (255, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, exercise, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    if result_landmarks.pose_landmarks:
        # 랜드마크 좌표 추출
        landmarks = result_landmarks.pose_landmarks.landmark

        left_hand = detection_body_part(landmarks, 'LEFT_WRIST')
        left_hand = (left_hand[0] * frame.shape[1], left_hand[1] * frame.shape[0])

        right_hand = detection_body_part(landmarks, 'RIGHT_WRIST')
        right_hand = (right_hand[0] * frame.shape[1], right_hand[1] * frame.shape[0])
        
        # 손목 좌표가 사각형 영역 안에 있는지 확인
        for exercise, (x1, y1, x2, y2) in exercise_boxes.items():
            if click_button(x1, y1, x2, y2, left_hand):
                selected_exercise = exercise
            elif click_button(x1, y1, x2, y2, right_hand):
                selected_exercise = exercise

        # 랜드마크 그리기
        mp_drawing.draw_landmarks(
            frame,
            result_landmarks.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        cv2.imshow('Video', frame)
        cv2.waitKey(10)

    return selected_exercise
