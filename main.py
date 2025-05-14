import pygame
import chess
import cv2
import mediapipe as mp
import numpy as np

# --- Initialization ---
pygame.init()
WIDTH, HEIGHT = 640, 640
SQUARE_SIZE = WIDTH // 8
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gesture Chess")

board = chess.Board()

# Mediapipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

colors = [pygame.Color("white"), pygame.Color("gray")]
font = pygame.font.SysFont(None, 36)

selected_square = None
gesture_active = False

def get_square_from_pos(x, y):
    col = int(x / SQUARE_SIZE)
    row = int(y / SQUARE_SIZE)
    if 0 <= col < 8 and 0 <= row < 8:
        square = chess.square(col, 7 - row)
        return square
    return None

def draw_board():
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, rect)

            piece = board.piece_at(chess.square(col, 7 - row))
            if piece:
                text = font.render(piece.symbol(), True, pygame.Color("black"))
                screen.blit(text, (col * SQUARE_SIZE + 20, row * SQUARE_SIZE + 20))

    # Highlight selected square
    if selected_square is not None:
        col = chess.square_file(selected_square)
        row = 7 - chess.square_rank(selected_square)
        highlight_rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        pygame.draw.rect(screen, pygame.Color("yellow"), highlight_rect, 4)

def detect_gesture(frame):
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    index_tip_pos = None
    gesture = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]

            h, w, _ = frame.shape
            x = int(index_tip.x * WIDTH)
            y = int(index_tip.y * HEIGHT)
            index_tip_pos = (x, y)

            # Gesture detection
            pinch_dist = np.linalg.norm(np.array([index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y]))
            fingers = [landmarks[i].y < landmarks[i - 2].y for i in [8, 12, 16, 20]]

            if all(not f for f in fingers):
                gesture = "fist"
            elif fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
                gesture = "peace"
            elif all(fingers):
                gesture = "open_palm"
            elif pinch_dist < 0.05:
                gesture = "pinch"

    cv2.imshow("Gesture View", frame)
    return index_tip_pos, gesture

# --- Game Loop ---
running = True
clock = pygame.time.Clock()

while running:
    screen.fill((0, 0, 0))
    draw_board()
    pygame.display.flip()
    clock.tick(30)

    ret, frame = cap.read()
    if not ret:
        break

    index_pos, gesture = detect_gesture(frame)

    if index_pos and gesture:
        x, y = index_pos
        square = get_square_from_pos(x, y)

        if gesture == "pinch" and not gesture_active:
            gesture_active = True  # lock pinch
            if square is not None:
                if selected_square is not None:
                    if square != selected_square:
                        move = chess.Move(selected_square, square)
                        if move in board.legal_moves:
                            board.push(move)
                            print("Move played:", move)
                        else:
                            print("Illegal move:", move)
                    else:
                        print("Same square selected again")
                    selected_square = None
                else:
                    selected_square = square
                    print("Selected:", square)

        elif gesture != "pinch":
            gesture_active = False  # unlock when not pinching

        if gesture == "open_palm":
            selected_square = None
            print("Selection cleared")

        elif gesture == "peace":
            if board.move_stack:
                board.pop()
                selected_square = None
                print("Move undone")

        elif gesture == "fist":
            board.reset()
            selected_square = None
            print("Board reset")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

cap.release()
cv2.destroyAllWindows()
pygame.quit()
