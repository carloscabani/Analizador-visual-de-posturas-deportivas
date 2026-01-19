import mediapipe as mp
import cv2
import numpy as np
from src.Captura_camara import CapturaCamara
from src.exercises.Ejercicio import Exercise
from src.utils import *
from src.exercises.Squat import draw_angle_visualization  # Import the required function

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class PushupBiomechanics:
    # ÁNGULO 1: CODO (Hombro-Codo-Muñeca)
    ELBOW_CORRECT_MIN = 45
    ELBOW_CORRECT_MAX = 90
    ELBOW_EXTENDED_MIN = 160  # Extensión completa

    # ÁNGULO 2: ESPALDA (Hombro-Cadera-Tobillo)
    BACK_CORRECT_MIN = 170  # Espalda recta

class Pushup(Exercise):
    def __init__(self):
        self.biomech = PushupBiomechanics()
        self.pushup_state = "standing"
        self.correct_count = 0
        self.incorrect_count = 0
        self.current_errors = []
        self.min_elbow_angle = 180
        self.prev_elbow_angle = None

    def exercise(self, source):
        captura = CapturaCamara(source)

        while True:
            success, image = captura.show_frame()
            if not success or image is None:
                break

            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            idx_coord = get_idx_coord(image, results)

            # Ángulo del codo
            elbow_angle = draw_angle_visualization(
                image, idx_coord,
                landmark_ids=(12, 14, 16),  # Lado derecho
                label_offset=(10, 0)
            )
            if elbow_angle is None:
                elbow_angle = draw_angle_visualization(
                    image, idx_coord,
                    landmark_ids=(11, 13, 15),  # Lado izquierdo
                    label_offset=(10, 0)
                )

            # Ángulo de la espalda
            back_angle = draw_angle_visualization(
                image, idx_coord,
                landmark_ids=(12, 24, 28),  # Lado derecho
                label_offset=(-30, 0)
            )
            if back_angle is None:
                back_angle = draw_angle_visualization(
                    image, idx_coord,
                    landmark_ids=(11, 23, 27),  # Lado izquierdo
                    label_offset=(-30, 0)
                )

            # Estados del push-up
            try:
                if elbow_angle is None or back_angle is None:
                    raise ValueError("No se detectaron ángulos")

                # Estado 1: de pie (posición inicial)
                if self.pushup_state == "standing":
                    if elbow_angle < self.biomech.ELBOW_EXTENDED_MIN:
                        self.pushup_state = "descending"
                        self.current_errors = []
                        self.min_elbow_angle = 180

                # Estado 2: descendiendo
                elif self.pushup_state == "descending":
                    if elbow_angle < self.min_elbow_angle:
                        self.min_elbow_angle = elbow_angle

                    if elbow_angle > self.prev_elbow_angle + 2:  # Comienza a subir
                        if self.min_elbow_angle > self.biomech.ELBOW_CORRECT_MAX:
                            self.current_errors.append("codo_no_bajo_suficiente")
                        if back_angle < self.biomech.BACK_CORRECT_MIN:
                            self.current_errors.append("espalda_no_recta")
                        self.pushup_state = "ascending"

                # Estado 3: subiendo
                elif self.pushup_state == "ascending":
                    if elbow_angle >= self.biomech.ELBOW_EXTENDED_MIN:
                        if len(self.current_errors) == 0:
                            self.correct_count += 1
                        else:
                            self.incorrect_count += 1
                        self.pushup_state = "standing"

                self.prev_elbow_angle = elbow_angle

            except:
                pass

            # Mostrar contadores
            cv2.putText(image, f"Correctas: {self.correct_count}", (10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 0), thickness=2)
            cv2.putText(image, f"Incorrectas: {self.incorrect_count}", (10, 60),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 255), thickness=2)

            cv2.imshow('Image', rescale_frame(image, percent=100))
            if cv2.waitKey(5) & 0xFF == 27:
                break

        captura.release()
        cv2.destroyAllWindows()
        pose.close()
