import mediapipe as mp
import cv2
import numpy as np
import time
from src.Captura_camara import CapturaCamara
from src.exercises.Ejercicio import Exercise
from src.utils import *
from src.exercises.Squat import draw_angle_visualization  # Importar la función necesaria

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


class PlankBiomechanics:
    # ÁNGULO: ESPALDA (Hombro-Cadera-Tobillo)
    BACK_CORRECT_MIN = 170  # Espalda recta

    # ÁNGULO: CODOS (Hombro-Codo-Mano)
    ELBOW_CORRECT_MIN = 70
    ELBOW_CORRECT_MAX = 95


class Plank(Exercise):
    def __init__(self):
        self.biomech = PlankBiomechanics()
        self.plank_timer = None
        self.plank_duration = 0

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

            # Ángulo de los codos
            elbow_angle = draw_angle_visualization(
                image, idx_coord,
                landmark_ids=(12, 14, 16),  # Lado derecho
                label_offset=(30, -30)
            )
            if elbow_angle is None:
                elbow_angle = draw_angle_visualization(
                    image, idx_coord,
                    landmark_ids=(11, 13, 15),  # Lado izquierdo
                    label_offset=(30, -30)
                )

            try:
                if back_angle is None or elbow_angle is None:
                    raise ValueError("No se detectaron todos los ángulos necesarios")

                # Verificar ángulos de la espalda y los codos
                if (back_angle >= self.biomech.BACK_CORRECT_MIN and
                        self.biomech.ELBOW_CORRECT_MIN <= elbow_angle <= self.biomech.ELBOW_CORRECT_MAX):
                    if self.plank_timer is None:
                        self.plank_timer = time.time()
                    self.plank_duration += time.time() - self.plank_timer
                    self.plank_timer = time.time()
                else:
                    self.plank_timer = None

                # Mostrar estado de los codos
                if self.biomech.ELBOW_CORRECT_MIN <= elbow_angle <= self.biomech.ELBOW_CORRECT_MAX:
                    cv2.putText(image, "Codos correctos", (10, 60),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 0), thickness=2)
                else:
                    cv2.putText(image, "Codos incorrectos", (10, 60),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 255), thickness=2)

            except:
                pass

            # Mostrar duración con decimales
            cv2.putText(image, f"Duracion: {round(self.plank_duration, 1)} seg", (10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 255, 0), thickness=2)

            cv2.imshow('Image', rescale_frame(image, percent=100))
            if cv2.waitKey(5) & 0xFF == 27:
                break

        captura.release()
        cv2.destroyAllWindows()
        pose.close()
