import mediapipe as mp
import cv2
import numpy as np

from src.Captura_camara import CapturaCamara
from src.exercises.Ejercicio import Exercise
from src.utils import *

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose_landmark_drawing = mp_drawing.DrawingSpec(thickness=5, circle_radius=2, color=(0, 0, 255))
pose_connection_drawing = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))


def draw_angle_visualization(image, idx_coord, landmark_ids, sagitta=15, label_offset=(0, 0)):
    try:
        id1, vertex_id, id2 = landmark_ids
        
        if id1 not in idx_coord or vertex_id not in idx_coord or id2 not in idx_coord:
            return None
        
        l1 = np.linspace(idx_coord[id1], idx_coord[vertex_id], 100)
        l2 = np.linspace(idx_coord[vertex_id], idx_coord[id2], 100)
        
        cv2.line(image, 
                 (int(l1[99][0]), int(l1[99][1])), 
                 (int(l1[69][0]), int(l1[69][1])), 
                 thickness=4, color=(0, 0, 255))
        
        cv2.line(image, 
                 (int(l2[0][0]), int(l2[0][1])), 
                 (int(l2[30][0]), int(l2[30][1])), 
                 thickness=4, color=(0, 0, 255))
        
        angle = ang((idx_coord[id1], idx_coord[vertex_id]),
                    (idx_coord[vertex_id], idx_coord[id2]))
        
        text_pos = (idx_coord[vertex_id][0] + label_offset[0], 
                    idx_coord[vertex_id][1] + label_offset[1])
        
        cv2.putText(image, str(round(angle, 1)), text_pos,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6, color=(0, 255, 0), thickness=2)
        
        # CAMBIO: Sagitta negativa para arco interno
        center, radius, start_angle, end_angle = convert_arc(l1[90], l2[10], sagitta=-sagitta)  # ← AQUÍ
        axes = (radius, radius)
        draw_ellipse(image, center, axes, -1, start_angle, end_angle, 255)
        
        return angle
        
    except Exception as e:
        return None


def draw_squat_landmarks(image, results):
    if not results.pose_landmarks:
        return
    
    # Se asume perfil derecho, si está de perfil izquierdo, usar landmarks izquierdos
    SQUAT_LANDMARKS = {12, 24, 26, 28}  # hombro, cadera, rodilla, tobillo

    SQUAT_CONNECTIONS = [
        (12, 24),  # Hombro → Cadera (tronco)
        (24, 26),  # Cadera → Rodilla (muslo)
        (26, 28),  # Rodilla → Tobillo (pantorrilla)
    ]
    
    image_height, image_width, _ = image.shape
    
    # Líneas verdes
    for connection in SQUAT_CONNECTIONS:
        start_idx, end_idx = connection
        
        start_landmark = results.pose_landmarks.landmark[start_idx]
        end_landmark = results.pose_landmarks.landmark[end_idx]
        
        start_point = (
            int(start_landmark.x * image_width),
            int(start_landmark.y * image_height)
        )
        end_point = (
            int(end_landmark.x * image_width),
            int(end_landmark.y * image_height)
        )
        
        cv2.line(image, start_point, end_point, 
                 color=(0, 255, 0), thickness=2)
    
    # Círculos rojos
    for idx in SQUAT_LANDMARKS:
        landmark = results.pose_landmarks.landmark[idx]
        
        point = (
            int(landmark.x * image_width),
            int(landmark.y * image_height)
        )
        
        cv2.circle(image, point, radius=3, 
                   color=(0, 0, 255), thickness=5)


class SquatBiomechanics:
    
    # ÁNGULO 1: RODILLA (Cadera-Rodilla-Tobillo) 
    KNEE_STANDING_MIN = 160          # Pierna casi recta
    KNEE_STANDING_TOLERANCE = 10     # Tolerancia para detectar de pie
    
    # RANGO ADECUADO
    KNEE_CORRECT_MAX = 70            
    KNEE_CORRECT_MIN = 45            
    
    # ÁNGULO 2: CADERA (Hombro-Cadera-Rodilla)
    HIP_STANDING_MIN = 160           # Cuerpo recto
    
    # RANGO ADECUADI CADERA
    HIP_CORRECT_MAX = 75             # Máximo para sentadilla correcta (debe bajar de aquí)
    HIP_CORRECT_MIN = 45             # Mínimo para sentadilla correcta (no debe bajar de aquí)


class Squat(Exercise):
    def __init__(self):
        self.biomech = SquatBiomechanics()
        self.squat_state = "standing"
        self.correct_count = 0
        self.incorrect_count = 0
        self.current_errors = []
        
        # Tracking de ángulos mínimos alcanzados
        self.min_knee_angle = 180
        self.min_hip_angle = 180
        
        # Tracking de ángulos anteriores para detectar cambio de dirección
        self.prev_knee_angle = None
        self.prev_hip_angle = None

    def exercise(self, source):
        captura = CapturaCamara(source)
    
        while True:
            success, image = captura.show_frame()

            if not success:
                print("Video terminado o error de captura.")
                break 
            
            if image is None:
                print("Frame vacío, finalizando.")
                break 

            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            draw_squat_landmarks(image, results)
            idx_coord = get_idx_coord(image, results)

            # Angulo rodillas
            knee_angle = draw_angle_visualization(
                image, idx_coord, 
                landmark_ids=(24, 26, 28),  # Lado derecho
                label_offset=(10, 0)
            )
            
            if knee_angle is None:
                knee_angle = draw_angle_visualization(
                    image, idx_coord,
                    landmark_ids=(23, 25, 27),  # Lado izquierdo
                    label_offset=(10, 0)
                )
            
            # Angulo cadera
            hip_angle = draw_angle_visualization(
                image, idx_coord,
                landmark_ids=(12, 24, 26),  # Lado derecho
                label_offset=(-30, 0)
            )
            
            if hip_angle is None:
                hip_angle = draw_angle_visualization(
                    image, idx_coord,
                    landmark_ids=(11, 23, 25),  # Lado izquierdo
                    label_offset=(-30, 0)
                )

            # Estados
            try:
                if knee_angle is None or hip_angle is None:
                    raise ValueError("No se detectaron ángulos")
                
                # Estado 1: de pie
                if self.squat_state == "standing":
                    if (knee_angle < (self.biomech.KNEE_STANDING_MIN - self.biomech.KNEE_STANDING_TOLERANCE) or
                        hip_angle < (self.biomech.HIP_STANDING_MIN - 10)):
                        
                        self.squat_state = "descending"
                        self.current_errors = []
                        self.min_knee_angle = 180
                        self.min_hip_angle = 180
                        self.prev_knee_angle = knee_angle
                        self.prev_hip_angle = hip_angle
                
                # Estado 2: descenso
                elif self.squat_state == "descending":
                    # Actualizar mínimos
                    if knee_angle < self.min_knee_angle:
                        self.min_knee_angle = knee_angle
                    if hip_angle < self.min_hip_angle:
                        self.min_hip_angle = hip_angle
                    
                    # Detectar si comienza a subir
                    if self.prev_knee_angle is not None:
                        # Si el ángulo aumenta (sube) antes de alcanzar el rango correcto
                        if knee_angle > self.prev_knee_angle + 2:  # Tolerancia de 2° para evitar ruido
                            # Verificar si NO llegó al rango correcto
                            if self.min_knee_angle > self.biomech.KNEE_CORRECT_MAX:
                                self.current_errors.append("rodilla_no_bajo_suficiente")
                            
                            if self.min_hip_angle > self.biomech.HIP_CORRECT_MAX:
                                self.current_errors.append("cadera_no_bajo_suficiente")
                            
                            # Verificar si bajó demasiado
                            if self.min_knee_angle < self.biomech.KNEE_CORRECT_MIN:
                                self.current_errors.append("rodilla_bajo_demasiado")
                            
                            if self.min_hip_angle < self.biomech.HIP_CORRECT_MIN:
                                self.current_errors.append("cadera_bajo_demasiado")
                            
                            self.squat_state = "ascending"
                    
                    # Si está dentro del rango correcto, bottom
                    if (self.biomech.KNEE_CORRECT_MIN <= knee_angle <= self.biomech.KNEE_CORRECT_MAX):
                        self.squat_state = "bottom"
                    
                    # Actualizar ángulos anteriores
                    self.prev_knee_angle = knee_angle
                    self.prev_hip_angle = hip_angle
                
                # Estado 3: bottom
                elif self.squat_state == "bottom":
                    # Actualizar mínimos
                    if knee_angle < self.min_knee_angle:
                        self.min_knee_angle = knee_angle
                    if hip_angle < self.min_hip_angle:
                        self.min_hip_angle = hip_angle
                    
                    # VALIDAR SI SE SALE DEL RANGO CORRECTO
                    # Si baja DEMASIADO (menor que el mínimo permitido)
                    if knee_angle < self.biomech.KNEE_CORRECT_MIN:
                        if "rodilla_bajo_demasiado" not in self.current_errors:
                            self.current_errors.append("rodilla_bajo_demasiado")
                    
                    if hip_angle < self.biomech.HIP_CORRECT_MIN:
                        if "cadera_bajo_demasiado" not in self.current_errors:
                            self.current_errors.append("cadera_bajo_demasiado")
                    
                    # DETECTAR ASCENSO (comienza a subir)
                    if self.prev_knee_angle is not None:
                        if knee_angle > self.prev_knee_angle + 2:  # Está subiendo
                            # Si está subiendo desde dentro del rango, pasar a ascending
                            self.squat_state = "ascending"
                    
                    # Actualizar ángulos anteriores
                    self.prev_knee_angle = knee_angle
                    self.prev_hip_angle = hip_angle
                
                # Estado 4: ASCENDIENDO
                elif self.squat_state == "ascending":
                    # Verificar si regresó a posición de pie
                    if (knee_angle >= (self.biomech.KNEE_STANDING_MIN - self.biomech.KNEE_STANDING_TOLERANCE) and
                        hip_angle >= (self.biomech.HIP_STANDING_MIN - 10)):
                        
                        # VALIDACIÓN FINAL COMPLETA
                        
                        # 1. Verificar si NO bajó suficiente (se quedó arriba)
                        if self.min_knee_angle > self.biomech.KNEE_CORRECT_MAX:
                            if "rodilla_no_bajo_suficiente" not in self.current_errors:
                                self.current_errors.append("rodilla_no_bajo_suficiente")
                        
                        if self.min_hip_angle > self.biomech.HIP_CORRECT_MAX:
                            if "cadera_no_bajo_suficiente" not in self.current_errors:
                                self.current_errors.append("cadera_no_bajo_suficiente")
                        
                        # 2. Verificar si bajó DEMASIADO (más allá del mínimo)
                        if self.min_knee_angle < self.biomech.KNEE_CORRECT_MIN:
                            if "rodilla_bajo_demasiado" not in self.current_errors:
                                self.current_errors.append("rodilla_bajo_demasiado")
                        
                        if self.min_hip_angle < self.biomech.HIP_CORRECT_MIN:
                            if "cadera_bajo_demasiado" not in self.current_errors:
                                self.current_errors.append("cadera_bajo_demasiado")
                        
                        # CONTAR REPETICIÓN
                        if len(self.current_errors) == 0:
                            self.correct_count += 1
                            feedback_color = (0, 255, 0)
                            
                            # Clasificar según profundidad alcanzada
                            avg_depth = (self.min_knee_angle + self.min_hip_angle) / 2
                            if avg_depth < 50:
                                feedback_text = "CORRECTA - MUY PROFUNDA!"
                            elif avg_depth < 60:
                                feedback_text = "CORRECTA - PROFUNDA"
                            else:
                                feedback_text = "CORRECTA - COMPLETA"
                        else:
                            self.incorrect_count += 1
                            feedback_color = (0, 0, 255)
                            feedback_text = "INCORRECTA"
                        
                        # Mostrar feedback
                        cv2.putText(image, feedback_text,
                                   (50, 120),
                                   fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                   fontScale=1.2, color=feedback_color, thickness=3)
                        
                        # Reset
                        self.squat_state = "standing"
                        self.current_errors = []
                        self.min_knee_angle = 180
                        self.min_hip_angle = 180
                        self.prev_knee_angle = None
                        self.prev_hip_angle = None
            
            except:
                pass
            
            # Contadores principales
            y_offset = 30
            cv2.putText(image, f"Correctas: {self.correct_count}",
                       (10, y_offset),
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.8, color=(0, 255, 0), thickness=2)
            
            y_offset += 30
            cv2.putText(image, f"Incorrectas: {self.incorrect_count}",
                       (10, y_offset),
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.8, color=(0, 0, 255), thickness=2)
            
            y_offset += 30
            total = self.correct_count + self.incorrect_count
            cv2.putText(image, f"Total: {total}",
                       (10, y_offset),
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.8, color=(255, 255, 255), thickness=2)
            
            # Mostrar errores
            if len(self.current_errors) > 0:
                y_error = 160
                cv2.putText(image, "ERRORES:",
                           (10, y_error),
                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale=0.7, color=(0, 165, 255), thickness=2)
                
                y_error += 25
                error_messages = {
                    "rodilla_no_bajo_suficiente": f"- Rodilla: No bajo de {self.biomech.KNEE_CORRECT_MAX}° (quedo en {round(self.min_knee_angle, 1)}°)",
                    "cadera_no_bajo_suficiente": f"- Cadera: No bajo de {self.biomech.HIP_CORRECT_MAX}° (quedo en {round(self.min_hip_angle, 1)}°)",
                    "rodilla_bajo_demasiado": f"- Rodilla: Bajo de {self.biomech.KNEE_CORRECT_MIN}° (llego a {round(self.min_knee_angle, 1)}°)",
                    "cadera_bajo_demasiado": f"- Cadera: Bajo de {self.biomech.HIP_CORRECT_MIN}° (llego a {round(self.min_hip_angle, 1)}°)"
                }
                
                for error in self.current_errors:
                    text = error_messages.get(error, f"- {error}")
                    cv2.putText(image, text,
                               (10, y_error),
                               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale=0.5, color=(0, 165, 255), thickness=2)
                    y_error += 25
            
            # Indicadores de ángulos en tiempo real
            if self.squat_state != "standing":
                y_debug = image.shape[0] - 95
                
                # Mostrar rango de rodilla
                knee_in_range = (self.biomech.KNEE_CORRECT_MIN <= self.min_knee_angle <= self.biomech.KNEE_CORRECT_MAX)
                knee_color = (0, 255, 0) if knee_in_range else (0, 0, 255)
                knee_status = "OK" if knee_in_range else ("BAJO" if self.min_knee_angle < self.biomech.KNEE_CORRECT_MIN else "ALTO")
                
                cv2.putText(image, f"Rodilla {knee_status}: {round(self.min_knee_angle, 1)}°",
                           (10, y_debug),
                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale=0.6, color=knee_color, thickness=2)
                
                # Mostrar rango de cadera
                hip_in_range = (self.biomech.HIP_CORRECT_MIN <= self.min_hip_angle <= self.biomech.HIP_CORRECT_MAX)
                hip_color = (0, 255, 0) if hip_in_range else (0, 0, 255)
                hip_status = "OK" if hip_in_range else ("BAJO" if self.min_hip_angle < self.biomech.HIP_CORRECT_MIN else "ALTO")
                
                cv2.putText(image, f"Cadera {hip_status}: {round(self.min_hip_angle, 1)}°",
                           (10, y_debug + 25),
                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale=0.6, color=hip_color, thickness=2)
            
            # Estado actual
            cv2.putText(image, f"Estado: {self.squat_state}",
                       (10, image.shape[0] - 20),
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=0.6, color=(255, 255, 255), thickness=1)
            
            cv2.imshow('Image', rescale_frame(image, percent=100))
            
            key = cv2.waitKey(5) & 0xFF
            if key == 27 or key == ord('q') or key == ord('x') or key == ord('X'):
                break

            try:
                if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            
        captura.release()
        cv2.destroyAllWindows()    
        pose.close()
        
        total = self.correct_count + self.incorrect_count
        
        print("RESUMEN DE ENTRENAMIENTO")
        print("="*40)
        print(f"Sentadillas correctas:   {self.correct_count}")
        print(f"Sentadillas incorrectas: {self.incorrect_count}")
        print(f"Total:                   {total}")
        if total > 0:
            accuracy = (self.correct_count / total) * 100
            print(f"Precisión:               {accuracy:.1f}%")
