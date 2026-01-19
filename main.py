import argparse

from src.exercises.Squat import Squat
from src.exercises.Plank import Plank  
from src.exercises.Pushup import Pushup  

class WorkoutAnalyzer:
    def __init__(self):
        self.squat = Squat()
        self.plank = Plank()
        self.pushup = Pushup()  

    def rep(self, type, source):
        if type.lower() == "squat":
            self.squat.exercise(source)
        elif type.lower() == "plank":
            self.plank.exercise(source)
        elif type.lower() == "pushup": 
            self.pushup.exercise(source)
        else:
            raise ValueError(f"Input {type} and/or {source} is not correct")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-type', "--type", required=True, help="Tipo de ejercicio",
                        type=str)
    parser.add_argument('-source', "--source", required=True, help="Ruta del video o ID de cámara", type=str)
    args = parser.parse_args()
    type = args.type
    source = args.source
    
    # ========== CONVERSIÓN INTELIGENTE DE SOURCE ========== 
    # Si source es un número (ej: "0", "1"), convertir a int para cámara
    # Si no, mantener como string para archivo de video
    try:
        source = int(source)  # Intenta convertir a int
        print(f"Usando cámara ID: {source}")
    except ValueError:
        # No es un número, es una ruta de archivo
        print(f"Usando video: {source}")
    
    gym = WorkoutAnalyzer()
    gym.rep(type, source)
