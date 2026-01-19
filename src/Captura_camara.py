from threading import Thread
import cv2, time


class CapturaCamara(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)

        if not self.capture.isOpened():
            raise ValueError(f"Error: No se pudo abrir la fuente de video: {src}")
        

        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.FPS = 1 / 30
        self.FPS_MS = int(self.FPS * 500)
        self.thread = Thread(target=self.update, args=())
        self.frame = None
        self.status = False 
        self.stopped = False 
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.5)


    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

                if not self.status:
                    self.stopped = True
                    break
            time.sleep(self.FPS)

        if self.capture.isOpened():
            self.capture.release()

    def show_frame(self):
        return self.status, self.frame  

    def show_live_analysis(self):
        """
        Display the live camera feed in a window for real-time analysis.
        Press 'ESC' to exit the live feed.
        """
        while not self.stopped:
            status, frame = self.show_frame()
            if not status or frame is None:
                break

            cv2.imshow('Live Analysis', frame)

            # Exit on pressing 'ESC'
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.release()
        cv2.destroyAllWindows()

    def release(self):
        self.stopped = True
        time.sleep(0.2)
        
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)

        if self.capture.isOpened():
            self.capture.release()
