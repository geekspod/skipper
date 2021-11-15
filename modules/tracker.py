import cv2


class Tracker:
    def __init__(self, video, initial_bbox=None, initial_frame=None):
        self.video = video
        self.initial_frame = initial_frame
        self.initial_bbox = initial_bbox
        self.tracker = cv2.TrackerKCF_create()

    def init_tracker(self, frame=None, bbox=None):
        if frame is None:
            frame = self.initial_frame
        if bbox is None:
            bbox = self.initial_bbox
        self.tracker.init(frame, bbox)

    def start_tracker(self):
        while True:
            frame = self.video.frame
            success, bbox = self.tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
