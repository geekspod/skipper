from time import sleep
from tkinter import *
from PIL import Image, ImageTk
import cv2

from modules import video
from modules.detector import Detector
from modules.tracker import Tracker

win = Tk()

win.geometry("700x500")

label = Label(win)
label.grid(row=0, column=0)

sv = StringVar()


def cb():
    global to_track
    print('cb called')
    if detected_objects is None or len(detected_objects) == 0:
        print("No detected objects")
        return
    to_track = sv.get()


text = Entry(win, width=10, textvariable=sv)
text.grid(row=1, column=0)

button = Button(win, text="Track", command=cb)
button.grid(row=1, column=1)

video_feed = video.Video()
video_feed.start()

sleep(2)

detector = Detector(video_feed)
tracker = None

to_track = False
detected_objects = []


def show_frames():
    global detected_objects
    global tracker
    frame = None
    if not to_track:
        frame = video_feed.frame

    if not to_track:
        detected_objects = detector.detect()
        frame = detector.draw_detected_objects(frame)
    else:
        print(detected_objects, to_track)
        if len(detected_objects) == 0:
            print("ERROR")
        if not tracker:
            tracker = Tracker(video_feed)
        tracker.init_tracker(video_feed.frame, detected_objects[int(to_track)])
        tracker.start_tracker()

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)

    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    label.after(20, show_frames)


show_frames()
win.mainloop()
