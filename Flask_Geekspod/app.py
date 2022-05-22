import flask_socketio
from flask import Flask, render_template
import cv2
from flask import Flask, render_template
from flask import Response,jsonify
import socket
import numpy as np
import sys
from PIL import Image
sys.path.insert(0, 'D:\yolov4-deepsort')
from tracker import ObjectTracker
from flask_socketio import send, emit, SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
video = cv2.VideoCapture(0)

old_list = []
@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    global my_list
    object = ObjectTracker()
    while True:
        success, image = video.read()
        output = object.track(image)
        my_list = output['tracks']
        send_list(my_list)
        ret, jpeg = cv2.imencode('.jpg', output["output"])
        outputImage = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + outputImage + b'\r\n')

@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(video),mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def test_connect():
    emit('after connect',  {'data':'Lets dance'})


def send_list(message):
    global old_list
    if message != old_list:
        old_list = message
        socketio.emit('list', message)





