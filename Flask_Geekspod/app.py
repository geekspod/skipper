import pickle
import socket
import struct
import sys
import cv2

from flask import Flask, render_template
from flask import Response
from tracker import ObjectTracker
from flask_socketio import emit, SocketIO


HOST = ''
PORT = 8089
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')
s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')
conn, addr = s.accept()
data = b''  ### CHANGED
payload_size = struct.calcsize("L")  ### CHANGED

sys.path.insert(0, 'D:\yolov4-deepsort')

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
video = cv2.VideoCapture(0)
old_list = []
object = ObjectTracker()


@app.route('/')
def index():
    return render_template('index.html')


def video_socket():
    global data, payload_size
    # Retrieve message size
    while len(data) < payload_size:
        data += conn.recv(4096)
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]  ### CHANGED
    # Retrieve all data based on message size
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    # Extract frame
    frame = pickle.loads(frame_data)
    return frame


def gen(camera):
    global my_list, object
    while True:
        image = video_socket()
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
    return Response(gen(video), mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def test_connect():
    emit('after connect', {'data': 'Lets dance'})


def send_list(message):
    global old_list
    if message != old_list:
        old_list = message
        socketio.emit('list', message)


@socketio.on('track_id')
def recv_list(track_id):
    print("Received track {0}".format(track_id))
    global object
    object.tracking_id = track_id

