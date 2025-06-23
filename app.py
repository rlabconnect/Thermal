from flask import Flask, render_template, request, send_file
import mimetypes
import os
from processor import process_video, first_frame
import shutil
import sys, socket

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
VIDEO_PATH = None  # Global variable to store video path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER


@app.route("/", methods=["GET", "POST"])
def upload_file():
    global VIDEO_PATH  # Declare as global to modify

    if request.method == "GET":
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        shutil.rmtree(RESULT_FOLDER, ignore_errors=True)
        shutil.rmtree("static/frames", ignore_errors=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(RESULT_FOLDER, exist_ok=True)   
        os.makedirs("static/frames", exist_ok=True)   

    if request.method == "POST":
        video = request.files["video_file"]
        if video:
            VIDEO_PATH = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
            video.save(VIDEO_PATH)
            
            # For the first frame only, sending to frontend
            first_frame_path = first_frame(VIDEO_PATH)
            if first_frame_path:
                return render_template("choosing_frame.html", frame_url="static/frames/firstframe.png")

    return render_template("index.html")


@app.route("/Image_coordinates", methods=["POST"])
def send_coordinates():
    global VIDEO_PATH  # Access the global variable

    coordinates = request.get_json()
    region_coordinates = coordinates.get("coordinates", [])
    headings_inputs = coordinates.get("headings_inputs")

    try:
        output_excel, graph_path = process_video(VIDEO_PATH, region_coordinates, headings_inputs)
        filename = os.path.basename(output_excel)
        graph_name = os.path.basename(graph_path)
        return {
            "status": "success",
            "download_url": f"/download/{filename}",
            "graph_url": f"/download/{graph_name}"
        }
    except Exception as e:
        print("Error in processing video:", e)
        return {"status": "error"}


@app.route("/download/<filename>")
def download_result(filename):
    file_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    mime_type, _ = mimetypes.guess_type(file_path)
    return send_file(file_path, mimetype=mime_type or 'application/octet-stream', as_attachment=True)


def find_free_port(default=3000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", default))
            return default
        except OSError:
            s.bind(('', 0))
            return s.getsockname()[1]


if __name__ == "__main__":
    is_frozen = getattr(sys, 'frozen', False)
    port = find_free_port()
    app.run(debug=not is_frozen, port=port)
