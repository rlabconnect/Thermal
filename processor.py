import re
import pandas as pd
import cv2
import pytesseract
import os
from datetime import datetime
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os


# for extracting the first frames only and sending it to the backend
def first_frame(video_path):
    # for reading the first frame
    capture = cv2.VideoCapture(video_path)
    ret, frame = capture.read()
    if ret:
        os.makedirs("static/frames", exist_ok=True)
        frame_output_path = os.path.join("static", "frames", "firstframe.png")
        cv2.imwrite(frame_output_path, frame)
        return frame_output_path
    return None

def process_video(video_path ,region_coordinates,headings_inputs):
    
    filename = f"VideoData_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = os.path.join("results", filename + ".xlsx")
    graph_path = os.path.join("results", filename + ".png")

    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    skip_frames = int(fps // 3)
    frame_count = 0

    # One list per region
    region_texts = [[] for _ in range(len(region_coordinates))]

    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            break

        if frame_count % skip_frames == 0:
            for i, region in enumerate(region_coordinates):
                try:
                    x1 = int(region["x"])
                    y1 = int(region["y"])
                    w = int(region["width"])
                    h = int(region["height"])
                    x2 = x1 + w
                    y2 = y1 + h

                    roi = frame[y1:y2, x1:x2]
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

                    custom_config = r'--oem 3 --psm 6'
                    text = pytesseract.image_to_string(thresh, config=custom_config)

                    msg=''
                    for ch in text:
                        if ch.isdigit() or ch == '.':
                            msg+=ch
                            flag=1
                        # text=float(msg)
                    if msg!='':
                        text=float(msg)
                        if text>45 or text<30:
                            text=0
                    else:
                        text=0


                    region_texts[i].append(text)
                except Exception as e:
                    print(f"Error in region {i}: {e}")
                    region_texts[i].append(0)

        frame_count += 1

    capture.release()
    cv2.destroyAllWindows()

    headings=headings_inputs.split(",")
    # Create DataFrame with columns Region 1, Region 2, ...
    data = {f"{headings[i]}": region_texts[i] for i in range(len(region_texts))}
    df = pd.DataFrame(data)
    for column in df.columns:
        non_zero_values = df[column][df[column] != 0]
        if not non_zero_values.empty:
            avg = non_zero_values.mean()
            df[column] = df[column].replace(0, avg)
    df.to_excel(output_path, index=False)

    # here comes the graph part
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, palette='tab10')
    plt.title('Temperature Readings Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(output_path,graph_path)
    return output_path,graph_path
