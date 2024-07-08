from collections import defaultdict
import cv2
import json

from ultralytics import YOLO, solutions
from ultralytics.utils import click

model = YOLO("yolov8m.pt", verbose=False)

video_path = "your_output.ts"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

_, frame = cap.read()
h, w = frame.shape[:2]

mask1 = click.click(frame, "mask1.txt", saveConfig=True)
mask2 = click.click(frame, "mask2.txt", saveConfig=True)

writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

region_line = mask1.allPts[0]  # tek çizgi
region_mask = mask2.mask  # en az 2 alan

color = (0, 255, 0)  # ekran yazısının rengi # B G R

counter = solutions.ObjectCounter(
    view_img=False,
    reg_pts=region_line,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2,
    mask=region_mask,
    fps=fps,
    dist=40  # bir kutunun başlangıcı ile diğer kutunun arasında kalan mesafe (metre)
)

track_history = defaultdict(lambda: [])
frame_num = 0

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    frame_num += 1

    tracks = model.track(im0, persist=True, show=False, verbose=False)

    im0 = counter.start_counting(im0, tracks, frame_num)
    cv2.putText(im0, json.dumps(counter.class_wise_count), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    cv2.putText(im0, f"current speed estimation: {round(counter.current_speed, 1)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2, cv2.LINE_AA)

    cv2.imshow("YOLOv8 Tracking", im0)
    writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"total in counts: {counter.in_counts}, total out counts: {counter.out_counts}")
counter.history.to_csv("./history.csv")
