import argparse
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolov8x-pose.pt")  # COCO-17
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    kxy_list, kconf_list = [], []

    for _ in tqdm(range(frames)):
        ok, frame = cap.read()
        if not ok:
            break
        res = model(frame, conf=args.conf, verbose=False)[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            # no detection: fill zeros
            kxy = np.zeros((17,2), dtype=np.float32)
            kconf = np.zeros((17,), dtype=np.float32)
        else:
            # pick largest person
            areas = (res.boxes.xyxy[:,2]-res.boxes.xyxy[:,0]) * (res.boxes.xyxy[:,3]-res.boxes.xyxy[:,1])
            i = int(areas.argmax().item())
            kxy = res.keypoints.xy[i].cpu().numpy().astype(np.float32)     # (17,2)
            kconf = res.keypoints.conf[i].cpu().numpy().astype(np.float32) # (17,)
        kxy_list.append(kxy)
        kconf_list.append(kconf)

    cap.release()

    kxy = np.stack(kxy_list, axis=0)      # (T,17,2)
    kconf = np.stack(kconf_list, axis=0)  # (T,17)

    np.savez_compressed(args.out, kxy=kxy, kconf=kconf, fps=fps)
    print("Saved:", args.out, "T=", kxy.shape[0], "fps=", fps)

if __name__ == "__main__":
    main()