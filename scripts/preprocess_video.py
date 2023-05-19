import cv2
from pathlib import Path

if __name__ == '__main__':
    video_path = Path('../data/outdoor.MOV')
    output_dir = Path('../data/outdoor_imgs')

    output_dir.mkdir(exist_ok=True)

    cam = cv2.VideoCapture(str(video_path))
    frame_count = 0
    while(True):
        ret, frame = cam.read()
        if ret:
            if frame_count % 5 == 0:
                output_path = output_dir / f"indoor_{frame_count:04d}.jpg"
                print(f"Creating {output_path}")

                cv2.imwrite(str(output_path), frame)
            frame_count += 1
        else:
            break