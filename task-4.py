import cv2
from ultralytics import YOLO

def main():
    # 1. Load a pre-trained YOLOv8 model 
    # 'yolov8n.pt' is the nano model (fastest, great for real-time webcam processing)
    model = YOLO('yolov8n.pt')

    # 2. Set up real-time video input using OpenCV (0 is typically the default webcam)
    # To use a video file instead, replace 0 with the file path (e.g., 'video.mp4')
    video_source = 0
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Starting video stream. Press 'q' to quit.")

    while True:
        # Read a frame from the video source
        success, frame = cap.read()
        if not success:
            print("Video stream ended or cannot be read.")
            break

        # 3 & 4. Process frame for detection and tracking
        # persist=True tells the tracker to remember objects across frames
        # You can specify tracker="botsort.yaml" or tracker="bytetrack.yaml" (default is bot-sort)
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)

        # 5. Display the output with bounding boxes, labels, and tracking IDs
        # results[0].plot() automatically draws the bounding boxes and IDs on the frame
        annotated_frame = results[0].plot()

        # Show the frame in a window
        cv2.imshow("YOLOv8 Real-Time Object Tracking", annotated_frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()