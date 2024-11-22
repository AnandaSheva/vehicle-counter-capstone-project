from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\lukad\vehicle-counter-capstone-project-1\ifdhal\models\best.pt")

results = model.predict(source=r"C:\Users\lukad\vehicle-counter-capstone-project-1\Untitled.mp4", save=True)

vehicle_classes = [0, 1, 2, 3]

for result in results:
    frame = result.orig_img  
    for box in result.boxes:
        class_id = int(box.cls)  
        if class_id in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            confidence = box.conf.item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{result.names[class_id]} {confidence:.2f}"  
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Filtered Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
