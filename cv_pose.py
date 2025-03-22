import cv2
import onnxruntime as ort
import numpy as np
import time

providers = ['CPUExecutionProvider']

def nms(boxes, scores, iou_threshold=0.5):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=iou_threshold) # tolist()变成数组
    return indices.flatten() if len(indices) > 0 else [] # flatten变成一维数组

def connect_keypoints(image, keypoints, skeleton, colors):
    for i, (start, end) in enumerate(skeleton):
        if keypoints[start][2] > 0.5 and keypoints[end][2] > 0.5:
            x1, y1 = int(keypoints[start][0]), int(keypoints[start][1])
            x2, y2 = int(keypoints[end][0]), int(keypoints[end][1])
            cv2.line(image, (x1, y1), (x2, y2), colors[i % len(colors)], 2)

def postprocess(output_boxs, keypoints, original_image, input_size, original_data, conf_thresh):
    input_h, input_w = input_size
    orig_h, orig_w = original_data
    scale_h, scale_w = orig_h / input_h, orig_w / input_w 

    boxes = []
    confidences = []
    kpts_list = []

    for i in range(output_boxs.shape[0]): # [8400, 5]
        conf = output_boxs[i][-1] # -1最后一个
        if conf >= conf_thresh:
            x_center = output_boxs[i][0] * scale_w
            y_center = output_boxs[i][1] * scale_h
            width = output_boxs[i][2] * scale_w
            height = output_boxs[i][3] * scale_h
            x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
            x2, y2 = int(x_center + width / 2), int(y_center + height / 2)


            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(conf)
            cur_kps = keypoints[i].reshape(-1, 3)
            cur_kps[:, 0] *= scale_w
            cur_kps[:, 1] *= scale_h
            kpts_list.append(cur_kps)

            indices = nms(boxes, confidences)

    skeleton = [
        (0, 1), (1, 3), (0, 2), (2, 4),  
        (0, 5), (5, 7), (7, 9),         
        (0, 6), (6, 8), (8, 10),        
        (5, 11), (11, 13), (13, 15),    
        (6, 12), (12, 14), (14, 16),    
        (11, 12) 
    ] # skeleton是一个列表，每个元素是一个元组（start, end）

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 128, 0), (128, 0, 128),
        (0, 128, 128), (128, 128, 128)
    ]

    for i in indices:
        box = boxes[i]
        x1, y1, w, h = box
        conf = confidences[i]
        cv2.rectangle(original_image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        cv2.putText(original_image, f"person {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        connect_keypoints(original_image, kpts_list[i], skeleton, colors)

    return original_image

def preprocess_image(frame, input_size=(640, 640)):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_data = frame_rgb.shape[:2]
    frame_resized = cv2.resize(frame_rgb, input_size)
    frame_nomalized = frame_resized / 255.0
    input_tensor = frame_nomalized.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
    return input_tensor, original_data

def pose_inference(model_path):
    session_option = ort.SessionOptions()
    session_option.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_option=session_option, providers=providers)
    return session

if __name__ == '__main__':
    model_path = "yolo11n-pose.onnx"
    session = pose_inference(model_path)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
    else:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("无法读取帧")
                break
            
            input_tensor, original_data = preprocess_image(frame)
            outputs = session.run([output_name], {input_name: input_tensor})
            output_rs = outputs[0][0].T

            boxs = output_rs[:, 0:5]
            keypoints = output_rs[:, 5:]

            input_size = (640, 640)

            result_frame = postprocess(boxs, keypoints, frame, input_size, original_data, conf_thresh=0.5)
    
            end_time = time.time()
            fps = 1 / (end_time - start_time)

            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示结果帧
            cv2.imshow("YOLOv11n-Pose Real-Time Detection", result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
