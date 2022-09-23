import cv2, torch, os
print('availabe:',torch.cuda.is_available() )
print('devices available', torch.cuda.device_count())
print('device id:',torch.cuda.current_device() )
print('device address', torch.cuda.device(0))
print('gpu model',torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

import cv2
print("DNN_BACKEND_CUDA",cv2.dnn.DNN_BACKEND_CUDA)
print("DNN_BACKEND_CUDA",cv2.dnn.DNN_TARGET_CUDA)


video_capture = cv2.VideoCapture(0)
anterior = 0
while True:
    if not video_capture.isOpened():
        print('Unable to load camera. Use the command "xhost +"')
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)