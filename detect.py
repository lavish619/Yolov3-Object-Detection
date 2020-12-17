import cv2
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights",default='./yolov3.weights',
                help="path to model weights file")
ap.add_argument("-m", "--model",default='./yolov3.cfg',
                help="path to pre-trained model cfg file")
ap.add_argument("-l", "--labels", default='./yolov3.txt',
                help="path to classes/labels file")
ap.add_argument("-c", "--confidence",  type=float, default=0.5,
                help="confidence threshold to filter weak detections")
ap.add_argument("-n", "--nms_threshold", type=float, default=0.4,
                help="non max suppression threshold")
ap.add_argument("-i", "--Image", help="path to Image to be detected")
ap.add_argument("-v", "--Video", default = 0, help="path to Video to be detected")
args = vars(ap.parse_args())

import os  
assert os.path.exists(args['labels']) == True
assert os.path.exists(args['weights']) == True
assert os.path.exists(args['model']) == True

if args['Image'] is not None:
    assert os.path.exists(args['Image']) == True
    filename = args['Image'].split('/')[-1].split('.')[0]
if args['Video']!=0:
    assert os.path.exists(args['Video']) == True
    filename = args['Video'].split('/')[-1].split('.')[0]

print("All files present...")

# read class names from text file
#pre-trained model on 80 classes of coco dataset
classes = None
with open(args['labels'], 'r') as f:
     classes = [line.strip() for line in f.readlines()]
print("classes read..")

scale = 0.00392
conf_threshold = args['confidence']
nms_threshold = args['nms_threshold']

# generate different colors for different classes 
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# function to get the output layer names 
# in the architecture
def get_output_layers(net): 
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])+"  "+"{:0.2f}".format(confidence*100)+'%'
    color = colors[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x+15,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

def get_detections(image, net):
    
    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)

    # run inference through the network
    # and gather predictions from output layers
    detections = net.forward(get_output_layers(net))
    return detections
#     print('detections', detections)

def filter_detection(detections, image):
    Width = image.shape[1]
    Height = image.shape[0]
    
    # initialization
    class_ids = []
    confidences = []
    boxes = []
    
    # for each detetion from each output layer get the confidence, class id, 
    #bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
    
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        
    return image


# capturing image/video input
if args['Image'] is not None:
    cap = cv2.VideoCapture(args['Image'])
else:
    cap = cv2.VideoCapture(args['Video'])
    
if (cap.isOpened() == False):  
    print("Error reading image/video file") 
    
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
if args['Image'] is None:
    writer = cv2.VideoWriter('./output/'+str(filename)+'_out.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (frame_width, frame_height)) 

# read pre-trained model and config file
net = cv2.dnn.readNet(args['weights'], args['model'])
print("Model Loaded")

print("Detecting objects...")
while True:
    ret, frame = cap.read()
    if ret:
        detections = get_detections(frame,net)
        image = filter_detection(detections, frame)
        
        # display output image    
        cv2.imshow(filename, image)
        
        # save output image to disk
        if args['Image'] is not None:
            cv2.waitKey(0)
            cv2.imwrite("./output/"+filename+"_out.jpg", image)
        else:
            writer.write(image)
            
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    else:
        break

# release resources
cap.release()
if args['Image'] is None:
    writer.release() 
cv2.destroyAllWindows()





















