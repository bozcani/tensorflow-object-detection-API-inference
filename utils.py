import cv2
import numpy as np

def display_predictions(img, predictions, min_score_threshold):
    img = img.copy()
    h,w = img.shape[:2]

    bboxes = predictions['bboxes'] 

    for i in range(len(bboxes)):
        if bboxes[i]['score']>min_score_threshold:
            x1 = int((bboxes[i]['xmin'])*w)
            y1 = int((bboxes[i]['ymin'])*h)
            x2 = int((bboxes[i]['xmax'])*w)
            y2 = int((bboxes[i]['ymax'])*h)

            cv2.putText(img, str(bboxes[i]["cat_name"])+":"+str(bboxes[i]["category"]), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,  
                    1, (0,0,255), 1, cv2.LINE_AA) 
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 1)

    return img

def filter_out_predictions(predictions, iou_threshold, min_score_threshold):

    bboxes = predictions['bboxes']

    eliminated_indices = []
    for i in range(len(bboxes)):
        ymin_i, xmin_i, ymax_i, xmax_i = bboxes[i]['ymin'], bboxes[i]['xmin'], bboxes[i]['ymax'], bboxes[i]['xmax']
        for j in range(len(bboxes)):
            if i==j:
                continue
            ymin_j, xmin_j, ymax_j, xmax_j = bboxes[j]['ymin'], bboxes[j]['xmin'], bboxes[j]['ymax'], bboxes[j]['xmax']
            iou = bb_intersection_over_union((xmin_i, ymin_i, xmax_i, ymax_i), ((xmin_j, ymin_j, xmax_j, ymax_j)))

            if iou>=iou_threshold:
                if bboxes[i]['score']>bboxes[j]['score']:
                    eliminated_indices.append(j)
                else:
                    eliminated_indices.append(i)

    new_bboxes=[]
    for k in range(len(bboxes)):
        if k not in eliminated_indices:
            new_bboxes.append(bboxes[k])

    predictions['bboxes']=new_bboxes
    return predictions


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou                