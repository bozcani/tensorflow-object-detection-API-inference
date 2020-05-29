from detector import Detector
import os, sys
import cv2
import numpy as np

from utils import filter_out_predictions, display_predictions

# Detection
param = {}
#param['num_classes'] = 90
param['num_classes'] = 49


param['categories'] = "COCO"
#param['model_name'] = "faster_rcnn_resnet101_coco_2018_01_28"
#param['model_name'] = "faster_rcnn_nas_coco_2018_01_28"
#param['model_name'] = "ssd_mobilenet_v1_coco_2018_01_28"
#param['model_name'] = "mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
#param['model_name'] = "alet_checkpoint"
param['model_name'] = "alet_faster_rcnn_resnet101_coco_checkpoint_808167"

#param['model_name']='alet'
#"mask_rcnn_resnet101_atrous_coco_2018_01_28"

param['model_parent'] = "models"
param['model_path'] = os.path.join(param['model_parent'], param['model_name'])
param['model_zoo_link'] = "http://download.tensorflow.org/models/object_detection/"
param['model_url'] = os.path.join(param['model_zoo_link'], param['model_name'])
param['path_to_ckpt'] = os.path.join(param['model_path'], "frozen_inference_graph.pb")
#param['path_to_labels'] = os.path.join('configs', 'mscoco_label_map.pbtxt')
param['path_to_labels'] = os.path.join('configs', 'alet_label_map.pbtxt')


def main(args):
    obj=Detector(param)

    
    #detections = obj.run_on_video("samples/test_video.mp4", 
    #                                display=True,
    #                                write_annotations=True)
    
    img = cv2.imread("samples/sample_folder/5070.png")
    img_ori = img.copy()
    
    temp = img.copy()
    predictions = obj.run_on_image(temp, display=False, show_mask=False)
    cv2.imwrite("res_builtin.jpg", temp) 

    temp = img.copy()
    disp = display_predictions(temp, predictions, min_score_threshold=0.05)
    cv2.imwrite("res.jpg", disp) 

    temp = img.copy()
    filtered_predictions = filter_out_predictions(predictions, iou_threshold=0.8, min_score_threshold=0.05)
    disp = display_predictions(temp, filtered_predictions, min_score_threshold=0.05)
    cv2.imwrite("res_filterd.jpg", disp) 


    """
    results = obj.run_on_image_folder(path_to_folder="samples/sample_folder_2",
                                        save_dir="./",
                                        display=False,
                                        show_mask=False)

    print(results)
    """

if __name__=='__main__':
    main(sys.argv)
   
