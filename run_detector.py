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
param['model_name'] = "alet_faster_rcnn_resnet101_coco_checkpoint_fixed_181432"

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

    """
    #detections = obj.run_on_video("samples/test_video.mp4", 
    #                                display=True,
    #                                write_annotations=True)
    """


    """
    # -------------------------------
    # ------ Run on single image ----
    # -------------------------------
    IMG_PATH="samples/6900.jpg"
    img = cv2.imread(IMG_PATH)
    
    predictions = obj.run_on_image(IMG_PATH, display=False, show_mask=False)

    disp = display_predictions(img, predictions, min_score_threshold=0.05)
    cv2.imwrite("res.jpg", disp) 

    
    filtered_predictions = filter_out_predictions(predictions, iou_threshold=0.5, min_score_threshold=0.05)
    disp = display_predictions(img, filtered_predictions, min_score_threshold=0.05)
    cv2.imwrite("res_filterd.jpg", disp) 
    # -------------------------------
    """


    # -------------------------------
    # ------ Run on image folder ----
    # -------------------------------
    results = obj.run_on_image_folder(path_to_folder="samples/sample_folder_2",
                                        save_dir="./",
                                        display=False,
                                        show_mask=False)

    i=0
    for predictions in results:
        img = cv2.imread(predictions['img_name'])
        filtered_predictions = filter_out_predictions(predictions, iou_threshold=0.5, min_score_threshold=0.5)

        disp = display_predictions(img, filtered_predictions, min_score_threshold=0.5)
        cv2.imwrite(str(i)+".png", disp)
        i += 1
    # -------------------------------



if __name__=='__main__':
    main(sys.argv)
   
