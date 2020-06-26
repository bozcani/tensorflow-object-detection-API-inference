from detector import Detector
import os, sys
import cv2
import numpy as np
import json
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
    IMG_PATH="samples/data/data/env1_m30_view/env1_m30_view_01422.png"
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
    """
    folder_names = ["env1_m30_view",
                    "env1_m45_view",
                    "env2_m30_view",
                    "env2_m45_view",
                    "env3_m30_view",
                    "env3_m45_view"]
    """
    folder_names = ["/home/ilkerbozcan/repos/uav-indoor-anomaly-detection/test_data/test1",
                    "/home/ilkerbozcan/repos/uav-indoor-anomaly-detection/test_data/test2",
                    "/home/ilkerbozcan/repos/uav-indoor-anomaly-detection/test_data/test3",
                    "/home/ilkerbozcan/repos/uav-indoor-anomaly-detection/test_data/test4"]

    for folder_name in folder_names:
        os.mkdir(folder_name+"_out")
        results = obj.run_on_image_folder(path_to_folder=folder_name,
                                            save_dir="./",
                                            display=False,
                                            show_mask=False)

        i=0

        filtered_results=[]
        for predictions in results:
            img = cv2.imread(predictions['img_name'])
            filtered_predictions = filter_out_predictions(predictions, iou_threshold=0.5, min_score_threshold=0.5)

            disp = display_predictions(img, filtered_predictions, min_score_threshold=0.5)
            cv2.imwrite(folder_name+"_out/"+str(i)+".png", disp)
            i += 1
            print(i)
            filtered_results.append(filtered_predictions)
        # -------------------------------
        print(filtered_results)
        with open(folder_name+"/"+"annotations.json", 'w+') as f:
            json.dump(filtered_results, f)
    

if __name__=='__main__':
    main(sys.argv)
   
