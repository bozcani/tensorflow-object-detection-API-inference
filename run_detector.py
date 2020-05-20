from detector import Detector
import os, sys
import cv2
import numpy as np

# Detection
param = {}
param['num_classes'] = 80
param['categories'] = "COCO"
#param['model_name'] = "faster_rcnn_resnet101_coco_2018_01_28"
#param['model_name'] = "faster_rcnn_nas_coco_2018_01_28"
param['model_name'] = "ssd_mobilenet_v1_coco_2018_01_28"
#param['model_name'] = "mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
#param['model_name']='alet'
#"mask_rcnn_resnet101_atrous_coco_2018_01_28"

param['model_parent'] = "models"
param['model_path'] = os.path.join(param['model_parent'], param['model_name'])
param['model_zoo_link'] = "http://download.tensorflow.org/models/object_detection/"
param['model_url'] = os.path.join(param['model_zoo_link'], param['model_name'])
param['path_to_ckpt'] = os.path.join(param['model_path'], "frozen_inference_graph.pb")
param['path_to_labels'] = os.path.join('configs', 'mscoco_label_map.pbtxt')


def main(args):
    obj=Detector(param)

    
    #detections = obj.run_on_video("samples/test_video.mp4", 
    #                                display=True,
    #                                write_annotations=True)
    
    img = cv2.imread("samples/6660.jpg")

    objects=obj.run_on_image(img, display=True, show_mask=False)

    cv2.imwrite("res.jpg", img) 

    print(objects)


if __name__=='__main__':
    main(sys.argv)
   
