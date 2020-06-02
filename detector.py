import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import cv2
import numpy as np
import tensorflow as tf
import wget
import tarfile
import json
from shutil import copyfile

# Object detection module imports
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from object_detection.utils import ops as utils_ops

class Detector:
    model_name = None # Placeholder
    detection_graph = None # Placeholder
    category_index = None # Placeholder
    param = None # Placeholder

    def __init__(self, param):
        """Default constructer.

        Args:
            param (param): A dictionary including config parameters.
        """

        self.param = param
        print(" [ INFO ] Detector.init: Task created with empty settings.")
        print(" [ INFO ] Detector.init: {} object categories including {} objects used.".format(self.param['categories'], self.param['num_classes']))
        print(" [ INFO ] Detector.init: Model is being set to {}".format(self.param['model_name']))
        self._set_model(self.param['model_name'])

        print(" [ INFO ] Detector.init: Model is being loaded from {}".format(param['model_path']))

        self._load_model()
        #self.sess = tf.Session(graph=detection_graph)

    def _set_model(self, model_name):
        """Set model.

        Args:
            model_name (str): Model name form Tensorflow Object Detector API model zoo.
                    model zoo: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
        """
        if self.model_name is not None:
            print(" [ INFO ] Detector.set_model: Model is changed to {} from {}.".format(self.model_name, model_name))
        else:
            print(" [ INFO ] Detector.set_model: Model is set first time: {}.".format(model_name))
        self.model_name = model_name

    def _load_model(self):
        """Load model from tensorflow frozen graph. Download it if does not exist."""
        
        if os.path.isdir(self.param['model_path']):
            pass

        else:
            print("\033[93m [ WARNING ] Detector.load_model: Model does not exist {}\033[0m.".format(self.param['model_path']))
            os.makedirs(self.param['model_parent'], exist_ok=True)
            print("\033[93m[ WARNING ] Detector.load_model: Model path created. {}\033[0m.".format(self.param['model_path']))


            print("\033[93m [ WARNING ] Downloading from {}\033[0m.".format(self.param['model_url']))
            fname = wget.download(self.param['model_url']+".tar.gz", out=self.param['model_parent'])

            tar = tarfile.open(fname, "r:gz")
            tar.extractall(path=self.param['model_parent'])
            tar.close()
            os.remove(os.path.join(self.param['model_parent'], self.param['model_name']+".tar.gz"))

            
        self.detection_graph = tf.compat.v1.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(self.param['path_to_ckpt'], 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            
            self.tensor_dict=tensor_dict
            self.image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

            ## Loading label map
            # Label maps map indices to category names, so that when our convolution network predicts `5`,
            # we know that this corresponds to `airplane`.  Here we use internal utility functions,
            # but anything that returns a dictionary mapping integers to appropriate string labels would be fine
            label_map = label_map_util.load_labelmap(self.param['path_to_labels'])
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.param['num_classes'], use_display_name=True)
            self.category_index = label_map_util.create_category_index(categories)


        self.sess = tf.compat.v1.Session(graph=self.detection_graph)




    def run_on_video(self, path_to_video, save=True, display=True, write_annotations=True):
        cap = cv2.VideoCapture(path_to_video)

        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("\033[91m [ ERROR ] Detector.run_on_video: Video does not exist: {}\033[0m.".format(path_to_video))

        dirname = path_to_video.split(".")[0]
        dest_folder = dirname+"_output"
        if not os.path.isdir(dest_folder):
            os.mkdir(dest_folder)
            os.mkdir(os.path.join(dest_folder, "images"))
        else:
            print("\033[91m [ ERROR ] Detector.run_on_video: Cannot create annotations. Annotation folder already exists: {}. Set write_annotations to False or remove the annotation folder to recreate.\033[0m".format(path_to_video))
            return {"annotations":[]}
        i=0
        output = {"annotations":[]}
        # Read until video is completed
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                h,w = frame.shape[:2]
                im_ori = frame.copy()
                detections  = self.run_on_image(frame, display=display)
                annotations = self.prepare_annotation(detections, w, h)

                cv2.imwrite(os.path.join(dest_folder, "images", "{:07d}.jpg".format(i)), im_ori)
                output["annotations"].append({"image":"{:07d}.jpg".format(i),
                                "annotations": annotations})
                i+=1
            else: 
                break

        
        # Write annotations.
        with open(os.path.join(dest_folder, 'annotations.json'), 'w+') as f:
            json.dump(output, f)

        # When everything done, release the video capture object
        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()
        return annotations

    def prepare_annotation(self, detections, im_width, im_height):
        annotations = []
        for detection in detections:
            annotations.append({"class_id": detection[0],
                                "score": str(detection[1]),
                                "ymin": int(detection[2][0]*im_height),
                                "xmin": int(detection[2][1]*im_width),
                                "ymax": int(detection[2][2]*im_height),
                                "xmax": int(detection[2][3]*im_width)})
        return annotations


    def run_on_image(self, img, display=True, show_mask=False):
        """Run object detector on given image.

        Args:
            img (np.array or str): Input image, numpy data or path.

        Returns:
            list: List of detected objects. 
        """
        assert (type(img)==np.ndarray) or (type(img)==str), "Invalid type: {}, should be np.array (image data) or str (image path)".format(type(img))      

        if type(img)==np.ndarray:
            image=img
        if type(img)==str:
            image=cv2.imread(img)

        # Run inference
        output_dict = self.sess.run(self.tensor_dict,
                            feed_dict={self.image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]


        if show_mask:
            instance_masks = output_dict.get('detection_masks').astype(np.float32)
        else:
            instance_masks = None    

        """
        # Visualization of the results of a detection.
        objects=vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=instance_masks,
            min_score_thresh=0.1,
            use_normalized_coordinates=True,
            line_thickness=8)


        if display==True:
            cv2.imshow("img", image)
            cv2.waitKey(0)
        """

        output_names = [self.category_index[c]['name'] for c in output_dict['detection_classes'][:output_dict['num_detections']]]
        #output_names = []

        output={}
        if type(img)==str:
            output['img_name']=img
        else:
            output['img_name']=None

        output['bboxes']=[]
        for i in range(output_dict['num_detections']):
            annot = {}
            annot["category"] = output_dict['detection_classes'][i]
            annot["score"] = output_dict['detection_scores'][i]
            annot["xmin"]  = output_dict['detection_boxes'][i][1]
            annot["ymin"]  = output_dict['detection_boxes'][i][0]
            annot["xmax"]  = output_dict['detection_boxes'][i][3]
            annot["ymax"]  = output_dict['detection_boxes'][i][2]
            annot["cat_name"] = output_names[i]

            output["bboxes"].append(annot)

        return output



    def run_on_image_folder(self, path_to_folder, save_dir=None, display=False, show_mask=False):
        """Run object detector on images in given folder.

        Args:
            path_to_folder (np.array): Path to image folder.

        Returns:
            list: List of detected objects. 
        """

        onlyfiles = [f for f in os.listdir(path_to_folder) if os.path.isfile(os.path.join(path_to_folder, f))]    

        results = []
        for fname in onlyfiles:
            predictions = self.run_on_image(os.path.join(path_to_folder, fname), display=display, show_mask=show_mask)
            results.append(predictions)
        return results    