from styx_msgs.msg import TrafficLight
import rospy
import rospkg
import numpy as np
import os
import tensorflow as tf

class TLClassifier(object):
    def __init__(self):
        self.threshold = 0.5
        self.load_model()

    def load_model(self):
        rospack = rospkg.RosPack()
        model_path = os.path.join(rospack.get_path("tl_detector"),"model.pb")

        rospy.loginfo('Loading model: %s', model_path)

        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph = graph)

        self.image_tensor = graph.get_tensor_by_name('image_tensor:0')
        self.boxes_tensor = graph.get_tensor_by_name('detection_boxes:0')
        self.scores_tensor = graph.get_tensor_by_name('detection_scores:0')
        self.classes_tensor = graph.get_tensor_by_name('detection_classes:0')
        self.detections_tensor = graph.get_tensor_by_name('num_detections:0')

    def light_detection(self, image):
        image = np.expand_dims(image, axis=0)
        ops = [self.classes_tensor, self.scores_tensor, self.boxes_tensor, self.detections_tensor]
        detection_classes, detection_scores, _, _ = self.sess.run(ops, feed_dict = { self.image_tensor : image })
    
        fin_classes = []
        fin_scores = []
        
        for cla, scr in zip(detection_classes[0], detection_scores[0]):
            if scr > self.threshold:
                fin_classes.append(cla)
                fin_scores.append(scr)
             
        if len(fin_classes) == 0:
            return TrafficLight.UNKNOWN
            
        mean_class = np.ceil(sum(np.multiply(fin_scores, fin_classes)) / sum(fin_scores))

        #print(fin_classes)
        
        if mean_class == 1:
            return TrafficLight.GREEN
        elif mean_class == 3:
            return TrafficLight.YELLOW
        elif mean_class == 2:
            return TrafficLight.RED            
        else:
            return TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
        image (cv::Mat): image containing the traffic light

        Returns:
        int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        return self.light_detection(image)

