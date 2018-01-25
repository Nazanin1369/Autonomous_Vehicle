#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import numpy as np
import random
import math

STATE_COUNT_THRESHOLD = 3
# this means how many frames the image is delay
# set DELAY=1 means there is no delay.
DELAY = 1
# if a light is more then 200m away from the car
# we'll ignore that light
MAX_DISTANCE_SQR = 40000

class Point:
    def __init__(self, t):
        self.x = t[0]
        self.y = t[1]

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.tl_wps = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        rospy.logwarn("The parameter string is : %s", config_string )
        self.config = yaml.load(config_string)

        self.upcoming_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        
        # debug: publisher of cross-correlation results
        self.ccresult_pub = rospy.Publisher('/traffic_ccresult', Image, queue_size=1)
        
                
        
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.count = 0
        self.camera_car_position = []
       
        
        # x-margin is the margin in pixels that corrspondes to a
        # 5 meter distance from the traffic light along the y-axis (x in image).
        # we use this margin to search for the red pixels instead of a fixed one        
        self.x_margin = 50 # default is 50 (Yuda)
        
        # dimensions of the red-light template in pixels
        self.template_x = 30 # (pixels in x) just a wild-guess for initial value. 
        self.template_y = 30 # (pixels in y)
        # the actual dimension of a single light as a swuare         
        self.TrafficLight_dim = 1.6# (suppose to be in meters but scaling is a bit hand-tuned) 
        
        
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
       
    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
#        self.camera_car_position.append(self.pose.pose.position)
#        if len(self.camera_car_position) > DELAY:
#            self.camera_car_position.pop(0)
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            if state == TrafficLight.UNKNOWN:        
                self.upcoming_light_pub.publish(Int32(-1))
                return
            self.last_wp = light_wp
            self.upcoming_light_pub.publish(Int32( light_wp | (self.state << 16) ))
        else:
            self.upcoming_light_pub.publish(Int32(self.last_wp | (self.last_state << 16) ))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_index = -1
        closest_dis = -1

        if self.waypoints is not None:
            wps = self.waypoints.waypoints
            for i in range(len(wps)):
                dis = (wps[i].pose.pose.position.x - pose.x) ** 2 + \
                    (wps[i].pose.pose.position.y - pose.y) ** 2

                if (closest_dis == -1) or (closest_dis > dis):
                    closest_dis = dis
                    closest_index = i
        return closest_index


    # George: Here's my version of project_to_image_plane
    #     I am avoiding the TransformListener object as I am not sure about how
    #     to configure it without having doubts. The transform is easy to 
    #     work-out directly from the pose vector and gives me control over the 
    #     coordinate frame.
    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """
        # Retreving camera intronsics
        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        # image size        
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # caching the world coordinates of the point
        Px = point_in_world.x
        Py = point_in_world.y
        Pz = point_in_world.z

        # Using the pose to obatin the camera frame as rotation matrix R and 
        # a world position p (NOTEL ASSUMING THAT THE CAMERA COINCIDES WITH 
        # THE CAR'S BARYCENTER 
#        Cx = self.camera_car_position[0].x
#        Cy = self.camera_car_position[0].y
#        Cz = self.camera_car_position[0].z
        Cx = self.pose.pose.position.x
        Cy = self.pose.pose.position.y
        Cz = self.pose.pose.position.z
        
        # print(Px, Py, Pz)
        # print(Cx, Cy, Cz)
        # print('=========')
        
        # get orientation (just the scalar part of the the quaternion)
        s = self.pose.pose.orientation.w # quaternion scalar
        
        # now obtaining orientation of the car (assuming rotation about z: [0;0;1])
        theta = 2 * np.arccos(s)
        # Constraining the angle in [-pi, pi)
        if theta > np.pi:
            theta = -(2 * np.pi - theta)

        # transforming the world point to the camera frame as:
        #
        #               Pc = R' * (Pw - C)

        #        where R' = [ cos(theta)  sin(theta)   0; 
        #                   -sin(theta)  cos(theta)   0;
        #                      0              0      1]
        #
        # Thus,
        p_camera = [ np.cos(theta) * (Px - Cx) + np.sin(theta) * (Py - Cy) , \
                     -np.sin(theta) * (Px - Cx) + np.cos(theta) * (Py - Cy) , \
                     Pz - Cz]
        # print(p_camera)                                                        


        # NOTE: From the simulator, it appears from the change in the angle 
        # that the positive direction of rotation is counter-clockwise. This 
        # means that there are two possible frame arrangements:
        #
        # a) A RIGHT-HAND frame: In this frame, z - points upwards (oposite to the 
        # image y axis)and y points to the left (oposite to the image x-axis)
        #
        # b) A LEFT_HAND frame: In this frame, z-points downwards (same as the
        #    image y-axis) and y points left (oposite to the image x-axis).
        #
        # thus, there are two ways of obtaining the image projection:
        
        
        # =============== Udacity Forums ====================================
        # see https://discussions.udacity.com/t/focal-length-wrong/358568
#        if fx < 10:
#            fx = 2574
#            fy = 2744
#            p_camera[2] -= 1.0  # to account for the elevation of the camera
#            c_x = image_width/2 - 30
#            c_y = image_height + 50
#        x = int(-fx * p_camera[1] / p_camera[0] + c_x)
#        y = int(-fy * p_camera[2] / p_camera[0] + c_y)
         # ===================================================================        
        
        
        # =============== Using actual (given) intrinsics ================
        if fx < 3 or fy < 3: 
            # This means that intrinsics have been normalized
            # so that they can be applied at any resolution                         
            fx = fx * image_width
            fy = fy * image_height
            # MAYBE, assume that camera is 1 m above the ground (not used)
            #p_camera[2] -= 1                       
        # Since we dont know the intersection of the optical axis with the image,
        # we assume it lies in the middle of it. It should work roughly....
        c_x = image_width / 2
        c_y = image_height / 2
        
        x = int(-fx * p_camera[1] / p_camera[0] + c_x)
        y = int(+fy * p_camera[2] / p_camera[0] + c_y)
        # ===============================================================

        # Set the ADAPTIVE search margin for the traffic light.
        self.x_margin = int(np.abs(6 * fx / p_camera[0])) # note the 5 metert factor
        #rospy.logwarn("X-Margin : %i", self.x_margin)
        if (self.x_margin < 20):
            self.x_margin = 20 # use default 20 pixels if too small margin
        if (self.x_margin > 200): # use maximum margin 200 pixels (just to avoid large roi images when out of range)
            self.x_margin = 200 
        
        # Now working out the size (dimension) of the traffic light 
        # (NOTE: ONLY a single light - i.e. on of the three) in pixels
        self.template_x = int( fx * self.TrafficLight_dim / p_camera[0] )
        if (self.template_x > 45 ):
            self.template_x = 45
            
        self.template_y = int( fy * self.TrafficLight_dim / p_camera[0] )
        if (self.template_y > 45 ):
            self.template_y = 45
        

        return (x,y)

    def in_image(self, x, y):
        if x is None or y is None:
            return False
        if x < 0 or x >= self.config['camera_info']['image_width']:
            return False
        if y < 0 or y >= self.config['camera_info']['image_height']:
            return False
        return True

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        x, y = self.project_to_image_plane(light.pose.pose.position)

        #TODO use light location to zoom in on traffic light in image
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # Actually this range is enough. 
        # But we just use a bigger rectangle to make sure the light is in this region
        # ret = cv2.rectangle(cv_image, (x-20,y-50), (x+20,y+50), (0, 0, 255))
        #left = x - 50
        left = x - self.x_margin        
        #right = x + 50
        right = x + self.x_margin        
        top = 20
        bottom = y + 100
        
        state = TrafficLight.UNKNOWN
              
        #if self.in_image(left, top) and self.in_image(right, bottom):
        roi = cv_image[top:bottom, left:right]
        # skip if roi not set (it may happen, despite the in_image tests)
        # NOTE: This 
        if (roi is None): 
            rospy.logwarn("ROI image is None!")
            return state
        # skip if roi is singular (i.e. single row or column)
        if (roi.shape[0] < 2 or roi.shape[1] < 2):
            return state
        # bridge will be useful publishing the classification results  
        bridge = CvBridge()
            
            
        if (self.template_x > 5) and (self.template_y):
            (ccres, state) = self.light_classifier.matchRedTemplate(cv_image, self.template_x, self.template_y)                    
            # publish the results
            if not (ccres is None):                
                self.ccresult_pub.publish(bridge.cv2_to_imgmsg(ccres, "bgr8"))
            
            
        self.count += 1
        # perform light state classification
        #state = self.light_classifier.get_classification(roi)
        #rospy.logwarn("TL state classified: %d, state count %d", state, self.state_count)
        # debug only
        # if self.count > STATE_COUNT_THRESHOLD and self.count < 10: # save some imgs, not all 
        #     cv2.imwrite('/home/student/Tests/imgs/' + ("%.3d-%d" % (self.count, state)) + '.jpg', roi)
        return state
        
    def track_index_diff(self,index1, index2):
        if (self.waypoints.waypoints is None):
            return -1        
        N = len(self.waypoints.waypoints)
        if index2 >= index1 :
            return index2 - index1
        else:
            return N - index1 - 1 + index2
   
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if (self.waypoints is None):
            return (-1, TrafficLight.UNKNOWN)
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        
        # now, for all stop_lines find nearest points (shoudl be done ONCE
        if (len(self.tl_wps)==0):           
            for i, stop_line in enumerate(stop_line_positions):
                tl_wp = self.get_closest_waypoint(Point(stop_line))
                self.tl_wps.append( (tl_wp+5) % len(self.waypoints.waypoints) ) # +1 to give extra margin towards the traffic light
            
        
        light = None
        
        if(self.pose):
            car_wp = self.get_closest_waypoint(self.pose.pose.position)

            
            # Now find the smallest distance to a traffic light wp from the car wp:
            minDist = self.track_index_diff(car_wp, self.tl_wps[0])
            light_index = 0
            for i in range(1, len(self.tl_wps)):
                dist = self.track_index_diff(car_wp, self.tl_wps[i])
                if dist > 150 / 0.63: #about 150 meters ON-TRACK 
                    continue
                if (dist < minDist):
                    minDist = dist                    
                    light_index = i
            
            light = self.lights[light_index]
            #print("Nearest Traffic light index  : ", light_index)
        
        if light:
            # print(light_wp, self.count)
            state = self.get_light_state(light)
            return self.tl_wps[light_index], state
        
        return -1, TrafficLight.UNKNOWN
       
   
    

    def process_traffic_lights_old(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose.position)

            #TODO find the closest visible traffic light (if one exists)            
            light_wp = -1           
            for i, stop_line in enumerate(stop_line_positions):
                # computing the distance from the car to the traffic-light stop line                
                dis = (stop_line[0] - self.pose.pose.position.x)**2 + \
                    (stop_line[1] - self.pose.pose.position.y)**2
                # if the distance beyond a threshold (200^2 = 40000) skip    
                if dis > MAX_DISTANCE_SQR:
                    continue
                # Now, if the traffic light is potentially visible, find the
                # closest waypoint to it                
                stop_line_wp = self.get_closest_waypoint(Point(stop_line))
                
                if stop_line_wp >= car_position:
                    if (light_wp == -1) or (light_wp > stop_line_wp):
                        print(stop_line_wp, car_position, light_wp)
                        light_wp = stop_line_wp
                        light = self.lights[i]

        #rospy.logwarn("Found closest traffic light (waypoint) : %i" , light_wp)
        if light:
            # print(light_wp, self.count)
            state = self.get_light_state(light)
            return light_wp, state
        # self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
