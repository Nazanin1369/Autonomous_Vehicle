from styx_msgs.msg import TrafficLight
import cv2
import numpy as np


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
#        self.lower_red_1 = np.array([0, 43, 46])
#        self.upper_red_1 = np.array([10, 255, 255])
#        self.lower_red_2 = np.array([156, 43, 46])
#        self.upper_red_2 = np.array([180, 255, 255])
        self.lower_red_1 = np.array([0, 100, 100])
        self.upper_red_1 = np.array([10, 255, 255])
        
        self.lower_red_2 = np.array([160, 100, 100])
        self.upper_red_2 = np.array([179, 255, 255])
        
        self.lower_green = np.array([35, 43, 46])
        self.upper_green = np.array([77, 255, 255])
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        
        # Generating templates for Red, Yellow and Greeb
        h = 30
        w = 30
        self.putative_red = np.zeros((h, w,  3), dtype = np.uint8)
        self.putative_yellow = np.zeros((h, w,  3), dtype = np.uint8)
        self.putative_green = np.zeros((h, w,  3), dtype = np.uint8)
                
        # now fill in the 1s in a dist of radius template_dim / 2 and
        # centrer (template_dim / 2, template_dim / 2)
        radius_x =  w / 2 - 4 # leave some margin for black pixels
        radius_y = h / 2 - 4         
        cx = w / 2
        cy = h / 2
        N = 30
         
        for deg in range(360):
            for i in range(N):
                r_x = (1.0 * radius_x) / N * (i + 1)
                r_y = (1.0 * radius_y) / N * (i + 1)
                                
                x = int(r_x * np.cos(deg * np.pi / 180) + cx )
                y = int(r_y * np.sin(deg * np.pi / 180) + cy )
                # NOTE: Create a BGR to acommodate OpenCV conventions....                
                self.putative_red[y][x][0] = 0
                self.putative_red[y][x][1] = 0
                self.putative_red[y][x][2] = 255
                # an yellow-slightly orange-ish color...
                self.putative_yellow[y][x][0] = 30
                self.putative_yellow[y][x][1] = 255
                self.putative_yellow[y][x][2] = 255
                # green
                self.putative_green[y][x][0] = 0
                self.putative_green[y][x][1] = 255
                self.putative_green[y][x][2] = 0
                
        
        #self.putative_yellow = cv2.imread("/home/george/SelfDrivingCar-Final/CarND-Capstone/data/yellow_tl.png")
        

    def matchRedTemplate(self, image, template_x, template_y):
        
        # why hald the template? Because we want to avoid delays....
        w = template_x / 2
        h = template_y / 2
        # first create a template of size template_dim
#        template = np.zeros((template_y, template_x,  3), dtype = np.uint8)
#        # now fill in the 1s in a dist of radius template_dim / 2 and
#        # centrer (template_dim / 2, template_dim / 2)
#        radius_x =  template_x / 2 - 2 # leave some margin for black pixels
#        radius_y = template_y /2 - 2         
#        cx = template_x / 2
#        cy = template_y / 2
#        N = 100
#         
#        for deg in range(360):
#            for i in range(N):
#                r_x = (1.0 * radius_x) / N * (i + 1)
#                r_y = (1.0 * radius_y) / N * (i + 1)
#                                
#                x = int(r_x * np.cos(deg * np.pi / 180) + cx )
#                y = int(r_y * np.sin(deg * np.pi / 180) + cy )
#                # NOTE: Create a BGR to acommodate OpenCV conventions....                
#                template[y][x][0] = 0
#                template[y][x][1] = 0
#                template[y][x][2] = 255
        
        template_red = cv2.resize(self.putative_red, (w, h) )
        template_yellow = cv2.resize(self.putative_yellow, (w, h))
        template_green = cv2.resize(self.putative_green, (w, h))
        
        # half the image too. It speeds up the template search without significant loses in performance
        image = cv2.resize(image, (image.shape[1] /2 , image.shape[0] / 2))
        
        # Convert images to hsv
        #template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
        #image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Now doing plain old template matching in hsv space!               
        # All the 6 distances
        #methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
        #           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        
        method = 'cv2.TM_CCORR_NORMED'
        display_img = image.copy()
        method = eval(method)

        # Apply template Matching
        res_red = cv2.matchTemplate(image, template_red, method)
        min_valr, max_valr, min_locr, max_locr = cv2.minMaxLoc(res_red)
        
        res_yellow = cv2.matchTemplate(image, template_yellow, method)
        min_valy, max_valy, min_locy, max_locy = cv2.minMaxLoc(res_yellow)
        
        res_green = cv2.matchTemplate(image, template_green, method)
        min_valg, max_valg, min_locg, max_locg = cv2.minMaxLoc(res_green)
                
        
        if (max_valr > 0.7): # if so, then its DEFINITELY a RED!
            state = TrafficLight.RED            
            min_loc = min_locr
            max_loc = max_locr
            color = (0, 0, 255)            
            
        elif(max_valg > 0.7):
            state = TrafficLight.GREEN
            color = (0, 255, 0)
            min_loc = min_locg
            max_loc = max_locg
        elif(max_valy > 0.757):
            state = TrafficLight.YELLOW
            color = (0, 255, 255)
            min_loc = min_locy
            max_loc = max_locy
        else:
            return(None, TrafficLight.UNKNOWN)
            

        #print("Maximum template score : ", 1.0 * max_val  )

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
    
        bottom_right = (top_left[0] + w , top_left[1] + h)

        cv2.rectangle(display_img, top_left, bottom_right, color, 2)

        

        return (display_img, state)


    
                    
