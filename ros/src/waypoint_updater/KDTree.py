
X_SPLIT = 0    
Y_SPLIT = 1
INF_ = 999999999.9

class KD2TreeNode(object):
    def __init__(self, x, y, s, d, index, axis):
          
          # The index of the node in the waypoint list
          self.index = index
          # The split axis of the node (X_SPLIT or Y_SPLIT)
          self.split_axis = axis
          # The point as a [x, y, s, d, index] entry         
          self.data = [x, y, s, d]
          # The left and right branches
          self.right = None
          self.left = None
          
                
    # Find the point with minimum designated coordinate
    def findMin(self, axis):
        
        if self.split_axis == axis:
            if self.left == None:
                return self
            else:
                return self.left.FindMin(axis)
        else:
            # creating two dummy entries points for comparisons
            min_left = None;
            min_right = None
            if self.left != None:  
                self.left.findMin(axis);
            
            if self.right != None:
                min_right = self.right.findMin(axis);
            # taking cases
            if min_left == None and min_right == None :
                return self
            elif min_left == None:
                if min_right.data[axis] < self.data[axis]:
                    return min_right
                else:
                    return self
            elif min_right == None:
                if min_left.data[axis] < self.data[axis]:
                    return min_left
                else:
                    return self
            else:
                if min_left.data[axis] < self.data[axis] and min_left.data[axis] < min_right.data[axis]:
                    return min_left
                elif self.data[axis] < min_left.data[axis] and self.point[axis] < min_right.data[axis]:
                    return self
                else:
                    return min_right
        #Should never be here
        return None
    
    # Insert a new point
    def insertNode(self, x, y, s, d, index):
        
        p = [x, y]
        if x == self.data[0] and y == self.data[1]:
            return False # duplicate entry attempt
        
        
        if p[self.split_axis] < self.data[self.split_axis]:
            # Inserting on the left            
            if self.left != None:
                return self.left.insertNode(x, y, s, d, index); # proceed recursively
            else: 
                #create the left branch and make it a leaf
                self.left = KD2TreeNode(x, y, s, d, index, (int)(not self.split_axis) );
                return True
        else: 
            # Inserting on the right. NOTE: If a point is on the split axis, then it goes right!!!!        
            if self.right != None:
                return self.right.insertNode(x, y, s, d, index) # proceed recursively
            else: 
                # create the right branch as a leaf
                self.right = KD2TreeNode(x, y, s, d, index, (int)(not self.split_axis))
                return True
        # Should never reach here...
        return False


    # Nearest neighbor search
    def NNSearch(self, point, min_distance, best_node):
        # (squared) distance to the data of the node
        local_best_node = best_node
        local_min_distance = min_distance
        dist = (self.data[0] - point[0] ) ** 2 + (self.data[1] - point[1] ) ** 2
        # update if this is a good distance
        if dist < local_min_distance:
            local_min_distance = dist;
            local_best_node = self
            if dist == 0:
                return (dist, local_best_node) # exact point match
            
        # Now (maybe) going through the left-right branches
        
        # CASE #1 : min_dist > distance-2-separator (split axis)
        #           This means we need to check both sides. 
        dist2barrier = abs(self.data[self.split_axis] - point[self.split_axis])
        if dist2barrier < local_min_distance:
            if self.left != None:
                (local_min_distance, local_best_node) = self.left.NNSearch(point, local_min_distance, local_best_node)
            
            if self.right != None:
                (local_min_distance, local_best_node) = self.right.NNSearch(point, local_min_distance, local_best_node)
        
        else: 
            # Need to search the side we or on ONLY
            if point[self.split_axis] < self.data[self.split_axis]: 
                # we are on the left
                if self.left != None:
                    (local_min_distance, local_best_node) = self.left.NNSearch(point, local_min_distance, local_best_node)
                else: 
                    # We are on the right
                    if self.right != None:
                        (local_min_distance, local_best_node) = self.right.NNSearch(point, local_min_distance, local_best_node)
        # unreachable region
        return (local_min_distance, local_best_node)
        
        
# Testing...   

#root = KD2TreeNode(-2, 1, 0, 0, 0, 0)
#root.insertNode(1, 2, 0, 0, 1)
#root.insertNode(3, 5, 0, 0, 2)
#root.insertNode(4, 1, 0, 0, 3)
#root.insertNode(1, -1, 0, 0, 4)
#(dist, nn) = root.NNSearch([-10, 3], INF_, root)
#print (" Distance : " , dist)
#print (" point : ", nn.data)

