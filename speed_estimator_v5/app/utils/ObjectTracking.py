from .BboxUtils import calculate_center_point, measure_euclidean_distance
from time import time 


class ObjectTracking(object):

    ''' Module to parse detection data, track them by assigning IDs and pruning them when no longer required. '''

    def __init__(self, euclidean_distance_threshold : float = 10, deregistration_time : int = 10, frame_rate : int = 30):

        '''
            Paramaters:
                * euclidean_distance_threshold : (float) : Pixel distance between two detections center points, if two points are 
                    within this set threshold, they are the same detection, requiring that dicitonary entry is updated. Otherwise,
                    register fresh detection.
                * deregistration_time : (int) : Whole number time in seconds representing how long it is taken before a detection
                    entry is pruned from the tracked_objects dictionary. 
                * frame_rate : (int) : Whole number representing frame rate of the media being processed.
        '''

        # Dictionary to store detection entries.
        self.tracked_objects = {}

        # Increment counter to assign unique ID values. 
        self.ID_increment_counter = 0

        # Constructor set Euclidean distance threshold.
        self.euclidean_distance_threshold = euclidean_distance_threshold

        # Constructor set deregistration time. 
        self.deregistration_time = deregistration_time

        # Constructor set frame rate. 
        self.frame_rate = frame_rate

        # Maximum length for center points history.
        self.max_center_points = 150

    
    def update_tracker(self, detections : list[dict]) -> list[dict]:

        ''' 
            Update the tracker by ingesting detections, comparing their center point values, checking whether or not they are the same. Otherwise,
            updating the entry classing it a fresh detection.
        
            Paramaters: 
                * detections : (list[dict]) : List encapsulating detection dictionary entries. 
            Returns:
                * parsed_detections : (list[dict]) : Parsed list of detection dictionaries with applied tracking processing.
        '''

        # If detections not of the expected type. Fail to run.
        if detections is None or not isinstance(detections, list):
            raise ValueError('Detections being parsed not a list of dictionaries.')

        # Get current time detections were being processed at. 
        updated_at = time()

        # Initialise list to append detections post processing.
        parsed_detections = []

        # Iterate over the current detections being ingested. 
        for current_detection in detections:
            
            # Calculate current, concerned detections center point value. 
            current_center_point = calculate_center_point(current_detection)

            # Check whether center point values match.
            matched_ID = self.match_detection_center_points(current_detection, current_center_point)

            # If an ID value is returned.
            if matched_ID is not None:
                # Update that detection object.
                self.update_object(matched_ID, current_detection, updated_at, current_center_point)
            else:
                # Otherwise, handle fresh detection.
                self.register_object(current_detection, updated_at, current_center_point)

            # Append that processed detection to the list. 
            parsed_detections.append(current_detection)

        # Check for objects that need pruning (exceed the threshold).
        self.prune_outdated_objects(updated_at)

        # Return list of parsed_detections.
        return parsed_detections
    

    def match_detection_center_points(self, detection : dict, current_center_point : tuple[float, float]) -> int | None:

        '''
            Iterate over stored detection entries and detections being parsed. Compare their values to see if they are the 
            same or not. ID value is only returned if data matched is within the set thresholds.

            Paramaters:
                * detection : (dict) : Current detection being processed in dictionary form with its keys representing its metadata. 
                * current_center_point : (tuple[float, float]) : Current center point being calculated within the moment of processing.
            Returns:
                * detection_ID : (int | None) : If ruleset is matched, return ID value. Otherwise, return None. 
        '''

        # Initalise variables prior to use. 
        closest_ID = None 
        shortest_distance = float('inf')

        # Iterate over detections in the tracked_objects dictionary. 
        for ID, prev_detection in self.tracked_objects.items():
            
            # Fetch center point prior to the current. 
            previous_center = prev_detection['center_points'][-1]

            # Get the straight line distance between both center points. 
            euclidean_distance_squared = measure_euclidean_distance(current_center_point, previous_center)

            # Accumulate the detections real world dimensions.
            avg_dimensions_width = detection['avg_class_dimensions']['width']
            # Get the detections bbox width and height.
            detection_width = abs(detection['x2'] - detection['x1'])

            # Scale the euclidean distance threshold in accordance to the detections dimensions. 
            scaled_euclidean_distance_threshold = self.euclidean_distance_threshold * (detection_width / avg_dimensions_width)

            # If distance within the threshold and threshold SQUARED less than shorted distance.
            if euclidean_distance_squared <= scaled_euclidean_distance_threshold and \
                euclidean_distance_squared < shortest_distance:

                # Assign ID value as the closest.
                closest_ID = ID 
                # Assign shortest distance as the current straight line distance.
                shortest_distance = euclidean_distance_squared

        # Return the ID value. 
        return closest_ID
    

    def register_object(self, detection : dict, seen_at : float, current_center_point : tuple[float, float]) -> None:

        '''
            Reigster fresh detection entry within the tracked_objects dictionary with current metadata
            at the time of processing. 

            Paramaters:
                * detection : (dict) : Current detection dictionary from ingested list.
                * seen_at : (float) : time detection is being processed at. 
                * current_center_point : (tuple[float, float]) : Current center x and y values at time of processing.
            Returns:
                * None
        '''

        # Fetch detection classname
        classname = detection['classname']

        # Assign ID value to dictionary entry and required metadata.
        self.tracked_objects[self.ID_increment_counter] = {
            'center_points' : [current_center_point],
            'first_detected' : seen_at,
            'last_detected' : seen_at,
            'classname' : classname
        }
        
        # Assign ID value towards detection dict. 
        detection['ID'] = self.ID_increment_counter

        # Increment counter to keep ID values unique.
        self.ID_increment_counter += 1
    

    def update_object(self, ID : int, detection : dict, updated_at : float, current_center_point : tuple[float, float]) -> None:

        '''
            Function to ingest detection data and update its relevant values whilst it is still concerned in the tracked objects
                dictionary.

            Paramaters:
                * ID : (int) : Detections unique identifier.
                * detection : (dict) : Detection dictionary encapsulating its metadata.
                * updated_at : (float) : Current time detection is being processed at.
                * current_center_point : (tuple[float, float]) : Current detections center point value at the time of processing.
            Returns:
                * None
        '''

        # Append current center point value to detection center points list. 
        self.tracked_objects[ID]['center_points'].append(current_center_point)
        # Update time detection was last seen.
        self.tracked_objects[ID]['last_detected'] = updated_at

        # Maintain a rolling window of last five velocity values. 
        if len(self.tracked_objects[ID]['center_points']) > self.max_center_points:
            self.tracked_objects[ID]['center_points'].pop(0)

        # Update detection dictionary with its entry within the tracked_objects dictionary.
        detection.update(self.tracked_objects[ID])

        # Ensure ID value is still the same.
        detection['ID'] = ID
         
    
    def prune_outdated_objects(self, updated_at : float) -> None:

        '''
            Iterate over parameterised detections and prune those exceeding the set time limit threshold.

            Parameters:
                * parsed_detections : list[dict] -> list of detection data entries.
            Returns:
                * None. 
        '''

        # Initialise list to store ID values of detections to be pruned. 
        stale_detections = [ID for ID, detection in self.tracked_objects.items()
                            if (updated_at - detection['last_detected']) > self.deregistration_time]

        # Iterate over the IDs present. 
        for ID in stale_detections:
            # Use IDs to delete entries from tracked objects. 
            del self.tracked_objects[ID]
