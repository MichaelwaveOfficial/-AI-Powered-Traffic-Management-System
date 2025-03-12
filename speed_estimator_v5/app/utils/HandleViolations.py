import time, datetime  
import cv2 
import os 
import numpy as np

from ..Settings import *

from .Annotations import Annotations


class HandleViolations(object):

    ''' Module encapsulating logic concerned with checking detections attributes and whether or not they violate
        traffic laws. In this particular instance, a vehicles speed against the set limit. '''

    def __init__(self, annotations : Annotations, speed_limit : int = 0, deregisration_time : int = 12):
        
        '''
            Paramaters:
                * annotations : (Annotations) : Annotations module to access some of its methods.
                * speed_limit : (int) : Set speed limit from the UI.
                * deregistration_time : (int) : Time taken in seconds until a detection is removed from storage.
        '''

        self.annotations = annotations
        self.speed_limit = speed_limit
        self.deregistration_time = deregisration_time

        # Dictionary to store captured detections in violation of traffic laws.
        self.captured_offenders = {}

    
    def handle_violations(self, frame : np.ndarray, detections : list[dict]) -> None:

        '''
            Process list of violating detections, capture the frame of the offense and append required data.

            Paramaters:
                * frame : (np.ndarray) : Raw frame to be processed.
                * detections : (list[dict]) : List of detection data.
            Returns:
                * captured_frame : (np.ndarray) : Processed frame with the captured detection.
        '''

        # Iterate over each detection within the list.
        for detection in detections:
            
            # Apply offending flag to detection for annotation purposes.
            detection['offender'] = True
            
            # Timestamp capture is being processed at. 
            captured_at = time.time()

            # Obtain processed frame with required data.
            captured_frame = self.capture_offense(frame=frame, detection=detection, captured_at=captured_at)
            
            # Save frame capture.
            self.save_capture_to_device(frame=captured_frame)

    
    def save_capture_to_device(self, captured_frame : np.ndarray, save_path : str = CAPTURES_DIR_PATH) -> None:

        '''
            Method capturing the datetime stamp of the incident, saves it to devices local storage.

            Paramaters:
                * captured_frame : (np.ndarray) : The processed frame to be saved.
                * save_path : (str) : String path variable where capture will be saved.
            Returns:
                * None
        '''

        # Incident timestamp.
        captured_at = datetime.datetime.now().strftime('%a-%b-%Y_%I-%M-%S%p')

        # Concatenate save_path and timestamp with .jpeg suffix.
        filename = os.path.join(save_path, f'{captured_at}.jpg')
       
        try:
            # Attempt to write to device.
            cv2.imwrite(filename, captured_frame)
            print('Capture Saved : ', filename)
        except Exception as e:
            print(f'Error occurded writing out capture to application directory! \n{e}')

    
    def check_detections_speeds(self, detections : list[dict]) -> list[dict]:

        '''
            Iterate over detections being parsed and see if they meet the pre-requisites to be classed
            as an offender for further processing.

            Paramaters:
                * detections : (list[dict]) : List of detection dictionary data to be ingested.
                * frame : (np.ndarray) : Current frame to be processed from media.
            Returns:
                * parsed_detections : (list[dict]) : List of processed detections that meet the validation requirements.
        '''

        # Initialise list to append detections that meet pre-requisites.
        parsed_detections = []

        # Timestamp detection was processed at.
        detected_at = time.time()

        # Iterate over each detection entry in the list.
        for detection in detections:

            # Fetch required detection metadata variables.
            ID = detection.get('ID')
            speed = detection.get('speed')

            if ID and speed is not None and self.validate_detection(speed):
                
                # If ID not currently present in dictionary.
                if ID not in self.captured_offenders:

                    # Create an entry
                    self.captured_offenders[ID] = {
                        'last_detected' : detected_at,
                        'already_captured' : False
                    }

                # Update current, present entries last seen timestamp.
                self.captured_offenders[ID]['last_detected'] = detected_at

                # If detection not already flagged as processed.
                if not self.captured_offenders[ID]['already_captured']:

                    # Update flag in dictionary.
                    self.captured_offenders[ID]['already_captured'] = True

                    # Append final detection to parsing list. 
                    parsed_detections.append(detection)

        # Check no defunct objects have overstayed their welcome.
        self.prune_outdated_objects(detected_at)

        # Return list of parsed detections exceeding speed limit.
        return parsed_detections


    def validate_detection(self, speed : float) -> bool:

        ''' Check detections provided speed violates set user speed limit. '''

        return speed > self.speed_limit
    

    def capture_offense(self, detection : dict, frame : np.ndarray, captured_at : float) -> np.ndarray | None:

        ''' Wrapper function for annotations offense capture function. '''

        return self.annotations.capture_traffic_violation(frame, detection, captured_at)


    def prune_outdated_objects(self, updated_at : float) -> None:

        '''
            Iterate over parameterised detections and prune those exceeding the set time limit threshold.

            Parameters:
                * parsed_detections : list[dict] -> list of detection data entries.
            Returns:
                * None. 
        '''

        # Initialise list to store ID values of detections to be pruned. 
        stale_detections = [ID for ID, detection in self.captured_offenders.items()
                            if (updated_at - detection['last_detected']) > self.deregistration_time]

        # Iterate over the IDs present. 
        for ID in stale_detections:
            # Use IDs to delete entries from tracked objects. 
            del self.captured_offenders[ID]
  