import numpy as np 
from collections import deque 
from time import time
from .BboxUtils import measure_euclidean_distance
from ..Settings import *


class SpeedEstimation(object):

    ''' Module to estimate a detections speed from given average dimensions, center point comparisons with calculations smoothed out
        to mitigate outliers. 
    '''

    def __init__(
        self,
        frame_rate : int = 30, 
        deregistration_time : int = 12,
        rolling_window_size : int = 5,
        ppm_smoothing_factor : float = 0.7
    ):
        
        '''
            Paramaters:
                * frame_rate : (int) : Frame rate from video being processed. Default set to 30.
                * deregistration_time : (int) : Time in seconds until a detection is deregistered from its dictionary. Defaults to 12.
                * rolling_window_size : (int) : Max length of measurements allowed to benefit smoothing of speed calculations. Defaults to 5.
                * ppm_smoothing_factor : (float) : Smoothing factor leveraged for pixels per meter measurements. Defaults to 0.7
        '''

        self.frame_rate = frame_rate
        self.deregistration_time = deregistration_time
        self.rolling_window_size = rolling_window_size
        self.ppm_smoothing_factor = ppm_smoothing_factor

        # Dictionary where detections will be stored along with a history of the calculated speeds.
        self.detection_speeds = {}

    def estimate_detections_speeds(self, detections : list[dict]) -> list[dict]:

        '''
            Pipeline function to ingest detections, create entries to store the history of their calculated speeds, apply the appropriate smoothing
                and apply the results.

            Paramaters:
                * detections : (list[dict]) : Current list of detections containing each entries dictionary.
            Returns:
                * detections : (list[dict]) : Same list of detections, with appended speeds after processing.
        '''

        # Current time detection was processed at. 
        updated_at = time()

        # Iterate over each entry in detections list.
        for detection in detections:

            # If validations not met (required key value pairs), coninue onto next detection.
            if not self.validate_detection(detection):
                continue
            
            # Fetch detection ID.
            ID = detection['ID']

            # Fetch detections last most center point.
            current_center_point = detection['center_points'][-1]
            
            # Calibrate the detections dimensions.
            detection_ppm = self.calibrate_ppm(detection)

            # If detection not already present, initalise entry and continue onto next detection.
            if ID not in self.detection_speeds:
                self.initialise_detection_speeds(ID, current_center_point, detection_ppm, updated_at)
                continue
            
            # Fetch prior detection entry, stored within the modules dicitonary. 
            previous_data = self.detection_speeds[ID]

            # Smooth out detection dimensions to mitigate erratic calculations.
            smoothed_detection_ppm = self.smooth_detection_ppm(previous_data['ppm'], detection_ppm)

            # Calculate a detections speed from cetner points frame by frame.
            detection_speed = self.calculate_frame_speed(previous_data, current_center_point, smoothed_detection_ppm, updated_at)

            # If a speed value has been returned.
            if detection_speed:
                
                # Update that detections entry with its updated value.
                self.update_detections_speed(ID, detection_speed, current_center_point, smoothed_detection_ppm, updated_at)

                # Assign value to detection dictionary entry, smooth and format as needed.
                detection['speed'] = round(float(np.median(self.detection_speeds[ID]['speeds']) * 10), 2)

        # Check dictionary has no outdated entries to mitigate overheads.
        self.prune_outdated_objects(updated_at)

        # Return list of detections.
        return detections
    

    def validate_detection(self, detection : dict) -> bool:

        ''' Helper function ensuring detection has key pair value pre-requisites met before processing. '''

        return (
            detection.get('ID') is not None and 
            'center_points' in detection and 
            len(detection['center_points']) >= 1
        )
    

    def initialise_detection_speeds(self, ID : int, center_point : tuple[float], ppm : float, updated_at : float) -> None:

        ''' 
            Create detection entry within the detection_speeds dictionary.

            Paramaters:
                * ID : (int) : Unique identifier for the given detection.
                * center_point : (tuple[float, float]) : Detections current center point value.
                * ppm : (float) : Pixels Per Meter value representing detection size. 
                * updated_at (float) : Timestamp detection is being processed at.
        '''

        self.detection_speeds[ID] = {
            'last_center' : center_point,
            'ppm' : ppm,
            'updated_at' : updated_at,
            'speeds' : deque(maxlen=self.rolling_window_size)
        }


    def smooth_detection_ppm(self, prev_ppm : float, curr_ppm : float) -> float:

        ''' 
            Smooth detections PPM value in order to try and cull erratic calucations and ensure 
                consistency across frames.

            Paramaters:
                * prev_ppm : (float) : PPM value from preivous iteration.
                * curr_ppm : (float) : PPM value from current iteration.
        '''

        return self.ppm_smoothing_factor * prev_ppm + (1 - self.ppm_smoothing_factor) * curr_ppm
    

    def calculate_frame_speed(self, prev_detection : dict, current_center_point : tuple[float, float], current_ppm : float, updated_at : float) -> float:

        '''
            Calculate a detections speed frame by frame by estimating the distance between current and prior center point values.

            Paramaters:
                * prev_detection : (dict) : Stored detection dictionary, representing detections prior status.
                * current_center_point : (tuple[float, float]) : Current, calculated center point at time of processing.
                * current_ppm : (float) :  Current, calculated pixels per meter value.
                * updated_at : (float) : Current timestamp at the time of processing.
            Returns:
                * calculated_speed : (float) : Detections speed converted into MPH, which for the UK, is the appropriate format.
        '''

        # Fetch required data points from stored detection dictionary entry.
        prev_center = prev_detection['last_center']
        prev_ppm = prev_detection['ppm']
        prev_time = prev_detection['updated_at']

        # Calculate straight line distance between both points.
        pixel_distance = measure_euclidean_distance(prev_center, current_center_point)

        # Calculate elapsed time between current and previously handled timestamps.
        elapsed_time = updated_at - prev_time

        # If required variables are not valid, return none.
        if elapsed_time <= 0 or pixel_distance < 2:
            return None 
        
        # Average out pixel per meter values.
        avg_ppm = (prev_ppm + current_ppm) / 2

        # Return calucalted speed, converted into a familial format. 
        return self.calculate_speed(pixel_distance, avg_ppm, elapsed_time)
    

    def update_detections_speed(self, ID : int, speed : float, center_point : tuple[float, float], ppm : float, updated_at : time) -> None:

        ''' 
            Update detection dictionary entries with relevant values.

            Paramaters:
                * ID : (int) : Unique detection identifier.
                * speed : (float) : Calculated the speed for a given detection.
                * center_point : (tuple[float, float]) : Detections current calculated center point.
                * ppm : (float) :  Detections pixels per meter value to estimate size. 
                * updated_at : (float) : Time detection was last processed at.
        '''

        # Append new speed value entry to speeds list.
        self.detection_speeds[ID]['speeds'].append(speed)
        
        # Update other values in the detection_speeds dictionary entry.
        self.detection_speeds[ID].update({
            'last_center' : center_point,
            'ppm' : ppm, 
            'updated_at' : updated_at
        })


    def calibrate_ppm(self, detection : dict) -> float:

        '''
            Attempt to calibrate pixels per meter by obtaining the average real-world dimensions for a detection and 
                leveraging it against the detections bounding box width and height ro determine its scale.

            Parameters:
                * detection : dict -> single detection dictionary containing metadata. 

            Returns:
                * float -> pixels per meter values for that detection to later be used for speed estimation.
        '''

        # Accumulate the detections real world dimensions.
        real_width, real_height = (
            detection['avg_class_dimensions']['width'],
            detection['avg_class_dimensions']['height']
        )

        # Get the detections bbox width and height.
        detection_width = max(abs(detection['x2'] - detection['x1']), 1)
        detection_height = max(abs(detection['y2'] - detection['y1']), 1)

        # Calculate its ppm width and height. 
        ppm_width = detection_width / real_width
        ppm_height = detection_height / real_height

        ppms = [ppm for ppm in [ppm_width, ppm_height] if ppm > 0]

        # Return detections scale.
        return sum(ppms) / len(ppms)
    
    
    def calculate_speed(self, pixel_distance : float, ppm : float, elapsed_time : float) -> float:

        ''' 
            Calculate a detections speed from the timestamp and center point values it was
            currently and last processed at.

            Paramaters:
                * pixel_distance : (float) : Distance covered between center points in pixels.
                * ppm : (float) : Detectiosn pixels per meter value representing size. 
                * elapsed_time : (float) : Time taken between center points being recorded. 
            Returns:
                * calculated_speed : (float) : Calculated speed converted into the required 
                    unit of measurement. 
        '''

        # If input values will lead to NaN, return 0 early.
        if elapsed_time <= 0 or ppm <= 0:
            return 0.0
        
        # Convert speed calculations into the appropriate units. 
        return self.unit_conversion(speed=(pixel_distance / ppm) / elapsed_time, measurement='mph')
    

    def unit_conversion(self, speed : float, measurement : str = 'mph') -> float:

        '''
            Simple helper function to convert speed measurement into a recognised unit of measurement determined by the user. 

            Parameters:
                * speed : float -> The detections calculated speed.
                * measurement : str -> The unit of measurement desired.

            Returns:
                * float -> speed multipled by the concered unit of measuremnt. 
        '''

        # Conversion values dictionary contianing key value pairs. 
        conversion_factors = {'mph': 2.23,  'kmh': 3.6}

        # If provided measurements not in conversions dictionary, let user know. 
        if measurement not in conversion_factors:
            raise ValueError(f"Unsupported measurement unit: {measurement}")

        # Return speed multiplied by specified conversion factor. 
        return speed * conversion_factors[measurement]

    
    def prune_outdated_objects(self, updated_at : float) -> None:

        '''
            Iterate over parameterised detections and prune those exceeding the set time limit threshold.

            Parameters:
                * parsed_detections : list[dict] -> list of detection data entries.
        '''

        # Initialise list to store ID values of detections to be pruned. 
        stale_detections = [ID for ID, detection in self.detection_speeds.items()
                            if (updated_at - detection['updated_at']) > self.deregistration_time]

        # Iterate over the IDs present. 
        for ID in stale_detections:
            # Use IDs to delete entries from tracked objects. 
            del self.detection_speeds[ID]
            