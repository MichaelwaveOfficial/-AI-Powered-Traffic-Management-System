import numpy as np

from .Settings import *

from .utils.ObjectDetection import ObjectDetection
from .utils.ObjectTracking import ObjectTracking
from .utils.SpeedEstimation import SpeedEstimation
from .utils.HandleViolations import HandleViolations
from .utils.ANPR import ANPR
from .utils.Annotations import Annotations

# Instantiate YOLO detection models.
vehicle_detection = ObjectDetection(model=DETECTION_MODEL_PATH, confidence_threshold=BASE_YOLO_CONFIDENCE_THRESHOLD)
license_plate_detection = ObjectDetection(model=PLATE_DETECTION_MODEL_PATH, confidence_threshold=PLATE_YOLO_CONFIDENCE_THRESHOLD)

# Intstantiate objects to mimic singleton pipeline, each module representing a step in the pipeline.
annotations = Annotations()
object_tracking = ObjectTracking()
speed_estimation = SpeedEstimation()
traffic_violation_checks = HandleViolations(annotations=annotations)
ANPR = ANPR(detection_model=license_plate_detection)

# Inform users whether hardware acceleration is being used or not. 
vehicle_detection.check_for_hardware_acceleration()


def process_video(
        frame : np.ndarray, 
        speed_limit : int = 0, 
        frame_rate : int = 30, 
        vision_type : str = 'object_detection', 
        confidence_threshold :float = BASE_YOLO_CONFIDENCE_THRESHOLD
    ) -> np.ndarray:
    
    '''
        Paramaters:
            * frame : (np.ndarray) : Input frame being processed for inference. 
            * speed_limit : (int) : Set speed limit from UI, default is 0 representing NO LIMIT SET. 
            * frame_rate : (int) : Currently processed medias frame rate, attempt to provide more accurate measurements. 
            * vision_type : (str) : Current vision mode set by user in UI, default set to 'object_detection'. 
            * confidence_threshold : (float) : Model confidence level to register a detection during inference.

        Returns:
            * annotated_frame : (np.ndarray) : Updated frame with annotations for model driven insights. 
    '''

    # Update module variables from users set UI preferences.
    vehicle_detection.confidence_threshold = confidence_threshold
    object_tracking.frame_rate = frame_rate
    speed_estimation.frame_rate = frame_rate
    traffic_violation_checks.speed_limit = speed_limit

    ''' Object Detection. '''

    # Obtain detections data by running inference leveraging YOLOV11 model on input media. 
    vehicle_detections : list[dict] = vehicle_detection.run_inference(frame=frame) 

    ''' Object Tracking '''

    # Assign IDs to detections and update their center point values.
    tracked_vehicle_detections : list[dict] = object_tracking.update_tracker(detections=vehicle_detections)

    ''' Speed Estimation. '''

    # Estimate a detections speed by comparing current and previous center points. 
    vehicle_speed_estimation_detections : list[dict] = speed_estimation.estimate_detections_speeds(detections=tracked_vehicle_detections)
    
    ''' Violation Checks. '''

    # Take updated detections with their speeds, check that they fall within the set requirements. 
    traffic_violation_detections : list[dict] = traffic_violation_checks.check_detections_speeds(detections=vehicle_speed_estimation_detections)

    # If any violating detections have been returned and the captured frame is valid.
    if len(traffic_violation_detections) > 0:

        ''' ANPR. '''

        # Read detections license plate.
        anpr_detections : list[dict] = ANPR.process_detection_plates(frame=frame, detections=traffic_violation_detections)

        # Take captured, plate read detections, capture the incident of the event.
        traffic_violation_checks.handle_violations(frame=frame, detections=anpr_detections)

    ''' Frame Annotation. '''

    # Supply the final step of processed data to be annotated for traffic insights. 
    annotated_frame : np.ndarray = annotations.annotate_frame(frame=frame, detections=vehicle_speed_estimation_detections, vision_type=vision_type)

    # Return frame whether modified or not. 
    return annotated_frame
