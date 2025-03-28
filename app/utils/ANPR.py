import easyocr
import cv2 
import re 
import time 
import numpy as np
from rapidfuzz import fuzz 

from .ObjectDetection import ObjectDetection


class ANPR(object):

    ''' Module leveraging YOLO object detection to extract license plates from a detections 
        bounding box, process, crop the region and read the plate text utilising OCR.
    '''

    def __init__(
            self,
            detection_model : ObjectDetection,
            ocr_lang : str = 'en',
            ocr_gpu : bool = True,
            deregistration_time : int = 12,
            plate_similarity_threshold : int = 70
        ):
        
        '''
            Paramaters:
                * detection_model : (ObjectDetection) : Defined ObjectDetection model to return a structured output from required classes.
                * ocr_lang : (str) : Language to be read by OCR model, set to English by default. 
                * ocr_gpu : (bool) : Boolean value whether or not OCR model should leverage GPU, True by default
                * deregistration_time : (int) : Time taken in seconds to discard of detection data, 12 seconds by default.
                * plate_similarity_threshold : (int) : Threshold value for license plate similarity. 
        '''

        self.detection_model : ObjectDetection = detection_model
        self.deregistration_time : int = deregistration_time
        self.plate_similarity_threshold : int = plate_similarity_threshold

        # Initialise OCR text model.
        self.ocr_reader : easyocr = easyocr.Reader([ocr_lang], gpu=ocr_gpu)

        # Maximum constraint for license plate length to match UK requirements.
        self.MAX_UK_PLATE_LENGTH : int = 7

        # Regular expression representing plate format. 
        self.UK_PLATE_REGEX =  re.compile(r'^([A-Z]{2})([0-9]{2})([A-Z]{3})$')


        # Dicitonary to store detection IDs with their corresponding license plates.
        self.detection_plates : dict = {}

        # Mapping dictionaries for characters that can be easily mistaken.
        self.char_2_int_dict = {'O': '0','I': '1','J': '3','A': '4','G': '6','S': '5'}
        self.int_2_char_dict = {'0': 'O','1': 'I','3': 'J','4': 'A','6': 'G','5': 'S'}

    
    def process_detection_plates(self, frame : np.ndarray, detections : list[dict]) -> list[dict]:

        '''
            Parse detections list assign license plate structure if not already present, detect, crop and process
            its license plate for later annotation in captures.

            Paramaters:
                * frame : (np.ndarray) : Raw frame to read license plates from.
                * detections : (list[dict]) : List of detection dictionaries containing metadata.
            Returns:
                * None
        '''
        
        # Fetch timestamp of operations.
        processed_at = time.time()

        # Initialise list to store detections and their read plates.
        plate_read_detections = []

        # Iterate over each detection in the paramaterised list. 
        for detection in detections:
            
            # Fetch detections ID value.
            ID = detection.get('ID')

            # Initalise structure of detection plates dictionary entry.
            if ID not in self.detection_plates:
                # Assign status OCCLUDED at plate text.
                self.detection_plates[ID] = {'plate_text' : 'OCCLUDED', 'last_seen' : 0}

            # If elapsed time less than deregistration time threshold.
            if processed_at - self.detection_plates[ID]['last_seen'] < self.deregistration_time:
                
                # Assign license plate value to detection dictionary.
                detection['license_plate'] = {'plate_text' : self.detection_plates[ID]['plate_text']}
                
                # Append processed detection to be returned by list if successful.
                plate_read_detections.append(detection)
                
                # Detection processing finished. Pass onto next iteration.
                continue
        
            # Crop vehicle from raw frame based on its bbox values.
            vehicle_crop = self.crop_frame_from_detection_data(frame, detection)

            # Attain license plate detection data by running YOLO inference on cropped vehicle.
            license_plate_detections = self.detection_model.run_inference(vehicle_crop)

            # If detection data is returned.
            if not license_plate_detections:

                self.detection_plates[ID]['plate_text'] = 'OCCLUDED'
                detection['license_plate'] = {'plate_text' : 'OCCLUDED'}
                continue

            # Sort plates in descending order by their confidence score.
            sorted_license_plates = sorted(
                license_plate_detections,
                key = lambda license_plate : license_plate['confidence_score'],
                reverse = True
            )

            # Initialise plate_found boolean flag.
            plate_found = False

            # Iterate over each SORTED license plate.
            for license_plate in sorted_license_plates:
                
                # Extract license plate text from the cropped license_plates.
                license_plate_text = self.extract_license_plate(frame, detection, license_plate)

                # If license plate text discerned.
                if license_plate_text:
                    
                    # Assign variables to detection_plates dictionary.
                    self.detection_plates[ID]['plate_text'] = license_plate_text
                    self.detection_plates[ID]['last_seen'] = processed_at

                    # Set flag to True as plate found and processed.
                    plate_found = True

                    # End this iteration if plate found.
                    break
            
            # If unsuccessful, mark license plate as occluded.
            if not plate_found:
                self.detection_plates[ID]['plate_text'] = 'OCCLUDED'
            
            # Update key value with updated plate text.
            detection['license_plate'] = {'plate_text' : self.detection_plates[ID]['plate_text']}
            
            # Append detection ONLY when successful.
            plate_read_detections.append(detection)

        # Ensure outdated detection data is pruned.
        self.prune_outdated_objects(processed_at)

        return plate_read_detections


    def extract_license_plate(self, frame : np.ndarray, detection : dict, license_plate : dict) -> str:

        '''
            Ingests the raw frame, the detections and license plates bounding box values so the license
            plate can be extracted in order to make plate processing easier. 

            Paramaters:
                * frame : (np.ndarray) : The raw frame to be operated upon.
                * detection (dict) : Detection dictionary containing required bbox values.
                * license_plate (dict) : License plate detection containing required bbox values.
            Returns:
                * corrected_plate_text : (str) : Returns plate text in the appropriated format. 
        '''

        # Absolute license plate coordinates within the vehicles detection bbox.
        abs_coords = (
            int(detection['x1'])  + int(license_plate['x1']),
            int(detection['y1'])  + int(license_plate['y1']),
            int(detection['x1'])  + int(license_plate['x2']),
            int(detection['y1'])  + int(license_plate['y2']),
        )

        # y1:y2, x1:x2
        cropped_license_plate = frame[abs_coords[1]:abs_coords[3], abs_coords[0]:abs_coords[2]]

        #cv2.imshow('cropped plate', cropped_license_plate)

        # Read plate text from the license plate crop.
        ocr_read_plate_text = self.read_license_plate(cropped_license_plate)
        
        # Return corrected plate text.
        return self.correct_plate_text(ocr_read_plate_text)


    def crop_frame_from_detection_data(self, frame, detection):

        x1, y1, x2, y2 = map(int, (detection['x1'], detection['y1'], detection['x2'], detection['y2']))

        return frame[y1: y2, x1:x2]
    

    def validate_plate_format(self, plate_text : str) -> bool:

        ''' Return true is plate text matches regular expression. Otherwise, false. '''

        return self.UK_PLATE_REGEX.match(plate_text) is not None


    def correct_plate_text(self, raw_plate_text : str) -> str | None:

        ''' 
            Function to process and clean OCR model output, rectifying plate text
                into expected, familial output matching UK license plate formats.

            Paramaters:
                * raw_plate_text : (str) : The raw, unprocessed output from the OCR model.
            Returns:
                * corrected_plate_text : (str) : Correct plate text, formatted in line with UK legislation.
        '''

        # If no raw plate text present or length of its characters is none, return early. 
        if not raw_plate_text or len(raw_plate_text) <= 0:
            return None

        # Clean parsed plate text removing spaces. trailing characters and converting to upper case.
        cleansed_plate_text = ''.join(raw_plate_text).upper().strip().replace(' ', '')
        #print('cleansed text', cleansed_plate_text)

        # Return early if validation already matches.
        if self.validate_plate_format(cleansed_plate_text):
            #print('early validation achieved ', cleansed_plate_text)
            return cleansed_plate_text
        
        # If too short or too long, cull result.
        if not 6 <= len(cleansed_plate_text) <= 8:
            #print(cleansed_plate_text, 'disposed, did not pass length checks.')
            return None 
        
        # Group areas of interest from parsed text compared to the regex.
        cleansed_plate_text_groups = self.UK_PLATE_REGEX.match(cleansed_plate_text)

        # If no groups, persist with fallback cleansing.
        if not cleansed_plate_text_groups:
            
            # If cleansed plate length complies with set threshold.
            if len(cleansed_plate_text) == self.MAX_UK_PLATE_LENGTH:
                
                # If groups cannot be discerned from regex. Slice manaually.
                area_code, reg_year, suffix = cleansed_plate_text[:2], cleansed_plate_text[2:5], cleansed_plate_text[5:]

                # Convert misinterpreted characters into expected group types.
                # area_code ++ suffix should be letters.
                # reg_year should be numeric.
                area_code = self.convert_text_groups(area_code, 'area_code')
                reg_year = self.convert_text_groups(reg_year, 'reg_year')
                suffix = self.convert_text_groups(suffix, 'suffix')

            else:
                #print('plate disposed ', cleansed_plate_text)
                return None

        else:               

            # Assign groups to variables for greater access ++ contextualisation.
            area_code, reg_year, suffix = cleansed_plate_text_groups.groups()
            #print(f'area_code:{area_code}, year:{reg_year}, suffix:{suffix}')

            # Convert misinterpreted characters into expected group types.
            # area_code ++ suffix should be letters.
            # reg_year should be numeric.
            area_code = self.convert_text_groups(area_code, 'area_code')
            reg_year = self.convert_text_groups(reg_year, 'reg_year')
            suffix = self.convert_text_groups(suffix, 'suffix')
        
        # Aggregrate groups together, form corrected plate text. 
        corrected_plate_text = area_code + reg_year + suffix
        #print('corrected character plate ', corrected_plate_text)

        # Generate similarity percentage from corrected and original text.
        plate_similarity_ratio = fuzz.ratio(corrected_plate_text, cleansed_plate_text)

        print(
            'input plate',
            cleansed_plate_text,
            'corrected_plate',
            corrected_plate_text,
            'at confidence', plate_similarity_ratio
        )
        
        # If similarity float meets set threshold ++ meets validation, return final plate.
        if plate_similarity_ratio >= self.plate_similarity_threshold and \
            self.validate_plate_format(corrected_plate_text):
            #print('final product ', corrected_plate_text)
            return corrected_plate_text
        
        # If no conditions met, return none.
        return None

    
    def preprocess_plate(self, cropped_plate : np.ndarray, kernel_size : int = 3, operation_iterations : int = 1) ->  np.ndarray:

        '''
            Function to leverage cv2s BIFs, processing the cropped license plate frame, improving the legibility 
            of the plate text to imrpove the accuracy of the OCR model later on in the pipeline.

            Paramaters:
                * cropped_plate : (np.ndarray) : The region of frame specific to the license plate.
                * kernel_size : (int) : The height and width properties of the kernel, defaults to 3.
                * operation_iterations : (int) : Number of times morphological operations will be applied, defaults to 1.
            Returns:
                * processed_license_plate : (np.ndarray) : Post processed plate with operations applied ready for
                    its text to be read by the OCR model. 
        '''

        # Size of the kernal in pixels (w, h). 
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Convert frame to greyscale, reducing colour channels as a separation of concerns. 
        grey_cropped_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)

        # Threshold the image to generate a greater contract making text more legible for the model. 
        thresholded_plate = cv2.threshold(
            grey_cropped_plate,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]

        # Apply morphological operations to mitigate artifacts generated by noise.
        morphologically_applied_plate = cv2.morphologyEx(thresholded_plate, cv2.MORPH_CLOSE, kernel, operation_iterations)

        # Apply a gaussian blur to smooth the image further. 
        smoothed_plate = cv2.GaussianBlur(morphologically_applied_plate, (kernel_size, kernel_size), 0)

        return smoothed_plate

    
    def read_license_plate(self, cropped_plate : np.ndarray) -> str | None:

        '''
            Ingest cropped frame pertaining to a detections license plate, feed that region 
            to the model to read the plate text with an OCR model.

            Paramaters:
                * cropped_plate : (np.ndarray) : Region of the frame specific to the vehicles license plate. 
            Returns:
                * raw_plate_text : (str | None) : If legible, raw license plate text will be returned.
        '''

        try:
            # Take cropped_plate frame and preprocess it for better model digestion.
            processed_plate = self.preprocess_plate(cropped_plate)

            # Use OCR model to read text from pre-processed plate.
            return self.ocr_reader.readtext(processed_plate, detail=0)
        
        except Exception as e:
            print(f'OCR model failed to read text.\n{e}')

        return None


    def prune_outdated_objects(self, updated_at : float) -> None:

        '''
            Iterate over parameterised detections and prune those exceeding the set time limit threshold.

            Parameters:
                * parsed_detections : list[dict] -> list of detection data entries.
            Returns:
                * None. 
        '''

        # Initialise list to store ID values of detections to be pruned. 
        stale_detections = [ID for ID, detection in self.detection_plates.items()
                            if (updated_at - detection['last_seen']) > self.deregistration_time]

        # Iterate over the IDs present. 
        for ID in stale_detections:
            # Use IDs to delete entries from tracked objects. 
            del self.detection_plates[ID]

    
    def convert_text_groups(self, text_grouping : str, group : str) -> str:

        '''
            Hones in on the specific regions of the UK license plate, iterates over its characters ensuring 
                that they are not mismatches. Conerts if necessary. 

            Pamaters:
                * text_grouping : (str) : The group of text to be iterated upon.
                * group : (str) : Specific license plate region. Accepted regions are "area_code", "suffix", "reg_year"
            Returns:
                * text_grouping : (str) : Text group with the correct characters.
        '''

        if group in ['area_code', 'suffix']:
            return ''.join(self.int_2_char_dict.get(char, char) for char in text_grouping)
        elif group == 'reg_year':
            return ''.join(self.char_2_int_dict.get(char, char) for char in text_grouping)
        else:
            return text_grouping