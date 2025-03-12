import customtkinter
import os
import cv2
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from .Settings import *
from .Pipeline import process_video


class TkinterUI(object):

    '''
        CustomTkinter UI for a traffic management application.
    '''

    def __init__(self, root):

        ''' Application Constants. '''

        self.APP_GEOMETRY = '1280x720'
        self.ICON_SIZE = 40
        self.DARK_BG = '#343a40'
        self.PADDING = {
            'min' : 5,
            'med' : 10, 
            'max' : 20, 
            'padx' : 15,
            'pady' : 10
        }

        # Key value pairs for current, national UK, speed limits in MPH.
        self.NATIONAL_SPEED_LIMITS = {
            'School Zone - 20mph' : 20,
            'Residential - 30mph' : 30,
            'Carriageway - 60mph' : 60,
            'Motorway - 70mph' : 70,
        }

        # List for vision modes.
        self.vision_modes = {
            'Object Detection' : 'object_detection',
            'Object Tracking' : 'object_tracking',
            'Speed Estimation' : 'speed_estimation',
        }
        
        ''' Root widget setup. '''

        self.root = root
        self.root.title('Speed Estimation Application')
        self.root.geometry(self.APP_GEOMETRY)
        
        ''' Application Variables. '''

        self.is_paused = False 
        self.is_stopped = False
        self.video = None  
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0
        self.current_speed_limit = None
        self.current_vision_mode = self.vision_modes['Object Detection']
        self.base_confidence = BASE_YOLO_CONFIDENCE_THRESHOLD

        ''' Application Widget Icons. '''

        self.directory_icon = customtkinter.CTkImage(
            dark_image=Image.open(os.path.join(ICONS_DIR_PATH , 'folder.png')), 
            size=(self.ICON_SIZE, self.ICON_SIZE),
        )
       
        self.search_icon = customtkinter.CTkImage(
            dark_image=Image.open(os.path.join(ICONS_DIR_PATH , 'search.png')), 
            size=(self.ICON_SIZE, self.ICON_SIZE),
        )
        self.stop_icon = customtkinter.CTkImage(
            dark_image=Image.open(os.path.join(ICONS_DIR_PATH , 'stop.png')), 
            size=(self.ICON_SIZE, self.ICON_SIZE),
        )

        self.play_icon = customtkinter.CTkImage(
            dark_image=Image.open(os.path.join(ICONS_DIR_PATH , 'play.png')), 
            size=(self.ICON_SIZE, self.ICON_SIZE),
        )

        self.pause_icon = customtkinter.CTkImage(
            dark_image=Image.open(os.path.join(ICONS_DIR_PATH , 'pause.png')), 
            size=(self.ICON_SIZE, self.ICON_SIZE),
        )

        ''' Application Widgets Frames. '''

        # Frame widget to enscapulate the media uploaded. 
        self.video_frame = customtkinter.CTkFrame(master=self.root)
        self.video_frame.pack(pady=self.PADDING['pady'], padx=self.PADDING['padx'], fill='both', expand=True)

        # Frame widget to encapsulate the video players controls.
        self.controls_frame = customtkinter.CTkFrame(master=self.root)
        self.controls_frame.pack( padx=self.PADDING['padx'], pady=self.PADDING['pady'], fill='x')

        # Frame to encapsulate video seek bar. 
        self.seek_bar_frame = customtkinter.CTkFrame(master=self.controls_frame)
        self.seek_bar_frame.pack(fill='x', padx=self.PADDING['padx'], pady=self.PADDING['pady'])

        # Frame widget to encapsulate video controls.
        self.player_controls_frame = customtkinter.CTkFrame(master=self.controls_frame)
        self.player_controls_frame.pack(fill='x', padx=self.PADDING['padx'], pady=self.PADDING['pady'])

        # Frame widget to encapsulate dropdowns
        self.player_controls_dropdowns = customtkinter.CTkFrame(master=self.player_controls_frame)
        self.player_controls_dropdowns.pack(pady=self.PADDING['pady'], padx=self.PADDING['padx'], side='left')

        # Frame widget for speeds.
        self.speed_limit_frame = customtkinter.CTkFrame(master=self.player_controls_dropdowns)
        self.speed_limit_frame.pack(pady=self.PADDING['pady'], padx=self.PADDING['padx'], side='left')

        # Frame widget for vision modes.
        self.vision_modes_frame = customtkinter.CTkFrame(master=self.player_controls_dropdowns)
        self.vision_modes_frame.pack(pady=self.PADDING['pady'], padx=self.PADDING['padx'], side='left')

        # Frame to enscapsulate model confidence seek bar.
        self.confidence_thresholds_frame = customtkinter.CTkFrame(master=self.player_controls_dropdowns)
        self.confidence_thresholds_frame.pack(pady=self.PADDING['pady'], padx=self.PADDING['padx'])

        '''     Overlay Frames.     '''

        self.video_canvas = customtkinter.CTkCanvas(master=self.video_frame)
        self.video_canvas.configure(bg=self.DARK_BG, highlightthickness=0)
        self.video_canvas.pack(fill='both', expand=True, pady=self.PADDING['pady'], padx=225)
        self.video_canvas.pack_propagate(False)

        '''     Button Widgets.     '''

        self.import_video_button = customtkinter.CTkButton(master=self.player_controls_frame, image=self.search_icon, text='', command=self.source_video_import)
        self.import_video_button.pack(pady=self.PADDING['pady'], padx=self.PADDING['padx'], fill='both', expand=True, side='right')

        self.play_pause_btn = customtkinter.CTkButton(master=self.player_controls_frame, image=self.play_icon, text='', command=self.handle_video_state)
        self.play_pause_btn.pack(pady=self.PADDING['pady'], padx=self.PADDING['padx'], fill='both', expand=True, side='right')

        self.stop_capture_button = customtkinter.CTkButton(master=self.player_controls_frame, image=self.stop_icon, text='', command=self.delete_import)
        self.stop_capture_button.pack(pady=self.PADDING['pady'], padx=self.PADDING['padx'], fill='both', expand=True, side='right')

        ''' Txt Label Widgets. '''

        self.lbl_set_limit = customtkinter.CTkLabel(master=self.speed_limit_frame, text='Set Speed Limit:')
        self.lbl_set_limit.pack()

        self.lbl_set_vision_mode = customtkinter.CTkLabel(master=self.vision_modes_frame, text='Set Vision Mode:')
        self.lbl_set_vision_mode.pack()

        self.lbl_model_threshold = customtkinter.CTkLabel(master=self.confidence_thresholds_frame, text=f'Set Model Confidence: {str(self.base_confidence)}')
        self.lbl_model_threshold.pack()

        '''     OptionMenu Widgets.     '''

        self.speed_limit_options_menu = customtkinter.CTkOptionMenu(master=self.speed_limit_frame, values=list(self.NATIONAL_SPEED_LIMITS.keys()), command=self.set_speed_limit)
        self.speed_limit_options_menu.pack(pady=self.PADDING['pady'], padx=10)

        self.vision_mode_options_menu = customtkinter.CTkOptionMenu(master=self.vision_modes_frame, values=list(self.vision_modes.keys()), command=self.set_vision_mode)
        self.vision_mode_options_menu.pack(pady=self.PADDING['pady'], padx=10)

        ''' Video Seeker. '''

        self.video_seek_bar = customtkinter.CTkSlider(master=self.seek_bar_frame, orientation=tk.HORIZONTAL, command=self.seek_video, from_=0, to=100)
        self.video_seek_bar.pack(pady=self.PADDING['pady'], padx=self.PADDING['padx'], fill='x')
        self.video_seek_bar.set(self.current_frame)

        ''' Confidence Threshold Seeker. '''

        self.confidence_seeker = customtkinter.CTkSlider(master=self.confidence_thresholds_frame, orientation=tk.HORIZONTAL, command=self.update_model_threshold, from_=0, to=1)
        self.confidence_seeker.pack(pady=self.PADDING['pady'], padx=self.PADDING['padx'], fill='x')
        self.confidence_seeker.set(self.base_confidence)

        ''' Video Seeker Timestamp Labels. '''

        self.video_time_label = customtkinter.CTkLabel(master=self.seek_bar_frame, text='0:00 / 0:00')
        self.video_time_label.pack()

    
    def source_video_import(self) -> None:
        
        ''' Helper function for user to source their desired video of choice within their hardwares file explorer. '''

        # Don't proceed with video import if no speed limit is set
        if self.current_speed_limit is None:
            messagebox.showwarning('Speed Limit Required', 'Please set a speed limit before importing a video :)')
            return

        # If media currently present playng, reset application.
        if self.video is not None:
            self.stop_video_import()
        
        # Get filepath from users chosen media file. 
        video_capture_path = filedialog.askopenfilename(filetypes=[('Video files', '.mp4; *.avi; *.MOV')])

        # If path exists and is valid. 
        if video_capture_path:

            self.is_stopped = False
            self.is_paused = False
            
            # Read video capture from given path. 
            self.video = cv2.VideoCapture(video_capture_path)

            if not self.video.isOpened():
                self.video = None 
                return

            # Obtain medias properties. 
            self.fps = self.video.get(cv2.CAP_PROP_FPS)
            self.frame_interval = max(1, int(1000 / self.fps))
            self.total_frames = max(1, int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)))

            # If total frames > than 0 attesting video is valid, update seek bar to match video length.
            if self.total_frames != 0:
                self.video_seek_bar.configure(from_=0, to=self.total_frames)

            self.video_canvas.delete('all')
            self.video_canvas.imgtk = None

            # Play video.
            self.read_video()
    

    def read_video(self) -> None:
        
        '''
            Read selected video, process frames from that video in the VideoProcessing pipeline and return that 
            processed frame into the tkinter canvas widget to be displayed to the user. 
        '''

        # If video is playing.
        if not self.is_paused and \
            not self.is_stopped:

            # Read from video capture whilst return == True. 
            ret, frame = self.video.read()

            if ret:
                
                # Update the play/pause button accordingly. 
                self.play_pause_btn.configure(image=self.pause_icon)

                # Update video slider. 
                self.current_frame += 1
                self.video_seek_bar.set(self.current_frame)

                frame = self.process_video_for_canvas(frame=frame)


    def fetch_video_meta_data(self) -> None:

        ''' Helper function to fetch meta data about the video being processed. '''

        # Get the current time within the video.
        current_video_time = int(self.video.get(cv2.CAP_PROP_POS_FRAMES)) / self.video.get(cv2.CAP_PROP_FPS)

        # Get total video duration.
        total_video_time = int(self.total_frames / self.video.get(cv2.CAP_PROP_FPS))

        ## Convert timestamps into strings. 
        current_video_time_str = time.strftime('%M:%S', time.gmtime(current_video_time))
        total_video_time_str = time.strftime('%M:%S', time.gmtime(total_video_time))
        
        # Update seekbar label to inform user of current video timestamp.
        self.video_time_label.configure(text=f'{current_video_time_str} / {total_video_time_str}')


    def process_video_for_canvas(self, frame) -> None:

        '''
            Helper function to take frame, parse it in the video processing pipeline and convert it 
            into a format suitable for tkinter canvas widgets. 
        '''

        # Get width and height of the canvas widget. 
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        # Run model inference pipeline on the retrieved frame.
        inference_frame = process_video(
            frame=frame,
            speed_limit=self.current_speed_limit,
            frame_rate=self.fps,
            vision_type=self.current_vision_mode,
            confidence_threshold=self.base_confidence
        )
        
        # Retrieve meta data about media being processed. 
        self.fetch_video_meta_data()

        # Process frame frame processing for tkinter canvas widget. 
        frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (canvas_width, canvas_height))
        frame = ImageTk.PhotoImage(Image.fromarray(frame))

        # Update canvas widget with current frame.
        self.video_canvas.imgtk = frame 
        self.video_canvas.create_image(0, 0, image=frame, anchor = tk.NW)
        self.video_canvas.after(self.frame_interval, self.read_video)


    def seek_video(self, frame_no) -> None:

        ''' Update seek bar so user can choose which parts of the video can be viewed. '''
        
        # Update current frame with parsed frame number.
        self.current_frame = int(float(frame_no))

        # Set video to the frame number being parsed. 
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

        # If video paused.
        if self.is_paused:
            
            # Read the video.
            ret, frame = self.video.read()

            # If frame is retrieved successfully. 
            if ret:

                frame = self.process_video_for_canvas(frame=frame)


    def handle_video_state(self) -> None:
        
        ''' Handle video state, controlled by user pressing either play/pause. '''

        if self.is_paused:
            self.is_paused = False
            self.read_video()
            self.play_pause_btn.configure(image=self.pause_icon)
        
        else:
            self.is_paused = True
            self.play_pause_btn.configure(image=self.play_icon)


    def set_speed_limit(self, speed_limit) -> None:
        
        ''' When users select a speed from the given limits, intialise that limit.'''

        self.current_speed_limit = self.NATIONAL_SPEED_LIMITS.get(self.speed_limit_options_menu.get(), 0)


    
    def set_vision_mode(self, vision_mode) -> None:

        ''' Helper function to handle vision mode selection from dropdown menu. '''

        self.current_vision_mode = self.vision_modes.get(self.vision_mode_options_menu.get(), 0)


    def update_model_threshold(self, threshold) -> None:
    
        ''' Helper function allowing users to update a models confidence threshold dynamically. '''

        self.base_confidence = float(threshold)
        self.lbl_model_threshold.configure(text=f'Set Model Confidence: {str(self.base_confidence)}')
    

    def delete_import(self) -> None:

        ''' When video is stopped, reset all variables to free resources. '''

        self.is_stopped = True
        self.is_paused = True
        self.current_frame = 0

        self.video_canvas.delete('all')
        self.video_canvas.imgtk = None
        self.video_seek_bar.set(self.current_frame)
        self.play_pause_btn.configure(image=self.play_icon)

        if self.video is not None:
            self.video.release()
            self.video = None
