import cv2

from .manager import OpenCVWindowManager, CaptureManager
from .utils import preprocess_frame, draw_bounding_box, draw_landmarks

from .correlationfilter.base import CFTracker
from .correlationfilter.utils import generate_bounding_box


class Tracker(object):
    r"""
    Notes
    ------
        Code based on the examples provided by Joseph Howse in his excellent
        book: "OpenCV Computer Vision with Python".
        http://nummist.com/opencv/
    """

    def __init__(self, video_file, target_centre, target_shape,
                 window_manager=OpenCVWindowManager):

        self._window_manager = window_manager(
            'Tracker', self.on_keypress)

        self._capture_manager = CaptureManager(
            cv2.VideoCapture(video_file), self._window_manager, True)

        # obtain first frame
        self._capture_manager.enter_frame()
        frame0 = self._capture_manager.frame
        if frame0 is not None:
            # pre-process first frame
            frame0 = preprocess_frame(frame0)
        self._capture_manager.exit_frame()

        # initialize cf tracker
        self._tracker = CFTracker(frame0, target_centre, target_shape)

        self._target_centre = target_centre
        self._target_shape = target_shape

    def track(self):
        r"""
        Tracking loop.
        """
        # create window
        self._window_manager.create_window()
        target_centre = self._target_centre
        target_shape = self._target_shape

        while self._window_manager.is_window_created:
            # obtain frame
            self._capture_manager.enter_frame()
            frame = self._capture_manager.frame
            # resize
            frame = cv2.resize(frame, (640, 360))
            self._capture_manager._frame = frame

            if frame is not None:

                # pre-process frame
                img = preprocess_frame(frame)
                # track target
                target_centre, _, _ = self._tracker.track(img, target_centre)
                # generate target bounding box
                target_bb = generate_bounding_box(target_centre, target_shape)
                # draw
                draw_landmarks(frame, target_centre.points)
                draw_bounding_box(frame, target_bb.points)

            self._capture_manager.exit_frame()
            self._window_manager.process_events()

    def on_keypress(self, key_code):
        r"""
        Handle a keypress.
            space  -> Take a screen-shot.
            tab    -> Start/stop recording a screen-cast.
            escape -> Quit.
        """
        if key_code == 32:  # space
            self._capture_manager.write_image('screen-shot.png')
        elif key_code == 9:  # tab
            if not self._capture_manager.is_writing_video:
                self._capture_manager.start_writing_video('screen-cast.avi')
            else:
                self._capture_manager.stop_writing_video()
        elif key_code == 27:  # escape
            self._window_manager.destroy_window()
