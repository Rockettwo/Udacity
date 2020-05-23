# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
import glob as glob
import matplotlib.image as mpimg

from HelperClasses import LaneDetection

# videos = glob.glob('../*.mp4')
#
# for fname in videos:
#     # New instance of LaneDetection to start from scratch and reset previous settings
#     lanDet = LaneDetection()
#
#     clip1 = VideoFileClip(fname)
#     white_clip = clip1.fl_image(lanDet.processImage)
#     white_clip.write_videofile("../output_videos/" + fname.split('\\')[-1], audio=False)


lanDet = LaneDetection()

img = mpimg.imread("../test_images/test6.jpg")
result = lanDet.processImage(img)
