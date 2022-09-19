import cv2 as cv
from pupil_apriltags import Detector

at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

vid = cv.VideoCapture(0)


while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    if not ret:
            break
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    tags = at_detector.detect(frame, estimate_tag_pose=False, camera_params=None, tag_size=None)
    
    print(tags.__len__())

    if tags.__len__() > 0:

        tag_family = str(tags[0].tag_family.decode('utf-8', 'ignore'))
        tag_id = str(tags[0].tag_id)


        print(tag_family+" "+tag_id)


    #print(tags[0].tag_family)
    # Display the resulting frame
    cv.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()