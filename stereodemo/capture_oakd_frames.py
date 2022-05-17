import cv2
import depthai as dai
import numpy as np
import json

# Adapted from LearnOpenCV
# https://github.com/spmallick/learnopencv/tree/master/oak-getting-started

def getFrame(queue):
  # Get frame from queue
  frame = queue.get()
  # Convert frame to OpenCV format and return
  return frame.getCvFrame()

def getMonoCamera(pipeline, isLeft):
  # Configure mono camera
  mono = pipeline.createMonoCamera()

  # Set Camera Resolution
  mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

  if isLeft:
      # Get left camera
      mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
  else :
      # Get right camera
      mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
  return mono


def getStereoPair(pipeline, monoLeft, monoRight):
    # Configure stereo pair for depth estimation
    stereo = pipeline.createStereoDepth()
    # Checks occluded pixels and marks them as invalid
    stereo.setLeftRightCheck(True)
    
    # Configure left and right cameras to work as a stereo pair
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    return stereo

if __name__ == '__main__':

    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Set up left and right cameras
    monoLeft = getMonoCamera(pipeline, isLeft = True)
    monoRight = getMonoCamera(pipeline, isLeft = False)

    # Combine left and right cameras to form a stereo pair
    stereo = getStereoPair(pipeline, monoLeft, monoRight)
    
    # Set XlinkOut for disparity, rectifiedLeft, and rectifiedRight
    xoutDisp = pipeline.createXLinkOut()
    xoutDisp.setStreamName("disparity")
    
    xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutDepth.setStreamName("depth")

    xoutRectifiedLeft = pipeline.createXLinkOut()
    xoutRectifiedLeft.setStreamName("rectifiedLeft")

    xoutRectifiedRight = pipeline.createXLinkOut()
    xoutRectifiedRight.setStreamName("rectifiedRight")

    stereo.disparity.link(xoutDisp.input)
    
    stereo.rectifiedLeft.link(xoutRectifiedLeft.input)
    stereo.rectifiedRight.link(xoutRectifiedRight.input)
    stereo.depth.link(xoutDepth.input)
    
    # Pipeline is defined, now we can connect to the device

    with dai.Device(pipeline) as device:

        calib = device.readCalibration()
        w, h = monoLeft.getResolutionSize()
        
        # The rectified stereo images intrinsics both correspond to the right camera intrinsics.
        disparityIntrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, dai.Size2f(w, h))
        baselineMeters = 1e-2 * calib.getBaselineDistance(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT)
        with open('stereodemo-calibration.json', 'w') as f:
            d = dict(baseline_meters=baselineMeters,
                     fx=disparityIntrinsics[0][0],
                     fy=disparityIntrinsics[1][1],
                     cx0=disparityIntrinsics[0][2],
                     cx1=disparityIntrinsics[0][2],
                     cy=disparityIntrinsics[1][2])
            f.write(json.dumps(d))

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        disparityQueue = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)
        rectifiedLeftQueue = device.getOutputQueue(name="rectifiedLeft", maxSize=1, blocking=False)
        rectifiedRightQueue = device.getOutputQueue(name="rectifiedRight", maxSize=1, blocking=False)
        depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

        # Calculate a multiplier for colormapping disparity map
        disparityMultiplier = 255 / stereo.getMaxDisparity()

        cv2.namedWindow("Stereo Pair")
        
        # Variable use to toggle between side by side view and one frame view.
        sideBySide = False

        print ("Press 's' to save an image")

        save_frame_id = 0
        while True:
            
            # Get disparity map
            disparityPixels = getFrame(disparityQueue)
            
            # Colormap disparity for display
            disparity = (disparityPixels * disparityMultiplier).astype(np.uint8)
            disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
            
            # Depth
            depthMm = getFrame(depthQueue)
            centralDepthMm = depthMm[h//2, w//2]
            centralDisp = disparityPixels[h//2, w//2]
            depthFromDispMeters = (baselineMeters * disparityIntrinsics[0][0]) / centralDisp
            # print (f"Central pixel depth = {centralDepthMm} disparity_raw = {centralDisp} depthFromDispMeters={depthFromDispMeters}")

            # Get left and right rectified frame
            leftFrame = getFrame(rectifiedLeftQueue)
            rightFrame = getFrame(rectifiedRightQueue)
            
            if sideBySide:
                # Show side by side view
                imOut = np.hstack((leftFrame, rightFrame))
            else :
                # Show overlapping frames
                imOut = np.uint8(leftFrame/2 + rightFrame/2)
                        
            imOut = cv2.cvtColor(imOut,cv2.COLOR_GRAY2RGB)
            cv2.imshow("Stereo Pair", imOut)
            cv2.imshow("Disparity", disparity)

            # Check for keyboard input
            key = cv2.waitKey(1)
            if key == ord('q'):
                # Quit when q is pressed
                break
            elif key == ord('t'):
                # Toggle display when t is pressed
                sideBySide = not sideBySide
            elif key == ord('s'):
                # Save the current frames  
                cv2.imwrite(f"img_{save_frame_id:03d}_left.png", leftFrame)
                cv2.imwrite(f"img_{save_frame_id:03d}_right.png", rightFrame)
                print (f"Wrote img_{save_frame_id:03d}_left/right.png")
                save_frame_id += 1

