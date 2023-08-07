import cv2
import numpy as np

# from firebase_admin import db
from firebase_admin import credentials, db, initialize_app


# Initialize Firebase
cred = credentials.Certificate(
    "unityfirebase-4db68-firebase-adminsdk-zzsdt-3f0c2b1cde.json"
)
initialize_app(
    cred,
    {
        "databaseURL": "https://unityfirebase-4db68-default-rtdb.asia-southeast1.firebasedatabase.app/"
    },
)

# Open video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for red color in HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Add this check
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Send the position data to Firebase
            ref = db.reference("ball_position")
            ref.set({"x": cX, "y": cY})

            # Draw a circle at the center of the ball
            cv2.circle(frame, (cX, cY), 10, (0, 255, 0), -1)

    # Display the resulting frame and mask
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
