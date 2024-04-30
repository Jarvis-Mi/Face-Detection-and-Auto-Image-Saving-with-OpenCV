import cv2
import os


def main():
    """
    This function captures and saves 500 images of a user's face using OpenCV.
    """

    # Initialize video capture object
    cap = cv2.VideoCapture(0)

    # Load pre-trained face cascade classifier for frontal face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Initialize image counter and get user input for folder name
    image_count = 0
    user_name = input("Enter a name for the folder (lowercase): ").lower()

    # Create folder path based on user input
    image_folder_path = os.path.join('./images/', user_name)

    # Check if folder already exists
    if os.path.exists(image_folder_path):
        print(f"Folder name '{user_name}' already exists. Please choose another name.")
        user_name = input("Enter a new name for the folder (lowercase): ").lower()
    else:
        # Create the folder if it doesn't exist
        os.makedirs(image_folder_path)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)

        for (x, y, w, h) in faces:
            # Increment image counter
            image_count += 1

            # Generate image filename with sequential numbering
            image_filename = os.path.join(image_folder_path, f"{image_count}.jpg")

            # Print image capture confirmation message
            print(f"Capturing image {image_count}...")

            # Save the detected face region as a JPG image
            cv2.imwrite(image_filename, frame[y:y + h, x:x + w])

            # Draw a green rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Display the frame with detected faces
        cv2.imshow("Face Capture Window", frame)

        # Exit loop if 'q' key is pressed or 500 images are captured
        if cv2.waitKey(1) & 0xFF == ord('q') or image_count > 499:
            break

    # Release the video capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

    print("Successfully captured 500 images!")


if __name__ == "__main__":
    main()
