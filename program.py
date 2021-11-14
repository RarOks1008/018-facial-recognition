import cv2

camera = cv2.VideoCapture(0)
face_c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    _, frame = camera.read()

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_c.detectMultiScale(grayscale, 1.1, 5)
    #Middle value is best kept between 1.1 to 1.6
    #Closer to 1.1 for shorter but better match, and closer to 1.6 for faster algorithm

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("My Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()