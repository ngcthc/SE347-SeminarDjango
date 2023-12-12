import cv2
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
import cv2, json, numpy as np, os
from keras.models import load_model
from scipy.spatial.distance import cosine

def index(request):
    template = loader.get_template('index.html')
    return HttpResponse(template.render({}, request))

def webcam_feed_stream():
    import cv2, json, numpy as np
    from keras.models import load_model
    from scipy.spatial.distance import cosine

    threshold = 0.92

    # Load haarcascades for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the pre-trained VGG model with the last two Dense layers removed
    model = load_model('webcam\data\mobilenetv2.h5')

    # Thư mục để lưu file JSON
    json_file_path = "webcam\data\\face_features.json"

    # Danh sách chứa thông tin khuôn mặt
    try: 
        with open(json_file_path, 'r') as file:
            face_features_list = json.load(file)
    except:
        face_features_list = []
        
    cap = cv2.VideoCapture(0)

    def cosine_similarity(feature1, feature2):
        return 1 - cosine(feature1, feature2)
    

    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)
        # Chuyển đổi ảnh sang ảnh đen trắng để tăng hiệu suất
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện khuôn mặt trong frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_roi = frame[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (224, 224))
            face_roi = np.expand_dims(face_roi, axis=0)
            face_features = model.predict(face_roi).flatten().tolist()

            if face_features_list is None: 
                cv2.putText(frame, f"Khong co du lieu khuon mat", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break

            max_similarity = 0
            best_match_name = ""
            for feature in face_features_list:
                name = feature["name"]
                stored_feature = feature["face_features"]
                similarity = cosine_similarity(face_features, stored_feature)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_name = name
                if similarity > threshold:
                    cv2.putText(frame, f"{best_match_name} - Similarity: {max_similarity:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)        

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def webcam(request):
    return StreamingHttpResponse(webcam_feed_stream(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def face_features_stream():

    # Load haarcascades for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Đọc frame từ webcam
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)
        # Chuyển đổi ảnh sang ảnh đen trắng để tăng hiệu suất
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện khuôn mặt trong frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_roi = frame[y:y+h, x:x+w]
            break

        global global_face_roi
        global_face_roi = face_roi

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

def create(request):
    return StreamingHttpResponse(face_features_stream(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def addface(request):
    if request.method == 'POST':
        name = request.POST.get('name', '')

        face_roi = global_face_roi
        # Thư mục để lưu ảnh khuôn mặt
        face_images_directory = "webcam\data\\face_images\\"
        if not os.path.exists(face_images_directory):
            os.makedirs(face_images_directory)

        # Thư mục để lưu file JSON
        json_file_path = "webcam\data\\face_features.json"

        # Load the pre-trained VGG model with the last two Dense layers removed
        model = load_model('webcam\data\mobilenetv2.h5')

        # Danh sách chứa thông tin khuôn mặt
        try: 
            with open(json_file_path, 'r') as file:
                face_features_list = json.load(file)
        except:
            face_features_list = []

        image_path = f'{face_images_directory}face_{name}.png'
        cv2.imwrite(image_path, face_roi)
        # print(image_path)
        face_roi = cv2.resize(face_roi, (224, 224))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_features = model.predict(face_roi).flatten().tolist()

        if face_features is not None:
            face_info = {
                "image_path": image_path,
                "name": name,
                "face_features": face_features
            }
            # Thêm vào danh sách
            face_features_list.append(face_info)

        with open(json_file_path, 'w') as json_file:
            json.dump(face_features_list, json_file, indent=4)

        # Thực hiện các thao tác khác cần thiết, ví dụ: lưu tên vào database
        return HttpResponse(f'Created face with name: {name}')
    else:
        return render(request, 'addface.html')  # Thay 'your_template_name' bằng tên thực tế của template
