from detector import init_detector,detect_faces
from embedder import BuffaloModel,open_numpy_as_tensor,get_vector_from_face

DETECTOR_MODEL_FILE = "scrfd_500m_bnkps.onnx"
EMBEDDER_MODEL_FILE='w600k_r50.onnx'
INPUT_IMAGE = "group_photo.png"

face_detector = init_detector(DETECTOR_MODEL_FILE, target_size=2048)
embedder=BuffaloModel(path=EMBEDDER_MODEL_FILE)
        
detected_faces = detect_faces(
    image_path=INPUT_IMAGE,
    detector=face_detector,
    conf_thresh=0.25
)

for face in detected_faces:
    tensor_face_img=open_numpy_as_tensor(face['image'])
    face_vector=get_vector_from_face(img=tensor_face_img,model=embedder)
    print(face_vector.shape)
    
    