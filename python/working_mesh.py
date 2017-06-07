import eos
import numpy as np
import pylab
import dlib
import cv2
import sys

PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)

def main():
    """Demo for running the eos fitting from Python."""
    image = cv2.imread(sys.argv[1])
    landmarks = np.ndarray.tolist(get_landmarks(image))
    landmark_ids = list(map(str, range(1, 69))) # generates the numbers 1 to 68, as strings
    image_width = image.shape[1] # Make sure to adjust these when using your own images!
    image_height = image.shape[0]

    model = eos.morphablemodel.load_model("../share/sfm_shape_3448.bin")
    blendshapes = eos.morphablemodel.load_blendshapes("../share/expression_blendshapes_3448.bin")
    landmark_mapper = eos.core.LandmarkMapper('../share/ibug_to_sfm.txt')
    edge_topology = eos.morphablemodel.load_edge_topology('../share/sfm_3448_edge_topology.json')
    contour_landmarks = eos.fitting.ContourLandmarks.load('../share/ibug_to_sfm.txt')
    model_contour = eos.fitting.ModelContour.load('../share/model_contours.json')

    (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(model, blendshapes,
        landmarks, landmark_ids, landmark_mapper,
        image_width, image_height, edge_topology, contour_landmarks, model_contour)

    texture = eos.render.extract_texture(mesh, pose, image, False, 2048)
    print(landmark_ids)
    cv2.imwrite(sys.argv[2] + '.isomap.png', texture)	
    eos.core.write_obj(mesh, sys.argv[2] + ".obj")

def get_landmarks(image):
    rects = DETECTOR(image, 0)
    if len(rects) == 0:
        raise NoFaces
    return np.matrix([[p.x, p.y] for p in PREDICTOR(image, rects[0]).parts()])


def read_pts(filename):
    """A helper function to read ibug .pts landmarks from a file."""
    lines = open(filename).read().splitlines()
    lines = lines[3:71]

    landmarks = []
    for l in lines:
        coords = l.split()
        landmarks.append([float(coords[0]), float(coords[1])])

    return landmarks

if __name__ == "__main__":
    main()
