from flask import Flask, request
from smart_move_analysis.utils import get_landmarks_from_angle, landmark_list_angles, EXERCISE_ANGLES, obtain_angles
from smart_move_analysis.reference_store import LandmarkData, ReferenceStore
from smart_move_analysis.analyzer import Analyzer



REPETITION_THRESHOLD = .85

reference_store = ReferenceStore('smart_move_analysis/data')

knn_models = {}
for exercise in reference_store.exercises():
    knn_models.update({exercise_half:Analyzer.from_exercise_references(
            exercise_references=reference_store.get(*exercise_half),
            exercise_angles=exercise if exercise in EXERCISE_ANGLES else None)
        for exercise_half in [
            (exercise, True), (exercise, False)
        ]
    })

            

app = Flask(__name__)
            

            
@app.route('/')
def hello():
    return 'Hello, World!'

@app.post('/exercise/analysis')
def exercise_analysis():
    first_half = bool(request.json.get('first_half'))
    exercise_category = request.json.get('exercise_category') \
            .lower()

    all_landmarks = request.json.get('landmarks')
    landmarks_coordinates = []
    for i in range(33):
        coord = [coord for coord in all_landmarks if coord["id"] == i]
        if coord:
            landmarks_coordinates.append({"x": coord[0]["x"], "y": coord[0]["y"], "z": coord[0]["z"]})

    if not landmarks_coordinates:
        return {"error_msg": f"No landmarks were specified or were not in the correct format."}, 400

    if (exercise_category, first_half) not in knn_models:
        return {"error_msg": f"The specified exercise {exercise_category} is not supported."}, 400

    knn_model = knn_models[(exercise_category, first_half)]
    angles_to_use = obtain_angles(exercise_category if exercise_category in EXERCISE_ANGLES else None)
    
    if knn_model:
        print(landmarks_coordinates)
        print(angles_to_use)
        landmark_angles = landmark_list_angles(
                [LandmarkData(visibility=None, **coordinates) for coordinates in landmarks_coordinates],
                angles=angles_to_use,
                d2=True
            )
        correctness, most_divergent_angle_idx, most_divergent_angle_ideal = knn_model.correctness(landmark_angles)
        most_divergent_angle_value = landmark_angles[most_divergent_angle_idx]
        progress = knn_model.progress(landmark_angles)
    else:
        return {"error_msg": f"The system is not trained for exercise {exercise_category}."}, 400

    landmark_first, landmark_middle, landmark_last = get_landmarks_from_angle(most_divergent_angle_idx, exercise_category)

    finished_repetition = False
    if progress > REPETITION_THRESHOLD:
        first_half = not first_half
        if first_half:
            finished_repetition = True

    # Convert to Python data types that can then be easily rendered into JSON (Example)
    response = {}
    response.update(
        correctness=correctness,
        progress=progress,
        finished_repetition=finished_repetition,
        first_half=first_half,
        most_divergent_angle_landmark_first=landmark_first,
        most_divergent_angle_landmark_middle=landmark_middle,
        most_divergent_angle_landmark_last=landmark_last,
        most_divergent_angle_value=most_divergent_angle_value,
        most_divergent_angle_ideal=most_divergent_angle_ideal)

    return response, 200

