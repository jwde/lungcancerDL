from evaluate import main


models = [
    ('alexslicer','alexslicer', '../input/3Darrays_visual/'),
    ('alexslicerMIL','alexslicerMIL', '../input/3Darrays_visual/'),
    ('alexslicerZMIL','alexslicerZMIL', '../input/3Darrays_visual/'),
    ('alexslicerZMIL','alexslicerZMILunbalancednotrim', '../input/arrays_notrim/'),
    ('alexslicer','alexslicerunbalancednotrim', '../input/arrays_notrim/'),
]

for model in models:
    main(['eval_all', model[0], model[1], '../models/' + model[1], model[2], '../input/stage1_solution_trim.csv'])
