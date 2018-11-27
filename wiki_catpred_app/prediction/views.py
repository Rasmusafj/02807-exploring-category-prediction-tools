from django.shortcuts import render

import sys
super_path = "/Users/arpelarpe/programming_projects/02807-exploring-category-prediction-tools"
sys.path.append(super_path)
from ml_models import LSHMinHash, SVCMachineLearningModel

directory_path = super_path + "/data/dataset/"
arguments = {
    "k_neighbours": 20,
    "k_hash_functions": 400,
    "n_shingles": 1,
    "bands": 200,
    "debug_number": 20,
    "directory_path": directory_path,
}
LSH = LSHMinHash(**arguments)

arguments = {
        "debug_number": 20,
        "normalize": True,
        "h": 2**11,
        "directory_path": directory_path
        }

SVC = SVCMachineLearningModel(**arguments)


def index(request):
    return render(request, 'index.html', {})


def predict(request):
    path = request.POST["path"]
    f = open(path, 'r', encoding="utf-8")
    line = f.read()
    line = line.rstrip("\n").split(",")
    LSH_prediction = LSH.predict_new([line])[0]
    SVC_prediction = SVC.predict_new([line])[0]

    contents = {
        "LSH": LSH_prediction,
        "SVC": SVC_prediction
    }

    return render(request, 'index.html', contents)
