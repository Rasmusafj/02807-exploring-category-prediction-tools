"""
Functions to evaluate the performance of the implemented models.

From project description:

    "At last, we will compare the created tools on benchmarks such as speed,
    accuracy, memory consumption and if time permits, generalization to documents not from Wikipedia."

"""
import tracemalloc
from ml_models import LSHMinHash, SVCMachineLearningModel
from utils import CustomTimer, total_allocated_memory


def get_model(model):
    if model == "LSH":
        arguments = {
            "k_neighbours": 15,
            "k_hash_functions": 400,
            "n_shingles": 1,
            "bands": 100,
            "debug_number": 0
        }
        return LSHMinHash(**arguments)
    if model == "SVM":
        arguments = {
            "C_id": 200.0,
            "kernel_id": 'linear',
            "debug_number": 100,
            "h": 2**11
        }
        return SVCMachineLearningModel(**arguments)

#LSHMinHash
tracemalloc.start()
custom_timer = CustomTimer()
custom_timer.start()
model = get_model("LSH")
total_time = custom_timer.stop_timer_and_get_result()
print("Took {0} seconds to initialize model and preprocess data.".format(total_time))

custom_timer.start()
model.evaluate_on_test()
total_time = custom_timer.stop_timer_and_get_result()
print("Took {0} seconds to evaluate model.".format(total_time))
snapshot = tracemalloc.take_snapshot()
total_allocated_memory(snapshot)
model.generate_confusion_matrix("Confusion matrix LSH", "LSH")

"""
#SVM
'''
tracemalloc.start()
custom_timer = CustomTimer()
custom_timer.start()
model = get_model("SVM")
total_time = custom_timer.stop_timer_and_get_result()
print("Took {0} seconds to initialize model and preprocess data.".format(total_time))

# KNN
custom_timer.start()
model.evaluate_on_test()
total_time = custom_timer.stop_timer_and_get_result()
print("Took {0} seconds to evaluate model.".format(total_time))
snapshot = tracemalloc.take_snapshot()
total_allocated_memory(snapshot)
<<<<<<< HEAD
'''

"""

k_neighbours_list = [3, 5, 10, 20, 40]

# Pipeline for the AbstractModel implementation
for k_neighbour in k_neighbours_list:
    model.k_neighbours = k_neighbour
    accuracy = model.evaluate_on_test()
    print("Accuracy for k={0} is: {1}".format(k_neighbour, accuracy))
"""
=======
"""
