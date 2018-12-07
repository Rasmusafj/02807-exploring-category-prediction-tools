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
            "bands": 50,
            "debug_number": 50
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

# SVC
tracemalloc.start()
custom_timer = CustomTimer()
custom_timer.start()
model = get_model("SVM")
total_time = custom_timer.stop_timer_and_get_result()
print("Took {0} seconds to initialize model and preprocess data.".format(total_time))