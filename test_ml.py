from ml_models import SVCMachineLearningModel

arguments = {
    "h": 2**11
    }

model = SVCMachineLearningModel(**arguments)

model.C_id = 200.0
model.kernel_id = 'linear'
model.fit_data()
accuracy = model.evaluate_on_test()
print("Accuracy is: {0}".format(accuracy))
print(model.model.n_support_)
model.generate_confusion_matrix("Confusion matrix SVM", "SVM")
