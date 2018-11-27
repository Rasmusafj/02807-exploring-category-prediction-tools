from models import SVCMachineLearningModel

arguments = {
    "normalize": True,
    "debug_number": 2000,
    "h": 2**11,
    }

model = SVCMachineLearningModel(**arguments)

print('Hej2')

# Works well C = 200.0, linear

for C in [200.0]:
    #for gamma in gamma_id_list:
    model.C_id = C
    print(C)
    model.kernel_id = 'linear'
    print(model.kernel_id)
    #model.degree_id = degree
    #model.gamma_id = gamma
    model.fit_data()
    accuracy = model.evaluate_on_test()
    print("Accuracy is: {0}".format(accuracy))

#or C in [0.5,1,5,10]:
#    #for gamma in gamma_id_list:
#    model.C_id = C
#    print(C)
#    model.kernel_id = 'poly'
#    print(model.kernel_id)
#    #model.degree_id = degree
#    #model.gamma_id = gamma
#    model.fit_data()
#    accuracy = model.evaluate_on_test()
#    print("Accuracy is: {0}".format(accuracy))
