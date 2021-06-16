from utils.hyperparameters import HyperParameters

a = HyperParameters
b = a.create_from_file("person.json")
c = 1

# import json
#
# personDict = {
#   'learning_rate': 999,
#   'gradient_clip': 999
# }
#
# with open('person.json', 'w') as json_file:
#   json.dump(personDict, json_file)

