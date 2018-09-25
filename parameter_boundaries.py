from growth import get_FL
from main import initiate_models

models = initiate_models()

# Get maximum global flowlength
FL_list = []
for model in models:
    this_FL = 0
    for cell in model.allVegetation:
        this_FL += cell.FL
    FL_list.append(this_FL)

FL_glob_max = max(FL_list)
print(FL_glob_max)

# Get maximum local flowlength
print(get_FL(200))
FL_list = []
for model in models:
    for cell in model.allVegetation:
        FL_list.append(cell.FL)

FL_loc_max = max(FL_list)
print(FL_loc_max)