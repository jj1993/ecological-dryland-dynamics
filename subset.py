from collections.__init__ import defaultdict
from fit import initiate_models, RUNOFF_RETURN

FL_RESOLUTION = 2

def alpha():
    # set-up models

    patch_types = ['RP', 'BP', 'MP', 'RG', 'BG', 'MG']
    patches_return = defaultdict(list)
    patches_no_return = defaultdict(list)
    for key in patch_types:
        for n, model in enumerate(models):
            if (n + 1) in RUNOFF_RETURN:
                patches_return[key] += model.patches[key]
            else:
                patches_no_return[key] += model.patches[key]

    sorted_patches = list(patches_return.values()) + list(patches_no_return.values())
    sorted_patches_new = []
    for category in sorted_patches:
        # Sorting patches on FL
        category.sort(key=lambda x: sum([cell.FL for cell in x]))
        # Splitting category in #FL_RESOLUTION equally sized groups of patches
        sorted_patches_new += np.array_split(category, FL_RESOLUTION)
    sorted_patches = sorted_patches_new

    # TODO: Only compare patches from same category

    # return alpha
    pass


def intra_comp():
    # set-up models
    models = initiate_models()

    # return r_rr, r_bb
    pass


def inter_comp():
    # set-up models
    models = initiate_models()

    # return r_rb, r_br
    pass


def gamma():
    # set-up models
    models = initiate_models()

    # return gamma
    pass


def beta():
    # set-up models
    models = initiate_models()

    # return beta
    pass

