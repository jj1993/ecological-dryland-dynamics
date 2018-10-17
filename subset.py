from collections.__init__ import defaultdict
import numpy as np

from model import RUNOFF_RETURN

RESOLUTION = 2
patch_types = ['RP', 'BP', 'MP', 'RG', 'BG', 'MG']

def alpha(patches):
    # split into runoff return, patch type, flowlength

    # subsets = [[], []]
    # # Filter on global flowlength
    # for patch in patches:
    #     if patch.model.runoff_return:
    #         subsets[0].append(patch)
    #     else:
    #         subsets[1].append(patch)

    subsets = [patches]
    new_subsets = []
    for subset in subsets:
        subset_dict = defaultdict(list)
        for patch in subset:
            subset_dict[patch.type].append(patch)
        new_subsets += subset_dict.values()
    subsets = new_subsets

    new_subsets = []
    for subset in subsets:
        # Sorting patches on FL
        subset.sort(key=lambda x: sum([cell.FL for cell in x.RL + x.BR]))
        # Splitting category in #FL_RESOLUTION equally sized groups of patches
        new_subsets += list(np.array_split(subset, RESOLUTION))
    subsets = new_subsets

    return subsets

# def beta(patches):
#     # match patches from corresponding plots
#     subsets = []
#     for runoff_nr in RUNOFF_RETURN.keys():
#         subsets_dict = defaultdict(list)
#         twin_nr = RUNOFF_RETURN[runoff_nr]
#         for patch in patches:
#             if patch.model.nr == runoff_nr or patch.model.nr == twin_nr:
#                 subsets_dict[patch.id[2:].strip("*")].append(patch)
#         subsets += subsets_dict.values()
#
#     return subsets

def gamma(patches):
    # split into runoff return, patch type, slope position

    # subsets = [[], []]
    # # Filter on global flowlength
    # for patch in patches:
    #     if patch.model.runoff_return:
    #         subsets[0].append(patch)
    #     else:
    #         subsets[1].append(patch)

    subsets = [patches]

    new_subsets = []
    for subset in subsets:
        subset_dict = defaultdict(list)
        for patch in subset:
            subset_dict[patch.type].append(patch)
        new_subsets += subset_dict.values()
    subsets = new_subsets

    new_subsets = []
    for subset in subsets:
        # Sorting patches on FL
        subset.sort(key=lambda x: sum([cell.pos[0] for cell in x.RL + x.BR]))
        # Splitting category in #FL_RESOLUTION equally sized groups of patches
        new_subsets += list(np.array_split(subset, RESOLUTION))
    subsets = new_subsets

    return subsets

def competition(patches):
    # split into runoff return, patch type, slope position

    # subsets = [[], []]
    # # Filter on global flowlength
    # for patch in patches:
    #     if patch.model.runoff_return:
    #         subsets[0].append(patch)
    #     else:
    #         subsets[1].append(patch)

    subsets = [patches]

    new_subsets = []
    for subset in subsets:
        # Sorting patches on FL
        subset.sort(key=lambda x: sum([cell.FL for cell in x.RL + x.BR]))
        # Splitting category in #FL_RESOLUTION equally sized groups of patches
        new_subsets += np.array_split(subset, RESOLUTION)
    subsets = new_subsets

    new_subsets = []
    for subset in subsets:
        # Sorting patches on FL
        subset = list(subset)
        subset.sort(key=lambda x: sum([cell.pos[0] for cell in x.RL + x.BR]))
        # Splitting category in #FL_RESOLUTION equally sized groups of patches
        new_subsets += np.array_split(subset, RESOLUTION)
    subsets = new_subsets

    intra_r, intra_b, inter = [], [], []
    for subset in subsets:
        subset_dict = defaultdict(list)
        for patch in subset:
            # Only sort on the type, not the size!!
            subset_dict[patch.type[0]].append(patch)
        intra_r.append(subset_dict['R'])
        intra_b.append(subset_dict['B'])
        inter.append(subset_dict['M'])

    return intra_r, intra_b, inter

def all(patches):
    return [alpha(patches), gamma(patches), *competition(patches)]