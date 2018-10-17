import data

def alpha(param):
    params_model, params_RL, params_BR, _ = data.get_params()
    params_model["alpha"] = param

    return params_model, params_RL, params_BR

def beta(param):
    params_model, params_RL, params_BR, _ = data.get_params()
    params_model["beta"] = param

    return params_model, params_RL, params_BR

def gamma(param):
    params_model, params_RL, params_BR, _ = data.get_params()
    params_model["gamma"] = param

    return params_model, params_RL, params_BR


def r_bb(param):
    params_model, params_RL, params_BR, _ = data.get_params()
    params_BR["r_ib"] = param

    return params_model, params_RL, params_BR

def r_rr(param):
    params_model, params_RL, params_BR, _ = data.get_params()
    params_RL["r_ir"] = param

    return params_model, params_RL, params_BR

def inter(param):
    params_model, params_RL, params_BR, _ = data.get_params()
    p1, p2 = param
    params_BR["r_ir"] = p1
    params_RL["r_ib"] = p2

    return params_model, params_RL, params_BR
