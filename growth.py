
def get_FL(y):
    return y * (y + 1) / 2

def f_pos(y, alpha, L, cell_size):
    return alpha * cell_size * y / L + (1 - alpha / 2)

def f_conn(FL_loc, FL_glob, p):
    width, height, beta, gamma = p["width"], p["height"], p["beta"], p["gamma"]
    FL_glob_max = width * sum([get_FL(y) for y in range(height)])
    FL_loc_max = get_FL(height)
    glob = beta * FL_glob / FL_glob_max
    loc = gamma * FL_loc / FL_loc_max
    return (1 - glob) * (1 + loc)

def f_comp(RL_eff, BR_eff, r_ir, r_ib):
    return RL_eff * r_ir + BR_eff * r_ib