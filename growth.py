def get_FL(y):
    return y * (y + 1) / 2

def f_pos(y, alpha, cell_size):
    # The sign is reversed because the y-coordinates go in reverse
    return 1 + alpha * (cell_size * y - 1)

def f_conn(FL_loc, FL_glob, FL_glob_max, FL_loc_max, beta, gamma):
    glob = beta * FL_glob / FL_glob_max
    loc = gamma * FL_loc / FL_loc_max
    return (1 - glob), (1 + loc)

def f_comp(RL_eff, BR_eff, c_ir, c_ib):
    return 2000 * RL_eff * c_ir, 600 * BR_eff * c_ib