def normalization(self, current_x, list_x):
    x_min = min(list_x)
    x_max = max(list_x)
    x_norm = (current_x - x_min) / ( x_max - x_min)
    return x_norm + 0.001

