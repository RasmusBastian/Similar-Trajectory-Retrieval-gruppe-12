import numpy as np

def onehot_encode(traj_list_onedim, num_classes):
    onehot_list = []
    for traj in traj_list_onedim:
        one_hot = np.zeros((len(traj), num_classes))
        one_hot[np.arange(len(traj)), traj] = 1.0
        onehot_list.append(one_hot.tolist())
    return onehot_list

def image_encode(x_list, y_list, x_num_classes, y_num_classes):
    # local change: one float32 array instead of nested Python lists (same values, far less memory)
    image_list = np.zeros((len(x_list), x_num_classes, y_num_classes), dtype=np.float32)
    for i in range(len(x_list)):
        image_list[i, x_list[i], y_list[i]] = 1.0
    return image_list