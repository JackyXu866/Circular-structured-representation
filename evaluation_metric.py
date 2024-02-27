# this is the code for six evaluation metrics in emotion distribution learning, added by a top-1 accuracy
# written by XDU Jingyuan Yang

import math
import numpy as np

def Evaluation_metrics(emo, dist_emo):
    # return 6 evalution metrics in label distribtion learing
    emo = emo.cpu().detach().numpy()  ######
    dist_emo = dist_emo.cpu().detach().numpy() ######
    # 1-Chebyshev
    Dis_1 = abs(dist_emo - emo).max(1)
    Dis_1 = np.average(Dis_1)
    # 2-Clark
    Dis_2 = np.sqrt((((dist_emo - emo) ** 2) / ((dist_emo + emo) ** 2)).sum(1))
    Dis_2 = np.average(Dis_2) / (np.sqrt(8)) # normalize
    # 3-Canberra
    Dis_3 = (abs(dist_emo - emo) / (dist_emo + emo)).sum(1)
    Dis_3 = np.average(Dis_3) / 8 # normalize
    # 4-Kullback-Leibler
    e = 1e-10 * np.ones((emo.shape[0], emo.shape[1]))
    Dis_4 = (dist_emo * np.log(dist_emo/emo + e)).sum(1)
    Dis_4 = np.average(Dis_4)
    # 5-Cosine
    Sim_1 = (dist_emo * emo).sum(1) / (np.sqrt((emo ** 2).sum(1)) * np.sqrt((dist_emo ** 2).sum(1)))
    Sim_1 = np.average(Sim_1)
    # 6-Intersection
    Sim_2 = (np.minimum(dist_emo, emo, out=None)).sum(1)
    Sim_2 = np.average(Sim_2)

    return Dis_1, Dis_2, Dis_3, Dis_4, Sim_1, Sim_2