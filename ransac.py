from p3p import P3P
from scipy.spatial.transform import Rotation as Rot
import numpy as np
from tools import reproject_square_sum

def RANSACPnP(points2D, points3D, cameraMatrix, distCoeffs, times = 10, specify=[]):
    best_reproject = np.inf
    best_T = None
    best_rvec = None
    if len(specify) != 0:
        P3P_c = P3P(cameraMatrix, distCoeffs)
        rvec, T = P3P_c.solve_P4P(points2D[specify], points3D[specify])
        R = Rot.from_rotvec(rvec).as_matrix()
        s = reproject_square_sum(points2D, points3D, R, T, cameraMatrix, distCoeffs)
        return rvec, T, best_reproject

    for i in range(times):
        try:
            random_sample = np.random.choice(points2D.shape[0], 4, replace=False)
            P3P_c = P3P(cameraMatrix, distCoeffs)
            output = P3P_c.solve_P4P(points2D[random_sample], points3D[random_sample])
            if len(output) == 0:
                continue
            else:
                rvec, T = output
            
            R = Rot.from_rotvec(rvec).as_matrix()
            s = reproject_square_sum(points2D, points3D, R, T, cameraMatrix, distCoeffs)
            
            if s < best_reproject:
                best_reproject = s
                best_T = T
                best_rvec = rvec
        except Exception as e:
            pass
    return best_rvec, best_T, best_reproject