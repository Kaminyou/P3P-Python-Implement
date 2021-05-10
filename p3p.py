# reference
# https://blog.csdn.net/baiyu9821179/article/details/78928776
# https://github.com/opencv/opencv/blob/master/modules/calib3d/src/p3p.cpp
import math
import numpy as np
from tools import getUndistortedPoints
from scipy.spatial.transform import Rotation as Rot

class P3P(object):
    def __init__(self, camera_matrix, distCoeffs):
        self.camera_matrix = camera_matrix
        self.distCoeffs = distCoeffs
        fx, fy, cx, cy = camera_matrix[0,0], camera_matrix[1,1], camera_matrix[0,2], camera_matrix[1,2]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.inverse_K()
        
    def inverse_K(self):
        self.inv_fx = 1 / self.fx
        self.inv_fy = 1 / self.fy
        self.cx_fx = self.cx / self.fx
        self.cy_fy = self.cy / self.fy
    
    def solve_P4P(self, pixel_points, world_points):
        """
        input 
            pixel_points: a 4*2 vector with float type
            world_points: a 4*3 vector with float type
        output
            a array with rvec and T
            if the length of array is 0, then there is no root found
        """
        # GET THE UNDISTORTED POINTS 
        undistorted_points = getUndistortedPoints(pixel_points, self.camera_matrix, self.distCoeffs)
        
        #undistorted_points = undistorted_points * np.array([self.inv_fx, self.inv_fy]) - np.array([self.cx_fx, self.cy_fy]) #4*2
        undistorted_points_extract = undistorted_points[:3] #3*2
        undistorted_points_extract = np.insert(undistorted_points_extract, 2, np.ones(len(undistorted_points_extract)), axis=1) #3*3
        #norm = np.sqrt(np.sum(np.power(undistorted_points_extract,2), axis=1))
        #mks = 1 / norm
        #undistorted_points_extract = (undistorted_points_extract.T/norm).T
        
        distances = np.zeros(3, dtype=float)
        X0, Y0, Z0, X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3 = world_points.flatten()
        distances[0] = math.sqrt((X1 - X2) * (X1 - X2) + (Y1 - Y2) * (Y1 - Y2) + (Z1 - Z2) * (Z1 - Z2))
        distances[1] = math.sqrt((X0 - X2) * (X0 - X2) + (Y0 - Y2) * (Y0 - Y2) + (Z0 - Z2) * (Z0 - Z2))
        distances[2] = math.sqrt((X0 - X1) * (X0 - X1) + (Y0 - Y1) * (Y0 - Y1) + (Z0 - Z1) * (Z0 - Z1))
        #print(distances)
        
        mu0, mv0, mk0, mu1, mv1, mk1, mu2, mv2, mk2 = undistorted_points_extract.flatten()
        len0, len1, len2 = np.sqrt(np.power(undistorted_points_extract, 2).sum(axis=1))
        #print(undistorted_points_extract)
        #print(len0, len1, len2)
        mu3, mv3 = undistorted_points[3]
        cosines = np.zeros(3, dtype=float)
        cosines[0] = (mu1 * mu2 + mv1 * mv2 + mk1 * mk2) / (len1 * len2)
        cosines[1] = (mu0 * mu2 + mv0 * mv2 + mk0 * mk2) / (len0 * len2)
        cosines[2] = (mu0 * mu1 + mv0 * mv1 + mk0 * mk1) / (len0 * len1)
        
        #print(cosines)
        
        lengths = self.length_solver(distances, cosines)
        #print("NUM OF SOL", len(lengths))
        if len(lengths) == 0:
            return []
        #print(lengths)
        
        reproj_errors = []
        Rs = []
        Ts = []
        for length in lengths:
            M_orig = np.tile(length,3).reshape(3,3).T * undistorted_points_extract

            R, T = self.align_solver(M_orig, X0, Y0, Z0, X1, Y1, Z1, X2, Y2, Z2)
            
            XYZ3p = np.dot(R, np.array([X3,Y3,Z3])) + T
            XYZ3p = XYZ3p / XYZ3p[2]
            mu3p, mv3p = XYZ3p[:2]
            
            reproj_error = (mu3p - mu3)**2 + (mv3p - mv3)**2
            reproj_errors.append(reproj_error)
            Rs.append(R)
            Ts.append(T)
        reproj_errors = np.array(reproj_errors)
        Rs = np.array(Rs)
        Ts = np.array(Ts)
        
        sorted_idx = np.argsort(reproj_errors)
        sorted_Rs = Rs[sorted_idx]
        sorted_Ts = Ts[sorted_idx]

        best_Rs = sorted_Rs[0]
        best_Ts = sorted_Ts[0]
        
        rotation_c = Rot.from_matrix(best_Rs)
        rvec = rotation_c.as_rotvec()

        return [rvec, best_Ts]
    

    def jacobi_4x4_solver(self,A):
        """
        CORRECTLY IMPLEMENTED
        input 
            A: a 1*16 vector with float type
        output
            D: eigenvalues
            U: transform matrix
        """

        U = np.eye(4, dtype=float).flatten()
        B = [A[0], A[5], A[10], A[15]]
        D = B.copy()
        Z = [0,0,0,0]

        for iteration in range(50):
            summ = abs(A[1]) + abs(A[2]) + abs(A[3]) + abs(A[6]) + abs(A[7]) + abs(A[11])
            if (summ == 0):
                return D, U

            tresh =  (0.2 * summ / 16) if (iteration < 3) else 0

            for i in range(3):
                loc = 5*i + 1
                for j in range(i+1, 4):
                    Aij = A[loc]
                    eps_machine = 100 * abs(Aij)
                    if (iteration > 3 and abs(D[i]) + eps_machine == abs(D[i]) and abs(D[j]) + eps_machine == abs(D[j])):
                        A[loc] = 0
                    elif (abs(Aij) > tresh):
                        hh = D[j] - D[i]
                        if (abs(hh) + eps_machine == abs(hh)):
                            t = Aij / hh
                        else:
                            theta = 0.5 * hh / Aij
                            t = 1 / (abs(theta) + math.sqrt(1 + theta * theta))
                            if (theta < 0):
                                t = -t

                        hh = t * Aij
                        Z[i] -= hh
                        Z[j] += hh
                        D[i] -= hh
                        D[j] += hh
                        A[loc] = 0

                        c = 1.0 / math.sqrt(1 + t * t)
                        s = t * c
                        tau = s / (1.0 + c)

                        for k in range(i):
                            g = A[k * 4 + i]
                            h = A[k * 4 + j]
                            A[k * 4 + i] = g - s * (h + g * tau)
                            A[k * 4 + j] = h + s * (g - h * tau)

                        for k in range(i+1, j):
                            g = A[i * 4 + k]
                            h = A[k * 4 + j]
                            A[i * 4 + k] = g - s * (h + g * tau)
                            A[k * 4 + j] = h + s * (g - h * tau)

                        for k in range(j+1, 4):
                            g = A[i * 4 + k]
                            h = A[j * 4 + k]
                            A[i * 4 + k] = g - s * (h + g * tau)
                            A[j * 4 + k] = h + s * (g - h * tau)

                        for k in range(4):
                            g = U[k * 4 + i]
                            h = U[k * 4 + j]
                            U[k * 4 + i] = g - s * (h + g * tau)
                            U[k * 4 + j] = h + s * (g - h * tau)

                    loc += 1
            for i in range(4):
                B[i] += Z[i]
            D = B.copy()
            Z = [0,0,0,0]
        return D, U

    def align_solver(self, M_end, X0, Y0, Z0, X1, Y1, Z1, X2, Y2, Z2):
        """
        CORRECTLY IMPLEMENTED
        input 
            M_end: a 3*3 vector with float type
            X0~Z2: float
        output
            R: rotation matrix
            T: transform matrix
        """
        R = np.zeros((3,3), dtype=float)
        T = np.zeros(3, dtype=float)

        # Centroids
        C_start = np.zeros(3, dtype=float)
        C_start[0] = (X0 + X1 + X2) / 3
        C_start[1] = (Y0 + Y1 + Y2) / 3
        C_start[2] = (Z0 + Z1 + Z2) / 3
        C_end = M_end.mean(axis=0)

        # Covariance matrix s
        s = np.zeros(9, dtype=float)
        for j in range(3):
            s[0 * 3 + j] = (X0 * M_end[0][j] + X1 * M_end[1][j] + X2 * M_end[2][j]) / 3 - C_end[j] * C_start[0]
            s[1 * 3 + j] = (Y0 * M_end[0][j] + Y1 * M_end[1][j] + Y2 * M_end[2][j]) / 3 - C_end[j] * C_start[1]
            s[2 * 3 + j] = (Z0 * M_end[0][j] + Z1 * M_end[1][j] + Z2 * M_end[2][j]) / 3 - C_end[j] * C_start[2]

        Qs = np.zeros(16, dtype=float)
        Qs[0 * 4 + 0] = s[0 * 3 + 0] + s[1 * 3 + 1] + s[2 * 3 + 2]
        Qs[1 * 4 + 1] = s[0 * 3 + 0] - s[1 * 3 + 1] - s[2 * 3 + 2]
        Qs[2 * 4 + 2] = s[1 * 3 + 1] - s[2 * 3 + 2] - s[0 * 3 + 0]
        Qs[3 * 4 + 3] = s[2 * 3 + 2] - s[0 * 3 + 0] - s[1 * 3 + 1]

        Qs[1 * 4 + 0] = Qs[0 * 4 + 1] = s[1 * 3 + 2] - s[2 * 3 + 1]
        Qs[2 * 4 + 0] = Qs[0 * 4 + 2] = s[2 * 3 + 0] - s[0 * 3 + 2]
        Qs[3 * 4 + 0] = Qs[0 * 4 + 3] = s[0 * 3 + 1] - s[1 * 3 + 0]
        Qs[2 * 4 + 1] = Qs[1 * 4 + 2] = s[1 * 3 + 0] + s[0 * 3 + 1]
        Qs[3 * 4 + 1] = Qs[1 * 4 + 3] = s[2 * 3 + 0] + s[0 * 3 + 2]
        Qs[3 * 4 + 2] = Qs[2 * 4 + 3] = s[2 * 3 + 1] + s[1 * 3 + 2]

        evs, U = self.jacobi_4x4_solver(Qs)

        # Looking for the largest eigen value
        ev_max = max(evs)
        i_ev = evs.index(ev_max)

        # Quaternion
        q = np.array(U).reshape(4,4)[:,i_ev]

        #From a quaternion to an orthogonal matrix
        # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        q_square = np.power(q,2)
        q0_1 = q[0] * q[1]
        q0_2 = q[0] * q[2]
        q0_3 = q[0] * q[3]
        q1_2 = q[1] * q[2]
        q1_3 = q[1] * q[3]
        q2_3 = q[2] * q[3]

        R[0][0] = q_square[0] + q_square[1] - q_square[2] - q_square[3]
        R[0][1] = 2. * (q1_2 - q0_3)
        R[0][2] = 2. * (q1_3 + q0_2)

        R[1][0] = 2. * (q1_2 + q0_3)
        R[1][1] = q_square[0] + q_square[2] - q_square[1] - q_square[3]
        R[1][2] = 2. * (q2_3 - q0_1)

        R[2][0] = 2. * (q1_3 - q0_2)
        R[2][1] = 2. * (q2_3 + q0_1)
        R[2][2] = q_square[0] + q_square[3] - q_square[1] - q_square[2]

        for i in range(3):
            T[i] = C_end[i] - (R[i][0] * C_start[0] + R[i][1] * C_start[1] + R[i][2] * C_start[2])

        return R, T

    def Quartic_solver(self, parameters):
        """
        CORRECTLY IMPLEMENTED
        input 
            parameters: 5 dim array indicate Quartic parameters
        output
            roots: the real root
        """
        out = []
        parameters = np.array(parameters, dtype=float)
        p = np.poly1d(parameters)
        r = np.roots(p)
        r = r[np.isreal(r)]
        for root in r:
            out.append(np.real(root))
        return out

    def length_solver(self, distances, cosines):
        """
        CORRECTLY IMPLEMENTED
        input 
            distances: a 3 dim vector with float type
            cosines: a 3 dim vector with float type
        output
            lengths: a 4* 3 array
        """
        p, q, r = cosines * 2
        d_p = np.power(distances, 2)
        a, b = (d_p/d_p[2])[:2]

        a2 = a**2
        b2 = b**2
        p2 = p**2
        q2 = q**2
        r2 = r**2
        pr = p*r
        pqr = pr*q

        #Check reality condition (the four points should not be coplanar)
        if (p2 + q2 + r2 - pqr - 1 == 0):
            return []

        ab = a * b
        a_2 = 2*a
        A = -2 * b + b2 + a2 + 1 + ab*(2 - r2) - a_2

        # Check reality condition
        if (A == 0):
            return []

        a_4 = 4*a
        B = q*(-2*(ab + a2 + 1 - b) + r2*ab + a_4) + pr*(b - b2 + ab)
        C = q2 + b2*(r2 + p2 - 2) - b*(p2 + pqr) - ab*(r2 + pqr) + (a2 - a_2)*(2 + q2) + 2
        D = pr*(ab-b2+b) + q*((p2-2)*b + 2 * (ab - a2) + a_4 - 2)
        E = 1 + 2*(b - a - ab) + b2 - b*p2 + a2
        

        temp = (p2*(a-1+b) + r2*(a-1-b) + pqr - a*pqr)
        b0 = b * temp * temp

        # Check reality condition
        if (b0 == 0):
            return []

        real_roots = self.Quartic_solver([A, B, C, D, E])

        if len(real_roots) == 0:
            return []

        r3 = r2*r
        pr2 = p*r2
        r3q = r3 * q
        inv_b0 = 1 / b0

        lengths = []
        # For each solution of x
        for x in real_roots:
            if (x <= 0):
                continue

            x2 = x**2
            b1 = ((1-a-b)*x2 + (q*a-q)*x + 1 - a + b) * (((r3*(a2 + ab*(2 - r2) - a_2 + b2 - 2*b + 1)) * x + (r3q*(2*(b-a2) + a_4 + ab*(r2 - 2) - 2) + pr2*(1 + a2 + 2*(ab-a-b) + r2*(b - b2) + b2))) * x2 + (r3*(q2*(1-2*a+a2) + r2*(b2-ab) - a_4 + 2*(a2 - b2) + 2) + r*p2*(b2 + 2*(ab - b - a) + 1 + a2) + pr2*q*(a_4 + 2*(b - ab - a2) - 2 - r2*b)) * x + 2*r3q*(a_2 - b - a2 + ab - 1) + pr2*(q2 - a_4 + 2*(a2 - b2) + r2*b + q2*(a2 - a_2) + 2) + p2*(p*(2*(ab - a - b) + a2 + b2 + 1) + 2*q*r*(b + a_2 - a2 - ab - 1)))

            # Check reality condition
            if (b1 <= 0):
                continue

            y = inv_b0 * b1
            v = x2 + y*y - x*y*r

            if (v <= 0):
                continue

            Z = distances[2] / math.sqrt(v)
            X = x * Z
            Y = y * Z

            lengths.append([X,Y,Z])

        return lengths