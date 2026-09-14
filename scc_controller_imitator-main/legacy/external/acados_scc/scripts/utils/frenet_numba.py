from numba.experimental import jitclass
import numpy as np
from numba import float64, int64

spec_spline = [
    ('x', float64[:]),
    ('y', float64[:]),
    ('nx', int64),
    ('a', float64[:]),
    ('b', float64[:]),
    ('c', float64[:]),
    ('d', float64[:]),
]

@jitclass(spec_spline)
class Spline:
    def __init__(self, x, y):
        self.x = np.ascontiguousarray(x)
        self.y = np.ascontiguousarray(y)
        self.nx = len(x)
        h = self.x[1:] - self.x[:-1]

        self.a = self.y.copy()
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)

        self.b = (self.a[1:] - self.a[:-1])/h - h*(2*self.c[:-1] + self.c[1:])/3
        self.d = (self.c[1:] - self.c[:-1])/(3*h)

    def calc(self, t_arr):
        idx = np.searchsorted(self.x, t_arr) - 1
        idx = np.minimum(np.maximum(idx, 0), self.nx - 2)
        dx = t_arr - self.x[idx]
        result = self.a[idx] + self.b[idx]*dx + self.c[idx]*dx**2 + self.d[idx]*dx**3
        return result

    def calcd(self, t_arr):
        idx = np.searchsorted(self.x, t_arr) - 1
        idx = np.minimum(np.maximum(idx, 0), self.nx - 2)
        dx = t_arr - self.x[idx]
        result = self.b[idx] + 2*self.c[idx]*dx + 3*self.d[idx]*dx**2
        return result

    def calcdd(self, t_arr):
        idx = np.searchsorted(self.x, t_arr) - 1
        idx = np.minimum(np.maximum(idx, 0), self.nx - 2)
        dx = t_arr - self.x[idx]
        result = 2*self.c[idx] + 6*self.d[idx]*dx
        return result

    def __calc_A(self, h):
        nx = self.nx
        A = np.zeros((nx, nx))
        A[0, 0] = 1.0
        A[nx - 1, nx - 1] = 1.0
        for i in range(1, nx - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2.0*(h[i - 1] + h[i])
            A[i, i + 1] = h[i]
        return A

    def __calc_B(self, h):
        nx = self.nx
        B = np.zeros(nx)
        for i in range(1, nx - 1):
            B[i] = 3.0*((self.a[i + 1] - self.a[i])/h[i] - (self.a[i] - self.a[i - 1])/h[i - 1])
        return B

spec_roadmap = [
    ('cx', float64[:]),
    ('cy', float64[:]),
    ('s', float64[:]),
    ('total_length', float64),
    ('ds', float64[:]),
    ('sx', Spline.class_type.instance_type),
    ('sy', Spline.class_type.instance_type),
]

@jitclass(spec_roadmap)
class RoadMap:
    def __init__(self, x, y):
        self.cx = np.ascontiguousarray(x)
        self.cy = np.ascontiguousarray(y)
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, self.cx)
        self.sy = Spline(self.s, self.cy)
        self.total_length = self.s[-1]

    def __calc_s(self, x, y):
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        self.ds = np.hypot(dx, dy)
        self.ds = np.where(self.ds < 1e-5, 1e-5, self.ds)
        s = np.zeros(len(self.ds) + 1)
        s[1:] = np.cumsum(self.ds)
        return s

    def ClosestWaypoint(self, x_arr, y_arr):
        num_waypoints = self.cx.size
        num_points = x_arr.size
        idx_closest = np.empty(num_points, dtype=np.int64)

        for j in range(num_points):
            min_dist = 1e12  # 매우 큰 값
            idx = -1
            x_target = x_arr[j]
            y_target = y_arr[j]
            for i in range(num_waypoints):
                dx = self.cx[i] - x_target
                dy = self.cy[i] - y_target
                dist = np.hypot(dx, dy)
                if dist < min_dist:
                    min_dist = dist
                    idx = i
            idx_closest[j] = idx

        return idx_closest

    def FrenetSCycle(self, s):
        return np.mod(s, self.total_length)

    def GetCurvature(self, s_arr):
        dx = self.sx.calcd(s_arr)
        ddx = self.sx.calcdd(s_arr)
        dy = self.sy.calcd(s_arr)
        ddy = self.sy.calcdd(s_arr)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**1.5)
        return k

    def GetYaw(self, s_arr):
        dx = self.sx.calcd(s_arr)
        dy = self.sy.calcd(s_arr)
        yaw = np.arctan2(dy, dx)
        return yaw
    
    
    def ToFrenet(self, x, y):
        x_arr = x
        y_arr = y
        num_points = x_arr.size
        frenet_s = np.empty(num_points, dtype=np.float64)
        frenet_d = np.empty(num_points, dtype=np.float64)

        wp1 = self.ClosestWaypoint(x_arr, y_arr)
        frenet_s = self.s[wp1]

        max_iter = 1000  # 최대 반복 횟수
        tolerance = 1e-2  # 수렴 기준

        for i in range(num_points):
            s = frenet_s[i]
            x_target = x_arr[i]
            y_target = y_arr[i]
            for _ in range(max_iter):
                x0 = self.sx.calc(s)
                y0 = self.sy.calc(s)
                dx = x_target - x0
                dy = y_target - y0
                slope = self.GetYaw(s)
                cos_slope = np.cos(slope)
                sin_slope = np.sin(slope)
                norm_distance = dx * cos_slope + dy * sin_slope
                s += norm_distance
                s = self.FrenetSCycle(s)
                if np.abs(norm_distance) < tolerance:
                    break
            frenet_s[i] = s

            # 프레네 d 값 계산
            x0 = self.sx.calc(s)
            y0 = self.sy.calc(s)
            dx = x_target - x0
            dy = y_target - y0
            slope = self.GetYaw(s)
            cos_slope = np.cos(slope)
            sin_slope = np.sin(slope)
            frenet_d[i] = -dx * sin_slope + dy * cos_slope

        return frenet_s, frenet_d

    def ToCartesian(self, s_arr, d_arr):
        x = self.sx.calc(s_arr)
        y = self.sy.calc(s_arr)
        slope = self.GetYaw(s_arr)
        cos_slope = np.cos(slope)
        sin_slope = np.sin(slope)
        x += d_arr * cos_slope
        y += d_arr * sin_slope
        return x, y
