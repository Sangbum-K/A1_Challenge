import numpy as np

class RoadMap:
    def __init__(self, x, y):
        self.cx = np.array(x)
        self.cy = np.array(y)
        self.s  = np.array(self.__calc_s(x, y))
        self.sx = Spline(self.s, self.cx)
        self.sy = Spline(self.s, self.cy)
        
        self.total_length = self.s[-1]
        
    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        self.ds[self.ds < 1e-6] = 1e-6
        s = np.concatenate(([0], np.cumsum(self.ds)))
        return s
    
    def ClosestWaypoint(self, x, y):
        """
        x, y: numpy arrays
        Returns indices of closest waypoints
        """
        dx = self.cx[:, np.newaxis] - x[np.newaxis, :]
        dy = self.cy[:, np.newaxis] - y[np.newaxis, :]
        ds = np.hypot(dx, dy)
        idx_closest = np.argmin(ds, axis=0)
        return idx_closest
    
    def FrenetSCycle(self, s):
        return np.mod(s, self.total_length)
    
    def GetCurvature(self, s):
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(1.5))
        return k
    
    def GetYaw(self, s):
        """
        Calculate yaw angle
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = np.arctan2(dy, dx)
        return yaw
    
    def ToFrenet(self, x, y):
        """
        Convert Cartesian coordinates to Frenet coordinates.
        x, y: numpy arrays
        Returns frenet_s, frenet_d: numpy arrays
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Initial guess for frenet_s using the closest waypoint
        wp1 = self.ClosestWaypoint(x, y)
        frenet_s = self.s[wp1]
        
        # Convert to arrays for vectorized computation
        frenet_s = frenet_s.astype(float)
        frenet_d = np.zeros_like(frenet_s)
        
        # Iterative process
        max_iter = 1000  # Limit iterations to prevent long computation time
        for j in range(max_iter):
            x0 = self.sx.calc(frenet_s)
            y0 = self.sy.calc(frenet_s)
            dx = x - x0
            dy = y - y0
            slope = self.GetYaw(frenet_s)
            cos_slope = np.cos(slope)
            sin_slope = np.sin(slope)
            norm_distance = dx * cos_slope + dy * sin_slope
            frenet_s += norm_distance
            frenet_s = self.FrenetSCycle(frenet_s)
            # Check convergence
            if np.all(np.abs(norm_distance) < 1e-2):
                break

        
        # Compute frenet_d after convergence
        x0 = self.sx.calc(frenet_s)
        y0 = self.sy.calc(frenet_s)
        dx = x - x0
        dy = y - y0
        slope = self.GetYaw(frenet_s)
        cos_slope = np.cos(slope)
        sin_slope = np.sin(slope)
        frenet_d = -dx * sin_slope + dy * cos_slope
        
        return frenet_s, frenet_d
    
    def ToCartesian(self, s, d):
        """
        Convert Frenet coordinates to Cartesian coordinates.
        """
        s = np.asarray(s)
        d = np.asarray(d)
        x = self.sx.calc(s)
        y = self.sy.calc(s)
        slope = self.GetYaw(s)
        cos_slope = np.cos(slope)
        sin_slope = np.sin(slope)
        x += d * cos_slope
        y += d * sin_slope
        return x, y
    
    def ToFrenetVelocity(self, vx, vy, frenet_s):
        slope = self.GetYaw(frenet_s)
        cos_slope = np.cos(slope)
        sin_slope = np.sin(slope)
        frenet_vs = vx * cos_slope + vy * sin_slope
        frenet_vd = -vx * sin_slope + vy * cos_slope
        return frenet_vs, frenet_vd
    
    def ToCartesianVelocity(self, vs, vd, frenet_s):
        slope = self.GetYaw(frenet_s)
        cos_slope = np.cos(slope)
        sin_slope = np.sin(slope)
        vx = vs * cos_slope - vd * sin_slope
        vy = vs * sin_slope + vd * cos_slope
        return vx, vy


class Spline:
    """
    Cubic Spline class
    """

    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.nx = len(x)
        h = np.diff(x)

        self.a = self.y.copy()
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)

        self.b = (np.diff(self.a) / h) - h * (2 * self.c[:-1] + self.c[1:]) / 3
        self.d = (self.c[1:] - self.c[:-1]) / (3 * h)

    def calc(self, t):
        """
        Calculate position
        """
        t = np.asarray(t)
        idx = np.searchsorted(self.x, t) - 1
        idx = np.clip(idx, 0, self.nx - 2)
        dx = t - self.x[idx]
        result = self.a[idx] + self.b[idx] * dx + \
            self.c[idx] * dx ** 2 + self.d[idx] * dx ** 3
        return result

    def calcd(self, t):
        """
        Calculate first derivative
        """
        t = np.asarray(t)
        idx = np.searchsorted(self.x, t) - 1
        idx = np.clip(idx, 0, self.nx - 2)
        dx = t - self.x[idx]
        result = self.b[idx] + 2 * self.c[idx] * dx + 3 * self.d[idx] * dx ** 2
        return result

    def calcdd(self, t):
        """
        Calculate second derivative
        """
        t = np.asarray(t)
        idx = np.searchsorted(self.x, t) - 1
        idx = np.clip(idx, 0, self.nx - 2)
        dx = t - self.x[idx]
        result = 2 * self.c[idx] + 6 * self.d[idx] * dx
        return result

    def __calc_A(self, h):
        """
        Calculate matrix A for spline coefficients
        """
        nx = self.nx
        A = np.zeros((nx, nx))
        A[0, 0] = 1
        A[nx - 1, nx - 1] = 1
        for i in range(1, nx - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
        return A

    def __calc_B(self, h):
        """
        Calculate vector B for spline coefficients
        """
        nx = self.nx
        B = np.zeros(nx)
        for i in range(1, nx - 1):
            B[i] = 3 * ((self.a[i + 1] - self.a[i]) / h[i] - (self.a[i] - self.a[i - 1]) / h[i - 1])
        return B
