import numpy as np

# Create a class-based physical system model
class constant_setpoint_system:
    def __init__(self, start, end, pv0, sp, sh, sl, outfunc):
        self.t = np.linspace(start, end, (end-start)*10)
        self.PV = [pv0]
        self.SP = np.ones(len(self.t))*sp
        self.OUT = []
        self.P = 1
        self.I = 0
        self.D = 0
        self.scale_hi = sh
        self.scale_lo = sl
        self.outfunc = outfunc

    def run_pid(self):
        for i in np.arange(0, len(self.t)):
            if i == 0:
                out = ( self.P*self.error(self.PV[i], self.SP[i]) )
            elif self.I != 0:
                out = ( self.P*self.error(self.PV[i], self.SP[i]) +
                      (1/self.I)*self.integ_error(self.PV[i-1], self.PV[i], self.SP[i]) +
                      self.D*self.deriv_error(self.PV[i-1], self.PV[i], self.SP[i]) )
            else:
                out = ( self.P*self.error(self.PV[i], self.SP[i]) +
                      self.D*self.deriv_error(self.PV[i-1], self.PV[i], self.SP[i]) )

            #out = out / (self.scale_hi - self.scale_lo)
            #if out < -0.5: out = -0.5
            #if out > 0.5: out = 0.5
            #out = out + 0.5

            self.OUT.append(out)
            self.PV.append(self.outfunc(self.PV[i], self.SP[i], self.OUT[i]))

        self.PV.pop()
        return self.PV, self.SP, self.OUT

    def error(self, pv, sp):
        return sp - pv

    def integ_error(self, pv0, pv1, sp):
        e0 = self.error(pv0, sp)
        e1 = self.error(pv1, sp)
        dt = self.t[1] - self.t[0]

        if sp < pv0 and sp < pv1:
            if e0 < e1:
                return 0.5 * (e1 - e0) * dt + e0 * dt
            else:
                return 0.5 * (e0 - e1) * dt + e1 * dt
        if sp > pv0 and sp > pv1:
            if e0 < e1:
                return 0.5 * (e1 - e0) * dt + e0 * dt
            else:
                return 0.5 * (e0 - e1) * dt + e1 * dt
        else:
            intercept = -e0/((e1-e0)/dt)
            return 0.5*e0*intercept + 0.5*e1*(dt-intercept)

    def deriv_error(self, pv0, pv1, sp):
        e0 = self.error(pv0, sp)
        e1 = self.error(pv1, sp)
        dt = self.t[1] - self.t[0]

        return (e1 - e0)/dt