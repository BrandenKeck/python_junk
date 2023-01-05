import numpy as np
import matplotlib.pyplot as plt
from pid_systems import constant_setpoint_system

def temperature_of(pv, sp, out): return pv + out/5 - np.sqrt(np.abs(out))/10 #+ np.random.normal(0, 0.01, 1)[0]
temperature_system = constant_setpoint_system(0, 60, 21, 23, 0, 150, temperature_of)
temperature_system.SP[100:] = 25
temperature_system.SP[400:] = 21
temperature_system.P = 1
temperature_system.I = 0.001
#temperature_system.D = 0.05

X1, X2, X3 = temperature_system.run_pid()
plt.plot(temperature_system.t, X1, 'g-')
plt.plot(temperature_system.t, X2, 'r-')
print(X3)
plt.show()

 