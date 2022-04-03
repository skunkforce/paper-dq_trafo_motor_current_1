import numpy as np
import numpy.typing as npt
def hello():
    print("hello")
    
"""DQ Transformation: Überführe dreiphasige Drehstrommaschine in zweiachsiges Koordinatensystem d und q
    Notation angelehnt an deutschsprachiges wikipedia
"""

def transform_dq_amplitude_invariant(three_phase_current: npt.ArrayLike, phi=0):
    """[I_d, I_q] = trafo_matrix * [I_U, I_V, I_W]
        Shape of three_phase_current: [1x3]
    """
    trafo_matrix = np.matrix([
        [ np.cos(phi),  np.cos(phi - 2/3*np.pi),  np.cos(phi - 4/3*np.pi)],
        [-np.sin(phi), -np.sin(phi - 2/3*np.pi), -np.sin(phi - 4/3*np.pi)]
    ])
    return 2/3 * np.dot(trafo_matrix, three_phase_current)
    

