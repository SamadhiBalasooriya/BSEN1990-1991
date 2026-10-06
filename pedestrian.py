import math
import numpy as np
from sympy import *


class Pedestrian:
    """
    Base Class for creating a Pedestrian
    """

    populationProperties = {}
    meanLognormalModel = 4.28  # mM
    sdLognormalModel = 0.21  # sM

    detK = 14110
    detVelocity = 1.25

    synchedPace = 0
    synchedPhase = 0

    def __init__(self, mass, damp, stiff, pace, phase, location, velocity, iSync):
        """
        this function introduce the properties when creating one pedestrian

        Parameters
        ----------
        mass: human mass
        damp : damping effect of pedesteian
        stiff : stiffness of humans
        pace : pacing frequency
        phase : phase angle
        location : location of mass
        velocity : velocity of travelling mass
        iSync : synchronization

        Returns
        -------
        None.

        """
        self.mass = mass
        self.damp = damp
        self.stiff = stiff
        self.pace = pace
        self.phase = phase
        self.location = location
        self.velocity = velocity
        self.iSync = iSync


def calcPedForce(self, t):
        # Question: What are all the commented out parts in matlab ped_force
        g = 9.81
        W = self.mass * g
        #x = self.location + self.velocity * t  # Position of Pedestrian at each time t
        numped = np.size(self.mass)
        # Young
        eta = np.array([0.41 * (self.pace - 0.95),
                        0.069 + 0.0056 * self.pace,
                        0.033 + 0.0064 * self.pace,
                        0.013 + 0.0065 * self.pace])
        phi = np.zeros((eta.shape[0], numped))

        # Now assemble final force, and include weight
        N = eta.shape[0]  # No. of additional terms in harmonic series
        F0 = W * np.vstack((np.ones((1, numped)), eta))  # Force amplitudes
        beta = 2 * math.pi * self.pace[np.newaxis, :] * np.arange(N + 1)[:, np.newaxis]  # Frequencies
        phi = np.vstack((np.zeros((1, numped)), phi)) + self.phase  # Phases

        omega = beta * t + phi
        Ft = np.sum(F0 * np.cos(omega), axis=0)
        
        return Ft  # Returns an array of forces for each pedestrian



def calc_nprime(density, xi, S):
    """
    Compute n' (equivalent number of pedestrians per unit area) [1/m^2]
    using the TC1–TC5 rules:

    - if d < 1.0:  n' = 10.8 * sqrt(xi * n) / S
    - if d >=1.0:  n' = 1.85 * sqrt(n) / S

    Parameters
    ----------
    density : float
        Pedestrian density d [ped/m^2]
    xi : float
        Modal damping ratio [-] (e.g., 0.005 = 0.5%)
    n : float
        Number of pedestrians on the bridge [-]
    S : float
        Bridge deck area [m^2]

    Returns
    -------
    nprime : float
        Equivalent pedestrians per unit area [1/m^2]
    """
    density = float(density)

    if np.isscalar(xi):
        xi = float(xi)
    else:
        xi = float(xi[0])
    S = float(S)

    if S <= 0:
        raise ValueError("S (deck area) must be > 0.")
    if xi < 0:
        raise ValueError("xi must be >= 0.")

    if density < 1.0:
        return 10.8 * np.sqrt(xi * density*S) / S
    else:
        return 1.85 * np.sqrt(density*S) / S
    
def psi_w_h1(x):
    x = np.asarray(x, dtype=float)
    psi = np.zeros_like(x)

    # 1.25–1.70 : 0 -> 1
    m = (x >= 1.25) & (x < 1.70)
    psi[m] = (x[m] - 1.25) / (1.70 - 1.25)

    # 1.70–2.10 : 1
    m = (x >= 1.70) & (x <= 2.10)
    psi[m] = 1.0

    # 2.10–2.30 : 1 -> 0
    m = (x > 2.10) & (x < 2.30)
    psi[m] = 1.0 - (x[m] - 2.10) / (2.30 - 2.10)

    # 2.3-2.5Hz
    #m = (x >=2.30) & (x <= 2.50)
    #psi[m] = 0.25

    # 2.50–3.40 : 0 -> 0.25
    m = (x > 2.50) & (x < 3.40)
    psi[m] = 0.25 * (x[m] - 2.50) / (3.40 - 2.50)

    # 3.40–4.20 : 0.25
    m = (x >= 3.40) & (x <= 4.20)
    psi[m] = 0.25

    # 4.20–4.60 : 0.25 -> 0
    m = (x > 4.20) & (x <= 4.60)
    psi[m] = 0.25 * (1.0 - (x[m] - 4.20) / (4.60 - 4.20))

    # Minimum rule:
    # If first vertical frequency is between 2.25 and 3.0 Hz,
    # psi_w should not be less than 0.25.
    m = (x >= 2.25) & (x <= 3.0)
    psi[m] = np.maximum(psi[m], 0.25)

    return psi

def psi_w_h1_modified(x):
    """
    Modified psi function used only when force reduction factor rf != 1.

    Shape:
    - 1.25–1.70 : 0 -> 1
    - 1.70–2.30 : 1
    - 2.30–2.50 : 1 -> 0
    - otherwise : 0
    """

    x = np.asarray(x, dtype=float)
    psi = np.zeros_like(x)

    # 1.25–1.70 : 0 -> 1
    m = (x >= 1.25) & (x < 1.70)
    psi[m] = (x[m] - 1.25) / (1.70 - 1.25)

    # 1.70–2.30 : 1
    m = (x >= 1.70) & (x <= 2.30)
    psi[m] = 1.0

    # 2.30–2.50 : 1 -> 0
    m = (x > 2.30) & (x <= 2.50)
    psi[m] = 1.0 - 0.75*(x[m] - 2.30) / (2.50 - 2.30)

    return psi

def psi_from_x(x):
    """
    Given x = first-harmonic frequency [Hz],
    second-harmonic frequency is 2x.
    Returns (psi1, psi2).
    """
    x = np.asarray(x, dtype=float)
    psi1 = psi_w_h1(x)
    return psi1

def force_2harm(t, x, density, xi, S, phase=(0.0, 0.0), rf=1.0):
    """
    Two-harmonic equivalent pedestrian stream force:

        p(t) = Pw * n' * [ ψ1(x)     * cos(2π * x     * t + φ1)
                         + ψ2(2x)    * cos(2π * (2x)  * t + φ2) ]

    where:
      - Pw = 280 (constant)
      - n' is computed from density, xi, n, S
      - ψ1 is from the 1st-harmonic graph using x
      - ψ2 is from the 2nd-harmonic graph using 2x

    Parameters
    ----------
    t : float
        Time [s]
    x : float or np.ndarray
        First-harmonic frequency [Hz] (usually the walking step frequency fs).
        Second harmonic is automatically 2*x.
    density : float
        Pedestrian density [ped/m^2]
    xi : float
        Modal damping ratio [-]
    n : float
        Number of pedestrians [-]
    S : float
        Deck area [m^2]
    phase : (φ1, φ2)
        Phase angles [rad] for harmonic 1 and 2

    Returns
    -------
    p : float or np.ndarray
        Force at time t. Scalar if x scalar; array if x array.
    """
    pw = 280.0  # Constant force amplitude [N]
    t = float(t)
    rf = float(rf)

    if np.isscalar(x):
        x = float(x)
    else:
        x = float(x[0])

    # n' [1/m^2]
    nprime = calc_nprime(density, xi, S)


    psi1 = psi_w_h1(x)     # modified psi for reduced-force model

    phi1, phi2 = phase

    p = rf * pw * nprime * (psi1 * np.cos(2.0 * np.pi * (x) * t + phi1))

    return float(p) if p.ndim == 0 else p


def calcPedForce2(self, t):
        # all mass pace are random seiris
        g = 9.81
        W = self.mass * g
        numped = np.size(self.mass)
        # Young
        eta = np.array([0.41 * (self.pace - 0.95),
                        0.069 + 0.0056 * self.pace,
                        0.033 + 0.0064 * self.pace,
                        0.013 + 0.0065 * self.pace])
        phi = np.zeros((eta.shape[0], numped))

        # Now assemble final force, and include weight
        N = eta.shape[0]  # No. of additional terms in harmonic series
        F0 = W * np.vstack((np.ones((1, numped)), eta))  # Force amplitudes
        beta = 2 * math.pi * self.pace[np.newaxis, :] * np.arange(N + 1)[:, np.newaxis]  # Frequencies
        phi = np.vstack((np.zeros((1, numped)), phi)) + self.phase  # Phases

        omega = beta * t + phi
        Ft = np.sum(F0 * np.cos(omega), axis=0)
        
        return Ft  # Returns an array of forces for each pedestrian


def calcDynamicPedForce(self, t):
        # Question: What are all the commented out parts in matlab ped_force
        g = 9.81

        W = self.mass * g
        #x = self.location + self.velocity * t  # Position of Pedestrian at each time t

        # Young
        eta = np.array([0.41 * (self.pace - 0.95),
                        0.069 + 0.0056 * self.pace,
                        0.033 + 0.0064 * self.pace,
                        0.013 + 0.0065 * self.pace])
        phi = np.zeros(4)

        # Now assemble final force, and include weight
        N = len(eta)  # No. of additional terms in harmonic series
        F0 = W * eta # Force amplitudes (constant amplitude for 1)
        beta = 2 * math.pi * self.pace * np.array([i+1 for i in range(N)])  # Frequencies
        phi  += self.phase  # Phases - enforce first phase as zero phase
       
        beta = beta[:, np.newaxis]  # Reshape to (N+1, 1)
        phi = phi[:, np.newaxis]    # Reshape to (N+1, 1)
        F0=F0[:, np.newaxis]
        omega = beta * t + phi
        Ft = np.sum( F0* np.cos(omega), axis=0)
        #Ft = sum(F0 * np.cos(omega))

        return  Ft
    # endregion    

class Crowd:

    populationProperties = {}
    """
    an empty dictionary is initialized to store population properties which will be introduced in the following 
    lines of code.  
    """

    def __init__(self, numPedestrians, length, width, sync):
        """
        initialization takes arguments numPedestrians, length, width and sync. Then set the corresponding attributes
        """
        # self.density = density
        self.numPedestrians = numPedestrians
        self.length = length
        self.width = width
        self.sync = sync

        self.area = self.length * self.width
        # self.numPedestrians = int(self.density * self.area)
        self.lamda = self.numPedestrians / self.length

        self.locations = []
        self.iSync = []
        self.pedestrians = []

        # Crowd synchronization
        self.determineCrowdSynchronisation()

    def determineCrowdSynchronisation(self):
        sync = self.sync/100
        self.iSync = np.random.choice([0, 1], size=self.numPedestrians, p=[1 - sync, sync])
        pace = np.random.normal(loc=self.populationProperties['meanPace'], scale=self.populationProperties['sdPace'])
        phase = (2 * math.pi) * (np.random.rand())
        Pedestrian.setPaceAndPhase(pace, phase)

    def addRandomPedestrian(self, location, synched):
        self.pedestrians.append(Pedestrian.randomPedestrian(location, synched))

    def addDeterministicPedestrian(self, location, synched):
        self.pedestrians.append(Pedestrian.deterministicPedestrian(location, synched))

    def addExactPedestrian(self, location, synched):
        """
        Temporary, for testing
        """
        self.pedestrians.append(Pedestrian.exactPedestrian(location, synched))

class SinglePedestrian(Pedestrian):    #update this to run on body damping and frequancy
    """
    Sub Class of Pedestrian
    """
    def __init__(self):
        """
        super().__init__(parameters)
            inherits the parameters from the Base class Pedestrian.

        reintroduce the parameters from Pedestrian Class which overrides the ones set under the parent class

        introduce two other Parameters
        numPedestrians
        """
        
        k = 14.11e3  

        pMass = self.populationProperties['meanMass']
        pDamp = self.populationProperties['meanDamping'] * 2 * math.sqrt(k * pMass)
        pStiff = k
        pPace = 2
        pPhase = 0
        pLocation = 0
        pVelocity = 1.25
        iSync = 0
        super().__init__(pMass, pDamp, pStiff, pPace, pPhase, pLocation, pVelocity, iSync)
        self.numPedestrians = 1
        self.pedestrians = [self] #???

    @classmethod
    def fromDict(cls, crowdOptions):
        return cls()


class DeterministicCrowd(Crowd):

    arrivalGap = 1      

    def __init__(self, numPedestrians, length, width, sync):
        super().__init__(numPedestrians, length, width, sync)
        self.generateLocations()
        self.populateCrowd()

    def generateLocations(self):
        self.locations = -self.arrivalGap*np.array(range(self.numPedestrians))

    def populateCrowd(self):
        for i in range(self.numPedestrians):
            self.addDeterministicPedestrian(self.locations[i], self.iSync[i])

    @classmethod
    def setArrivalGap(cls, arrivalGap):
        cls.arrivalGap = arrivalGap


class RandomCrowd(Crowd):
    def __init__(self, numPedestrians, length, width, sync):
        super().__init__(numPedestrians, length, width, sync)
        self.generateLocations()
        self.populateCrowd()

    def generateLocations(self):
        gaps = np.random.exponential(1 / self.lamda, size=self.numPedestrians)
        self.locations = np.cumsum(gaps, axis=None, dtype=None, out=None)

    def populateCrowd(self):
        for i in range(self.numPedestrians):
            self.addRandomPedestrian(self.locations[i], self.iSync[i])
