import numpy as np

class Functions():

    FUNCTION = "GAUSS_SQUARED"
    STDDAYS = 11
    FREQSPAN = 3 * STDDAYS

    @staticmethod
    def getFrequencyLifeSpan(t0, freqspan):
        return [int(t0 - freqspan), int(t0 + freqspan + 1)]
    
    @staticmethod
    def getLogMultiplicity(col_date, time, tbase=np.inf):
        '''
        Returns the log of node multiplicity at a given time point time
        
        Parameters:
            time: time at which the function is evaluated
            tbase: optional, used if tree is cut after prediction time
        '''
        [t1, t2] = Functions.getFrequencyLifeSpan(col_date,33)
        t2 = min(t2, tbase)

        mul = -np.inf

        if col_date <= tbase and t1 <= time < t2:
            mul = Functions.weightFunction(col_date, time) + np.log(float(1.0))

        return mul

    @staticmethod
    def weightFunction(t, t0):
        if Functions.FUNCTION == "GAUSS_SQUARED":
            return Functions.weightFunction_gauss_squared(t, t0)
        else:
            return 1

    @staticmethod
    def weightFunction_gauss_squared(t, t0):
        w = -np.inf
        if abs(t0 - t) <= Functions.FREQSPAN:
            x = (t - t0)
            std = Functions.STDDAYS
            w = float(-(x * x * x * x)) / float(2 * std * std * std * std)
        return w

    @staticmethod
    def logSum(logv):
        '''
        Computes the logarithm of the sum of exponents of numbers on the list,
        log(sum(exp(logv)))

        :param logv: list of floats

        :return: float
        '''
        if len(logv) == 0 or max(logv) == -np.inf:
            return -np.inf
        ma = np.max(logv)
        return np.log(np.sum([np.exp(x - ma) for x in  logv])) + ma