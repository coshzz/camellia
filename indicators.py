import numpy as np
import math
from numba import njit

@njit
def simple_moving_average_1m_delay(t_1m, p_1m, tf, period):
    N = len(p_1m)
    sma = np.zeros((N,), dtype='float32')
    
    ts0 = t_1m[0]
    tsn = t_1m[-1]
    N_buff = (tsn-ts0)//tf + 1
    tf_buff = np.zeros((N_buff, 4), dtype='float32')
    tf_0 = ts0 // tf
    
    #print(f"tf_0: {tf_0}, N_buff: {N_buff}")
    
    for i in range(N):
        # update tf_buff
        ts = t_1m[i]
        p = p_1m[i]
        t_n = ts // tf
        r_n = ts % tf
        tf_i = int(t_n) - tf_0
        
        #print(tf_i)
        
        if r_n==0:
            tf_buff[tf_i][:] = p
        else:
            tf_buff[tf_i][1] = max(p[1], tf_buff[tf_i][1])
            tf_buff[tf_i][2] = min(p[2], tf_buff[tf_i][2])
            tf_buff[tf_i][3] = p[3]
        
        # compute indicator
        if tf_i>=(period-1):
            samples = tf_buff[tf_i-period+1:tf_i+1, 3]
        else:
            samples = tf_buff[:tf_i+1, 3]
        
        # output
        if i<N-1:
            sma[i+1] = samples.mean()

    #print(f"last_i: {i}, tf_i: {tf_i}")
    
    return sma


@njit
def simple_moving_average_tf_delay(t_1m, p_1m, tf, period):
    N = len(p_1m)
    sma = np.zeros((N,), dtype='float32')

    ts0 = t_1m[0]
    tsn = t_1m[-1]
    N_buff = (tsn-ts0)//tf + 1
    tf_buff = np.zeros((N_buff, 4), dtype='float32')
    tf_0 = ts0 // tf
    
    for i in range(N):
        # update tf_buff
        ts = t_1m[i]
        p = p_1m[i]
        t_n = ts // tf
        r_n = ts % tf
        tf_i = int(t_n) - tf_0
        
        if r_n==0:
            tf_buff[tf_i][:] = p
        else:
            tf_buff[tf_i][1] = max(p[1], tf_buff[tf_i][1])
            tf_buff[tf_i][2] = min(p[2], tf_buff[tf_i][2])
            tf_buff[tf_i][3] = p[3]
        
        # compute indicator
        if tf_i>=period:
            samples = tf_buff[tf_i-period:tf_i, 3]
        elif tf_i==0:
            samples = tf_buff[:1, 3]
        else:
            samples = tf_buff[:tf_i, 3]
        
        # output
        if i<N-1:
            sma[i+1] = samples.mean()

    return sma


@njit
def exponential_moving_average_1m_delay(t_1m, p_1m, tf, period):
    N = len(p_1m)
    k = 2 / (period + 1)
    ema = np.zeros((N,), dtype='float32')

    ts0 = t_1m[0]
    tsn = t_1m[-1]
    N_buff = (tsn-ts0)//tf + 1
    tf_buff = np.zeros((N_buff, 4), dtype='float32')
    tf_0 = ts0 // tf
    last_tf_i = 0
    
    ema_0 = 0
    ema_1 = 0
    
    for i in range(N):
        # update tf_buff
        ts = t_1m[i]
        p = p_1m[i]
        t_n = ts // tf
        r_n = ts % tf
        tf_i = int(t_n) - tf_0
        
        if r_n==0:
            tf_buff[tf_i][:] = p
        else:
            tf_buff[tf_i][1] = max(p[1], tf_buff[tf_i][1])
            tf_buff[tf_i][2] = min(p[2], tf_buff[tf_i][2])
            tf_buff[tf_i][3] = p[3]
        
        # compute indicator
        if tf_i==0:
            ema_0 = tf_buff[0][3]
            ema_1 = ema_0
        elif tf_i==last_tf_i:
            ema_0 = tf_buff[tf_i][3]*k + ema_1*(1-k)
        else:
            ema_0 = tf_buff[tf_i][3]*k + ema_0*(1-k)
            ema_1 = ema_0
            last_tf_i = tf_i
            
        # output
        if i<N-1:
            ema[i+1] = ema_0
        
    return ema
    
    
@njit
def exponential_moving_average_tf_delay(t_1m, p_1m, tf, period):
    N = len(p_1m)
    k = 2 / (period + 1)
    ema = np.zeros((N,), dtype='float32')

    ts0 = t_1m[0]
    tsn = t_1m[-1]
    N_buff = (tsn-ts0)//tf + 1
    tf_buff = np.zeros((N_buff, 4), dtype='float32')
    tf_0 = ts0 // tf
    last_tf_i = 0
    
    ema_0 = 0
    ema_1 = 0
    
    for i in range(N):
        # update tf_buff
        ts = t_1m[i]
        p = p_1m[i]
        t_n = ts // tf
        r_n = ts % tf
        tf_i = int(t_n) - tf_0
        
        if r_n==0:
            tf_buff[tf_i][:] = p
        else:
            tf_buff[tf_i][1] = max(p[1], tf_buff[tf_i][1])
            tf_buff[tf_i][2] = min(p[2], tf_buff[tf_i][2])
            tf_buff[tf_i][3] = p[3]
    
        # compute indicator
        if tf_i==0:
            ema_0 = tf_buff[0][3]
            ema_1 = ema_0
        elif tf_i==last_tf_i:
            ema_0 = tf_buff[tf_i-1][3]*k + ema_1*(1-k)
        else:
            ema_0 = tf_buff[tf_i-1][3]*k + ema_0*(1-k)
            ema_1 = ema_0
            last_tf_i = tf_i
            
        # output
        if i<N-1:
            ema[i+1] = ema_0
        
    return ema



@njit
def bollinger_band_1m_delay(t_1m, p_1m, tf, period, multiplier):
    N = len(p_1m)
    bb = np.zeros((N, 5), dtype='float32')

    ts0 = t_1m[0]
    tsn = t_1m[-1]
    N_buff = (tsn-ts0)//tf + 1
    tf_buff = np.zeros((N_buff, 4), dtype='float32')
    tf_0 = ts0 // tf
    
    for i in range(N):
        # update tf_buff
        ts = t_1m[i]
        p = p_1m[i]
        t_n = ts // tf
        r_n = ts % tf
        tf_i = int(t_n) - tf_0
        
        if r_n==0:
            tf_buff[tf_i][:] = p
        else:
            tf_buff[tf_i][1] = max(p[1], tf_buff[tf_i][1])
            tf_buff[tf_i][2] = min(p[2], tf_buff[tf_i][2])
            tf_buff[tf_i][3] = p[3]
        
        # compute indicator
        if tf_i>=(period-1):
            samples = tf_buff[tf_i-period+1:tf_i+1, 3]
        else:
            samples = tf_buff[:tf_i+1, 3]
            
        mean = samples.mean()
        std = samples.std() + 1e-8
        upper = mean + multiplier*std
        lower = mean - multiplier*std
        norm = (samples[-1] - mean) / std
        
        # output
        if i<N-1:
            #bb[i+1] = (mean, std, upper, lower, norm)
            bb[i+1, 0] = mean
            bb[i+1, 1] = std
            bb[i+1, 2] = upper
            bb[i+1, 3] = lower
            bb[i+1, 4] = norm
        
    return bb
    
@njit
def bollinger_band_tf_delay(t_1m, p_1m, tf, period, multiplier):
    N = len(p_1m)
    bb = np.zeros((N, 5), dtype='float32')

    ts0 = t_1m[0]
    tsn = t_1m[-1]
    N_buff = (tsn-ts0)//tf + 1
    tf_buff = np.zeros((N_buff, 4), dtype='float32')
    tf_0 = ts0 // tf

    for i in range(N):
        # update tf_buff
        ts = t_1m[i]
        p = p_1m[i]
        t_n = ts // tf
        r_n = ts % tf
        tf_i = int(t_n) - tf_0
        
        if r_n==0:
            tf_buff[tf_i][:] = p
        else:
            tf_buff[tf_i][1] = max(p[1], tf_buff[tf_i][1])
            tf_buff[tf_i][2] = min(p[2], tf_buff[tf_i][2])
            tf_buff[tf_i][3] = p[3]
        
        # compute indicator
        if tf_i>=period:
            samples = tf_buff[tf_i-period:tf_i, 3]
        elif tf_i==0:
            samples = tf_buff[:1, 3]
        else:
            samples = tf_buff[:tf_i, 3]
            
        mean = samples.mean()
        std = samples.std() + 1e-8
        upper = mean + multiplier*std
        lower = mean - multiplier*std
        norm = (samples[-1] - mean) / std
        
        # output
        if i<N-1:
            #bb[i+1] = (mean, std, upper, lower, norm)
            bb[i+1, 0] = mean
            bb[i+1, 1] = std
            bb[i+1, 2] = upper
            bb[i+1, 3] = lower
            bb[i+1, 4] = norm
        
    return bb


@njit
def keltner_channel_1m_delay(t_1m, p_1m, tf, ema_period, atr_period, multiplier):
    N = len(p_1m)
    k = 2 / (ema_period + 1)
    kc = np.zeros((N, 5), dtype='float32')

    ts0 = t_1m[0]
    tsn = t_1m[-1]
    N_buff = (tsn-ts0)//tf + 1
    tf_buff = np.zeros((N_buff, 4), dtype='float32')
    tf_0 = ts0 // tf
    last_tf_i = 0

    ema_0 = 0
    ema_1 = 0
    atr_0 = 0
    atr_1 = 0
    p_0 = 0
    p_1 = 0
    
    for i in range(N):
        # update tf_buff
        ts = t_1m[i]
        p = p_1m[i]
        t_n = ts // tf
        r_n = ts % tf
        tf_i = int(t_n) - tf_0
        
        if r_n==0:
            tf_buff[tf_i][:] = p
        else:
            tf_buff[tf_i][1] = max(p[1], tf_buff[tf_i][1])
            tf_buff[tf_i][2] = min(p[2], tf_buff[tf_i][2])
            tf_buff[tf_i][3] = p[3]
            
        # compute indicator
        if tf_i==0:
            ema_0 = p[3]
            ema_1 = ema_0
            p_0 = tf_buff[0]
            p_1 = p_0
            tr = max(p_0[1], p_1[3]) - min(p_0[2], p_1[3])
            atr_0 = tr / atr_period
            atr_1 = atr_0
            
        elif tf_i==last_tf_i:
            ema_0 = tf_buff[tf_i][3]*k + ema_1*(1-k)
            p_0 = tf_buff[tf_i]
            p_1 = tf_buff[tf_i-1]
            tr = max(p_0[1], p_1[3]) - min(p_0[2], p_1[3])
            atr_0 = (atr_1*(atr_period-1) + tr) / atr_period
        else:
            ema_0 = tf_buff[tf_i][3]*k + ema_0*(1-k)
            ema_1 = ema_0
            p_0 = tf_buff[tf_i]
            p_1 = tf_buff[tf_i-1]
            tr = max(p_0[1], p_1[3]) - min(p_0[2], p_1[3])
            atr_0 = (atr_0*(atr_period-1) + tr) / atr_period
            atr_1 = atr_0
            last_tf_i = tf_i
            
        upper = ema_0 + multiplier*atr_0
        lower = ema_0 - multiplier*atr_0
        norm = (p[3] - ema_0) / atr_0
        
        # output
        if i<N-1:
            #kc[i+1] = (ema_0, atr_0, upper, lower, norm)
            kc[i+1, 0] = ema_0
            kc[i+1, 1] = atr_0
            kc[i+1, 2] = upper
            kc[i+1, 3] = lower
            kc[i+1, 4] = norm
        
    return kc
    
@njit
def keltner_channel_tf_delay(t_1m, p_1m, tf, ema_period, atr_period, multiplier):
    N = len(p_1m)
    k = 2 / (ema_period + 1)
    kc = np.zeros((N, 5), dtype='float32')

    ts0 = t_1m[0]
    tsn = t_1m[-1]
    N_buff = (tsn-ts0)//tf + 1
    tf_buff = np.zeros((N_buff, 4), dtype='float32')
    tf_0 = ts0 // tf
    last_tf_i = 0

    ema_0 = 0
    ema_1 = 0
    atr_0 = 0
    atr_1 = 0
    p_0 = 0
    p_1 = 0
    
    for i in range(N):
        # update tf_buff
        ts = t_1m[i]
        p = p_1m[i]
        t_n = ts // tf
        r_n = ts % tf
        tf_i = int(t_n) - tf_0
        
        if r_n==0:
            tf_buff[tf_i][:] = p
        else:
            tf_buff[tf_i][1] = max(p[1], tf_buff[tf_i][1])
            tf_buff[tf_i][2] = min(p[2], tf_buff[tf_i][2])
            tf_buff[tf_i][3] = p[3]
            
        # compute indicator
        if tf_i<=1:
            ema_0 = p[3]
            ema_1 = ema_0
            p_0 = tf_buff[0]
            p_1 = p_0
            tr = max(p_0[1], p_1[3]) - min(p_0[2], p_1[3])
            atr_0 = tr / atr_period
            atr_1 = atr_0
            
        elif tf_i==last_tf_i:
            ema_0 = tf_buff[tf_i-1][3]*k + ema_1*(1-k)
            p_0 = tf_buff[tf_i-1]
            p_1 = tf_buff[tf_i-2]
            tr = max(p_0[1], p_1[3]) - min(p_0[2], p_1[3])
            atr_0 = (atr_1*(atr_period-1) + tr) / atr_period
        else:
            ema_0 = tf_buff[tf_i-1][3]*k + ema_0*(1-k)
            ema_1 = ema_0
            p_0 = tf_buff[tf_i-1]
            p_1 = tf_buff[tf_i-2]
            tr = max(p_0[1], p_1[3]) - min(p_0[2], p_1[3])
            atr_0 = (atr_0*(atr_period-1) + tr) / atr_period
            atr_1 = atr_0
            last_tf_i = tf_i
            
        upper = ema_0 + multiplier*atr_0
        lower = ema_0 - multiplier*atr_0
        norm = (p[3] - ema_0) / atr_0
        
        # output
        if i<N-1:
            #kc[i+1] = (ema_0, atr_0, upper, lower, norm)
            kc[i+1, 0] = ema_0
            kc[i+1, 1] = atr_0
            kc[i+1, 2] = upper
            kc[i+1, 3] = lower
            kc[i+1, 4] = norm
        
    return kc



@njit
def distance_1m_delay(t_1m, p_1m, tf, period):
    N = len(p_1m)
    dist = np.zeros((N, 2), dtype='float32')

    ts0 = t_1m[0]
    tsn = t_1m[-1]
    N_buff = (tsn-ts0)//tf + 1
    tf_buff = np.empty((N_buff, 4), dtype='float32')
    tf_buff[:] = np.nan  
    d_buff = np.empty((N_buff, 1), dtype='float32')
    d_buff[:] = np.nan
    tf_0 = ts0 // tf

    for i in range(N):
        # update tf_buff
        ts = t_1m[i]
        p = p_1m[i]
        t_n = ts // tf
        r_n = ts % tf
        tf_i = int(t_n) - tf_0

        if np.all(p>0):
            logp = np.log10(p)
        else:
            logp = np.array(4*[np.nan])

        if r_n==0:
            tf_buff[tf_i][:] = logp
        else:
            tf_buff[tf_i][1] = max(logp[1], tf_buff[tf_i][1])
            tf_buff[tf_i][2] = min(logp[2], tf_buff[tf_i][2])
            tf_buff[tf_i][3] = logp[3]

        # compute indicator
        if tf_i>=(period-1):
            d_buff[tf_i][0] = math.fabs(tf_buff[tf_i][3] - tf_buff[tf_i-1][3])
            samples = d_buff[tf_i-period+1:tf_i+1]
        elif tf_i>0:
            d_buff[tf_i][0] = math.fabs(tf_buff[tf_i][3] - tf_buff[tf_i-1][3])
            samples = d_buff[:tf_i+1]
        else:
            samples = d_buff[:1]

        # output
        if i<N-1:
            dist[i+1, 0] = samples.sum()
            
        if tf_i>=period:
            dist[i+1, 1] = tf_buff[tf_i][3] - tf_buff[tf_i-period][0]
        else:
            dist[i+1, 1] = tf_buff[tf_i][3] - tf_buff[0][0]

    return dist


@njit
def distance_tf_delay(t_1m, p_1m, tf, period):
    N = len(p_1m)
    dist = np.zeros((N, 2), dtype='float32')

    ts0 = t_1m[0]
    tsn = t_1m[-1]
    N_buff = (tsn-ts0)//tf + 1
    tf_buff = np.empty((N_buff, 4), dtype='float32')
    tf_buff[:] = np.nan  
    d_buff = np.empty((N_buff, 1), dtype='float32')
    d_buff[:] = np.nan
    tf_0 = ts0 // tf

    for i in range(N):
        # update tf_buff
        ts = t_1m[i]
        p = p_1m[i]
        t_n = ts // tf
        r_n = ts % tf
        tf_i = int(t_n) - tf_0

        if np.all(p>0):
            logp = np.log10(p)
        else:
            logp = np.array(4*[np.nan])

        if r_n==0:
            tf_buff[tf_i][:] = logp
        else:
            tf_buff[tf_i][1] = max(logp[1], tf_buff[tf_i][1])
            tf_buff[tf_i][2] = min(logp[2], tf_buff[tf_i][2])
            tf_buff[tf_i][3] = logp[3]

        # compute indicator
        if tf_i>=period:
            d_buff[tf_i][0] = math.fabs(tf_buff[tf_i][3] - tf_buff[tf_i-1][3])
            samples = d_buff[tf_i-period:tf_i]
        elif tf_i>0:
            d_buff[tf_i][0] = math.fabs(tf_buff[tf_i][3] - tf_buff[tf_i-1][3])
            samples = d_buff[:tf_i]
        else:
            samples = d_buff[:1]

        # output
        if i<N-1:
            dist[i+1, 0] = samples.sum()
            
        if tf_i>=period:
            dist[i+1, 1] = tf_buff[tf_i][3] - tf_buff[tf_i-period][0]
        else:
            dist[i+1, 1] = tf_buff[tf_i][3] - tf_buff[0][0]

    return dist
