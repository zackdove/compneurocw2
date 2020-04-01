import random as rnd
import numpy as np
from bisect import bisect

# constants
Hz=1.0
sec=1.0
ms=0.001

def get_spike_train(rate,big_t,tau_ref):
    if 1<=rate*tau_ref:
        print("firing rate not possible given refractory period f/p")
        return []
    exp_rate=rate/(1-tau_ref*rate)
    spike_train=[]
    t=rnd.expovariate(exp_rate)
    while t< big_t:
        spike_train.append(t)
        t+=tau_ref+rnd.expovariate(exp_rate)
    return spike_train


# calculate fano factor of spike count
def get_fano_factor(spike_train, big_t, window_size):
    spike_count_list = []
    window_starts = []
    for i in np.arange(0, big_t, window_size*ms):
        window_starts.append(i)
        spike_count_list.append(0)
    # print("window starts length=", len(window_starts), " spike list len=",len(spike_count_list))
    for spike in spike_train:
        window_for_spike = bisect(window_starts, spike) -1
        # print(window_for_spike)
        spike_count_list[window_for_spike] += 1
    mean = np.mean(spike_count_list)
    # print("mean = ", mean)
    var = np.var(spike_count_list)
    fano = var/mean
    print("fano window size",window_size,"=",fano)
    return fano

# calculate coefficient of variance of inter spike interval
def get_cov(spike_train):
    interval_times = []
    interval_times.append(spike_train[0])
    for i in range(1, len(spike_train)):
        interval = spike_train[i] - spike_train[i-1]
        interval_times.append(interval)
    # print(interval_times)
    std = np.std(interval_times)
    mean = np.mean(interval_times)
    cov = std/mean
    # print("cov mean = ", mean)
    print ("cov=",cov)
    return cov

def load_data(filename,T):
    data_array = [T(line.strip()) for line in open(filename, 'r')]
    return data_array

# # variables:
# rate=10.0 *Hz
# # refractory period
# tau_ref=0*ms
# # total time
# big_t=10*sec
#
# spike_train=get_spike_train(rate,big_t,tau_ref)

# print(len(spike_train)/big_t)

# print(spike_train)

# 1000s, 35hz, no refractory
big_t = 1000*sec
rate = 100*Hz
tau_ref = 0*ms
spike_train = get_spike_train(rate, big_t, tau_ref)
print("1000s, 35hz, no refractory")
# fano factor 10ms windows
get_fano_factor(spike_train, big_t, 100)
# fano factor 50ms windows
get_fano_factor(spike_train, big_t, 50)
# fano factor 100ms windows
get_fano_factor(spike_train, big_t, 100)
# cov of interspike interval 1000s, 35Hz, no refractory
get_cov(spike_train)

# 1000s, 35hz, 5ms refractory
big_t = 1000*sec
rate = 35*Hz
tau_ref = 5*ms
spike_train = get_spike_train(rate, big_t, tau_ref)
print("1000s, 35hz, 5ms refractory")
# fano factor 10ms windows
get_fano_factor(spike_train, big_t, 10)
# fano factor 50ms windows
get_fano_factor(spike_train, big_t, 50)
# fano factor 100ms windows
get_fano_factor(spike_train, big_t, 100)
# cov of interspike interval 1000s, 35Hz, no refractory
get_cov(spike_train)


def vec_to_train(vec, f):
    f = 500*Hz
    p = 1/f
    # print(len(vec), t*f)
    train = []
    for index, s in enumerate(vec):
        if (s):
            train.append(index*p)
    return train

rho_vec=load_data("rho.dat",int)
rho_train = vec_to_train(rho_vec, 500*Hz)
print("rho:")
# fano factor 10ms windows
get_fano_factor(rho_train, 20*60*sec, 100)
# fano factor 50ms windows
get_fano_factor(rho_train, 20*60*sec, 50)
# fano factor 100ms windows
get_fano_factor(rho_train, 20*60*sec, 100)
# cov of interspike interval 1000s, 35Hz, no refractory
get_cov(spike_train)

# stimulus=load_data("stim.dat",float)
#
# print(len(stimulus))
# print(stimulus[0:5])
