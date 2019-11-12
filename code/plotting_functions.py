#
# Spec2Vec
#
# Copyright 2019 Netherlands eScience Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
from matplotlib import pyplot as plt


## ----------------------------------------------------------------------------------------
## ---------------------------- Plotting functions ----------------------------------------
## ----------------------------------------------------------------------------------------


def plot_precentile(Arr_sim, Arr_ref, num_bins = 1000, show_top_percentile = 1.0):
    """ Plot top percentile (as specified by show_top_percentile) of best restults
    in Arr_sim and compare against reference values in Arr_ref.
    
    Args:
    -------
    Arr_sim: numpy array
        Array of similarity values to evaluate.
    Arr_ref: numpy array
        Array of reference values to evaluate the quality of Arr_sim.
    num_bins: int
        Number of bins to divide data (default = 1000)   
    show_top_percentile
        Choose which part to plot. Will plot the top 'show_top_percentile' part of
        all similarity values given in Arr_sim. Default = 1.0
    """
    start = int(Arr_sim.shape[0]*show_top_percentile/100)
    idx = np.argpartition(Arr_sim, -start)
    starting_point = Arr_sim[idx[-start]]
    if starting_point == 0:
        print("not enough datapoints != 0 above given top-precentile")
        
    # Remove all data below show_top_percentile
    low_As = np.where(Arr_sim < starting_point)[0]

    length_selected = Arr_sim.shape[0] - low_As.shape[0] #start+1
    
    Data = np.zeros((2, length_selected))
    Data[0,:] = np.delete(Arr_sim, low_As)
    Data[1,:] = np.delete(Arr_ref, low_As)
    Data = Data[:,np.lexsort((Data[1,:], Data[0,:]))]

    ref_score_cum = []
    
    for i in range(num_bins):
        low = int(i * length_selected/num_bins)
        #high = int((i+1) * length_selected/num_bins)
        ref_score_cum.append(np.mean(Data[1,low:]))
    ref_score_cum = np.array(ref_score_cum)
                         
    fig, ax = plt.subplots(figsize=(6,6))
    plt.plot((show_top_percentile/num_bins*(1+np.arange(num_bins)))[::-1], ref_score_cum, color='black')
    plt.xlabel("Top percentile of spectral similarity score g(s,s')")
    plt.ylabel("Mean molecular similarity (f(t,t') within that percentile)")
    
    return ref_score_cum