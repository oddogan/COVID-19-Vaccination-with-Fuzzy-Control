"""
    Onat Deniz DOGAN
    2304467
    This is the script for implementation of the Vaccination v1.
"""

import numpy as np
import skfuzzy as fuzz
import skfuzzy.membership as mf
import matplotlib.pyplot as plt

def controller1(vaccination):
    current_percentage = np.arange(0,1.01,.01)
    control = np.arange(-0.25, 0.25005, .005)

    curr_percentage_low = mf.trimf(current_percentage, [0, 0, .6])
    curr_percentage_med = mf.trimf(current_percentage, [.55, .6, .65])
    curr_percentage_high = mf.trimf(current_percentage, [.6, 1, 1])
    
    control_slow = mf.trimf(control, [-0.25, -0.25, 0])
    control_zero = mf.trimf(control, [-0.1, 0, 0.1])
    control_fast = mf.trimf(control, [0, 0.25, 0.25])

    """
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(6,10))
    
    ax0.plot(current_percentage, curr_percentage_low, 'r', linewidth = 2, label = 'Low')
    ax0.plot(current_percentage, curr_percentage_med, 'g', linewidth = 2, label = 'Medium')
    ax0.plot(current_percentage, curr_percentage_high, 'b', linewidth = 2, label = 'High')
    ax0.set_title('Current Percentage')
    ax0.legend()

    ax1.plot(control, control_slow, 'r', linewidth = 2, label = 'Slow Down')
    ax1.plot(control, control_zero, 'g', linewidth = 2, label = 'Stay Same')
    ax1.plot(control, control_fast, 'b', linewidth = 2, label = 'Speed Up')
    ax1.set_title('Vaccination Percentage Control')
    ax1.legend()

    plt.tight_layout()
    plt.show()
    """

    # Get the current data
    input_current_percentage, _ = vaccination.checkVaccinationStatus()
    
    # Get the membership values for the current data
    curr_percentage_fit_low = fuzz.interp_membership(current_percentage, curr_percentage_low, input_current_percentage)
    curr_percentage_fit_med = fuzz.interp_membership(current_percentage, curr_percentage_med, input_current_percentage)
    curr_percentage_fit_high = fuzz.interp_membership(current_percentage, curr_percentage_high, input_current_percentage)

    # # RULE 1: If CURRENTPERCENTAGE is LOW then CONTROL will SPEEDUP
    # # RULE 2: If CURRENTPERCENTAGE IS MED then CONTROL will ZERO
    # # RULE 3: If CURRENTPERCENTAGE IS HIGH then CONTROL will SLOWDOWN

    # Measure the rule outputs
    rule1 = np.fmin(curr_percentage_fit_low, control_fast)
    rule2 = np.fmin(curr_percentage_fit_med, control_zero)
    rule3 = np.fmin(curr_percentage_fit_high, control_slow)
    out_speedup = rule1
    out_staysame = rule2
    out_slowdown = rule3

    # Defuzzification
    out_control = np.fmax(np.fmax(out_speedup, out_staysame), out_slowdown)
    defuzzified = fuzz.defuzz(control, out_control, 'centroid')

    result = fuzz.interp_membership(control, out_control, defuzzified)

    # Vaccinate people
    vaccination.vaccinatePeople(defuzzified)