"""
    Onat Deniz DOGAN
    2304467
    This is the script for implementation of the Vaccination v2.
"""

import numpy as np
import skfuzzy as fuzz
import skfuzzy.membership as mf
import matplotlib.pyplot as plt

def controller2(vaccination):
    current_percentage = np.arange(0,1.01,.01)
    current_effective_rate = np.arange(-1,1.02,.02)
    control = np.arange(-0.25, 0.25005, .005)

    curr_percentage_low = mf.trimf(current_percentage, [0, 0, .6]) # The current percentage is LOW
    curr_percentage_med = mf.trimf(current_percentage, [.55, .6, .65]) # The current percentage is MED
    curr_percentage_high = mf.trimf(current_percentage, [.6, 1, 1]) # The current percentage is HIGH

    curr_eff_rate_low = mf.trimf(current_effective_rate, [-1, -1, 0]) # The current effective rate is NEG
    curr_eff_rate_med = mf.trimf(current_effective_rate, [-.5, 0, .5]) # The current effective rate is ZERO
    curr_eff_rate_high = mf.trimf(current_effective_rate, [0, 1, 1]) # The current effective rate is POS

    control_slower = mf.trimf(control, [-0.25, -0.25, -0.125]) # The control output should be slowER
    control_slow = mf.trimf(control, [-0.25, -0.125, 0]) # The control output should be slow
    control_zero = mf.trimf(control, [-0.125, 0, 0.125]) # The control output should be same
    control_fast = mf.trimf(control, [0, 0.125, 0.25]) # The control output should be fast
    control_faster = mf.trimf(control, [0.125, 0.25, 0.25]) # The control output should be fastER

    # The code block to print the plots for membership functions
    """
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(6,10))

    ax0.plot(current_percentage, curr_percentage_low, 'r', linewidth = 2, label = 'Low')
    ax0.plot(current_percentage, curr_percentage_med, 'g', linewidth = 2, label = 'Medium')
    ax0.plot(current_percentage, curr_percentage_high, 'b', linewidth = 2, label = 'High')
    ax0.set_title('Current Percentage')
    ax0.legend()

    ax1.plot(current_effective_rate, curr_eff_rate_low, 'r', linewidth = 2, label = 'Low')
    ax1.plot(current_effective_rate, curr_eff_rate_med, 'g', linewidth = 2, label = 'Medium')
    ax1.plot(current_effective_rate, curr_eff_rate_high, 'b', linewidth = 2, label = 'High')
    ax1.set_title('Current Effective Rate')
    ax1.legend()

    ax2.plot(control, control_slower, 'r', linewidth = 2, label = 'Slower')
    ax2.plot(control, control_slow, 'g', linewidth = 2, label = 'Slow')
    ax2.plot(control, control_zero, 'b', linewidth = 2, label = 'Stay Same')
    ax2.plot(control, control_fast, 'y', linewidth = 2, label = 'Fast')
    ax2.plot(control, control_faster, 'm', linewidth = 2, label = 'Faster')
    ax2.set_title('Vaccination Percentage Control')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    """

    # Get the current data
    input_current_percentage, input_current_effective_rate = vaccination.checkVaccinationStatus()
    
    # Get the membership values for the current data
    curr_percentage_fit_low = fuzz.interp_membership(current_percentage, curr_percentage_low, input_current_percentage)
    curr_percentage_fit_med = fuzz.interp_membership(current_percentage, curr_percentage_med, input_current_percentage)
    curr_percentage_fit_high = fuzz.interp_membership(current_percentage, curr_percentage_high, input_current_percentage)

    curr_eff_rate_fit_low = fuzz.interp_membership(current_effective_rate, curr_eff_rate_low, input_current_effective_rate)
    curr_eff_rate_fit_med = fuzz.interp_membership(current_effective_rate, curr_eff_rate_med, input_current_effective_rate)
    curr_eff_rate_fit_high = fuzz.interp_membership(current_effective_rate, curr_eff_rate_high, input_current_effective_rate)

    # # RULE 1: If CURRENTPERCENTAGE is LOW and EFFRATE is LOW then CONTROL will be FASTER
    # # RULE 2: If CURRENTPERCENTAGE is LOW and EFFRATE is MED then CONTROL will be FASTER
    # # RULE 3: If CURRENTPERCENTAGE IS LOW and EFFRATE is HIGH then CONTROL will be FAST
    # # RULE 4: If CURRENTPERCENTAGE is MED and EFFRATE is LOW then CONTROL will be FAST
    # # RULE 5: If CURRENTPERCENTAGE IS MED and EFFRATE is MED then CONTROL will be ZERO
    # # RULE 6: If CURRENTPERCENTAGE IS MED and EFFRATE is HIGH then CONTROL will be SLOW
    # # RULE 7: If CURRENTPERCENTAGE is HIGH and EFFRATE is LOW then CONTROL will be SLOW
    # # RULE 8: If CURRENTPERCENTAGE IS HIGH and EFFRATE is MED then CONTROL will be SLOWER
    # # RULE 9: If CURRENTPERCENTAGE IS HIGH and EFFRATE is HIGH then CONTROL will be SLOWER

    # Measure the rule outputs
    rule1 = np.fmin(np.fmin(curr_percentage_fit_low, curr_eff_rate_fit_low), control_faster)
    rule2 = np.fmin(np.fmin(curr_percentage_fit_low, curr_eff_rate_fit_med), control_faster)
    rule3 = np.fmin(np.fmin(curr_percentage_fit_low, curr_eff_rate_fit_high), control_fast)
    rule4 = np.fmin(np.fmin(curr_percentage_fit_med, curr_eff_rate_fit_low), control_fast)
    rule5 = np.fmin(np.fmin(curr_percentage_fit_med, curr_eff_rate_fit_med), control_zero)
    rule6 = np.fmin(np.fmin(curr_percentage_fit_med, curr_eff_rate_fit_high), control_slow)
    rule7 = np.fmin(np.fmin(curr_percentage_fit_high, curr_eff_rate_fit_low), control_slow)
    rule8 = np.fmin(np.fmin(curr_percentage_fit_high, curr_eff_rate_fit_med), control_slower)
    rule9 = np.fmin(np.fmin(curr_percentage_fit_high, curr_eff_rate_fit_high), control_slower)

    out_faster = np.fmax(rule2, rule1)
    out_fast = np.fmax(rule3, rule4)
    out_zero = rule5
    out_slow = np.fmax(rule6, rule7)
    out_slower = np.fmax(rule9, rule8)

    # Defuzzification
    out_control = np.fmax(np.fmax(np.fmax(out_faster, out_fast), np.fmax(out_slower, out_slow)), out_zero)
    defuzzified = fuzz.defuzz(control, out_control, 'centroid')

    result = fuzz.interp_membership(control, out_control, defuzzified)

    # Vaccinate people
    vaccination.vaccinatePeople(defuzzified)