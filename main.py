"""
    Onat Deniz DOGAN
    2304467
    This is the main script to run the vaccination program.
"""

import vaccination
from vaccination_controller_part1 import *
from vaccination_controller_part2 import *

# Create the vaccination object
vaccination = vaccination.Vaccination()

# The iteration count
days = 20

prev_percentage = 0
cost = 0
steady = False
for tick in range(days*10+1):
    # Apply controller and get the current status
    # Use controller1(obj) or controller2(obj) for different versions of vaccination
    controller2(vaccination)
    current_percentage, current_rate = vaccination.checkVaccinationStatus()

    if not steady:
        cost += current_rate
        # Check the steady state
        if (abs(current_percentage - 0.6) < 0.003) and (abs(current_percentage - prev_percentage) < 0.002):
            ss_point = tick
            steady = True
        
    prev_percentage = current_percentage

# Get the results
vaccination.viewVaccination(point_ss=ss_point, 
                            vaccination_cost=cost,
                            save_dir='', 
                            filename='vaccinationpart2')