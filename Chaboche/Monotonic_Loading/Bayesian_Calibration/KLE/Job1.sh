#!/bin/bash

# Use 'nohup' to keep running process after SSH session is closed
# run MATLAB script from cli and exit
# save command line outputs to full_Bayesian.log

nohup matlab -r "run('./chaboche_calibration_elastic.m'); exit;" >> Job1.log &
