The code in this zip was used as a basis for the paper:
"Hybrid feature tweaking - combining random forest similarity tweaking with CLPFD"
In order to re-run the experiment in the paper or just use the feature tweaking
algorithms the following steps needs to be done:
1. Download and install sicstus prolog from here: https://sicstus.sics.se,
note that you can download an evaluation copy of the software, just try it out.
2. In file "actionable_features_clpfd.py" the Path in line 19, 
needs to be change to where you unpacked this software.
3. In file "communication_w_solver.py" the Path in line 109,
you need to specify the Path to where SICTUS is installed, i.e. where "sicstus.exe" is located.
4. In file "communication_w_solver.py" the Path in line 113, 
needs to be change to where you unpacked this software + /clpfd_tweaking/src/clpfd_server.pl.

Now you are ready to run the program. 
Open a command prompt and type 'python actionable_features_clpfd.py'and hit enter.

Note that as it is now, it will start with a small dataset,
'zoo', but build 4 models (sizes of 10, 50, 100 and 250) and tweak test examples using 
four different time-out values, 15 s, 45 s, 2 min and 5 min. So while the data set is quite small,
it will take som time to run, hence be patient.

The code should be fairly easy to follow. The code is organized as follows:
+ Pyhton
	++ "actionable_features_clpfd.py" - starts experiments and collects performance measures
	++ "communication_w_solver.py" - starts prolog process and handle communication with prolog
	++ "cost.py" - distance measure 
	++ "featureTweakPy.py" - Actionable Feature Tweaking (AFT)
	++ "rf_distance_measures" - Random Forest Similarity Tweaking (RFST) and vector direction calculation
+ Prolog
	++ "clpfd_server.pl" - server for the clpfd solver
	++ "clpfd_tweaking.pl" - CLPFD-FT and Hybrid Feature Tweaking (HFT)	 
 
Best regards,

Anonymus