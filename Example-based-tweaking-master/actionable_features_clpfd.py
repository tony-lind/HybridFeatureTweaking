# -*- coding: utf-8 -*-
"""
Code for testing different algorithmss feature tweaking 

@author: anonymous
"""
import numpy as np
import pandas as pd
import random
import time
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from communication_w_solver import start_solver, stop_solver, setup_problem, tweak_example, restart_needed
from rf_distance_measures import random_forest_tweaking, feature_direction
from featureTweakPy import feature_tweaking
from cost import cost_func #, to_closest_int neighbour_tweaking,
from prompt_toolkit.search import stop_search

Path = "YOUR PATH"

datasets = [# Categorical data sets               
            # ("balance_scale", Path + "data/balance_scale.csv","yes")
            # ("car", Path + "data/car.csv","yes")
            # ("HIV_746", Path + "data/HIV_746.csv","yes")
            # ("HIV_1625", Path + "data/HIV_1625.csv","yes")
            # ("impens", Path + "data/impens.csv","yes")
            # ("KRKP7", Path + "data/KRKP7.csv","yes")          
            # ("promoters", Path + "data/promoters.csv","yes")
            # ("schilling", Path + "data/schilling.csv","yes") 
            # ("shuttle", Path + "data/shuttle.csv","yes")
            # ("tic_tac_toe", Path +"/data/tictactoe.csv","yes")                               
            ("zoo", Path + "data/zoo.csv","yes")            
            ]

# List to save results in
results_clpfd = []
results_clpfd_combo = []
results_ft = []
# Number of trees in each forest
forest_size = [10, 50, 100, 250] 
# start the clpfd solver 
proc = start_solver(True) #True if we want output   
# Handle each data set in sequence
for (d_name, d_s, trans) in datasets:
# read in data 
    df = pd.read_csv(d_s)
# Turn categorical values into integers (incl. classes)
    if(trans == "yes"):
        df['class-_'] = pd.factorize(df['class-_'])[0]
        for i in range(0, len(df.columns) - 1):  # class always last column
            t_name = df.iloc[:, i].name
            if('categoric-' in t_name):
                df[t_name] = pd.factorize(df[t_name])[0]      
                   
#covert int64 to int32                       
    values = df.values.astype(int)
# Split into X and y - note that class label MUST be in the LAST column
    X = values[:, 0:len(df.columns) - 1] 
    y = values[:, len(df.columns) - 1]

# Split into train and test set
    test_size = 0.2
    seed = 42
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)

# Build RF - Vary the forest size: [10, 50, 100, 250]  
    our_models = []
    clf_10 = RandomForestClassifier(n_estimators=forest_size[0], criterion="entropy", n_jobs=-1)
    clf_10 = clf_10.fit(X_train, y_train)
    print("Build model of size 10")
    our_models.append(("10", clf_10))   
    
    clf_50 = RandomForestClassifier(n_estimators=forest_size[1], criterion="entropy", n_jobs=-1)
    clf_50 = clf_50.fit(X_train, y_train)
    print("Build model of size 50")
    our_models.append(("50", clf_50))
    
    clf_100 = RandomForestClassifier(n_estimators=forest_size[2], criterion="entropy", n_jobs=-1)
    clf_100 = clf_100.fit(X_train, y_train)
    print("Build model of size 100")
    our_models.append(("100", clf_100))
    
    clf_250 = RandomForestClassifier(n_estimators=forest_size[3], criterion="entropy", n_jobs=-1)
    clf_250 = clf_250.fit(X_train, y_train)
    print("Build model of size 250")
    our_models.append(("250", clf_250))  
    
# Find min and max-values for all features
    minMax = []
    cost_list = []
    for i in range(0, len(X_train[0])):
        cost_list.append(1)
        tMin = X_train[:,i].min()
        tMax = X_train[:,i].max()
        minMax.append((int(tMin), int(tMax)))    
        
# Hyperparameters for the clpfd solver                         
    y_labels = np.unique(y)
    y_tweak = []
    percentage = 51
    epsilon = 0.5
    # 15 s, 45 s, 2 min, 5 min
    timeL = [15000, 45000, 120000, 300000] 
    print("Starting evaluation of clpfd tweaking")
    for (m_size, our_model) in our_models:
# Setup model in clpf_solver   
        setup_res = setup_problem(proc, our_model, y_labels) 
#sleep a while for clpfd to be ready
        time.sleep(4)
# Reset performance metric values  
        clpfd_missed = [0,0,0,0]
        clpfd_optimality = [0,0,0,0]
        clpfd_success = [0,0,0,0]
        tot_clpfd_cost = [0,0,0,0]  
        clpfd_combo_missed = [0,0,0,0]
        clpfd_combo_optimality = [0,0,0,0]
        clpfd_combo_solo_optimality = [0,0,0,0]
        clpfd_combo_success = [0,0,0,0]
        tot_clpfd_combo_cost = [0,0,0,0]
        missed_ft = 0
        tot_ft_cost = 0  
        print("Model size: ", m_size)
        print("evaluation of tweaking starts")    
        for i in range(len(X_test)):
            x = X_test[i]       
            print("Ex no: ", i)
            y = int(y_test[i])            
            wish_class = random.randint(0, len(y_labels) - 1)
            if(y == wish_class):
                if(wish_class == 0):
                    wish_class = 1
                else:
                    wish_class = wish_class - 1
            #print("x: ", x)
            #print("y: ", y)
            #print("wish_class: ", wish_class) 
            y_tweak.append(wish_class)                
# FT performance
            x_new_ft = feature_tweaking(our_model, x, y_labels, wish_class, epsilon, cost_func)
            #TODO: fix this it could be come wrong if it same value but classified as wished class, i.e. miss classification 
            #print("FT: ", x_new_ft)
            pred_val = our_model.predict([x])
            if(np.equal(x_new_ft, x).all() and pred_val[0] != wish_class):  
                #print("FT missed")
                missed_ft += 1
            else:
                #print("FT: ", x_new_ft)
                #print("FT: ", int_x_ft)               
                ft_dist = cost_func(x_new_ft, x)        
                #print("ft cost: " + str(ft_dist))           
                tot_ft_cost = tot_ft_cost + ft_dist                              
            
# CLPFD combo tweaking performance  
            #check if we should restart server
            restart = restart_needed(proc)
            if restart:
                proc.kill()
                time.sleep(4)
                proc = start_solver(True)
                time.sleep(4)
                setup_res = setup_problem(proc, our_model, y_labels) 
                time.sleep(4)
                
            sim_cnt, sim_X, sim_y = random_forest_tweaking(our_model, [x], wish_class, X_train, y_train, True)
            if len(sim_X) > 0:
                feature_direction_list = feature_direction(sim_X[0], x)
            else:
                feature_direction_list = []  
            for i, t in enumerate(timeL):
                if (feature_direction_list == []):       
                    x_start = x.astype(int)           
                    combo_solo = False              
                    clpfd_ex, clpfd_outcome = tweak_example(proc, x_start.tolist(), timeL[i], minMax, wish_class, [], percentage, cost_list)
                    time.sleep(2)
                    if clpfd_outcome == 'error':
                        print("Here we need to re-start the prolog server due to error (1)!")
                        proc.kill()
                        time.sleep(4)
                        proc = start_solver(True)
                        time.sleep(4)
                        setup_res = setup_problem(proc, our_model, y_labels) 
                        time.sleep(4)
                        clpfd_outcome = "time_out"
                        clpfd_combo_outcome = "time_out"                   
                    else:    
                        #print("CLPFD and COMBO SAME: ", clpfd_ex)
                        clpfd_combo_outcome = clpfd_outcome
                        clpfd_ex_arr = clpfd_combo_ex_arr = np.asanyarray(clpfd_ex)                    
                else:
                    x_start = sim_X[0].astype(int)
                    x_goal = x.astype(int) 
                    combo_solo = True
                    clpfd_combo_ex, clpfd_combo_outcome = tweak_example(proc, [x_start.tolist(), x_goal.tolist()], timeL[i], minMax, wish_class, feature_direction_list, percentage, cost_list)
                    time.sleep(2)
                    if clpfd_combo_outcome == 'error':
                        print("Here we need to re-start the prolog server due to error (2)!")
                        proc.kill()
                        time.sleep(4)
                        proc = start_solver(True)
                        time.sleep(4)
                        setup_res = setup_problem(proc, our_model, y_labels) 
                        time.sleep(4)                     
                        clpfd_combo_outcome = "time_out"                       
                    else:
                        #print("CLPFD COMBO: ", clpfd_combo_ex)
                        clpfd_combo_ex_arr = np.asanyarray(clpfd_combo_ex)      
                        clpfd_ex, clpfd_outcome = tweak_example(proc, x_goal.tolist(), timeL[i], minMax, wish_class, [], percentage, cost_list) 
                        time.sleep(2)
                        if clpfd_outcome == 'error':
                            print("Here we need to re-start the prolog server due to error (3)!")
                            proc.kill()
                            time.sleep(4)
                            proc = start_solver(True)
                            time.sleep(4)
                            setup_res = setup_problem(proc, our_model, y_labels) 
                            time.sleep(4)                          
                            clpfd_outcome = "time_out"                                                          
                        else:
                            #print("CLPFD: ", clpfd_ex)
                            clpfd_ex_arr = np.asanyarray(clpfd_ex)             
#Record CLPFD result
                if clpfd_outcome == "optimality":
                    clpfd_optimality[i] += 1
                    clpfd_dist = cost_func(clpfd_ex_arr, x)
                    #print("clpfd optimal cost: " + str(val_1))
                    tot_clpfd_cost[i] = tot_clpfd_cost[i] + clpfd_dist
                elif clpfd_outcome == "success":
                    clpfd_success[i] += 1
                    clpfd_dist = cost_func(clpfd_ex_arr, x)
                    #print("clpfd success cost: " + str(val_1))
                    tot_clpfd_cost[i] = tot_clpfd_cost[i] + clpfd_dist
                else:
                    clpfd_missed[i] += 1    
# Record CLPFD combo result
                if clpfd_combo_outcome == "optimality":
                    if combo_solo:
                        clpfd_combo_solo_optimality[i] += 1
                    else: 
                        clpfd_combo_optimality[i] +=1    
                    clpfd_combo_dist = cost_func(clpfd_combo_ex_arr, x)
                    #print("clpfd optimal cost: " + str(val_1))
                    tot_clpfd_combo_cost[i] = tot_clpfd_combo_cost[i] + clpfd_combo_dist
                elif clpfd_combo_outcome == "success":
                    clpfd_combo_success[i] += 1
                    clpfd_combo_dist = cost_func(clpfd_combo_ex_arr, x)
                    #print("clpfd success cost: " + str(val_1))
                    tot_clpfd_combo_cost[i] = tot_clpfd_combo_cost[i] + clpfd_combo_dist
                else:
                    clpfd_combo_missed[i] += 1  
# Normalize FT values                    
        no_ex = len(X_test)         
        if (no_ex - missed_ft) != 0:
            norm_val_3 = tot_ft_cost / (no_ex - missed_ft)           
        else:
            norm_val_3 = 0  
# Save FT values in list
        results_ft.append(("feature_tweaking", no_ex, m_size, d_name, missed_ft, 0, tot_ft_cost, norm_val_3))   

# Normalize clpfd values  
        norm_val_1 = [0,0,0,0]
        no_ex = len(X_test)   
        for i, t in enumerate(timeL):    
            if (no_ex - clpfd_missed[i]) != 0:
                norm_val_1[i] = tot_clpfd_cost[i] / (no_ex - clpfd_missed[i])
            else:
                norm_val_1[i] = 0        
# Save clpfd values in list                      
        results_clpfd.append(("clfd_tweaking_15s", no_ex, m_size, d_name, clpfd_missed[0], clpfd_optimality[0], tot_clpfd_cost[0], norm_val_1[0])) 
        results_clpfd.append(("clfd_tweaking_45s", no_ex, m_size, d_name, clpfd_missed[1], clpfd_optimality[1], tot_clpfd_cost[1], norm_val_1[1])) 
        results_clpfd.append(("clfd_tweaking_2m", no_ex, m_size, d_name, clpfd_missed[2], clpfd_optimality[2], tot_clpfd_cost[2], norm_val_1[2])) 
        results_clpfd.append(("clfd_tweaking_5m", no_ex, m_size, d_name, clpfd_missed[3], clpfd_optimality[3], tot_clpfd_cost[3], norm_val_1[3]))

# Normalize clpfd combo values  
        norm_val_2 = [0,0,0,0]
        no_ex = len(X_test)   
        for i, t in enumerate(timeL):    
            if (no_ex - clpfd_combo_missed[i]) != 0:
                norm_val_2[i] = tot_clpfd_combo_cost[i] / (no_ex - clpfd_combo_missed[i])
            else:
                norm_val_2[i] = 0        
# Save clpfd combo values in list                      
        results_clpfd_combo.append(("clfd_combo_tweaking_15s", no_ex, m_size, d_name, clpfd_combo_missed[0], clpfd_combo_optimality[0], clpfd_combo_solo_optimality[0], tot_clpfd_combo_cost[0], norm_val_2[0])) 
        results_clpfd_combo.append(("clfd_combo_tweaking_45s", no_ex, m_size, d_name, clpfd_combo_missed[1], clpfd_combo_optimality[1], clpfd_combo_solo_optimality[1], tot_clpfd_combo_cost[1], norm_val_2[1])) 
        results_clpfd_combo.append(("clfd_combo_tweaking_2m", no_ex, m_size, d_name, clpfd_combo_missed[2], clpfd_combo_optimality[2], clpfd_combo_solo_optimality[2], tot_clpfd_combo_cost[2], norm_val_2[2])) 
        results_clpfd_combo.append(("clfd_combo_tweaking_5m", no_ex, m_size, d_name, clpfd_combo_missed[3], clpfd_combo_optimality[3], clpfd_combo_solo_optimality[3], tot_clpfd_combo_cost[3], norm_val_2[3]))

# write results to file
    with open(Path + "results_ft", 'a+') as file_handler:
            for item in results_ft:
                file_handler.write("{}\n".format(item))  
                 
    with open(Path + "results_clpfd", 'a+') as file_handler:
            for item in results_clpfd:
                file_handler.write("{}\n".format(item))                      

    with open(Path + "results_clpfd_combo", 'a+') as file_handler:
            for item in results_clpfd_combo:
                file_handler.write("{}\n".format(item))                      

# stop clpfd solver when we are ready                
stop_solver(proc)