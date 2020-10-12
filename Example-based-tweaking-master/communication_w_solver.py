"""
Sending problem to contraint solver

@author: anonymous
"""
import os
import sys
import json
import subprocess
from _dummy_thread import error

def get_data_structure(ensemble_classifier, class_labels):
    paths = []
    for i_est in range(0, len(ensemble_classifier.estimators_)):            
        paths.append([])        
    for i_est in range(0, len(ensemble_classifier.estimators_)):      
        for c_label in class_labels:        
            paths[i_est].append([]) 
    return paths        

def get_example_path(ensemble_classifier, x, class_labels):
    paths = get_data_structure(ensemble_classifier, class_labels)
    
    tree_index = 0
    for estimator in ensemble_classifier:
        dp = estimator.decision_path([x])
        leave_id = estimator.apply([x])
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        value = estimator.tree_.value
        path_tuples = []
        for node_id in dp.indices:
            if (leave_id[0] != node_id):
                node_feature = feature[node_id]
                node_threshold = threshold[node_id]
                int_threshold = int(node_threshold)
                path_tuples = path_tuples.copy()
                if (x[node_feature] <= node_threshold):                                        
                    path_tuples.append((node_feature, "<=", int_threshold))
                else:    
                    path_tuples.append((node_feature, ">", int_threshold)) 
            else: # leave
                node_value = value[node_id]
                node_v = node_value.ravel()
                node_class = node_v.argmax()            
                leave_tuple = path_tuples.copy()
                leave_tuple.append(("class", "=", node_class))  #redundant ok for now
                paths[tree_index][node_class].append(leave_tuple.copy())
        tree_index += 1         
    return paths

def get_all_class_paths(ensemble_classifier, class_labels):    
    paths = get_data_structure(ensemble_classifier, class_labels)
   
    tree_index = 0    
    for estimator in ensemble_classifier:
        #From current node extract information of children nodes
        children_left = estimator.tree_.children_left  
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        value = estimator.tree_.value
        
        #node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        stack = [(0, -1, [])]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth, path_tuples = stack.pop()
            #node_depth[node_id] = parent_depth + 1
            # Testnode
            if (children_left[node_id] != children_right[node_id]):
                node_feature = int(feature[node_id])
                node_threshold = threshold[node_id]
                left_path_tuples = path_tuples.copy()
                int_threshold = int(node_threshold)
                left_path_tuples.append((node_feature, "<=", int_threshold))
                right_path_tuples = path_tuples.copy() 
                right_path_tuples.append((node_feature, ">", int_threshold)) 
                stack.append((children_left[node_id], parent_depth + 1, left_path_tuples))
                stack.append((children_right[node_id], parent_depth + 1, right_path_tuples))
            else: # leave
                node_value = value[node_id]
                node_v = node_value.ravel()
                node_class = int(node_v.argmax())            
                leave_tuple = path_tuples.copy()
                leave_tuple.append(("class", "=", node_class))  #redundant ok for now
                paths[tree_index][node_class].append(leave_tuple.copy())
        tree_index += 1         
    return paths

def stop_solver(proc):
    empty_data = []
    quit = json.dumps({'command':'quit','data':empty_data})
    proc.stdin.write(quit);
    proc.stdin.flush()
    r = proc.stdout.readline(); 
    x = json.loads(r); 
    print("response is: " + str(x["result"]))
    
def start_solver(logg_param): 
    default_logging = logg_param
    logging = (os.environ.get('LOGGING', str(default_logging)).lower() in ["true"])

    def log(arg):
        if logging:
            print("Log: " + str(arg))
        return arg

    #Paths
    sicstus_path = 'C:/Program Files/SICStus Prolog VC15 4.5.1/bin/'
    exe_path = sicstus_path + 'sicstus' 
    #where you installed this program
    pl_path = 'YOUR PATH' + '/clpfd_tweaking/src/clpfd_server.pl' 

    # Make it possible to pass path to SICStus, and path to Prolog code, on command line
    #if __name__ == "__main__":
    if len(sys.argv) > 1:
        exe_path = sys.argv[1];
    if len(sys.argv) > 2:
        pl_path = sys.argv[2]
        
    # Start the Prolog server as a sub-process
    # The Prolog stderr will go to the console (which is helpful during troubleshooting), unless you set stderr=subprocess.NULL
    # SICStus will be less verbose if you pass --nologo and/or --noinfo
    #

    proc = subprocess.Popen([exe_path,'-l', pl_path,'--goal', 'start_clpfd_server.'] + ([] if logging else ['--nologo' ,'--noinfo']),
                            stdout=subprocess.PIPE,
                            stdin=subprocess.PIPE,
                            stderr=(None if logging else subprocess.DEVNULL),
                            encoding='UTF-8'
                        )
    return proc

def tweak_example(proc, x, timeOut, minMaxVals, wish_class, feature_direction_list, percentage, cost_list):
    # Send in weights for cost function etc.
    #print("Set up clpfd problem with example to be tweaked and cost function")
    #print("Define constraint problem - do this for each example")    
    example_information = []
    example_information.append({'ex':x, 'whish_c':wish_class, 'time':timeOut, 'min_max':minMaxVals, 'direction':feature_direction_list, 'probability':percentage, 'cost_f':cost_list})      
    tweak_example = json.dumps({'command':'tweak_example', 'information':example_information})
    #print(tweak_example)
    proc.stdin.write(tweak_example); 
    proc.stdin.flush();
    resp = proc.stdout.readline(); 
    #catch error in decode 
    try:       
        if (resp != ''):
            obj_resp = json.loads(resp);
        else:
            return [], error
    except ValueError:
        print("response is: " + str(resp))
        return [], error   
    else:  
        return obj_resp["ex"], obj_resp["outcome"]

def setup_problem(proc, ensemble_classifier, class_labels):     
    print("Get all paths - do once for a rf-model ")
    all_paths = get_all_class_paths(ensemble_classifier, class_labels) 
    #no_trees = len(ensemble_classifier.estimators_) 
    #print("Get example paths - do this for each example")
    #example_paths = get_example_path(ensemble_classifier, x, decimals, class_labels)
    print("Forward rf-model to constraint solver")    
    setup_problem = json.dumps({'command':'setup_problem', 'model':all_paths})
    print(setup_problem)
    #send set_problem command to prolog
    proc.stdin.write(setup_problem); 
    proc.stdin.flush()
    r = proc.stdout.readline(); 
    x = json.loads(r); 
    print("response is: " + str(x["result"]))

def restart_needed(proc):
    print("See if we need to re-start server ")
    restart_needed = json.dumps({'command':'need_to_restart', 'data':{}})
    proc.stdin.write(restart_needed); 
    proc.stdin.flush()
    r = proc.stdout.readline(); 
    x = json.loads(r); 
    print("response is: " + str(x["result"])) 
    if str(x["result"]) == "ok":
        return False
    else:
        return True
