# -*- coding:utf-8 -*-
import sys
import subprocess  
import os 
from jinja2 import Template, Environment, FileSystemLoader
from collections import deque
import numpy as np
from math import ceil
import prune_util
from sklearn.tree._tree import TREE_UNDEFINED  # -2
import time
import copy
import math
import json
import multiprocessing

# open

NUM_STAGES = 5
ALPHA = 0.9
ROUND = 4
RULES_PER_STAGE = {16:4050, 8:1067, 1:810, 0:410} # bit = 16, 8, 1, 0
# RULES_PER_STAGE = {0:410} # bit = 16, 8, 1, 0
FEATURE_NAMES = ['total_len', 'protocol', 'flags_1_1_', 'ttl', 'srcport', 'dstport', 'flag_5_5_', 'flag_6_6_']
FEATURE_BITS = [16, 8, 1, 8, 16, 16, 1, 1]
PORTS = [1, 2]

def manage_depth(tree, bits, stages, rules_per_stage, tree_max_depth, tree_feature, tree_children_left, tree_children_right):
    queues = [deque([0]), deque()]
    capability = [rules_per_stage]*stages
    
    for level in range(0, tree_max_depth+1):
        tmpque = queues[level%len(queues)]
        while tmpque:
            curr_node = tmpque.popleft()
            if tree_feature[curr_node] == TREE_UNDEFINED:
                # leaf
                capability[level%stages] -= 1
            else:
                if FEATURE_BITS[tree_feature[curr_node]] > bits: # range
                    capability[level%stages] -= 2 # left & right branch
                else: # ternary
                    capability[level%stages] -= 2*FEATURE_BITS[tree_feature[curr_node]] # in the worst case, ternary is equal to bits
                
                queues[(level+1)%len(queues)].append(int(tree_children_left[curr_node]))
                queues[(level+1)%len(queues)].append(int(tree_children_right[curr_node]))
        if capability[level%stages] < 0:
            # print('   OVERFLOW in rule size ', level-1)
            return level-1
    return level

def cal_latency(loop):
    min_latency = 1/100
    if loop == 1:
        latency =  1/753
    elif loop == 2:
        latency = 1/(753+117)
    else:
        latency = 1/(753+658*loop)
    return latency/min_latency

def cal_metric(tree, metric_type=1):
    TP, TN, FP, FN = prune_util.get_leaves_confusion_matrix(tree)
    if metric_type == 0: # Acc
        return (TP + TN) / (TP + TN + FP + FN)
    elif metric_type == 1: # F1
        return 2 * TP / (TP + FP + TP + FN)
    elif metric_type == 2: # Precision
        return TP / (TP + FP)
    else: # Recall
        return TP / (TP + FN)

# should exam evaluator carefully!
def evaluator(tree, stages=NUM_STAGES, alpha=ALPHA):
    # stage is level, not depth!
    depth = []
    bonus = []

    for loop in range(1, ROUND+1):
        maxdepth, ternary_idx, maxbonus = -1, -1, -1
        # if not normal, one stage is used for extra process(resub/recir)
        nowstages = stages if loop == 1 else stages-1
        # deep copy before prune
        tmptree = copy.deepcopy(tree)
        oldstage = nowstages*(loop-1)-1
        # depth = level-1
        for nowdepth in range(nowstages*loop-1, oldstage, -1):
            tmptree = prune_util.hard_prune(tmptree, 0, nowdepth)
            tmptree = prune_util.soft_prune_mark(tmptree)
            tmptree = prune_util.soft_prune(tmptree)
            tree_max_depth, _, _, _,  _, tree_feature, _, _, tree_children_left, tree_children_right = prune_util.tree_attributes(tmptree)
            
            for bits, rules in RULES_PER_STAGE.items():
                realdepth = manage_depth(tmptree, bits, nowstages, rules, tree_max_depth, tree_feature, tree_children_left, tree_children_right)
                if realdepth > maxdepth:
                    maxdepth = realdepth
                    if realdepth == tree_max_depth:
                        ternary_idx = bits
                        maxbonus = alpha*cal_metric(tmptree) + (1-alpha)*cal_latency(loop)
                # print('   nowdepth, realdepth, maxdepth, ternary_idx', nowdepth, realdepth, maxdepth, ternary_idx)
            # if maxdepth == nowstages*loop-1 or nowdepth <= maxdepth:
            #     break
        depth.append((maxdepth, ternary_idx))
        bonus.append(maxbonus)
        # if maxbonus == -1:      # if found the optimal solution in middle of stage, no need for trying further loop
        #     break
        
        # print('---loop %f, bonus %f, metric %f, latencyf %f'%(loop, bonus[-1], cal_metric(tmptree), cal_latency(loop)))    
    
    loop = np.argmax(bonus)
    print('\n---depth list: ', depth)
    print('---bonus list: ',bonus)
    print('---type(0:normal, 1:resub, 2:recir), depth, bonus:\n\t\t', min(loop, 2), depth[loop], bonus[loop])
    # >= 2 is recir template
    return min(loop, 2), depth[loop][0], depth[loop][1]

Manager = multiprocessing.Manager()
THDDEPTH = Manager.list()
THDBONUS = Manager.list()
FOUNDED = Manager.list()
LOCK = multiprocessing.Lock()

def eva_loop_thread(threadid, tree, stages, alpha):
    global THDDEPTH
    global THDBONUS
    global FOUNDED
    global LOCK
    
    # pid = os.getpid()
    # cpu_list = os.sched_getaffinity(pid)
    # cpu_count = len(cpu_list)
    # localCpu = threadid % cpu_count
    # cpu_list.clear()
    # cpu_list.add(localCpu)
    # os.sched_setaffinity(pid, cpu_list)
    
    maxdepth, ternary_idx, maxbonus = -1, -1, -1
    # if not normal, one stage is used for extra process(resub/recir)
    nowstages = stages if threadid == 1 else stages-1
    # deep copy before prune
    tmptree = copy.deepcopy(tree)
    oldstage = nowstages*(threadid-1)-1
    # depth = level-1
    for nowdepth in range(nowstages * threadid-1, oldstage, -1):
        tmptree = prune_util.hard_prune(tmptree, 0, nowdepth)
        tmptree = prune_util.soft_prune_mark(tmptree)
        tmptree = prune_util.soft_prune(tmptree)
        tree_max_depth, _, _, _,  _, tree_feature, _, _, tree_children_left, tree_children_right = prune_util.tree_attributes(tmptree)
        
        for bits, rules in RULES_PER_STAGE.items():
            if len(FOUNDED) > 0 and threadid >= max(FOUNDED):
                return
            realdepth = manage_depth(tmptree, bits, nowstages, rules, tree_max_depth, tree_feature, tree_children_left, tree_children_right)
            if realdepth > maxdepth:
                maxdepth = realdepth
                if realdepth == tree_max_depth:
                    ternary_idx = bits
                    maxbonus = alpha*cal_metric(tmptree) + (1-alpha)*cal_latency(threadid)
        if maxdepth == nowstages*threadid-1 or nowdepth <= maxdepth:
            break
    # if len(THDDEPTH) > 0 and threadid > 1 and maxdepth <= THDDEPTH[-1][0]:
    #     # if we have reached the maxdepth, more tries are useless
    #     return
    with LOCK:
        if maxbonus == -1:      # if found the optimal solution in middle of stage, no need for trying further threadid
            FOUNDED.append(threadid)
        THDDEPTH.append((maxdepth, ternary_idx, threadid))
        THDBONUS.append(maxbonus)

# should exam mevaluator and thread carefully!
def evaluator_mthread(tree, stages=NUM_STAGES, alpha=ALPHA):
    global THDDEPTH
    global THDBONUS
    
    threads = []
    for loop in range(1, ROUND+1):
        t = multiprocessing.Process(target = eva_loop_thread, args=(loop, tree, stages, alpha))
        t.start()
        threads.append(t)
    
    # for t in threads:
    #     t.start()
    
    for t in threads:
        t.join()
    
    loop = np.argmax(THDBONUS)
    print('\n---depth list: ', THDDEPTH)
    print('---bonus list: ',THDBONUS)
    print('---type(0:normal, 1:resub, 2:recir), depth, bonus:\n\t\t', min(loop, 2), THDDEPTH[loop], THDBONUS[loop])
    return min(THDDEPTH[loop][2]-1, 2), THDDEPTH[loop][0], THDDEPTH[loop][1]

def range2ternary(range_begin, range_end, mask_width):
    list_mask_value = []
    def tcam_range(range_begin, range_end, mask_width, list_mask_value):
        for i in range(64):
            mask = 0x1 << i
            if ((range_begin & mask) or (mask > range_end)):
                break

        for j in range( i, -1, -1):
            stride = (1 << j) - 1
            if (range_begin + stride == range_end):
                tuple_mask_value = (~(range_begin ^ range_end) & ((1 << mask_width)-1), range_begin)
                list_mask_value.append(tuple_mask_value)
                return
            elif (range_begin + stride < range_end) :
                tcam_range(range_begin, range_begin + stride, mask_width, list_mask_value)
                tcam_range(range_begin + stride + 1, range_end, mask_width, list_mask_value)
                return
            else :
                continue
                
    if (range_begin <= range_end):
        tcam_range(range_begin, range_end, mask_width, list_mask_value)
    else:
        input('=== Error range2ternary')
       
    return list_mask_value

def export_p4_rules(tree, stages, type_, ternary_idx, command_file):
    # [curr_node, prev_node, thresh_flag]
    queues = [deque([(0, 0, 0)]), deque()]

    tree_max_depth, _, tree_n_outputs, tree_n_classes, \
    tree_classes, tree_feature, tree_threshold, tree_value, \
    tree_children_left, tree_children_right = prune_util.tree_attributes(tree)

    # ternary
    def gen_ternary(idx, level_stages, prev_node, thresh_flag, curr_node, 
        left, right, less_than_feature, bit_width, leftbranch):
        if FEATURE_BITS[idx] > ternary_idx:
            # right = right+1 if leftbranch else right
            ternaries = [(right, left)]
        else:
            ternaries = range2ternary(left, right, bit_width)
        for mask_value in ternaries:
            str_ = ('bfrt.simple_l3_test.pipe.Ingress.' + \
                   'level%d.node.add_with_CheckFeature(%d, %d, ')%(
                   level_stages, prev_node, thresh_flag)

            paras = []
            for i in range(len(FEATURE_NAMES)):
                if i != idx and FEATURE_BITS[i] > ternary_idx:
                    paras.append(0)
                    paras.append((1<<FEATURE_BITS[i])-1)
                elif i != idx and FEATURE_BITS[i] <= ternary_idx:
                    paras.extend([0, 0])
                else:
                    paras.append(mask_value[1]) # value
                    paras.append(mask_value[0]) # mask
            str_ += ','.join(map(str, paras))

            # MATCH_PRIORITY, node_id, less_than_feature
            str_ += ', 0, %d, %d)\n'%(curr_node, less_than_feature)
            command_file.write(str_)

    for level in range(0, tree_max_depth+1):
        tmpque = queues[level%len(queues)]
        while tmpque:
            curr_node, prev_node, thresh_flag = tmpque.popleft()
            if tree_feature[curr_node] == TREE_UNDEFINED: # leaf
                if tree_n_outputs == 1:
                    value = tree_value[curr_node][0]
                else:
                    value = tree_value[curr_node].T[0]
                class_id = np.argmax(value)
                if (tree_n_classes != 1 and tree_n_outputs == 1):
                    class_id = int(tree_classes[class_id])

                # prev_node_id, threshold_flag
                str_ = ('bfrt.simple_l3_test.pipe.Ingress.' + \
                       'level%d.node.add_with_SetClass(%d, %d')%(
                       level%stages, prev_node, thresh_flag)
                for i in FEATURE_BITS:
                    if i > ternary_idx:
                        str_ += ', %d, %d'%(0, (1<<i)-1)
                    else:
                        str_ += ', 0, 0'
                # MATCH_PRIORITY, node_id, class_id
                str_ += ', 0, %d, %d)\n'%(curr_node, PORTS[class_id])
                command_file.write(str_)
            else: # children
                feature_id = tree_feature[curr_node]
                threshold = int(float(tree_threshold[curr_node]))
                gen_ternary(feature_id, level%stages, prev_node, thresh_flag, curr_node,
                             # less_than_feature=1
                    0, threshold, 1, FEATURE_BITS[feature_id], leftbranch=True)
                gen_ternary(feature_id, level%stages, prev_node, thresh_flag, curr_node,
                    threshold+1, (1<<FEATURE_BITS[feature_id])-1, 
                # less_than_feature=0
                    0, FEATURE_BITS[feature_id], leftbranch=False)

                # children
                queues[(level+1)%len(queues)].append((
                    int(tree_children_left[curr_node]), curr_node, 1))
                queues[(level+1)%len(queues)].append((
                    int(tree_children_right[curr_node]), curr_node, 0))

def create_new_p4(stage_num, loop, ternary_idx):
    type_name = ['normal', 'resubmit', 'recirculate']
    match_name = [ 'range' if bits > ternary_idx else 'ternary'
     for bits in FEATURE_BITS]
    file = './hardware_configure/template/%s_tmplate.p4'%(type_name[loop])

    levels=[]
    for i in range(stage_num):
        level_tmp='level'+'%d' % i
        levels.append(level_tmp)
    with open(file) as f:
        template_str = f.read()
        template = Environment(
            loader=FileSystemLoader('./hardware_configure/template/')
            ).from_string(template_str)
        result = template.render(levels=levels, match_name=match_name, 
            table_num=RULES_PER_STAGE[ternary_idx]) 
    result.encode(encoding='utf-8')
    p4_filepath = './hardware_configure/simple_l3_test.p4' 
    with open(p4_filepath,'w') as f2:
        f2.write(result)

def complier(tree, loop, ternary_idx, stages=NUM_STAGES):
    # write p4
    nowstages = stages if loop == 0 else stages-1
    create_new_p4(nowstages, loop, ternary_idx)
    # gen rules
    with open('./hardware_configure/command_p4.txt', 'w') as command_file:
        export_p4_rules(tree, nowstages, loop, ternary_idx, command_file)

if __name__ == '__main__':
    tree = prune_util.load_model()

    start = time.time()
    loop, depth, ternary_idx = evaluator_mthread(tree, stages=NUM_STAGES)
    print('---time evaluator MTHREAD cost',time.time() - start,'s')


    start = time.time()
    loop, depth, ternary_idx = evaluator(tree, stages=NUM_STAGES)
    print('---time evaluator SINGLE cost',time.time() - start,'s')

    start = time.time()
    tree = prune_util.hard_prune(tree, 0, depth)
    print('---time hard prune cost',time.time() - start,'s')

    start = time.time()
    tree = prune_util.soft_prune_mark(tree)
    tree = prune_util.soft_prune(tree)
    print('---time soft prune cost',time.time() - start,'s')
    
    start = time.time()
    complier(tree, loop, ternary_idx, stages=NUM_STAGES)
    print('---time complier cost',time.time() - start,'s')

    start = time.time()
    x_test, y_test = prune_util.load_data()
    count = {'0':0, '1':0}
    for idx, data in enumerate(x_test):
        result = prune_util.predict(tree, ['0', '1'], data)
        count[result] += 1

    print('tree predict:', count)
    print('---time predict cost',time.time() - start,'s')

    prune_util.output_testing_metrics(tree, x_test, y_test, ['0', '1'])
