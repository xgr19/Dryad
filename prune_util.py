# -*- coding:utf-8 -*-
import pickle
import json
import copy
import numpy as np
import time
import sklearn.tree as st
import os
# import psutil

feature_list = [
    'Total length', 'Protocol', 'IPV4 Flags (DF)', 'Time to live',
    'Src Port', 'Dst Port', 'TCP flags (Reset)', 'TCP flags (Syn)'
]

def set_node_idx(json_model, node_index):
    if "children" not in json_model:  # 叶节点
        json_model['id'] = '{}'.format(node_index)
        return node_index
    else:  # 非叶节点
        json_model['id'] = '{}'.format(node_index)
        children = json_model["children"]
        for child in children:
            node_index = set_node_idx(child, node_index + 1)
    return node_index

def tree_attributes(tree):
    # 进行属性初始化
    now_depth = 0
    max_depth = 0
    node_count = 1
    n_outputs_ = 0
    n_classes_ = 0
    classes_ = []
    feature = []
    threshold = []
    value = []
    children_left = []
    children_right = []
    set_node_idx(tree, 0)
    return get_tree_attributes(
        tree, now_depth, max_depth, node_count, n_outputs_, n_classes_, classes_,
        feature, threshold, value, children_left, children_right)

def get_tree_attributes(json_model, now_depth, max_depth, node_count, n_outputs_, n_classes_, classes_,
    feature, threshold, value, children_left, children_right):
    if "children" not in json_model:
        if now_depth > max_depth:
            max_depth = now_depth
        n_outputs_ = 1
        n_classes_ = len(json_model["value"])
        classes_ = [0, 1]
        feature.append(-2)
        threshold.append(-2)
        value.append([json_model["value"]])
        children_left.append(-1)
        children_right.append(-1)
        return max_depth, node_count, n_outputs_, n_classes_, classes_, \
               feature, threshold, value, children_left, children_right
    feature.append(feature_list.index(json_model["feature"]))
    threshold.append(json_model["threshold"])
    value.append([json_model["value"]])
    children = json_model["children"]

    children_left.append(children[0]["id"])
    children_right.append(children[1]["id"])

    # 左子树
    max_depth, node_count, n_outputs_, n_classes_, classes_, feature, threshold, value, children_left, children_right = \
        get_tree_attributes(children[0], now_depth + 1, max_depth, node_count + 1, n_outputs_, n_classes_, classes_,
                            feature, threshold, value, children_left, children_right)

    # 右子树
    max_depth, node_count, n_outputs_, n_classes_, classes_, feature, threshold, value, children_left, children_right = \
        get_tree_attributes(children[1], now_depth + 1, max_depth, node_count + 1, n_outputs_, n_classes_, classes_,
                            feature, threshold, value, children_left, children_right)

    return max_depth, node_count, n_outputs_, n_classes_, classes_, \
           feature, threshold, value, children_left, children_right

# 将sklearn模型转化为json模型
def sklearn2json(model, class_names, node_index=0):
    json_model = {}
    if model.tree_.children_left[node_index] == -1:  # 叶子节点
        count_labels = zip(model.tree_.value[node_index, 0], class_names)
        json_model['value'] = [count for count, label in count_labels]
    else:  # 非叶节点
        count_labels = zip(model.tree_.value[node_index, 0], class_names)
        json_model['value'] = [count for count, label in count_labels]
        feature = feature_list[model.tree_.feature[node_index]]
        threshold = model.tree_.threshold[node_index]
        json_model['name'] = '{} <= {}'.format(feature, threshold)
        json_model['feature'] = '{}'.format(feature)
        json_model['threshold'] = '{}'.format(threshold)
        left_index = model.tree_.children_right[node_index]
        right_index = model.tree_.children_left[node_index]
        json_model['children'] = [sklearn2json(model,  class_names, right_index),
                                  sklearn2json(model,  class_names, left_index)]
    return json_model

# 计算树的叶节点数
def get_tree_leaves_count(json_model, count):
    if "children" not in json_model:
        return 1
    children = json_model["children"]
    for child in children:
        count += get_tree_leaves_count(child, 0)
    return count

# 计算树的最大深度以及节点数（叶节点+非叶节点）
def get_tree_max_depth_and_nodes_count(json_model):
    nodes_count = 0
    max_depth = 0
    del_count = 0
    stack1 = [json_model]  # 从根节点0开始
    stack2 = [0]  # 根节点的深度为0
    while len(stack1) > 0:
        json_model = stack1.pop()  # pop保证每个节点只会被访问一次
        depth = stack2.pop()
        if depth > max_depth:
            max_depth = depth
        nodes_count += 1
        if "tobedel" in json_model:
            del_count = del_count + json_model["tobedel"]
        if "children" in json_model:  # 是非叶节点
            children = json_model["children"]
            for child in children:
                stack1.append(child)  # 将孩子存入，并且深度加1
                stack2.append(depth + 1)
    return max_depth, nodes_count, del_count

# 输出模型结构
def output_model_structure(json_model):
    max_depth, nodes_count, del_count = get_tree_max_depth_and_nodes_count(json_model)
    leaves_count = get_tree_leaves_count(json_model, 0)
    print('The true depth of the tree =', max_depth)
    print('The number of leaves =', leaves_count)
    print('The number of all nodes =', nodes_count)
    print('The number of nodes to be delete =', del_count)
    print('-----------------------------')
    
# 计算TP、TN、FP、FN
def get_node_confusion_matrix(json_model):
    value = json_model['value']
    if value[0] >= value[1]:
        class_name = 0
    else:
        class_name = 1
    TP = class_name * max(value)
    TN = (1 - class_name) * max(value)
    FP = class_name * min(value)
    FN = (1 - class_name) * min(value)
    return TP, TN, FP, FN

# 计算叶节点的混淆矩阵指标之和
def get_leaves_confusion_matrix(json_model, TP=0, TN=0, FP=0, FN=0):
    if "children" not in json_model:  # 叶节点
        return get_node_confusion_matrix(json_model)
    children = json_model["children"]
    for child in children:
        TP_, TN_, FP_, FN_ = get_leaves_confusion_matrix(child)
        TP += TP_
        TN += TN_
        FP += FP_
        FN += FN_
    return TP, TN, FP, FN

# 输出精度的评估指标
def output_metrics(TP, TN, FP, FN):
    print('TP =', TP)
    print('TN =', TN)
    print('FP =', FP)
    print('FN =', FN)
    print('Accuracy =', format((TP + TN) / (TP + TN + FP + FN), '.6f'))
    print('Precision score =', format(TP / (TP + FP), '.6f'))
    print('Recall score =', format(TP / (TP + FN), '.6f'))
    print('F1 score =', format(2 * TP / (TP + FP + TP + FN), '.6f'))
    print('-----------------------------')

# 得到数据对应叶节点的value
def classify(json_model, data):
    if "children" not in json_model:
        return json_model["value"]  # 到达叶子节点，完成测试

    feature = json_model["feature"]
    threshold = float(json_model["threshold"])
    feature_value = data[feature_list.index(feature)]
    if float(feature_value) <= threshold:
        child = json_model["children"][0]
        value = classify(child,  data)
    else:
        child = json_model["children"][1]
        value = classify(child,  data)

    return value

# 得到数据对应的class
def predict(json_model, class_names, data):
    value = classify(json_model, data)
    class_names_index = value.index(max(value))
    predict_result = class_names[class_names_index]
    return predict_result

# 输出测试精度
def output_testing_metrics(json_model, X, Y, class_names):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for index, data in enumerate(X):
        predict_result = predict(json_model, class_names, data)
        if predict_result == '1' and str(Y[index]) == '1':
            TP += 1
        if predict_result == '1' and str(Y[index]) == '0':
            FP += 1
        if predict_result == '0' and str(Y[index]) == '1':
            FN += 1
        if predict_result == '0' and str(Y[index]) == '0':
            TN += 1

    output_metrics(TP, TN, FP, FN)

# 得到节点所属的类别
def get_node_class_name(json_model):
    value = json_model['value']
    if value[0] >= value[1]:
        class_name = 0
    else:
        class_name = 1
    return class_name

def load_data():
    with open('./model_data/x_test.pkl', 'rb') as tf:
        x_test = pickle.load(tf)
    with open('./model_data/y_test.pkl', 'rb') as tf:
        y_test = pickle.load(tf)
    print('Size of x_test = %d x %d' % (len(x_test), len(x_test[0])))
    print('Size of y_test = %d x 1' % len(y_test))
    return x_test, y_test

def load_model():
    json_file = './model_data/first_soft_pruned_tree.json'
    with open(json_file, 'r') as f:
        best_soft_json_model = json.load(f)

    # print('---硬剪枝前的模型结构---')
    # output_model_structure(best_soft_json_model)
    # print('---硬剪枝前的训练精度---')
    # TP, TN, FP, FN = get_leaves_confusion_matrix(best_soft_json_model)
    # output_metrics(TP, TN, FP, FN)
    return best_soft_json_model

def hard_prune(json_model, now_depth, limit_depth):
    json_model["tobedel"] = 0
    if "leafcount" in json_model:
        json_model["leafcount"][0] = 0
        json_model["leafcount"][1] = 0
    else:
        json_model["leafcount"] = []
        json_model["leafcount"].append(0)
        json_model["leafcount"].append(0)
        
    if "children" not in json_model:  # 叶节点
        json_model["leafcount"][0] = 1
        return
    else:  # 非叶节点
        children = json_model["children"]
        if now_depth == limit_depth:  # 找到要剪枝的部分，将其删除， 删除后即为叶子节点
            del json_model["children"]
            json_model["leafcount"][0] = 1
        else:
            for child in children:
                hard_prune(child, now_depth + 1, limit_depth)
    return json_model

def soft_prune_mark(json_model):
    classNameStack = []

    jsonNode = json_model
    jsonNodeStack = []
    while jsonNodeStack or jsonNode:
        while jsonNode:
            jsonNodeStack.append(jsonNode)
            if "children" in jsonNode:
                jsonNode = jsonNode["children"][0]  # all assume to be binary tree
            else:
                jsonNode = None
       
        # visit current node 
        currentNode = jsonNodeStack.pop()           # turn to last node
        if "children" in currentNode and len(currentNode["children"]) > 0:
            currentNode["leafcount"][0] = currentNode["children"][0]["leafcount"][0] + currentNode["children"][0]["leafcount"][1]
            currentNode["leafcount"][1] = currentNode["children"][1]["leafcount"][0] + currentNode["children"][1]["leafcount"][1]
            
            classname = np.argmax(currentNode['value'])
            flag = 1
            count = currentNode["leafcount"][0] + currentNode["leafcount"][1]
            for childClassName in classNameStack[-count:]:
                if classname != childClassName:
                    flag = 0
                    break
            if flag == 1:
                currentNode["tobedel"] = 1
        else:
            classNameStack.append(np.argmax(currentNode['value'])) # push leaf node's class name
                
        
        # turn to current node's brother right node
        if jsonNodeStack and jsonNodeStack[-1]["children"][0] is currentNode:
            jsonNode = jsonNodeStack[-1]["children"][1]
        else:
            jsonNode = None
    return json_model
 

def soft_prune(json_model):
    if json_model is None:
        return None
        
    if json_model["tobedel"] == 1 :
        del json_model["children"]
    else:
        if "children" in json_model:
            children = json_model["children"]
            for child in children:
                soft_prune(child)
                
    return json_model
    