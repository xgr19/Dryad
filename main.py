# -*- coding:utf-8 -*-
# 得到第一次软剪枝后的json模型
import pickle
import json
import copy
import numpy as np
import time
from collections import Counter
from sklearn.tree._tree import TREE_LEAF
import sklearn.tree as st
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'D:/Graphviz/bin'

# 将sklearn模型转化为json模型
def sklearn2json(model, feature_list, class_names, node_index=0):
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
        json_model['children'] = [sklearn2json(model, feature_list, class_names, right_index),
                                  sklearn2json(model, feature_list, class_names, left_index)]
    return json_model


# 将sklearn模型和json模型进行同步
def pruned_sklearn_model(sklearn_model, index, json_model):
    if "children" not in json_model:
        sklearn_model.children_left[index] = TREE_LEAF
        sklearn_model.children_right[index] = TREE_LEAF
    else:
        pruned_sklearn_model(sklearn_model, sklearn_model.children_left[index], json_model["children"][0])
        pruned_sklearn_model(sklearn_model, sklearn_model.children_right[index], json_model["children"][1])


# 决策树可视化
def draw_file(model, feature_list, class_names, pdf_file):
    dot_data = st.export_graphviz(
        model,
        out_file=None,
        feature_names=feature_list,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        impurity=False,
    )
    graph = graphviz.Source(dot_data)
    graph.render(pdf_file)  # 在同级目录下生成tree.pdf文件
    print("The tree has been drawn in " + pdf_file + '.pdf')


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
    stack1 = [json_model]  # 从根节点0开始
    stack2 = [0]  # 根节点的深度为0
    while len(stack1) > 0:
        json_model = stack1.pop()  # pop保证每个节点只会被访问一次
        depth = stack2.pop()
        if depth > max_depth:
            max_depth = depth
        nodes_count += 1
        if "children" in json_model:  # 是非叶节点
            children = json_model["children"]
            for child in children:
                stack1.append(child)  # 将孩子存入，并且深度加1
                stack2.append(depth + 1)
    return max_depth, nodes_count


# 输出模型结构
def output_model_structure(json_model):
    max_depth, nodes_count = get_tree_max_depth_and_nodes_count(json_model)
    leaves_count = get_tree_leaves_count(json_model, 0)
    rules = leaves_count + (nodes_count - leaves_count) * 2
    print('The true depth of the tree =', max_depth)
    print('The number of leaves =', leaves_count)
    print('The number of all nodes =', nodes_count)
    print('The number of rules =', rules)

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
    print('%d/%d' % (TP+TN, TP + TN + FP + FN))
    print('Accuracy =', format((TP + TN) / (TP + TN + FP + FN), '.6f'))
    print('Precision score =', format(TP / (TP + FP), '.6f'))
    print('Recall score =', format(TP / (TP + FN), '.6f'))
    print('F1 score =', format(2 * TP / (TP + FP + TP + FN), '.6f'))


# 得到数据对应叶节点的value
def classify(json_model, feature_list, data):
    if "children" not in json_model:
        return json_model["value"]  # 到达叶子节点，完成测试

    feature = json_model["feature"]
    threshold = float(json_model["threshold"])
    feature_value = data[feature_list.index(feature)]
    if float(feature_value) <= threshold:
        child = json_model["children"][0]
        value = classify(child, feature_list, data)
    else:
        child = json_model["children"][1]
        value = classify(child, feature_list, data)

    return value


# 得到数据对应的class
def predict(json_model, feature_list, class_names, data):
    value = classify(json_model, feature_list, data)
    class_names_index = value.index(max(value))
    predict_result = class_names[class_names_index]
    return predict_result



# 输出测试精度
def output_testing_metrics(json_model, X, Y, feature_list, class_names):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for index, data in enumerate(X):
        predict_result = predict(json_model, feature_list, class_names, data)
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


# 得到叶节点所属的类别list
def get_leaves_class_name(json_model):
    stack = [json_model]  # 从根节点0开始
    class_name_list = []  # 记录每个叶节点的class
    while len(stack) > 0:
        json_model = stack.pop()  # pop保证每个节点只会被访问一次
        if "children" in json_model:  # 非叶节点
            children = json_model["children"]
            for child in children:
                stack.append(child)  # 将孩子存入
        else:  # 叶节点
            class_name_list.append(np.argmax(json_model['value']))

    return class_name_list


# 判断是否可以进行软剪枝
def can_be_simplified(json_model):
    class_name = np.argmax(json_model['value'])  # 得到节点所属的类别
    class_name_list = get_leaves_class_name(json_model)  # 得到叶节点所属的类别list  子树节点数
    flag = 1  # 判断是否可以进行软剪枝，1为可以，0为不可以
    for i_class_name in class_name_list:  # 叶节点数
        if i_class_name != class_name:  # class不属于同一类
            flag = 0
            break
    return flag


def load_data():
    with open("./8_features_20211202/x_train.pkl", "rb") as tf:
        x_train = pickle.load(tf)
    with open("./8_features_20211202/y_train.pkl", "rb") as tf:
        y_train = pickle.load(tf)
    with open("./8_features_20211202/x_test.pkl", "rb") as tf:
        x_test = pickle.load(tf)
    with open("./8_features_20211202/y_test.pkl", "rb") as tf:
        y_test = pickle.load(tf)
    print('Size of x_train = %d x %d' % (len(x_train), len(x_train[0])))
    print('Size of y_train = %d x 1' % len(y_train))
    print('Size of x_test = %d x %d' % (len(x_test), len(x_test[0])))
    print('Size of y_test = %d x 1' % len(y_test))
    print(Counter(y_test))
    print(Counter(y_train))
    return x_train, y_train, x_test, y_test


# 原始版本
def hard_prune2(json_model, now_depth, limit_depth):  # O(n) n:总结点数
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
        if now_depth == limit_depth:  # 找到要剪枝的部分，将其删除，删除后即为叶子节点
            del json_model["children"]
            json_model["leafcount"][0] = 1
        else:
            for child in children:
                hard_prune(child, now_depth + 1, limit_depth)
    return json_model


# 修改版本
def hard_prune(json_model, now_depth, limit_depth):  # O(n) n:总结点数
    jsonNode = json_model
    jsonNodeQueue = []
    depthQueue = []
    jsonNodeQueue.append(jsonNode)
    depthQueue.append(0)

    while jsonNodeQueue:
        jsonNode = jsonNodeQueue.pop(0)
        depth = depthQueue.pop(0)
        jsonNode["tobedel"] = 0
        if "leafcount" in jsonNode:
            jsonNode["leafcount"][0] = 0
            jsonNode["leafcount"][1] = 0
        else:
            jsonNode["leafcount"] = []
            jsonNode["leafcount"].append(0)
            jsonNode["leafcount"].append(0)

        if "children" not in jsonNode:  # 叶节点
            jsonNode["leafcount"][0] = 1
        else:  # 非叶节点
            left_child = jsonNode["children"][0]
            right_child = jsonNode["children"][1]
            if depth == limit_depth:  # 找到要剪枝的部分，将其删除，删除后即为叶子节点
                del jsonNode["children"]
                jsonNode["leafcount"][0] = 1
            else:
                jsonNodeQueue.append(left_child)
                depthQueue.append(depth + 1)
                jsonNodeQueue.append(right_child)
                depthQueue.append(depth + 1)

    return json_model


def soft_prune(json_model):
    classNameStack = []
    jsonNode = json_model
    jsonNodeStack = []

    while jsonNodeStack or jsonNode:
        while jsonNode:  # 一直往左走，走到最左的节点
            jsonNodeStack.append(jsonNode)
            if "children" in jsonNode:
                jsonNode = jsonNode["children"][0]
            else:
                jsonNode = None

        # 访问当前节点
        currentNode = jsonNodeStack.pop()  # 转到最后一个节点
        if "children" in currentNode and len(currentNode["children"]) > 0:  # 如果该节点有子节点
            currentNode["leafcount"][0] = currentNode["children"][0]["leafcount"][0] + \
                                          currentNode["children"][0]["leafcount"][1]
            currentNode["leafcount"][1] = currentNode["children"][1]["leafcount"][0] + \
                                          currentNode["children"][1]["leafcount"][1]

            classname = np.argmax(currentNode['value'])  # 得到当前节点所属的类别

            # 判断该节点是否可以被软剪枝
            flag = 1
            count = currentNode["leafcount"][0] + currentNode["leafcount"][1]  # 得到当前节点左右子节点叶子数之和
            for childClassName in classNameStack[-count:]:
                if classname != childClassName:
                    flag = 0
                    break
            if flag == 1:
                currentNode["tobedel"] = 1
                del currentNode["children"]
        else:  # 如果该节点没有子节点
            classNameStack.append(np.argmax(currentNode['value']))  # push叶节点所属的类别

        # turn to current node's brother right node
        if jsonNodeStack and jsonNodeStack[-1]["children"][0] is currentNode:
            jsonNode = jsonNodeStack[-1]["children"][1]
        else:
            jsonNode = None

    return json_model


if __name__ == '__main__':
    # 加载训练集和测试集（8个特征）
    print('**********Loading data (start)**********')
    # feature_list = [
    #     'Total length', 'Protocol', 'IPV4 Flags (DF)', 'IPV4 Flags (MF)', 'Header length',
    #     'Time to live', 'Src Port', 'Dst Port', 'TCP flags (Acknowledgment)',
    #     'TCP flags (Push)', 'TCP flags (Reset)', 'TCP flags (Syn)', 'TCP flags (Fin)'
    # ]
    feature_list = [
        'Total length', 'Protocol', 'IPV4 Flags (DF)', 'Time to live',
        'Src Port', 'Dst Port', 'TCP flags (Reset)', 'TCP flags (Syn)'
    ]
    class_names = np.array(['0', '1'])
    x_train, y_train, x_test, y_test = load_data()  # 加载数据
    print('**********Loading data (end)**********\n')

    # 训练模型
    max_depth = 12
    model = st.DecisionTreeClassifier(max_depth=max_depth, random_state=5)
    time_start = time.time()  # 记录开始时间
    model.fit(x_train, y_train)  # 进行训练
    time_end = time.time()  # 记录结束时间
    print('The time of training = ' + str(time_end - time_start) + 's')  # 输出训练时间

    # 将sklearn模型转化为json模型
    print('**********Transforming to json model (start)**********')
    json_model = sklearn2json(model, feature_list, class_names)
    json_model = hard_prune(json_model, 0, max_depth)  # 为json模型添加属性
    time_end = time.time()  # 记录结束时间
    print('The time of training = ' + str(time_end - time_start) + 's')  # 输出训练时间

    print('---剪枝前的模型结构---')
    output_model_structure(json_model)
    print('---剪枝前的训练精度---')
    TP, TN, FP, FN = get_leaves_confusion_matrix(json_model)
    output_metrics(TP, TN, FP, FN)
    print('---剪枝前的测试精度---')
    output_testing_metrics(json_model, x_test, y_test, feature_list, class_names)
    # print('---剪枝前的模型可视化---')
    # pruned_sklearn_model(model.tree_, 0, json_model)  # 将sklearn模型和json模型进行同步
    # draw_file(model, feature_list, class_names, pdf_file='./results/unpruned_tree')
    print('**********Transforming to json model (end)**********\n')

    # 保存未剪枝的json模型
    print('**********Saving unpruned tree model (start)**********')
    json_file = './p4/unpruned_tree_depth_' + str(max_depth) + '.json'
    with open(json_file, 'w') as f:
        f.write(json.dumps(json_model))
    print("The unpruned tree's json model is saved in", json_file)
    print('**********Saving unpruned tree model (end)**********\n')

    # 软剪枝操作
    print('**********Soft pruning (start)**********')
    time_start = time.time()  # 记录开始时间
    best_soft_json_model = soft_prune(json_model)
    time_end = time.time()  # 记录结束时间
    print('The time of soft prune = ' + str(time_end - time_start) + 's')  # 输出训练时间
    pruned_sklearn_model(model.tree_, 0, best_soft_json_model)  # 将sklearn模型和json模型进行同步
    print('---软剪枝后的模型结构---')
    output_model_structure(best_soft_json_model)
    print('---软剪枝后的训练精度---')
    TP, TN, FP, FN = get_leaves_confusion_matrix(best_soft_json_model)
    output_metrics(TP, TN, FP, FN)
    # print('---软剪枝后的模型可视化---')
    # pruned_sklearn_model(model.tree_, 0, best_soft_json_model)  # 将sklearn模型和json模型进行同步
    # draw_file(model, feature_list, class_names, pdf_file='./results/soft_pruned_best_tree')
    print('**********Soft pruning (end)**********\n')

    # 保存软剪枝后的json模型
    print('**********Saving first soft pruned tree model (start)**********')
    json_file = './results/first_soft_pruned_tree.json'
    with open(json_file, 'w') as f:
        f.write(json.dumps(best_soft_json_model))
    print("The first soft pruned tree's json model is saved in", json_file)
    print('**********Saving first soft pruned tree model (end)**********\n')

    # 硬剪枝+第二次软剪枝
    print('**********Pruning (start)**********')
    # limit_depth_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38,
    #                     40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, int(1e10)]
    # limit_depth_list = [4, 12, 20, 28, 36]
    limit_depth_list = [4]
    for limit_depth in limit_depth_list:
        # 硬剪枝
        print('---limit_depth = %d---' % limit_depth)
        soft_pruned_json_tree = copy.deepcopy(best_soft_json_model)
        sklearn_model = copy.deepcopy(model)
        time_start = time.time()  # 记录开始时间
        best_hard_json_model = hard_prune(soft_pruned_json_tree, 0, limit_depth)
        time_end = time.time()  # 记录结束时间
        print('The time of hard prune = ' + str(time_end - time_start) + 's')  # 输出硬剪枝花费的时间
        print('---硬剪枝后的模型结构---')
        output_model_structure(best_hard_json_model)
        print('---硬剪枝后的训练精度---')
        TP, TN, FP, FN = get_leaves_confusion_matrix(best_hard_json_model)
        output_metrics(TP, TN, FP, FN)
        print('---硬剪枝后的测试精度---')
        output_testing_metrics(best_hard_json_model, x_test, y_test, feature_list, class_names)
        # print('---硬剪枝后的模型可视化---')
        # pruned_sklearn_model(sklearn_model.tree_, 0, best_hard_json_model)  # 将sklearn模型和json模型进行同步
        # draw_file(sklearn_model, feature_list, class_names, pdf_file='./results/hard_pruned_best_tree')

        # 第二次软剪枝
        time_start = time.time()  # 记录开始时间
        best_final_json_model = soft_prune(best_hard_json_model)
        time_end = time.time()  # 记录结束时间
        print('The time of soft prune = ' + str(time_end - time_start) + 's')  # 输出训练时间
    best_final_json_model = json_model
    print('---最终的模型结构---')
    output_model_structure(best_final_json_model)
    print('---最终的训练精度---')
    TP, TN, FP, FN = get_leaves_confusion_matrix(best_final_json_model)
    output_metrics(TP, TN, FP, FN)
    print('---最终的测试精度---')
    output_testing_metrics(best_final_json_model, x_test, y_test, feature_list, class_names)
    # print('---最终的模型可视化---')
    # pruned_sklearn_model(sklearn_model.tree_, 0, best_final_json_model)  # 将sklearn模型和json模型进行同步
    # draw_file(sklearn_model, feature_list, class_names, pdf_file='./results/final_best_tree_depth_11')
    print('**********Pruning (end)**********\n')
