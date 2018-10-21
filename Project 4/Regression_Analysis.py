import os
import numpy as np
import matplotlib.pyplot as plt
from util import load_data_network
from sklearn.model_selection import KFold
import string
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsRegressor


from sklearn.ensemble import RandomForestRegressor

"""
def linear_regression(features_scalar, labels_scalar):
    linear_rmse_test_set = []
    linear_rmse_train_set = []
    kf = KFold(n_splits=10, random_state=0)

    for train_index, test_index in kf.split(features_scalar):
        features_scalar_train, labels_scalar_train = features_scalar[train_index], labels_scalar[train_index]
        features_scalar_test, labels_scalar_test = features_scalar[test_index], labels_scalar[test_index]
        linear_model = LinearRegression()
        linear_model.fit(features_scalar_train, labels_scalar_train)
        # testset
        labels_test_predict = linear_model.predict(features_scalar_test)
        linear_rmse_test = np.sqrt(metrics.mean_squared_error(labels_scalar_test, labels_test_predict))
        # trainset
        labels_train_predict = linear_model.predict(features_scalar_train)
        linear_rmse_train = np.sqrt(metrics.mean_squared_error(labels_scalar_train, labels_train_predict))

        linear_rmse_test_set.append(linear_rmse_test)
        linear_rmse_train_set.append(linear_rmse_train)

    return linear_rmse_test_set, linear_rmse_train_set
"""

def random_forest(features, labels, num_trees, num_features, max_depth):
    rf_rmse_train_set = []
    rf_rmse_test_set = []
    rf_feature_importance_set = []
    rf_predict = np.array([])
    rf_ground_truth = np.array([])
    kf = KFold(n_splits=10, random_state = 0)

    for train_index, test_index in kf.split(features):
        features_train, labels_train = features[train_index], labels[train_index]
        features_test, labels_test = features[test_index], labels[test_index]
        rf = RandomForestRegressor(n_estimators=num_trees, max_depth=max_depth, max_features=num_features, oob_score=True)
        rf.fit(features_train, labels_train)
        #trainset
        labels_train_predict = rf.predict(features_train)
        rf_rmse_train_set.append(np.sqrt(metrics.mean_squared_error(labels_train, labels_train_predict)))
        #testset
        labels_test_predict = rf.predict(features_test)
        labels_test = labels_test.transpose()
        labels_test = labels_test[0]
        rf_predict = np.concatenate((rf_predict, labels_test_predict))
        rf_ground_truth = np.concatenate((rf_ground_truth, labels_test))
        rf_rmse_test_set.append(np.sqrt(metrics.mean_squared_error(labels_test, labels_test_predict)))
        rf_feature_importance_set.append(rf.feature_importances_)
    plt.scatter(rf_ground_truth, rf_predict)
    plt.xlabel('true values')
    plt.ylabel('fitted values')
    plt.show()
    plt.scatter(rf_predict, rf_predict-rf_ground_truth)
    plt.xlabel('fitted values')
    plt.ylabel('residual values')
    plt.show()
    feature_names = ["week", "day_of_week", "backup_start_time", "workflow_id", "filename"]
    export_graphviz(rf.estimators_[0], feature_names=feature_names)
    os.system('dot -Tpng tree.dot -o tree.png')
    print "feature importances"
    print sum(rf_feature_importance_set)/10.0
    return sum(rf_rmse_test_set)/10.0, sum(rf_rmse_train_set)/10.0, 1-rf.oob_score_


def knn(features_scalar, labels_scalar, n_neighbor):
    linear_rmse_test_set = 0
    linear_rmse_train_set = 0
    kf = KFold(n_splits=10, random_state=0)

    for train_index, test_index in kf.split(features_scalar):
        features_scalar_train, labels_scalar_train = features_scalar[train_index], labels_scalar[train_index]
        features_scalar_test, labels_scalar_test = features_scalar[test_index], labels_scalar[test_index]
        k = KNeighborsRegressor(n_neighbors=n_neighbor)
        k.fit(features_scalar_train, labels_scalar_train)
        # testset
        labels_test_predict = k.predict(features_scalar_test)
        linear_rmse_test = metrics.mean_squared_error(labels_scalar_test, labels_test_predict)
        # trainset
        labels_train_predict = k.predict(features_scalar_train)
        linear_rmse_train = metrics.mean_squared_error(labels_scalar_train, labels_train_predict)

        linear_rmse_test_set+=linear_rmse_test
        linear_rmse_train_set+=linear_rmse_train

    return np.sqrt(linear_rmse_test_set/10.0), np.sqrt(linear_rmse_train_set/10.0)
    """
    plt.scatter(knn_ground_truth, knn_predict)
    plt.xlabel('true values')
    plt.ylabel('fitted values')
    plt.show()
    plt.scatter(knn_predict, knn_predict - knn_ground_truth)
    plt.xlabel('fitted values')
    plt.ylabel('residual values')
    plt.show()
    """

    return sum(knn_rmse_test_set) / 10.0, sum(knn_rmse_train_set) / 10.0

def main():
    dataset = load_data_network('network_backup_dataset.csv')
    week = dataset.week
    day_of_week = dataset.day_of_week
    backup_start_time = dataset.backup_start_time
    workflow_id = dataset.workflow_id
    filename = dataset.file_name
    size_of_backup = dataset.size_of_backup
    backup_time = dataset.backup_time
    """
    # data_list=[week, day_of_week, backup_start_time, workflow_id,filename, size_of_backup, backup_time]
    workflow_list = ['work_flow_0', 'work_flow_1', 'work_flow_2', 'work_flow_3', 'work_flow_4']
    week_list = [i + 1 for i in range(15)]
    day_of_week_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    workflow_count = [[0 for i in range(20)] for i in range(5)]
    i = 0
    while (week[i] - 1) * 7 + day_of_week_list.index(day_of_week[i]) + 1 <= 20:
        workflow_count[workflow_list.index(workflow_id[i])][
            (week[i] - 1) * 7 + day_of_week_list.index(day_of_week[i])] += size_of_backup[i]
        i += 1
    for i in range(5):
        plt.plot([j + 1 for j in range(20)], workflow_count[i], label=workflow_list[i])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    # plt.show()

    workflow_count = [[0 for i in range(105)] for i in range(5)]
    i = 0
    while i < len(week):
        workflow_count[workflow_list.index(workflow_id[i])][
            (week[i] - 1) * 7 + day_of_week_list.index(day_of_week[i])] += size_of_backup[i]
        i += 1
    for i in range(5):
        plt.plot([j + 1 for j in range(105)], workflow_count[i], label=workflow_list[i])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    # plt.show()
    """
    # question a: Linear regression

    # part 1: scalarize the variable(all the scalar variables will have a subfix of "scalar)

    #####################
    #### dictionary #####
    #####################
    day_of_week_dict = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6,
                        "Sunday": 7}
    workflow_id_dict = {'work_flow_0': 1, 'work_flow_1': 2, 'work_flow_2': 3, 'work_flow_3': 4, 'work_flow_4': 5}

    # scalar variables ##
    #####################
    week_scalar = week
    day_of_week_scalar = []
    backup_start_time_scalar = backup_start_time
    workflow_id_scalar = []
    filename_scalar = []

    for day in day_of_week:
        day_of_week_scalar.append(day_of_week_dict[day])

    for ID in workflow_id:
        workflow_id_scalar.append(workflow_id_dict[ID])

    def filename_remove_string(filename):
        return int(string.replace(filename, "File_", "")) + 1

    for name in filename:
        filename_scalar.append(filename_remove_string(name))

    ##   directly use linear regression   ##
    ########################################
    features_scalar = np.array(
        [week_scalar, day_of_week_scalar, backup_start_time_scalar, workflow_id_scalar, filename_scalar])
    features_scalar = features_scalar.transpose()
    labels_scalar = np.array([size_of_backup])
    labels_scalar = labels_scalar.transpose()
    """
    linear_rmse_test_set, linear_rmse_train_set = linear_regression(features_scalar, labels_scalar)
    
    ##   standardize   ##
    #####################
    standard = StandardScaler()
    result = standard.fit_transform(zip(week_scalar, labels_scalar))
    week_scalar_standard = result[:, 0]
    result = standard.fit_transform(zip(day_of_week_scalar, labels_scalar))
    day_of_week_scalar_standard = result[:, 0]
    result = standard.fit_transform(zip(backup_start_time_scalar, labels_scalar))
    backup_start_time_scalar_standard = result[:, 0]
    result = standard.fit_transform(zip(workflow_id_scalar, labels_scalar))
    workflow_id_scalar_standard = result[:, 0]
    result = standard.fit_transform(zip(filename_scalar, labels_scalar))
    filename_scalar_standard = result[:, 0]

    features_scalar = np.array([week_scalar_standard, day_of_week_scalar_standard, backup_start_time_scalar_standard,
                                workflow_id_scalar_standard, filename_scalar_standard])
    features_scalar = features_scalar.transpose()
    linear_rmse_test_set, linear_rmse_train_set = linear_regression(features_scalar, labels_scalar)

    #  three features  ##
    #####################

    Fval, pval = f_regression(features_scalar, labels_scalar)
    mval = mutual_info_regression(features_scalar, labels_scalar)
    print(Fval)
    print(mval)

    # 32 combinations of features ##
    ################################
    for i in range(1, 32):
        features_scalar = []
        if (i & 1):
            features_scalar.append(week_scalar_standard)
        if (i >> 1 & 1):
            features_scalar.append(day_of_week_scalar_standard)
        if (i >> 2 & 1):
            features_scalar.append(backup_start_time_scalar_standard)
        if (i >> 3 & 1):
            features_scalar.append(workflow_id_scalar_standard)
        if (i >> 4 & 1):
            features_scalar.append(filename_scalar_standard)

        features_scalar = list(zip(*features_scalar))
        features_scalar = np.array(features_scalar)
        linear_rmse_test_set, linear_rmse_train_set = linear_regression(features_scalar, labels_scalar)
        print(linear_rmse_test_set)
        print(linear_rmse_train_set)
    """
    


    """
    # Question b: Random Forest Method

    # subquestion i:
    print 'Test RMSE, Train RMSE, OOB error: '
    print random_forest(features=features_scalar, labels=labels_scalar, num_trees=20, num_features=5, max_depth=4)

    # subquestion ii:
    ## first is oob error
    tree_range = range(1, 201)
    feature_range = range(1, 6)
    for num_feature in feature_range:
        feature_oob_error = []
        for num_tree in tree_range:
            rmse_test, rmse_train, oob_error = random_forest(features=features_scalar, labels=labels_scalar, num_trees=num_tree, num_features=num_feature, max_depth=4)
            feature_oob_error.append(oob_error)
        plt.plot(tree_range, feature_oob_error, label=str(num_feature))
    plt.title("OOB error")
    plt.legend(loc='lower right')
    plt.show()
    
    ## second is test rmse error
    tree_range = range(1, 201)
    feature_range = range(1, 6)
    for num_feature in feature_range:
        feature_test_rmse_error = []
        for num_tree in tree_range:
            rmse_test, rmse_train, oob_error = random_forest(features=features_scalar, labels=labels_scalar,
                                                             num_trees=num_tree, num_features=num_feature, max_depth=4)
            feature_test_rmse_error.append(rmse_test)
        plt.plot(tree_range, feature_test_rmse_error, label=str(num_feature))
    plt.title("Test RMSE error")
    plt.legend(loc='lower right')
    plt.show()
    
    # subquestion iii:
    depth_range = range(1, 21)
    depth_oob_error = []
    depth_test_error = []
    for depth in depth_range:
        rmse_test, rmse_train, oob_error = random_forest(features=features_scalar, labels=labels_scalar,
                                                         num_trees=25, num_features=4, max_depth=depth)
        depth_oob_error.append(oob_error)
        depth_test_error.append(rmse_test)
    plt.plot(depth_range, depth_oob_error)
    plt.title("OOB Error")
    plt.show()

    plt.plot(depth_range, depth_test_error)
    plt.title("Test RMSE Error")
    plt.show()
    """
    
    # subquestion iv:
    #random_forest(features=features_scalar, labels=labels_scalar, num_trees=25, num_features=4, max_depth=4)

    # question e knn regression
    neighbors_range = range(1, 51)
    feature_train_error = []
    feature_test_error = []
    for neighbor in neighbors_range:
        rmse_test, rmse_train = knn(features_scalar=features_scalar, labels_scalar=labels_scalar, n_neighbor=neighbor)
        feature_test_error.append(rmse_test)
        feature_train_error.append(rmse_train)

    plt.plot(neighbors_range, feature_test_error, label="Test Error")
    plt.plot(neighbors_range, feature_train_error, label="Train Error")
    plt.legend(loc='lower right')
    plt.show()

    knn(features_scalar=features_scalar, labels_scalar=labels_scalar, n_neighbor=1)







if __name__ == "__main__":
    main()