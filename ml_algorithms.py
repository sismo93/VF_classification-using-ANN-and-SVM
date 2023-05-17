from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


from sklearn.model_selection import cross_val_score,StratifiedKFold

from sklearn.svm import SVC

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix,precision_recall_curve, RocCurveDisplay

from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, KFold






import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

def SVM_Grid_search(gammas, Cs, X, y, output_path,name):
    list_accuracy = []
    list_train_sizes = []
    list_train_scores_mean = []
    list_train_scores_std = []
    list_test_scores_mean = []
    list_test_scores_std = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    for gamma in gammas:
        for C in Cs:
            clf = SVC(kernel='rbf', gamma=gamma, C=C, probability=True)
            clf.fit(X_train, y_train)
            Y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, Y_pred)
            y_pred_proba = clf.predict_proba(X_test)[::,1]
            classification_rep = classification_report(y_test, Y_pred, target_names=["VF", "NotVF"])
            print(classification_rep)
            
            # Save classification report to a file
            with open(f"{output_path}/{name}_svm_classification_report_gamma={gamma}_C={C}.txt", "w") as f:
                f.write(classification_rep)
                
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label='VF')
            roc_auc = auc(fpr, tpr)*100
            
            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f%%)' % roc_auc)            
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC curve (gamma={gamma}, C={C})')
            plt.legend(loc="lower right")
            
            # Save ROC plot to a file
            plt.savefig(f"{output_path}/{name}_svm_roc_gamma={gamma}_C={C}.png")
            plt.close()
            
            accuracy = float(cm.diagonal().sum()) / len(y_test)
            print("Accuracy of SVM for the given dataset:", accuracy)
            list_accuracy.append(accuracy)

            # Plot confusion matrix using heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["VF", "NotVF"], yticklabels=["VF", "NotVF"])
            plt.title(f"Confusion Matrix (gamma={gamma}, C={C})")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            
            # Save confusion matrix plot to a file
            plt.savefig(f"{output_path}/{name}_SVM_confusion_matrix_gamma={gamma}_C={C}.png")
            plt.close()

         



"""
def SVM_Grid_search(gammas, Cs, X, y, output_path):
    list_accuracy = []
    list_train_sizes = []
    list_train_scores_mean = []
    list_train_scores_std = []
    list_test_scores_mean = []
    list_test_scores_std = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    for gamma in gammas:
        for C in Cs:
            clf = SVC(kernel='rbf', gamma=gamma, C=C, probability=True)
            clf.fit(X_train, y_train)
            Y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, Y_pred)
            y_pred_proba = clf.predict_proba(X_test)[::,1]
            print(classification_report(y_test, Y_pred, target_names=["VF", "NotVF"]))
            accuracy = float(cm.diagonal().sum()) / len(y_test)
            print("Accuracy of SVM for the given dataset:", accuracy)
            list_accuracy.append(accuracy)

            # Plot confusion matrix using heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["VF", "NotVF"], yticklabels=["VF", "NotVF"])
            plt.title(f"Confusion Matrix (gamma={gamma}, C={C})")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.savefig(f"{output_path}/confusion_matrix_gamma={gamma}_C={C}.png")
            plt.close()

            # Plot learning curves for detecting overfitting and underfitting
            train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5, train_sizes=np.linspace(.1, 1.0, 5))
            list_train_sizes.append(train_sizes)
            list_train_scores_mean.append(np.mean(train_scores, axis=1))
            list_train_scores_std.append(np.std(train_scores, axis=1))
            list_test_scores_mean.append(np.mean(test_scores, axis=1))
            list_test_scores_std.append(np.std(test_scores, axis=1))
            plt.figure(figsize=(8, 6))
            plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training score")
            plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Cross-validation score")
            plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                             np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1, color="r")
            plt.fill_between(train_sizes, np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                             np.mean(test_scores, axis=1) + np.std(test_scores, axis=1), alpha=0.1, color="g")
            plt.title(f"Learning Curve (gamma={gamma}, C={C})")
            plt.xlabel("Training Examples")
            plt.ylabel("Score")
            plt.legend(loc="best")
            plt.savefig(f"{output_path}/learning_curve_gamma={gamma}_C={C}.png")


def SVM_Grid_search(gammas,Cs,X,y):

    list_accuracy=[] 

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=1) # get train test split
        
    for gamma in gammas:
        for C in Cs:
            clf = SVC(kernel='rbf', gamma=gamma, C=C,probability=True)
            clf.fit(X_train, y_train)
            Y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test,Y_pred)
            y_pred_proba = clf.predict_proba(X_test)[::,1]
            print(classification_report(y_test, Y_pred, target_names=["VF","NotVF"]))
            accuracy = float(cm.diagonal().sum())/len(y_test)
            print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)
            list_accuracy.append(accuracy)

"""

def SVM_Cross_val(gammas,Cs,X,y):

    for gamma in gammas:
        for C in Cs:
        
            model=SVC(kernel='rbf', gamma=gamma,C=C,probability=True,random_state = np.random.RandomState(0))
            kfold=StratifiedKFold(n_splits=3)
            score=cross_val_score(model,X,y,cv=kfold,scoring='accuracy')
            print("Cross Validation score are: {}".format(score))
            print("Average Cross score :{}".format(score.mean()))







def create_model(neuron1=4, neuron2=2, learning_rate=0.1, activation_function='relu', init='normal'):
    # Define the architecture of the neural network
    model = Sequential()
    model.add(Dense(neuron1, input_dim=X.shape[1], kernel_initializer=init, activation=activation_function))
    model.add(Dense(neuron2, kernel_initializer=init, activation=activation_function))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile the model
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def ANN_grid_search(X,y):
    # Create the model
    a = StandardScaler()
    a.fit(X)
    X_standardized = a.transform(X)
    model = KerasClassifier(build_fn = create_model, verbose = 0)

    # Define the grid search parameters
    batch_size = [10, 20, 40]
    epochs = [50, 100, 150]
    learning_rate = [0.001, 0.01, 0.1]
    activation_function = ['relu', 'sigmoid']
    init = ['normal', 'uniform']
    neuron1 = [4, 8, 16]
    neuron2 = [2, 4, 8]

    # Make a dictionary of the grid search parameters
    param_grids = dict(batch_size=batch_size, epochs=epochs, learning_rate=learning_rate,
                    activation_function=activation_function, init=init, neuron1=neuron1, neuron2=neuron2)

    # Build and fit the GridSearchCV
    grid = GridSearchCV(estimator=model, param_grid=param_grids, cv=KFold(), verbose=10)
    grid_result = grid.fit(X_standardized, y)

    # Summarize the results
    print('Best score: {}, using {}'.format(grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('{}, {} with: {}'.format(mean, stdev, param))