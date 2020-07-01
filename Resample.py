
import numpy as np
import pandas as pd
import seaborn as sns
import timeit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import statistics
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import resample
dataset = pd.read_excel('Mesothelioma.xlsx')
dataset['class of diagnosis'].value_counts()
major = dataset[dataset['class of diagnosis']==1]
minor = dataset[dataset['class of diagnosis']==2]
minor['class of diagnosis'] = minor['class of diagnosis'].replace(2, 1)
major['class of diagnosis'] = major['class of diagnosis'].replace(1, 0)
minor_sampled = resample(minor, replace=True, n_samples=228, random_state=100) 
sns.heatmap(dataset.corr())
minor_sampled = pd.concat([major, minor_sampled])
minor_sampled = minor_sampled.drop(minor_sampled.columns[6], axis=1)
minor_sampled.to_csv('datab.csv', index = False)
data = pd.read_csv('datab.csv')
data.columns

X = data.iloc[:,:-1] 
y = data.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, 
                                                    random_state = 42,)

####################Random Forest#################################

from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators = 200, random_state = 100,
                                   criterion="gini", max_depth=None, 
                                   min_samples_split=2, bootstrap=False,)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])

specificity = cm[1,1]/(cm[1,0]+cm[1,1])

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)

from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_test, y_pred)

print(ac)
print(mcc)

########################KF ON RANDOM FOREST#################

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import statistics
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

rf = []
f1 = []
auc = []
mcc = []

if __name__ == '__main__':

    kf = KFold(n_splits=10, shuffle = True, random_state = 42)

    predicted_y = []
    expected_y = [] 
    X = pd.DataFrame(X)

    for train_index, test_index in kf.split(X, y):

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier = RandomForestClassifier(n_estimators = 150, random_state = 42,
                                            criterion="gini", max_depth=None,min_samples_split=2,
                                            bootstrap=False,)
        classifier.fit(X_train, y_train)

        predicted_y.extend(classifier.predict(X_test))

        expected_y.extend(y_test)
        rf.append(metrics.accuracy_score(expected_y, predicted_y))
        accuracy = metrics.accuracy_score(expected_y, predicted_y)
        
        f1.append(f1_score( expected_y,  predicted_y))
        auc.append(roc_auc_score(expected_y,  predicted_y))
        mcc.append(matthews_corrcoef( expected_y,  predicted_y))
                
    print("Accuracy: " + statistics.mean(rf).__str__())
    print("AUC: " + statistics.mean(auc).__str__())
    

####################Support Vector########################

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold

regressor = SVC(kernel = 'rbf',random_state = 10)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
specificity = cm[1,1]/(cm[1,0]+cm[1,1])

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)

from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_test, y_pred)
print(ac)
print(mcc)

        
#####################kf on Support Vector########################

rf = []
f1 = []
auc = []
mcc = []

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.svm import SVC
if __name__ == '__main__':
    kf = KFold(n_splits=10, shuffle = True, random_state = 42)
    predicted_y = []
    expected_y = [] 
    X = pd.DataFrame(X)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = SVC(kernel = 'rbf', random_state = 10)
        classifier.fit(X_train, y_train)
        predicted_y.extend(classifier.predict(X_test))
        expected_y.extend(y_test)
        
        rf.append(metrics.accuracy_score(expected_y, predicted_y))
        accuracy = metrics.accuracy_score(expected_y, predicted_y)
        
        f1.append(f1_score( expected_y,  predicted_y))
        auc.append(roc_auc_score(expected_y,  predicted_y))
        mcc.append(matthews_corrcoef( expected_y,  predicted_y))
                
    print("Accuracy: " + statistics.mean(rf).__str__())
    print("AUC: " + statistics.mean(auc).__str__())
    print("f1: " + statistics.mean(f1).__str__())
    print("MCCC: " + statistics.mean(mcc).__str__())

      
###########################Decision Tree#######################


from sklearn.tree import DecisionTreeClassifier
regressor = DecisionTreeClassifier( criterion = "entropy", random_state = 42,)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])

specificity = cm[1,1]/(cm[1,0]+cm[1,1])

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)

from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_test, y_pred)
print(ac)
print(mcc)

###########################KF on Decision Tree#######################

rf = []
f1 = []
auc = []
mcc = []

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
if __name__ == '__main__':

    kf = KFold(n_splits=10, shuffle = True, random_state = 42)

    predicted_y = []
    expected_y = [] 
    X = pd.DataFrame(X)

    for train_index, test_index in kf.split(X, y):

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regressor = DecisionTreeClassifier( criterion = "entropy", random_state = 42,)
        regressor.fit(X_train, y_train)
        
        predicted_y.extend(regressor.predict(X_test))
        expected_y.extend(y_test)

        rf.append(metrics.accuracy_score(expected_y, predicted_y))
        accuracy = metrics.accuracy_score(expected_y, predicted_y)
        
        f1.append(f1_score( expected_y,  predicted_y))
        auc.append(roc_auc_score(expected_y,  predicted_y))
        mcc.append(matthews_corrcoef( expected_y,  predicted_y))
                
    print("Accuracy: " + statistics.mean(rf).__str__())
    print("AUC: " + statistics.mean(auc).__str__())
    print("f1: " + statistics.mean(f1).__str__())
    print("MCCC: " + statistics.mean(mcc).__str__())
    
###########################KF on KNN#######################

rf = []
f1 = []
auc = []
mcc = []

from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
from sklearn.model_selection import KFold
if __name__ == '__main__':

    kf = KFold(n_splits=10, shuffle = True, random_state = 42)

    predicted_y = []
    expected_y = [] 
    X = pd.DataFrame(X)

    for train_index, test_index in kf.split(X, y):

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regressor = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train)
        regressor.fit(X_train, y_train)
        
        predicted_y.extend(regressor.predict(X_test))
        expected_y.extend(y_test)

        rf.append(metrics.accuracy_score(expected_y, predicted_y))
        accuracy = metrics.accuracy_score(expected_y, predicted_y)
        
        f1.append(f1_score( expected_y,  predicted_y))
        auc.append(roc_auc_score(expected_y,  predicted_y))
        mcc.append(matthews_corrcoef( expected_y,  predicted_y))
                
              
    print("Accuracy: " + statistics.mean(rf).__str__())
    print("AUC: " + statistics.mean(auc).__str__())
    print("f1: " + statistics.mean(f1).__str__())
    print("MCCC: " + statistics.mean(mcc).__str__())
    
#############################Naive Bayes##########################


from sklearn.naive_bayes import GaussianNB
regressor = GaussianNB( )

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])

specificity = cm[1,1]/(cm[1,0]+cm[1,1])

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)

from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_test, y_pred)


#############################KF ON Naive Bayes##########################

rf = []
f1 = []
auc = []
mcc = []

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import KFold
if __name__ == '__main__':

    kf = KFold(n_splits=10, shuffle = True, random_state = 42)

    predicted_y = []
    expected_y = [] 
    X = pd.DataFrame(X)

    for train_index, test_index in kf.split(X, y):

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        regressor = GaussianNB( )
        regressor.fit(X_train, y_train)
        
        predicted_y.extend(regressor.predict(X_test))
        expected_y.extend(y_test)

        rf.append(metrics.accuracy_score(expected_y, predicted_y))
        accuracy = metrics.accuracy_score(expected_y, predicted_y)
        
        f1.append(f1_score( expected_y,  predicted_y))
        auc.append(roc_auc_score(expected_y,  predicted_y))
        mcc.append(matthews_corrcoef( expected_y,  predicted_y))
                
              
    print("Accuracy: " + statistics.mean(rf).__str__())
    print("AUC: " + statistics.mean(auc).__str__())
    print("f1: " + statistics.mean(f1).__str__())
    print("MCCC: " + statistics.mean(mcc).__str__())


############################ensemble#############################

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


log_clf = LogisticRegression(solver="liblinear", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma="auto", random_state=42)


voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')


voting_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])

specificity = cm[1,1]/(cm[1,0]+cm[1,1])

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)

from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_test, y_pred)
print(ac)
print(mcc)

############################KF ON ensemble_1#############################

rf = []
f1 = []
auc = []
mcc = []

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import metrics

log_clf = LogisticRegression(solver="liblinear", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=150, random_state=42)
svm_clf = SVC(gamma="auto", random_state=42)


kf = KFold(n_splits=10, shuffle = True, random_state = 42)
predicted_y = []
expected_y = [] 
X = pd.DataFrame(X)
for train_index, test_index in kf.split(X, y):

    X_train_pca, X_test_pca = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    classifier =VotingClassifier(
                estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
                voting='hard')
    classifier.fit(X_train_pca, y_train)
    predicted_y.extend(classifier.predict(X_test_pca))
    expected_y.extend(y_test)
    rf.append(metrics.accuracy_score(expected_y, predicted_y))
    accuracy = metrics.accuracy_score(expected_y, predicted_y)
    
    f1.append(f1_score( expected_y,  predicted_y))
    auc.append(roc_auc_score(expected_y,  predicted_y))
    mcc.append(matthews_corrcoef( expected_y,  predicted_y))
            
print("Accuracy: " + statistics.mean(rf).__str__())
    
from sklearn.metrics import accuracy_score
ac = []
for clf in (log_clf, rnd_clf, svm_clf, classifier):
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    print(accuracy_score(y_pred, y_test))
   
from sklearn.metrics import confusion_matrix
cm = confusion_matrix( expected_y,  predicted_y)

from sklearn.metrics import accuracy_score
ac = accuracy_score( expected_y,  predicted_y)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])

specificity = cm[1,1]/(cm[1,0]+cm[1,1])

from sklearn.metrics import f1_score
f1 = f1_score( expected_y,  predicted_y)

from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef( expected_y,  predicted_y)
print(ac)
print(mcc)



############################ensemble_2#############################


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

gbc_clf = GradientBoostingClassifier(n_estimators=200, random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
dt_clf = DecisionTreeClassifier(criterion = "entropy", random_state = 42,)
log_clf = LogisticRegression(solver="liblinear", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma="auto", random_state=42)

voting_clf = VotingClassifier(
    estimators=[('svm', svm_clf), ('rf', rnd_clf), ('gbc', gbc_clf)],
    voting='hard')

voting_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

for clf in (gbc_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])

specificity = cm[1,1]/(cm[1,0]+cm[1,1])

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)

from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_test, y_pred)
print(ac)
print(mcc)

####################KF ON ENSEMBLE_2####################

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.svm import SVC

rf = []
f1 = []
auc = []
mcc = []

gbt_clf = GradientBoostingClassifier(max_depth=None, n_estimators=100, learning_rate=1.0, random_state=42)
rnd_clf = RandomForestClassifier(n_estimators = 200)
svm_clf = SVC(gamma="auto", random_state=42)


if __name__ == '__main__':

    kf = KFold(n_splits=8, shuffle = True, random_state = 42)

    predicted_y = []
    expected_y = [] 
    X = pd.DataFrame(X)

    for train_index, test_index in kf.split(X, y):

        X_train_pca, X_test_pca = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        classifier = VotingClassifier(
            estimators=[('gbt', gbt_clf), ('rf', rnd_clf), ('svc', svm_clf)],
            voting='hard')
        
        classifier.fit(X_train_pca, y_train)
        predicted_y.extend(classifier.predict(X_test_pca))
        expected_y.extend(y_test)
        rf.append(metrics.accuracy_score(expected_y, predicted_y))
        accuracy = metrics.accuracy_score(expected_y, predicted_y)
        
        f1.append(f1_score( expected_y,  predicted_y))
        auc.append(roc_auc_score(expected_y,  predicted_y))
        mcc.append(matthews_corrcoef( expected_y,  predicted_y))
                
    print("Accuracy: " + statistics.mean(rf).__str__())
    

from sklearn.metrics import accuracy_score

ac = []
for clf in (gbt_clf, rnd_clf, svm_clf, classifier):
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    acc=accuracy_score(y_pred, y_test)
    print(accuracy_score(y_pred, y_test))
    
       
from sklearn.metrics import confusion_matrix
cm = confusion_matrix( expected_y,  predicted_y)

from sklearn.metrics import accuracy_score
ac = accuracy_score( expected_y,  predicted_y)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])

specificity = cm[1,1]/(cm[1,0]+cm[1,1])

from sklearn.metrics import f1_score
f1 = f1_score( expected_y,  predicted_y)

from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef( expected_y,  predicted_y)
print(ac)
print(mcc)

########################GBRT###########################
    
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(max_depth=None, n_estimators=100, learning_rate=1.0, random_state=42)
gbrt.fit(X_train, y_train)

gbrt_slow = GradientBoostingClassifier(max_depth= None, n_estimators=100, learning_rate=0.001, random_state=42)
gbrt_slow.fit(X_train, y_train)

y_pred_gbrt = gbrt.predict(X_test)
y_pred = gbrt_slow.predict(X_test)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred_gbrt))
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm1 =confusion_matrix(y_test, y_pred_gbrt)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
ac1 = accuracy_score(y_test, y_pred_gbrt)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])

specificity = cm[1,1]/(cm[1,0]+cm[1,1])
specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])

from sklearn.metrics import f1_score
f = f1_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred_gbrt)

from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_test, y_pred)
mcc1=matthews_corrcoef(y_test, y_pred_gbrt)

########################KF on GBRT###########################
    

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn import metrics

rf = []
f1 = []
auc = []
mcc = []

if __name__ == '__main__':

    kf = KFold(n_splits=8, shuffle = True, random_state = 42)

    predicted_y = []
    expected_y = [] 
    X = pd.DataFrame(X)

    for train_index, test_index in kf.split(X, y):

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        gbrt_slow = GradientBoostingClassifier(max_depth= None, n_estimators=200, learning_rate=1, random_state=42)
        gbrt_slow.fit(X_train, y_train)
        predicted_y.extend(gbrt_slow.predict(X_test))
        expected_y.extend(y_test)
        rf.append(metrics.accuracy_score(expected_y, predicted_y))
        accuracy = metrics.accuracy_score(expected_y, predicted_y)
        
        f1.append(f1_score( expected_y,  predicted_y))
        auc.append(roc_auc_score(expected_y,  predicted_y))
        mcc.append(matthews_corrcoef( expected_y,  predicted_y))
                
    print("Accuracy: " + statistics.mean(rf).__str__())
    print("AUC: " + statistics.mean(auc).__str__())

   
from sklearn.metrics import confusion_matrix
cm = confusion_matrix( expected_y,  predicted_y)

from sklearn.metrics import accuracy_score
ac = accuracy_score( expected_y,  predicted_y)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])

specificity = cm[1,1]/(cm[1,0]+cm[1,1])

from sklearn.metrics import f1_score
f1 = f1_score( expected_y,  predicted_y)

from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef( expected_y,  predicted_y)

############################Stacking#########################

from sklearn.linear_model import SGDClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.classification import accuracy_score, log_loss

clf1 = SGDClassifier(alpha=0.001, penalty='l2', loss='log', class_weight='balanced', random_state=42)
clf1.fit(X_train, y_train)

clf2 = SGDClassifier(alpha=1, penalty='l2', loss='hinge', class_weight='balanced', random_state=42)
clf2.fit(X_train, y_train)

clf3 = RandomForestClassifier(n_estimators = 100, random_state = 42, criterion="gini", max_depth=None, 
                                   min_samples_split=2, bootstrap=True,)
clf3.fit(X_train, y_train)

sig_claccuracy = CalibratedClassifierCV(clf1, method="sigmoid")
sig_clf2 = CalibratedClassifierCV(clf2, method="sigmoid")
sig_clf3 = CalibratedClassifierCV(clf3, method="sigmoid")
sclaccuracy = StackingCVClassifier(classifiers = [sig_claccuracy, sig_clf2, sig_clf3],
                    shuffle = True, cv=10,
                    meta_classifier = LogisticRegression())
sclaccuracy.fit(X_train, y_train)  
y_pred = sclaccuracy.predict(X_test)
ac= accuracy_score(y_pred, y_test)
print(ac)       

######################ANN########################

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', 
                     activation = 'relu', input_dim = 33))
#
#classifier.add(Dense(units = 10, kernel_initializer = 'uniform',activation = 'sigmoid'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))

classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
y_pred = classifier.predict(X_test)
y_pred = np.round(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
ac_ann = accuracy_score(y_pred, y_test)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
specificity = cm[1,1]/(cm[1,0]+cm[1,1])

from sklearn.metrics import f1_score
f1 = f1_score( y_test, y_pred)

from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef( y_test, y_pred)

########################8 Fold KF ON ANN#################

from sklearn import metrics
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense

rf = []
f1 = []
auc = []
mcc = []

classifier = Sequential()
if __name__ == '__main__':
    kf = KFold(n_splits=10, shuffle = True, random_state = 42)
    predicted_y = []
    expected_y = [] 
    X = pd.DataFrame(X)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.add(Dense(units = 100, kernel_initializer = 'uniform', 
                     activation = 'relu', input_dim = 33))
#        classifier.add(Dense(units = 10, kernel_initializer = 'uniform', 
#                   activation = 'relu'))
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))
        classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])
                        
        classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
        predicted_y.extend(np.round(classifier.predict(X_test)))
        expected_y.extend(y_test)
        rf.append(metrics.accuracy_score(expected_y, predicted_y))
        accuracy = metrics.accuracy_score(expected_y, predicted_y)
        
        f1.append(f1_score( expected_y,  predicted_y))
        auc.append(roc_auc_score(expected_y,  predicted_y))
        mcc.append(matthews_corrcoef( expected_y,  predicted_y))
                
    print("Accuracy: " + statistics.mean(rf).__str__())
from sklearn.metrics import confusion_matrix
cm = confusion_matrix( expected_y,  predicted_y)

from sklearn.metrics import accuracy_score
ac = accuracy_score( expected_y,  predicted_y)

sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
specificity = cm[1,1]/(cm[1,0]+cm[1,1])

from sklearn.metrics import f1_score
f1 = f1_score( expected_y,  predicted_y)

from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef( expected_y,  predicted_y)

#########################10 Fold KF ON ANN###################

from sklearn import metrics
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense

rf = []
f1 = []
auc = []
mcc = []

classifier = Sequential()
if __name__ == '__main__':
    kf = KFold(n_splits=10, shuffle = True, random_state = 42)
    predicted_y = []
    expected_y = [] 
    X = pd.DataFrame(X)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.add(Dense(units = 100, kernel_initializer = 'uniform', 
                     activation = 'relu', input_dim = 33))
#        classifier.add(Dense(units = 10, kernel_initializer = 'uniform', 
#                   activation = 'relu'))
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))
        classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])
                        
        classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
        predicted_y.extend(np.round(classifier.predict(X_test)))
        expected_y.extend(y_test)
        rf.append(metrics.accuracy_score(expected_y, predicted_y))
        accuracy = metrics.accuracy_score(expected_y, predicted_y)
        
        f1.append(f1_score( expected_y,  predicted_y))
        auc.append(roc_auc_score(expected_y,  predicted_y))
        mcc.append(matthews_corrcoef( expected_y,  predicted_y))
                
    print("Accuracy: " + statistics.mean(rf).__str__())
    print("AUC: " + statistics.mean(auc).__str__())
    print("f1: " + statistics.mean(f1).__str__())
    print("MCCC: " + statistics.mean(mcc).__str__())
    






































