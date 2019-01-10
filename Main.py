import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from scipy.stats import pearsonr
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from tkinter import *



# df = pd.read_csv('C:\Users\www\Google 云端硬盘\Education\Personal Project\Moive Preference\Data\data1 .csv')

df = pd.read_csv("data2.csv")
# df = pd.read_csv("UBC_Grade.csv")

# renaming the column
def renameFeature(feature,to):
    df.rename(columns={feature: to}, inplace=True)

#renaming the columns
# renameFeature("Your Education Level","Education")
# df.rename(columns={'Your Nationality':"Nation"},inplace=True)
# df.rename(columns={'Income Level ':"Income"},inplace=True)
# df.rename(columns={'Are you religious? ':"Religion"},inplace=True)
# df.rename(columns={'Your Gender? ':"Gender"},inplace=True)
# df.rename(columns={'Moive1':"Movie1"},inplace=True)
# df.rename(columns={'What is one of your three most favourite movie ? (Movie #2) ':"Movie2"},inplace=True)
# df.rename(columns={'What Is one of your most favourite movie?  (Movie #3)':"Movie3"},inplace=True)
# df.rename(columns={"Among the three most favourite movie you just mentioned, which one rank the first or which one you like the most? ":"BestMovie"},inplace=True)
# df.rename(columns={"Where were you when you watch the movie that you like the most among the three? ":"Location"},inplace=True)
# df.rename(columns={"What's the genre of the movie that you like the most among the three? ":"Genre"},inplace=True)
# df.rename(columns={"What age did you watch  the movie that you like the most among the three? ":"Age"},inplace=True)
# df.rename(columns={"What was your mood before you watch the movie that you like the most among the three? ":"MoodBefore"},inplace=True)
# df.rename(columns={"What was your mood after you watch the movie that you like the most? ":"MoodBefore"},inplace=True)
# df.rename(columns={"Artistic, like to have novel experience, like adventure and travel, curious, creative":"O_Personality"},inplace=True)
# df.rename(columns={"Always prepared, pay attention to details, organized, follow schedule, finish the tasks ":"C_Personality"},inplace=True)
# df.rename(columns={"like party, don't mind being the center of attention, like to start conversations, talkitive":"E_Personality"},inplace=True)
# df.rename(columns={"easily disturbed, moody, irritated easily, upset easily, worry about things":"N_Personality"},inplace=True)
# df.rename(columns={"I am interested in people, sympathize with others' feelings, have soft heart, take time out for others, don't like conflict":"A_Personality"},inplace=True)

# print(df.head())  # print the first 5 rows
# print(df["Genre"])  # print specific columns
# print(df.columns) # print all the columns' name
# df.to_csv('data2.csv') #saving a new csv file


# ====================================== correlation =====================================================
# calculate the correlation between two variable
def correlation(a,b):
    # data = df[[a,b]]
    # correlation = data.corr(method='pearson')
    r,pvalue = pearsonr(df[a],df[b])
    print('correlation of',a,'and',b,'is',r,'pvalue is',pvalue)



# ===================================== t-test ==========================================
def t_test(feature,independentVariable1,independentVariable2,dependentVariable):

    x = df[df[feature]==independentVariable1] # select category
    y = df[df[feature]==independentVariable2]
    x_mean = x[dependentVariable].mean()
    y_mean = y[dependentVariable].mean()
    t_value, pvalue = ttest_ind(x[dependentVariable], y[dependentVariable])

    print(independentVariable1,'mean:',x_mean,"                ",independentVariable2,'mean:',y_mean)
    print("mean comparison test",independentVariable1,independentVariable2,'|| t-value:',t_value,'p-value:',pvalue)




# getting the data for personality and labels===============
# features_Personality = [0]*66
# labels_Genre = []
# print(feature)

# getting the list of features (X) with personality
# for i in range(df.shape[0]):
#     data = []
#     data.append(df['O_Personality'][i])
#     data.append(df['C_Personality'][i])
#     data.append(df["E_Personality"][i])
#     data.append(df["A_Personality"][i])
#     data.append(df["N_Personality"][i])
#     #print(data)
#     features_Personality[i] = data


# changing value in a column (if regex = x, value = y, x->y)
def changeValueInColumns(feature,originalValue,to):
    df[feature] = df[feature].replace(regex=originalValue, value=to)


changeValueInColumns('Genre','Horror','Thriller')
changeValueInColumns('Genre','biopic','Drama')
changeValueInColumns('Gender','Female',0)
changeValueInColumns('Gender','Male',1)
changeValueInColumns('Religion','Yes',1)
changeValueInColumns('Religion','No',0)
changeValueInColumns('Religion',"Don't Know",3)
changeValueInColumns('Education','Elementary',1)
changeValueInColumns('Education','High school',2)
changeValueInColumns('Education','College',3)
changeValueInColumns('Education','University',4)
changeValueInColumns('Education','Graduate School',5)
changeValueInColumns('Education','PHD',6)




# selecting columns of the data frame
X = df[['O_Personality','C_Personality',"E_Personality","A_Personality","N_Personality","Age","Gender","Religion","Education"]]
Y = df['Genre']

# X = df[['level','department','year','credit','class_average','class_size']]
# Y = df['grade']

# count the frequency of different genre
# counts = labels_Genre.value_counts().to_dict()
# print(counts)



# get a list of y
# for y in df["Genre"]:
#     labels_Genre.append(y)


# random partition the training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)


# ============================== decision tree model ========================================================
def decisionTreeClassifier(X_train,X_test,y_train,y_test):
    decisionTreeClassifier = tree.DecisionTreeClassifier()
    decisionTreeClassifier = decisionTreeClassifier.fit(X_train, y_train)
    predictions = decisionTreeClassifier.predict(X_test)
    print("accuracy for decisionTreeClassifier: ", accuracy_score(y_test,predictions))
    return accuracy_score(y_test, predictions)



# ============================== naive-bayes model ========================================================
def naiveBayesClassifier(X_train,X_test,y_train,y_test):
    naive_bayesClassifier = naive_bayes.GaussianNB()
    naive_bayesClassifier = naive_bayesClassifier.fit(X_train, y_train)
    predictions = naive_bayesClassifier.predict(X_test)
    print("accuracy for NaiveBayesClassifier: ", accuracy_score(y_test, predictions))
    return accuracy_score(y_test, predictions)


# ============================== KNN model ========================================================
def KNNClassifier(X_train,X_test,y_train,y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    KNNClassifier =model.fit(X_train,y_train)
    predictions = KNNClassifier.predict(X_test)
    # print(predictions)
    print("accuracy for KNNClassifier: ", accuracy_score(y_test, predictions))
    return accuracy_score(y_test, predictions)


# ===================logistic Regression =========================================================
def LRClassifier(X_train,X_test,y_train,y_test):
    model = LogisticRegression()
    LRClassifier =model.fit(X_train,y_train)
    predictions = LRClassifier.predict(X_test)
    # print(predictions)
    print("accuracy for LRClassifier: ", accuracy_score(y_test, predictions))
    return accuracy_score(y_test, predictions)


# =========================== Neural Network ======================================================
def NNClassifier(X_train,X_test,y_train,y_test):
    model = MLPClassifier()
    NNClassifier =model.fit(X_train,y_train)
    predictions = NNClassifier.predict(X_test)

    print("accuracy for NNClassifier: ", accuracy_score(y_test, predictions))
    return accuracy_score(y_test, predictions)


def calculateAccurateOfModel():

    A=[]
    B=[]
    C=[]
    D=[]

    for i in range(100):
        a = decisionTreeClassifier(X_train, X_test, y_train, y_test)
        b = naiveBayesClassifier(X_train, X_test, y_train, y_test)
        c = KNNClassifier(X_train, X_test, y_train, y_test)
        d = NNClassifier(X_train, X_test, y_train, y_test)

        A.append(a)
        B.append(b)
        C.append(c)
        D.append(d)

    print("the mean for target is 16")

    print("the mean for decision tree is: ", np.mean(A))
    print("the std for decision tree is: ",np.std(A))

    print("the mean for naive bayes is: ", np.mean(B))
    print("the std for naive bayes is: ", np.std(B))

    print("the mean for KNN is: ", np.mean(C))
    print("the std for KNN is: ", np.std(C))

    print("the mean for NN is: ", np.mean(D))
    print("the std for NN is: ", np.std(D))



def predictGenre(o,c,e,a,n,age,gender,religion,education):

    result = []

    decisionTreeClassifier = tree.DecisionTreeClassifier()
    decisionTreeClassifier = decisionTreeClassifier.fit(X, Y)
    print("decisionTreePredict", decisionTreeClassifier.predict([[o,c,e,a,n,age,gender,religion,education]]))

    result.append(decisionTreeClassifier.predict([[o,c,e,a,n,age,gender,religion,education]])[0])

    naive_bayesClassifier = naive_bayes.GaussianNB()
    naive_bayesClassifier = naive_bayesClassifier.fit(X, Y)
    print("NaiveBayes predict", naive_bayesClassifier.predict([[o, c, e, a, n, age, gender, religion, education]]))
    result.append(naive_bayesClassifier.predict([[o, c, e, a, n, age, gender, religion, education]])[0])

    KNNClassifier = KNeighborsClassifier(n_neighbors=5)
    KNNClassifier = KNNClassifier.fit(X, Y)
    print("KNN predict", KNNClassifier.predict([[o, c, e, a, n, age, gender, religion, education]]))
    result.append(KNNClassifier.predict([[o, c, e, a, n, age, gender, religion, education]])[0])

    NNClassifier = MLPClassifier()
    NNClassifier = NNClassifier.fit(X, Y)
    print("NN predict", NNClassifier.predict([[o, c, e, a, n, age, gender, religion, education]]))
    result.append(NNClassifier.predict([[o, c, e, a, n, age, gender, religion, education]])[0])

    LRClassifier = LogisticRegression()
    LRClassifier = LRClassifier.fit(X, Y)
    print("LR predict", LRClassifier.predict([[o, c, e, a, n, age, gender, religion, education]]))
    result.append(LRClassifier.predict([[o, c, e, a, n, age, gender, religion, education]])[0])

    print(result)

    return result



# 6 genre in total, 17% accuracy should be
def predictGrade(level,department,year,credit,class_average,class_size):

    result = []

    decisionTreeClassifier = tree.DecisionTreeClassifier()
    decisionTreeClassifier = decisionTreeClassifier.fit(X, Y)
    print("decisionTreePredict", decisionTreeClassifier.predict([[level,department,year,credit,class_average,class_size]]))


    naive_bayesClassifier = naive_bayes.GaussianNB()
    naive_bayesClassifier = naive_bayesClassifier.fit(X, Y)
    print("NaiveBayes predict", naive_bayesClassifier.predict([[level,department,year,credit,class_average,class_size]]))

    KNNClassifier = KNeighborsClassifier(n_neighbors=5)
    KNNClassifier = KNNClassifier.fit(X, Y)
    print("KNN predict", KNNClassifier.predict([[level,department,year,credit,class_average,class_size]]))

    NNClassifier = MLPClassifier()
    NNClassifier = NNClassifier.fit(X, Y)
    print("NN predict", NNClassifier.predict([[level,department,year,credit,class_average,class_size]]))




# ========================= testing =================================================================
# correlation("E_Personality","C_Personality")
# t_test("Gender",0,1,"Education")
decisionTreeClassifier(X_train,X_test,y_train,y_test)
naiveBayesClassifier(X_train,X_test,y_train,y_test)
KNNClassifier(X_train,X_test,y_train,y_test)
LRClassifier(X_train,X_test,y_train,y_test)
NNClassifier(X_train,X_test,y_train,y_test)
# o,c,e,a,n,age,gender,religion,education
predictGenre(4,4,3,4,2,27,1,0,5)
# calculateAccurateOfModel()



# python graphical interface:=====================================

class predictButton:
    def __init__(self,master):
        # o,c,e,a,n,age,gender,religion,education
        self.label_0 = Label(root, text="let's predict your favourite movie types !!!!")
        self.label_1 = Label(root, text="openness level 1 to 5")
        self.label_2 = Label(root, text="conciousness level 1 to 5")
        self.label_3 = Label(root, text="extravert level 1 to 5")
        self.label_4 = Label(root, text="Agreeable level 1 to 5")
        self.label_5 = Label(root, text="Neurotic level 1 to 5")
        self.label_6 = Label(root, text="age, by year's of old")
        self.label_7 = Label(root, text="gender: 1(male), 0(female)")
        self.label_8 = Label(root, text="religion: 1(religious) 0(nonreligious)")
        self.label_9 = Label(root, text="education: 1(elementary) to 6(phd)")
        self.label_decisionTree = Label(root, text="decision tree predict: " )
        self.label_naiveBayes = Label(root, text="naiveBayes predict: " )
        self.label_knn = Label(root, text="KNN predict: " )
        self.label_nn = Label(root, text="Neural Network predict: " )
        self.label_lr = Label(root, text="Logistic Regression predict: ")


        self.entry_1 = Entry(root)
        self.entry_2 = Entry(root)
        self.entry_3 = Entry(root)
        self.entry_4 = Entry(root)
        self.entry_5 = Entry(root)
        self.entry_6 = Entry(root)
        self.entry_7 = Entry(root)
        self.entry_8 = Entry(root)
        self.entry_9 = Entry(root)


        self.button_predict = Button(root, text="predict movie", command=self.getEntry)
        self.button_reset = Button(root, text="reset everything", command=self.resetEntry)

        self.label_0.grid(row=0,column=0)
        self.label_1.grid(row=1, column=0, sticky=E)
        self.label_2.grid(row=2, column=0, sticky=E)
        self.label_3.grid(row=3, column=0, sticky=E)
        self.label_4.grid(row=4, column=0, sticky=E)
        self.label_5.grid(row=5, column=0, sticky=E)
        self.label_6.grid(row=6, column=0, sticky=E)
        self.label_7.grid(row=7, column=0, sticky=E)
        self.label_8.grid(row=8, column=0, sticky=E)
        self.label_9.grid(row=9, column=0, sticky=E)


        self.entry_1.grid(row=1, column=1)
        self.entry_2.grid(row=2, column=1)
        self.entry_3.grid(row=3, column=1)
        self.entry_4.grid(row=4, column=1)
        self.entry_5.grid(row=5, column=1)
        self.entry_6.grid(row=6, column=1)
        self.entry_7.grid(row=7, column=1)
        self.entry_8.grid(row=8, column=1)
        self.entry_9.grid(row=9, column=1)

        self.button_predict.grid(row=10, column=2)
        self.button_reset.grid(row=11,column=2)

    def getEntry(self):
        # o,c,e,a,n,age,gender,religion,education
        result =[]

        o = int(self.entry_1.get())
        c = int(self.entry_2.get())
        e = int(self.entry_3.get())
        a = int(self.entry_4.get())
        n = int(self.entry_5.get())
        age = int(self.entry_6.get())
        gender = int(self.entry_7.get())
        religion = int(self.entry_8.get())
        education = int(self.entry_9.get())

        result = predictGenre(o,c,e,a,n,age,gender,religion,education)

        self.label_decisionTree.destroy()
        self.label_naiveBayes.destroy()
        self.label_knn.destroy()
        self.label_nn.destroy()
        self.label_lr.destroy()

        self.label_decisionTree = Label(root, text= "decision tree predict: "+ result[0])
        self.label_naiveBayes = Label(root, text= "naiveBayes predict: "+ result[1])
        self.label_knn = Label(root, text="KNN predict: " + result[2])
        self.label_nn = Label(root, text="Neural Network predict: " + result[3])
        self.label_lr = Label(root, text="Logistic Regression predict: " + result[4])

        self.label_decisionTree.grid(row=11, column=0)
        self.label_naiveBayes.grid(row=12, column=0)
        self.label_knn.grid(row=13, column=0)
        self.label_nn.grid(row=14, column=0)
        self.label_lr.grid(row=15, column=0)

    def resetEntry(self):
        self.entry_1.delete(0,END)
        self.entry_2.delete(0, END)
        self.entry_3.delete(0, END)
        self.entry_4.delete(0, END)
        self.entry_5.delete(0, END)
        self.entry_6.delete(0, END)
        self.entry_7.delete(0, END)
        self.entry_8.delete(0, END)
        self.entry_9.delete(0, END)




root = Tk()
b = predictButton(root)
root.mainloop()














