#####

import nltk
import sklearn.utils._typedefs
import sklearn.neighbors._partition_nodes
import sklearn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import pandas as pa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import sys

#####


# Input variables from user
selected_doctor_form_file = None
selected_net_form_file = None
selected_model = None
selected_features = None
x_train, y_train, x_test, y_test = None, None, None, None
xn, yn, xd, yd = None, None, None, None
text_train_x, text_train_y = None, None
text_model = None


# for read the ham and spam words from the file
def get_ham_spam_words(ham_path, spam_path):
    ham_words = {}
    spam_words = {}

    with open(ham_path, "rb") as f:
        for line in f.readlines():
            if str(line) != '\n':
                line = str(line).split(" ")
                key = line[0][2:]
                value = int(line[1][:-3])
                ham_words[key] = value

    with open(spam_path, "rb") as f:
        for line in f.readlines():
            if str(line) != '\n':
                line = str(line).split(" ")
                key = line[0][2:]
                value = int(line[1][:-3])
                spam_words[key] = value

    return ham_words, spam_words


# for divide the features and the classifier
def get_x_y(csv_pa, features_labels):
    x = csv_pa[features_labels]
    y = csv_pa.Type
    return x, y


# for make Decision Tree Model
def tree_model(x_train, y_train):
    tree_model = tree.DecisionTreeClassifier()
    tree_model.fit(x_train, y_train)
    return tree_model


# for make MLP Model
def network_model(x_train, y_train):
    network_model = MLPClassifier()
    network_model.fit(x_train, y_train)
    return network_model


# for make Gaussian Naive Bias Model
def naive_bias_model(x_train, y_train):
    naive_bias_model = GaussianNB()
    naive_bias_model.fit(x_train, y_train)
    return naive_bias_model


def read_swear_words(file_path):
    words = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            line = str(line)
            if line != "\n":
                words.append(line[:-1])

    return words


swear_words = read_swear_words("C:/Users/Ibrah/Desktop/Machinelearning Script/fucking")


# define the ham and spam words as list
qualities_words, spams_words = get_ham_spam_words("C:/Users/Ibrah/Desktop/Machinelearning Script/ham_words_file", "C:/Users/Ibrah/Desktop/Machinelearning Script/spam_words_file")

################### Features Labels ##############################
all_features_labels = ["following", "followers", "actions", "is_retweet", "location", "length_before",
                       "num_words_before", "length_after", "num_words_after", "num_urls", "num_hashtags",
                       "num_minations", "num_click", "swear", "bias"]

text_features_labels = ["length_before", "num_words_before", "length_after", "num_words_after", "num_urls",
                        "num_hashtags", "num_minations", "num_click", "swear", "bias"]

best_features_labels = ["following", "followers", "actions"]

best_text_features_labels = ["length_before", "num_words_before", "length_after", "num_words_after"]
##################################################################

# for read the train and test files
csv_train_file = pa.read_csv("C:/Users/Ibrah/Desktop/Machinelearning Script/train_file.csv")
csv_test_file = pa.read_csv("C:/Users/Ibrah/Desktop/Machinelearning Script/test_file.csv")


# for check if the tweet has swear words or not
def contains_profanity(tweet):
    for word in tweet.split(" "):
        if word in swear_words:
            return True
    return False



# for clean the words that not important
def clean_text(list_words):
    regex1 = re.compile(r'@[A-Za-z0-9]+|#|@|^http[s]*|[0-9]')
    filtered1 = [i for i in list_words if not regex1.search(i)]

    regex2 = re.compile('[%s]' % re.escape(string.punctuation))
    filtered2 = [i for i in filtered1 if not regex2.search(i)]

    return filtered2


# for return the tweet words to its roots and delete the duplicate
def get_roots(tweet):
    check_types = nltk.pos_tag(tweet)
    root = WordNetLemmatizer()
    root_words = set()
    for word in check_types:
        type = word[1][0]
        if type == 'N':
            root_words.add(root.lemmatize(word[0], pos="n"))
        elif type == "V":
            root_words.add(root.lemmatize(word[0], pos="v"))
        elif type == "A":
            root_words.add(root.lemmatize(word[0], pos="a"))
        elif type == "S":
            root_words.add(root.lemmatize(word[0], pos="s"))
        else:
            root_words.add(root.lemmatize(word[0], pos="n"))

    return list(root_words)


# for get the tweet features depend on the text
def preprocessing(tweet):
    tweet = str(tweet)

    # create length_before feature
    length_tweet_before = len(tweet)

    # create num_words_before feature
    num_of_words_before = len(tweet.split(" "))

    # create num_urls feature
    url = re.findall("(http[s]:\/\/)?([\w-]+\.)+([a-z]{2,5})(\/+\w+)?", tweet)
    num_of_urls = len(url)

    # create swear feature
    contain_swear_words = contains_profanity(tweet)
    if contain_swear_words:
        contain_swear_words = 1
    else:
        contain_swear_words = 0

    # for delete the stop words
    stop_words = set(stopwords.words("english"))
    divide_tweet_to_words = list(word_tokenize(str(tweet).lower()))
    remaining_words = [word for word in divide_tweet_to_words if word not in stop_words]

    # create num_hashtags feature
    num_of_hashtags = remaining_words.count("#")

    # create num_minations feature
    num_of_minations = remaining_words.count("@")

    # create num_click feature
    num_of_click_word = remaining_words.count("click")

    # for clean the tweet
    cleaning_tweet = clean_text(remaining_words)

    # create num_words_after feature
    num_of_words_after = len(cleaning_tweet)

    # create length_after feature
    length_tweet_after = len("".join(cleaning_tweet))

    # for get the roots of the tweet words
    tweet_roots = get_roots(cleaning_tweet)
    tweet_roots = " ".join(tweet_roots)

    return length_tweet_before, length_tweet_after, num_of_words_before, num_of_words_after, num_of_urls, num_of_hashtags, num_of_minations, contain_swear_words, num_of_click_word, tweet_roots


# for save the spam and ham words in dictionaries
def categorize_words(csv_data):
    spam_words = {}
    ham_words = {}

    for line in csv_data['roots'][csv_data['Type'] == 1]:
        for word in str(line).split(" "):
            if word in spam_words.keys():
                spam_words[word] += 1
            else:
                spam_words[word] = 1

    for line in csv_data['roots'][csv_data['Type'] == 0]:
        for word in str(line).split(" "):
            if word in ham_words.keys():
                ham_words[word] += 1
            else:
                ham_words[word] = 1

    return spam_words, ham_words


# for make bias feature
def bias_feature(roots, spam_words, ham_words):
    roots = str(roots)
    ham = 0
    spam = 0

    for word in roots.split(" "):
        if word in spam_words.keys():
            spam += spam_words[word]
        if word in ham_words.keys():
            ham += ham_words[word]

    if ham > spam:
        return 0

    return 1


# the created file:
# split_data_file_inaddition_to_new_features("train.csv", "train_file.csv", "test_file.csv", "ham_words_file", "spam_words_file")
# this function for make all features and split the train and test data into two csv files
def split_data_file_inaddition_to_new_features(file_path, new_train_path, new_test_path, ham_words_path,
                                               spam_words_path, test_size=0.2, random_state=10):
    from sklearn.preprocessing import LabelEncoder

    df = pa.read_csv(file_path)

    df_GF = df[["Tweet", "following", "followers", "actions", "is_retweet", "location", "Type"]]

    le = LabelEncoder()
    df_update = pa.DataFrame.copy(df_GF)
    # for replace the location with 1 if exist and 0 if null
    df_update['location'] = df_update['location'].apply(lambda x: 1 if not pa.isnull(x) else 0)
    # for replace the null with zero
    df_update['actions'] = df_update['actions'].replace(np.nan, 0)
    df_update['following'] = df_update['following'].replace(np.nan, 0)
    df_update['followers'] = df_update['followers'].replace(np.nan, 0)
    df_update['is_retweet'] = df_update['is_retweet'].replace(np.nan, 0)

    # for replace the ham with 0
    # and the spam with 1
    df_update['Type'] = le.fit_transform(df_update['Type'])

    # for get the features vectors for all tweet
    length_before_column = []
    length_after_column = []
    num_words_before_column = []
    num_words_after_column = []
    num_urls_column = []
    num_hashtags_column = []
    num_minations_column = []
    num_click_column = []
    swear_column = []
    root_column = []

    count = 0

    tweet_column = list(df_update["Tweet"])
    for tweet in tweet_column:
        lb, la, nwb, nwa, nu, nh, nm, nc, s, r = preprocessing(tweet)
        length_before_column.append(lb)
        length_after_column.append(la)
        num_words_before_column.append(nwb)
        num_words_after_column.append(nwa)
        num_urls_column.append(nu)
        num_hashtags_column.append(nh)
        num_minations_column.append(nm)
        num_click_column.append(nc)
        swear_column.append(s)
        root_column.append(r)
        # print(count)
        count += 1

    # print("features extraction done")

    new_csv = pa.DataFrame({"Type": df_update['Type'], "Tweet": df_update['Tweet'], "roots": root_column,
                            "following": df_update['following'],
                            "followers": df_update['followers'], "actions": df_update['actions'],
                            "is_retweet": df_update['is_retweet'],
                            "location": df_update['location'], "length_before": length_before_column,
                            "length_after": length_after_column, "num_words_before": num_words_before_column,
                            "num_words_after": num_words_after_column, "num_urls": num_urls_column,
                            "num_hashtags": num_hashtags_column,
                            "num_minations": num_minations_column,
                            "num_click": num_click_column, "swear": swear_column})

    # for split the data into training and testing csv files
    x = new_csv.drop('Type', axis=1)
    y = new_csv.Type

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    print('size of test dataset = {}, size of traing data = {}, percentage = {}%'.format(len(x_test), len(x_train),
                                                                                         len(x_test) * 100 / (
                                                                                                 len(x_test) + len(
                                                                                             x_train))))

    train_csv_file = pa.DataFrame(x_train)
    train_csv_file["Type"] = y_train

    test_csv_file = pa.DataFrame(x_test)
    test_csv_file["Type"] = y_test

    # for create the bias feature for all tweets
    spam_words, ham_words = categorize_words(train_csv_file)
    train_csv_file["bias"] = train_csv_file['roots'].apply(lambda b: bias_feature(b, spam_words, ham_words))
    test_csv_file["bias"] = test_csv_file['roots'].apply(lambda b: bias_feature(b, spam_words, ham_words))

    # for save the train and test csv files
    train_csv_file.to_csv(new_train_path, index=False)
    test_csv_file.to_csv(new_test_path, index=False)

    # for save the ham words in the file
    file_for_save_ham_words = open(ham_words_path, "wb")
    for element in ham_words:
        file_for_save_ham_words.write(bytes(element, 'utf-8'))
        file_for_save_ham_words.write(bytes(" ", 'utf-8'))
        file_for_save_ham_words.write(bytes(str(ham_words[element]), 'utf-8'))
        file_for_save_ham_words.write(bytes("\n", 'utf-8'))
    file_for_save_ham_words.close()

    # for save the spam words in the file
    file_for_save_spam_words = open(spam_words_path, "wb")
    for element in spam_words:
        file_for_save_spam_words.write(bytes(element, 'utf-8'))
        file_for_save_spam_words.write(bytes(" ", 'utf-8'))
        file_for_save_spam_words.write(bytes(str(spam_words[element]), 'utf-8'))
        file_for_save_spam_words.write(bytes("\n", 'utf-8'))
    file_for_save_spam_words.close()


# for show the information gain
def show_IG_for_features(x_train, y_train):
    # configure to select all features
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    # learn relationship from data
    fs.fit(x_train, y_train)

    for i in range(len(fs.scores_)):
        print('Feature[%d]: %s = %f' % (i, fs.feature_names_in_[i], fs.scores_[i]))


# for print the information gain on the console
def print_ig_on_console(x_train, y_train):
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    # learn relationship from data
    fs.fit(x_train, y_train)

    for i in range(len(fs.scores_)):
        print('%s %f' % (fs.feature_names_in_[i], fs.scores_[i]))


# for plot all features information gain
def plot_information_gain(csv_file, x_train, y_train):
    from sklearn.feature_selection import mutual_info_classif

    importances = mutual_info_classif(x_train, y_train)
    feat_importances = pa.Series(importances, csv_file.columns[2:len(csv_file.columns) - 1])
    feat_importances.plot(kind='barh', color='teal')
    plt.show()


# for get the text features from the tweet
def get_text_features(tweet, spam_words, ham_words):
    lb, la, nwb, nwa, nu, nh, nm, nc, s, r = preprocessing(tweet)
    bias = bias_feature(r, spam_words, ham_words)
    return lb, la, nwb, nwa, nu, nh, nm, nc, s, r, bias


# for print if the tweet is ham or spam
def print_model_result(model, features_type, spam_words, ham_words, tweet, following=0, followers=0, actions=0,
                       is_retweet=0, location=""):
    if location == "":
        location = 0
    else:
        location = 1
    lb, la, nwb, nwa, nu, nh, nm, nc, s, r, bias = get_text_features(tweet, spam_words, ham_words)

    features_vector = None

    if features_type == "text":
        features_vector = pa.DataFrame({"length_before": [lb], "num_words_before": [nwb],
                                        "length_after": [la], "num_words_after": [nwa], "num_urls": [nu],
                                        "num_hashtags": [nh], "num_minations": [nm], "num_click": [nc], "swear": [s],
                                        "bias": [bias]})
    elif features_type == "best_text":
        features_vector = pa.DataFrame({"length_before": [lb], "num_words_before": [nwb],
                                        "length_after": [la], "num_words_after": [nwa]})
    elif features_type == "all":
        features_vector = pa.DataFrame({"following": [following], "followers": [followers], "actions": [actions],
                                        "is_retweet": [is_retweet], "location": [location],
                                        "length_before": [lb], "num_words_before": [nwb], "length_after": [la],
                                        "num_words_after": [nwa], "num_urls": [nu], "num_hashtags": [nh],
                                        "num_minations": [nm], "num_click": [nc], "swear": [s], "bias": [bias]
                                        })
    elif features_type == "best_all":
        features_vector = pa.DataFrame({"following": [following], "followers": [followers], "actions": [actions]})
    else:
        print("Wrong features type parameter")

    # print("Tweet: {0}\nResult: {1}".format(tweet, model.predict(features_vector)))
    print(int(model.predict(features_vector)))


# first column should be Tweet
# second column should be following
# third column should be followers
# forth column should be actions
# fifth column should be is_retweet
# sixth column should be location
# seventh column should be Type
def read_from_test_file_with_doctor_features(test_file_path, spam_words, ham_words):
    from sklearn.preprocessing import LabelEncoder

    df = pa.read_csv(test_file_path)

    df_GF = df[["Tweet", "following", "followers", "actions", "is_retweet", "location", "Type"]]

    le = LabelEncoder()
    df_update = pa.DataFrame.copy(df_GF)

    df_update['location'] = df_update['location'].apply(lambda x: 1 if not pa.isnull(x) else 0)
    df_update['actions'] = df_update['actions'].replace(np.nan, 0)
    df_update['following'] = df_update['following'].replace(np.nan, 0)
    df_update['followers'] = df_update['followers'].replace(np.nan, 0)
    df_update['is_retweet'] = df_update['is_retweet'].replace(np.nan, 0)

    df_update['Type'] = le.fit_transform(df_update['Type'])

    length_before_column = []
    length_after_column = []
    num_words_before_column = []
    num_words_after_column = []
    num_urls_column = []
    num_hashtags_column = []
    num_minations_column = []
    num_click_column = []
    swear_column = []
    root_column = []

    count = 0

    tweet_column = list(df_update["Tweet"])
    for tweet in tweet_column:
        lb, la, nwb, nwa, nu, nh, nm, nc, s, r = preprocessing(tweet)
        length_before_column.append(lb)
        length_after_column.append(la)
        num_words_before_column.append(nwb)
        num_words_after_column.append(nwa)
        num_urls_column.append(nu)
        num_hashtags_column.append(nh)
        num_minations_column.append(nm)
        num_click_column.append(nc)
        swear_column.append(s)
        root_column.append(r)
        # print(count)
        count += 1

    # print("features extraction done")

    new_csv = pa.DataFrame({"Type": df_update['Type'], "Tweet": df_update['Tweet'], "roots": root_column,
                            "following": df_update['following'],
                            "followers": df_update['followers'], "actions": df_update['actions'],
                            "is_retweet": df_update['is_retweet'],
                            "location": df_update['location'], "length_before": length_before_column,
                            "length_after": length_after_column, "num_words_before": num_words_before_column,
                            "num_words_after": num_words_after_column, "num_urls": num_urls_column,
                            "num_hashtags": num_hashtags_column,
                            "num_minations": num_minations_column,
                            "num_click": num_click_column, "swear": swear_column})

    new_csv["bias"] = new_csv['roots'].apply(lambda b: bias_feature(b, spam_words, ham_words))
    new_csv["bias"] = new_csv['roots'].apply(lambda b: bias_feature(b, spam_words, ham_words))

    return new_csv


# first column should be Tweet
# second column should be Type
def read_from_test_file_just_tweets(test_file_path, spam_words, ham_words):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    df = pa.read_csv(test_file_path)

    df_GF = df[["Tweet", "Type"]]
    df_update = pa.DataFrame.copy(df_GF)
    df_update['Type'] = le.fit_transform(df_update['Type'])

    length_before_column = []
    length_after_column = []
    num_words_before_column = []
    num_words_after_column = []
    num_urls_column = []
    num_hashtags_column = []
    num_minations_column = []
    num_click_column = []
    swear_column = []
    root_column = []

    count = 0

    tweet_column = list(df_update["Tweet"])
    for tweet in tweet_column:
        lb, la, nwb, nwa, nu, nh, nm, nc, s, r = preprocessing(tweet)
        length_before_column.append(lb)
        length_after_column.append(la)
        num_words_before_column.append(nwb)
        num_words_after_column.append(nwa)
        num_urls_column.append(nu)
        num_hashtags_column.append(nh)
        num_minations_column.append(nm)
        num_click_column.append(nc)
        swear_column.append(s)
        root_column.append(r)
        # print(count)
        count += 1

    # print("features extraction done")

    new_csv = pa.DataFrame({"Type": df_update['Type'], "Tweet": df_update['Tweet'], "roots": root_column,
                            "length_before": length_before_column,
                            "length_after": length_after_column, "num_words_before": num_words_before_column,
                            "num_words_after": num_words_after_column, "num_urls": num_urls_column,
                            "num_hashtags": num_hashtags_column,
                            "num_minations": num_minations_column,
                            "num_click": num_click_column, "swear": swear_column})

    new_csv["bias"] = new_csv['roots'].apply(lambda b: bias_feature(b, spam_words, ham_words))
    new_csv["bias"] = new_csv['roots'].apply(lambda b: bias_feature(b, spam_words, ham_words))

    return new_csv


# for print the number of hams ans spams tweets in the csv file
def print_spams_qualities_info(csv_file_path):
    data = pa.read_csv(csv_file_path)

    spam = data[data.Type == "Spam"]
    No_spam = spam.shape[0]
    quality = data[data.Type == "Quality"]
    No_quality = quality.shape[0]
    Per_spam = No_spam / (No_spam + No_quality)

    print(data.info())
    print('spam = {}, quality = {} , Percentage of spam = {} %'.format(No_spam, No_quality, Per_spam * 100))


# for print the mean for all features the the csv files group by type
def print_mean_group_by_type(csv_file_path, features_names):
    data = pa.read_csv(csv_file_path)
    for f in features_names:
        print(data.groupby('Type').mean()[f])


# for get the the classification model result
# and calculate the accuracy
# and precision and recall and f1 scores
def cal_scores(model, x, y_true):
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score

    y_pred = model.predict(x)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return acc, recall, precision, f1, y_pred


# for print
# the accuracy
# and precision and recall and f1 scores
def print_score(model, x, y_true):
    from sklearn.metrics import classification_report, confusion_matrix

    acc, recall, precision, f1, y_pred = cal_scores(model, x, y_true)

    print("Accuracy: ", np.round(acc, 2))
    print("Recall: ", np.round(recall, 2))
    print("Precision: ", np.round(precision, 2))
    print("F1: ", np.round(f1, 2))
    print("confusion_matrix: \n", confusion_matrix(y_true, y_pred))
    print("classification_report: \n", classification_report(y_true, y_pred))


# for print
# the accuracy
# and precision and recall and f1 scores
# on the console
def print_score_on_console(model, x, y_true):
    from sklearn.metrics import confusion_matrix

    acc, recall, precision, f1, y_pred = cal_scores(model, x, y_true)

    print("accuracy %.2f" % acc)
    print("recall %.2f" % recall)
    print("precision %.2f" % precision)
    print("f1 %.2f" % f1)
    print("tp ", list(confusion_matrix(y_true, y_pred))[0][0])
    print("fp ", list(confusion_matrix(y_true, y_pred))[0][1])
    print("fn ", list(confusion_matrix(y_true, y_pred))[1][0])
    print("tn ", list(confusion_matrix(y_true, y_pred))[1][1])

# for create just text spam model
def create_text_model():
    global text_train_x, text_train_y, text_model
    # for make text_features model
    text_train_x, text_train_y = get_x_y(csv_train_file, text_features_labels)
    text_model = tree_model(text_train_x, text_train_y)


def pick_selected_features(sel_features):
    global selected_features

    if sel_features == "af":
        selected_features = all_features_labels
    elif sel_features == "at":
        selected_features = text_features_labels
    elif sel_features == "ba":
        selected_features = best_features_labels
    elif sel_features == "bt":
        selected_features = best_text_features_labels

def pick_selected_model(sel_model, x_train, y_train):
    global selected_model

    if sel_model == "tree":
        selected_model = tree_model(x_train, y_train)
    elif sel_model == "network":
        selected_model = network_model(x_train, y_train)
    elif sel_model == "naive_bias":
        selected_model = naive_bias_model(x_train, y_train)

# create system based on which selected from user
def create_the_system(model, sel_features):
    global selected_doctor_form_file, selected_net_form_file, selected_model, selected_features
    global x_train, y_train, x_test, y_test
    global xn, yn, xd, yd

    pick_selected_features(sel_features)

    x_train, y_train = get_x_y(csv_train_file, selected_features)
    x_test, y_test = get_x_y(csv_test_file, selected_features)

    pick_selected_model(model, x_train, y_train)




# for cleaning the text from punctuations and stopwords
def text_processing(text):


    text = [char for char in text if char not in string.punctuation]
    text = ''.join(text)
    text = [word for word in text.split() if word.lower() not in stopwords.words("english")]

    return text


# for create naive bias model with multiple features
def create_naive_bias_mul_features_model_then_test_tweet(tweet):
    from sklearn.preprocessing import LabelEncoder

    text = pa.DataFrame({"Tweet": [tweet]})
    tweets_types_columns = csv_train_file[["Tweet", "Type"]]
    tweets_file = pa.DataFrame.copy(tweets_types_columns)

    le = LabelEncoder()

    # for replace the ham with 0
    # and the spam with 1
    tweets_file['Type'] = le.fit_transform(tweets_file['Type'])

    count_vector = CountVectorizer(analyzer=text_processing)

    training_data = count_vector.fit_transform(tweets_file['Tweet'])
    test_text = count_vector.transform(text)

    naive_bias_text_model = MultinomialNB()
    naive_bias_text_model.fit(training_data, tweets_file['Type'])

    print(int(naive_bias_text_model.predict(test_text)))

# for handling the arguments which request from the java code
# and make response for it by print on the console
def handler(args):
    global xd, yd, xn, yn
    global selected_doctor_form_file, selected_net_form_file


    # following=0, followers=0, actions=0,
    #                  is_retweet=0, location=""
    # when the user enter tweet and want to check it if spam or ham
    # you should enter
    # args[2] --> tf: use text features model, nbf: use multi features by naive bias, af: use all features tree model
    # args[3] = tweet
    # args[4,5,6,7,8] : just if you pass "af" --> 4: following, 5: followers, 6: actions, 7: is_retweet, 8: location
    if args[1] == "tweet":
        if args[2] == "tf":
            create_text_model()
            print_model_result(text_model, "text", spams_words, qualities_words, tweet=args[3])
        elif args[2] == "nbf":
            create_naive_bias_mul_features_model_then_test_tweet(args[3])
        elif args[2] == "af":
            create_the_system("tree", "af")
            print_model_result(selected_model, "all", spams_words, qualities_words, tweet=args[3], following=int(
                args[4]),
                               followers=int(args[5]), actions=int(args[6]), is_retweet=int(args[7]), location=args[8])

    # when the user want to know the information gain for the features
    # you should pass
    # args[2] = the features_label selected by user (af, at, ba, bt)
    elif args[1] == "ig":
        pick_selected_features(args[2])
        x_ig, y_ig = get_x_y(csv_train_file, selected_features)
        print_ig_on_console(x_ig, y_ig)

    # when the user want to show the accuracy of the test file
    # you should pass
    # args[2] = the model selected by user (tree, network, naive_bias)
    # args[3] = the features_label selected by user (af, at, ba, bt)
    elif args[1] == "atf":
        create_the_system(args[2], args[3])
        print_score_on_console(selected_model, x_test, y_test)

    # when the user want to show the accuracy of the sample file
    # you should pass
    # args[2] = the model selected by user (tree, network, naive_bias)
    # args[3] = the features_label selected by user (af, at, ba, bt)
    # args[4] = the path of the sample file
    elif args[1] == "adf":
        create_the_system(args[2], args[3])
        selected_doctor_form_file = read_from_test_file_with_doctor_features(args[4], spams_words, qualities_words)
        xd, yd = get_x_y(selected_doctor_form_file, selected_features)
        print_score_on_console(selected_model, xd, yd)

    # when the user want to show the accuracy of the net file
    # you should pass
    # args[2] = the path of the tweets file
    # not required to the selected model and features selected by user
    elif args[1] == "anf":
        create_text_model()
        selected_net_form_file = read_from_test_file_just_tweets(args[2], spams_words, qualities_words)
        xn, yn = get_x_y(selected_net_form_file, text_features_labels)
        print_score_on_console(text_model, xn, yn)


# print_score_on_console(text_model, text_train_x, text_train_y)
# print_model_result(text_model, "text", spams_words, qualities_words, "click click click click click click ")
# print_ig_on_console(text_train_x, text_train_y)
# handler(list(["", "anf", "spam_short.csv"]))

#create_naive_bias_mul_features_model_then_test_tweet("Fuck you")

handler(sys.argv)