from utils import *

# Set the path to the training and test data 
train_path = 'data/train/train/' # set the train path
test_path = 'data/test/test/'    # set the test path

# Generate our DataFrame with the Filepaths and Classes
filepaths_train = []
for filename in os.listdir(train_path):
    filepaths_train.append(train_path + filename)

filepaths_test = []
for filename in os.listdir(test_path):
    filepaths_test.append(test_path + filename)

id_train = []
labels_train = []
for filename in os.listdir(train_path):
    filepath = train_path + filename
    labels_train.append(filename[:2]) 
    id_train.append(get_label(filepath))

id_test = []
labels_test = []
for filename in os.listdir(test_path):
    filepath = test_path + filename
    labels_test.append(filename[:2])
    id_test.append(get_label(filepath))
    
df_train = pd.DataFrame({"Filepath":filepaths_train, "Class ID":id_train, "Label":labels_train})
df_test = pd.DataFrame({"Filepath":filepaths_test, "Class ID":id_test, "Label":labels_test})
print("Train - ", df_train.shape)
print("Test - ", df_test.shape)
print(df_train.head(5))

df_train_full = load_mfcc_features(df_train)
df_test_full = load_mfcc_features(df_test)

X_train = np.array(df_train_full['mfcc'].tolist())
y_train = np.array(df_train_full['Class ID'].tolist())
X_test = np.array(df_test_full['mfcc'].tolist())
y_test = np.array(df_test_full['Class ID'].tolist())

np.save('datasets/X_train_mfcc.npy', X_train)
np.save('datasets/y_train_mfcc.npy', y_train)
np.save('datasets/X_test_mfcc.npy', X_test)
np.save('datasets/y_test_mfcc.npy', y_test)

print("Preprocessing Complete")
