import pandas as pd
import numpy as np
import seaborn as sns

from model.ctabgan import CTABGAN
from model.eval.evaluation import get_utility_metrics
from random import randint

from matplotlib import pyplot as plt

df = pd.read_csv("Real_Datasets/Epileptic.csv")
print(len(df), len(df.columns))

#df.drop('ID', acis=1, inplace=True) #Cervical
#df = df.replace(r'?', np.nan) #Cervical

#df.drop(['Unnamed'], axis=1, inplace=True) #Epileptic
#df['y'] = (df['y'] == 1).replace(True,1).replace(False,0) #Epileptic

print(len(df), len(df.columns))

#for col in df.columns:
    #print(col)
    #print(pd.unique(df[col]))
    #print(df[col].value_counts())

#print(df.dtypes)

#print(df.columns[:-1].values)

#df.to_csv("Real_Datasets/Epileptic.csv" ,index=False)

#for col in ['Age', 'Number of sexual partners', 'First sexual intercourse','Num of pregnancies', 'Smokes (years)', 'Smokes (packs/year)','Hormonal Contraceptives (years)','IUD (years)','STDs (number)', 'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis', 'STDs: Number of diagnosis']: #['ID', 'Age', 'Experience', 'Income', 'ZIP Code', 'Mortgage', 'CCAvg']: #['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week']: #
    #sns.distplot(df[col])
    #plt.show()

#for col in ['Smokes','Hormonal Contraceptives','IUD','STDs','STDs:condylomatosis', 'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis','STDs:pelvic inflammatory disease', 'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV','STDs:Hepatitis B', 'STDs:HPV', 'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller','Citology', 'Biopsy']: #['Family', 'Education', 'Personal Loan', 'Securities Account', 'CD Account', 'Online', 'CreditCard']: #['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']: #
    #sns.countplot(x=col, data=df)
    #plt.show()

#model = CTABGAN()

#model = CTABGAN(raw_csv_path = "Real_Datasets/Loan.csv",
                 #test_ratio = 0.20,
                 #categorical_columns = ['Family', 'Education', 'Personal Loan', 'Securities Account', 'CD Account', 'Online', 'CreditCard'], 
                 #log_columns = [],
                 #mixed_columns= {'Mortgage':[0.0]},
                 #general_columns = ['Age', 'Experience', 'Income', 'CCAge', 'Personal Loan', 'Securities Account', 'CD Account', 'Online', 'CreditCard'],
                 #non_categorical_columns = ['CCAvg'],
                 #integer_columns = ['Age', 'Experience', 'Income', 'ZIP Code', 'Mortgage'],
                 #problem_type= {"Classification": "Personal Loan"})

#model = CTABGAN(raw_csv_path = "Real_Datasets/Cervical.csv",
                 #test_ratio = 0.20,
                 #categorical_columns = ['Smokes','Hormonal Contraceptives','IUD','STDs','STDs:condylomatosis', 'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis','STDs:pelvic inflammatory disease', 'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV','STDs:Hepatitis B', 'STDs:HPV', 'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller','Citology', 'Biopsy'], 
                 #log_columns = [],
                 #mixed_columns= {'Num of pregnancies':[0,0], 'Smokes (years)':[0.0], 'Smokes (packs/year)':[0.0], 'Hormonal Contraceptives (years)':[0.0], 'IUD (years)':[0.0], 'STDs (number)':[0.0], 'STDs: Number of diagnosis':[0.0]},
                 #general_columns = ['Age', 'First sexual intercourse'],
                 #non_categorical_columns = [],
                 #integer_columns = ['Age', 'Number of sexual partners', 'First sexual intercourse','Num of pregnancies', 'Smokes (years)', 'Smokes (packs/year)','Hormonal Contraceptives (years)','IUD (years)','STDs (number)', 'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis', 'STDs: Number of diagnosis'],
                 #problem_type= {"Classification": "Biopsy"})
                #OBS. times since have many missing values: impossible to decide dist, 'Number of sexual partners' (log,normal)

model = CTABGAN(raw_csv_path = "Real_Datasets/Epileptic.csv",
                 test_ratio = 0.20,
                 categorical_columns = ['y'], 
                 log_columns = [],
                 mixed_columns= {},
                 general_columns = [],
                 non_categorical_columns = [],
                 integer_columns = df.columns[:-1].values,
                 problem_type= {"Classification": "y"})

model.fit()

sample = model.generate_samples()

sample.to_csv('Fake_Datasets/Epileptic.csv', index=False)

'''real = pd.read_csv('Real_Datasets/Cervical.csv')

print(real.shape)

#diff, real_results, fake_results = get_utility_metrics('Real_Datasets/Cervical.csv', ['Fake_Datasets/Cervical_batch64_nomixedgeneral_correct.csv'], scaler="MinMax", classifiers=["lr","svm","dt","rf","mlp"], test_ratio=.20, random_state=42)
#print(diff, real_results, fake_results[0]) #

all_experiments_real = []
all_experiments_fake = []

for i in range(3): #only one fake data
    rand = randint(0, 100)
    diff, real_results, fake_results = get_utility_metrics('Real_Datasets/Cervical.csv', ['Fake_Datasets/Cervical.csv'], scaler="MinMax", classifiers=["lr","svm","dt","rf","mlp"], test_ratio=.20, random_state=rand)
    all_experiments_real.append(real_results)
    all_experiments_fake.append(fake_results[0])
    #print(diff, real_results, fake_results[0]) #

print(np.array(all_experiments_real).mean(axis=0), np.array(all_experiments_fake).mean(axis=0))'''