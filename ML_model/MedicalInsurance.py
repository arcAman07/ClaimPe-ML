#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


train = pd.read_csv("D:/ML/Dataset/MedicalInsurance/Train-1542865627584.csv")
beneficiary = pd.read_csv("D:/ML/Dataset/MedicalInsurance/Train_Beneficiarydata-1542865627584.csv")
inpatient = pd.read_csv("D:/ML/Dataset/MedicalInsurance/Train_Inpatientdata-1542865627584.csv")
outpatient = pd.read_csv("D:/ML/Dataset/MedicalInsurance/Train_Outpatientdata-1542865627584.csv")

tt = pd.read_csv("D:/ML/Dataset/MedicalInsurance/Test-1542969243754.csv")
tb = pd.read_csv("D:/ML/Dataset/MedicalInsurance/Test_Beneficiarydata-1542969243754.csv")
ti = pd.read_csv("D:/ML/Dataset/MedicalInsurance/Test_Inpatientdata-1542969243754.csv")
to = pd.read_csv("D:/ML/Dataset/MedicalInsurance/Test_Outpatientdata-1542969243754.csv")


# In[3]:


df_procedures1 =  pd.DataFrame(columns = ['Procedures'])
df_procedures1['Procedures'] = pd.concat([inpatient["ClmProcedureCode_1"], inpatient["ClmProcedureCode_2"], inpatient["ClmProcedureCode_3"], inpatient["ClmProcedureCode_4"], inpatient["ClmProcedureCode_5"], inpatient["ClmProcedureCode_6"]], axis=0, sort=True).dropna()
df_procedures1['Procedures'].head(10)


# In[4]:


df_procedures1.shape


# In[5]:


grouped_procedure_df = df_procedures1['Procedures'].value_counts()
grouped_procedure_df


# In[6]:


df_diagnosis = pd.DataFrame(columns = ['Diagnosis'])
df_diagnosis['Diagnosis'] = pd.concat([inpatient["ClmDiagnosisCode_1"], inpatient["ClmDiagnosisCode_2"], inpatient["ClmDiagnosisCode_3"], inpatient["ClmDiagnosisCode_4"], inpatient["ClmDiagnosisCode_5"], inpatient["ClmDiagnosisCode_6"], inpatient["ClmDiagnosisCode_7"], inpatient["ClmDiagnosisCode_8"], inpatient["ClmDiagnosisCode_9"], inpatient["ClmDiagnosisCode_10"]], axis=0, sort=True).dropna()
df_diagnosis['Diagnosis'].head(10)


# In[7]:


df_diagnosis.shape


# In[8]:


grouped_diagnosis_df = df_diagnosis['Diagnosis'].value_counts()
grouped_diagnosis_df


# In[9]:


grouped_procedure_df1 = grouped_procedure_df.to_frame()
grouped_procedure_df1


# In[10]:


grouped_procedure_df1.columns = ['count']
grouped_procedure_df1


# In[11]:


grouped_procedure_df1['Procedure'] = grouped_procedure_df1.index
grouped_procedure_df1


# In[12]:


grouped_procedure_df1['Percentage'] = (grouped_procedure_df1['count']/sum(grouped_procedure_df1['count']))*100
grouped_procedure_df1['Percentage']


# In[13]:


grouped_diagnosis_df = grouped_diagnosis_df.to_frame()
grouped_diagnosis_df.columns = ['count']
grouped_diagnosis_df['Diagnosis'] = grouped_diagnosis_df.index
grouped_diagnosis_df['Percentage'] = (grouped_diagnosis_df['count']/sum(grouped_diagnosis_df['count']))*100
grouped_diagnosis_df['Percentage']


# In[14]:


# taking only top 20 

plot_procedure_df1 = grouped_procedure_df1.head(20)
plot_diagnosis_df1 = grouped_diagnosis_df.head(20)


# In[15]:


# Plotting the most commonly used diagnosis and procedures 
from matplotlib import pyplot as plt
plot_procedure_df1['Procedure'] = plot_procedure_df1['Procedure'].astype(str)
plot_procedure_df1.sort_values(by=['Percentage'])
plot_procedure_df1.plot(x ='Procedure', y='Percentage', kind='bar', color ='green',
                  title='Procedure Distribution- Inpatient', figsize=(15,10));


# In[16]:


plot_diagnosis_df1['Diagnosis'] =  plot_diagnosis_df1['Diagnosis'].astype(str)
plot_diagnosis_df1.sort_values(by=['Percentage'])
plot_diagnosis_df1.plot(x ='Diagnosis', y='Percentage', kind='bar', color ='green',
                  title='Diagnosis Distribution- Inpatient', figsize=(15,10));


# In[17]:


df_procedures2 =  pd.DataFrame(columns = ['Procedures'])
df_procedures2['Procedures'] = pd.concat([outpatient["ClmProcedureCode_1"], outpatient["ClmProcedureCode_2"], outpatient["ClmProcedureCode_3"], outpatient["ClmProcedureCode_4"], outpatient["ClmProcedureCode_5"], outpatient["ClmProcedureCode_6"]], axis=0, sort=True).dropna()
df_procedures2['Procedures'].head(10)


# In[18]:


grouped_procedure_df2 = df_procedures2['Procedures'].value_counts()


# In[19]:


df_diagnosis2 = pd.DataFrame(columns = ['Diagnosis'])
df_diagnosis2['Diagnosis'] = pd.concat([outpatient["ClmDiagnosisCode_1"], outpatient["ClmDiagnosisCode_2"], outpatient["ClmDiagnosisCode_3"], outpatient["ClmDiagnosisCode_4"], outpatient["ClmDiagnosisCode_5"], outpatient["ClmDiagnosisCode_6"], outpatient["ClmDiagnosisCode_7"],  outpatient["ClmDiagnosisCode_8"], outpatient["ClmDiagnosisCode_9"], outpatient["ClmDiagnosisCode_10"]], axis=0, sort=True).dropna()
df_diagnosis2['Diagnosis'].head(10)
grouped_diagnosis_df2 = df_diagnosis2['Diagnosis'].value_counts()


# In[20]:


grouped_procedure_df_op = grouped_procedure_df2.to_frame()
grouped_procedure_df_op.columns = ['count']
grouped_procedure_df_op['Procedure'] = grouped_procedure_df_op.index
grouped_procedure_df_op['Percentage'] = (grouped_procedure_df_op['count']/sum(grouped_procedure_df_op['count']))*100
grouped_procedure_df_op['Percentage']


# In[21]:


grouped_diagnosis_df_op = grouped_diagnosis_df2.to_frame()
grouped_diagnosis_df_op.columns = ['count']
grouped_diagnosis_df_op['Diagnosis'] = grouped_diagnosis_df_op.index
grouped_diagnosis_df_op['Percentage'] = (grouped_diagnosis_df_op['count']/sum(grouped_diagnosis_df_op['count']))*100
grouped_diagnosis_df_op['Percentage']


# In[22]:


# taking only top 20 

plot_procedure_df2 = grouped_procedure_df_op.head(20)
plot_diagnosis_df2 = grouped_diagnosis_df_op.head(20)


# In[23]:


# Plotting the most commonly used diagnosis and procedures 
from matplotlib import pyplot as plt


plot_procedure_df2['Procedure'] = plot_procedure_df2['Procedure'].astype(str)
plot_procedure_df2.sort_values(by=['Percentage'])
plot_procedure_df2.plot(x ='Procedure', y='Percentage', kind='bar', color ='yellow',
                   title='Procedure Distribution- Outpatient', figsize=(15,7));


# In[24]:


plot_diagnosis_df2['Diagnosis'] = plot_diagnosis_df2['Diagnosis'].astype(str)
plot_diagnosis_df2.sort_values(by=['Percentage'])
plot_diagnosis_df2.plot(x ='Diagnosis', y='Percentage', kind='bar', color ='yellow',
                   title='Diagnosis Distribution- Outpatient', figsize=(15,7))


# In[25]:


T_fraud = train['PotentialFraud'].value_counts()
grouped_train_df = T_fraud.to_frame()

grouped_train_df.columns = ['count']
grouped_train_df['Fraud'] = grouped_train_df.index
grouped_train_df['Percentage'] = (grouped_train_df['count']/sum(grouped_train_df['count']))*100
grouped_train_df['Percentage'].plot( kind='bar',color = "blue", title = 'Distribution')


# In[26]:


Train_f =  pd.DataFrame(columns = ['PotentialFraud', 'Provider'])
Train_f = train.loc[(train['PotentialFraud'] == 'Yes')]
Train_f


# In[27]:


fraud_provider_ip_df = pd.merge(inpatient, Train_f, how='inner', on='Provider')
fraud_provider_ip_df


# In[28]:


len(fraud_provider_ip_df)


# In[29]:


(len(fraud_provider_ip_df)/len(inpatient)) * 100


# In[30]:


fraud_provider_op_df = pd.merge(outpatient, Train_f, how='inner', on='Provider')
fraud_provider_op_df


# In[31]:


len(fraud_provider_op_df)


# In[32]:


(len(fraud_provider_op_df)/len(outpatient))*100


# In[33]:


df_procedures2 =  pd.DataFrame(columns = ['Procedures'])
df_procedures2['Procedures'] = pd.concat([fraud_provider_ip_df["ClmProcedureCode_1"], fraud_provider_ip_df["ClmProcedureCode_2"], fraud_provider_ip_df["ClmProcedureCode_3"], fraud_provider_ip_df["ClmProcedureCode_4"], fraud_provider_ip_df["ClmProcedureCode_5"], fraud_provider_ip_df["ClmProcedureCode_6"]], axis=0, sort=True).dropna()
df_procedures2['Procedures'].head(10)


# In[34]:


grouped_F_procedure_df = df_procedures2['Procedures'].value_counts()
grouped_F_procedure_df


# In[35]:


grouped_F_procedure_df2 = grouped_F_procedure_df.to_frame()
grouped_F_procedure_df2.columns = ['count']
grouped_F_procedure_df2['Procedure'] = grouped_F_procedure_df2.index
grouped_F_procedure_df2['Percentage'] = (grouped_F_procedure_df2['count']/sum(grouped_F_procedure_df2['count']))*100
grouped_F_procedure_df2['Percentage']


# In[36]:


df_diagnosis2 = pd.DataFrame(columns = ['Diagnosis'])
df_diagnosis2['Diagnosis'] = pd.concat([fraud_provider_ip_df["ClmDiagnosisCode_1"], fraud_provider_ip_df["ClmDiagnosisCode_2"], fraud_provider_ip_df["ClmDiagnosisCode_3"], fraud_provider_ip_df["ClmDiagnosisCode_4"], fraud_provider_ip_df["ClmDiagnosisCode_5"], fraud_provider_ip_df["ClmDiagnosisCode_6"], fraud_provider_ip_df["ClmDiagnosisCode_7"],  fraud_provider_ip_df["ClmDiagnosisCode_8"], fraud_provider_ip_df["ClmDiagnosisCode_9"], fraud_provider_ip_df["ClmDiagnosisCode_10"]], axis=0, sort=True).dropna()
df_diagnosis2['Diagnosis'].head(10)


# In[37]:


grouped_F_diagnosis_df = df_diagnosis2['Diagnosis'].value_counts()
grouped_F_diagnosis_df


# In[38]:


grouped_F_diagnosis_df2 = grouped_F_diagnosis_df.to_frame()
grouped_F_diagnosis_df2.columns = ['count']
grouped_F_diagnosis_df2['Diagnosis'] = grouped_F_diagnosis_df2.index
grouped_F_diagnosis_df2['Percentage'] = (grouped_F_diagnosis_df2['count']/sum(grouped_F_diagnosis_df2['count']))*100
grouped_F_diagnosis_df2['Percentage']


# In[39]:


plot_F_procedure_df1 = grouped_F_procedure_df2.head(20)

plot_F_diagnosis_df1 = grouped_F_diagnosis_df2.head(20)


# In[40]:


plot_F_procedure_df1.plot(x ='Procedure', y='Percentage', kind = 'bar', color ='g', figsize=(15,7))


# In[41]:


plot_F_diagnosis_df1.plot(x ='Diagnosis', y='Percentage', kind = 'bar', color ='y', figsize=(15,7))


# In[42]:


df_procedures_op2 =  pd.DataFrame(columns = ['Procedures'])
df_procedures_op2['Procedures'] = pd.concat([fraud_provider_op_df["ClmProcedureCode_1"], fraud_provider_op_df["ClmProcedureCode_2"], fraud_provider_op_df["ClmProcedureCode_3"], fraud_provider_op_df["ClmProcedureCode_4"], fraud_provider_op_df["ClmProcedureCode_5"], fraud_provider_op_df["ClmProcedureCode_6"]], axis=0, sort=True).dropna()
df_procedures_op2['Procedures'].head(10)


# In[43]:


grouped_F_procedure_op_df = df_procedures_op2['Procedures'].value_counts()
grouped_F_procedure_op_df.head()


# In[44]:


grouped_F_procedure_opdf2 = grouped_F_procedure_op_df.to_frame()
grouped_F_procedure_opdf2.columns = ['count']
grouped_F_procedure_opdf2['Procedure'] = grouped_F_procedure_opdf2.index
grouped_F_procedure_opdf2['Percentage'] = (grouped_F_procedure_opdf2['count']/sum(grouped_F_procedure_opdf2['count']))*100
grouped_F_procedure_opdf2['Percentage'].head()


# In[45]:


df_diagnosis_op2 = pd.DataFrame(columns = ['Diagnosis'])
df_diagnosis_op2['Diagnosis'] = pd.concat([fraud_provider_op_df["ClmDiagnosisCode_1"], fraud_provider_op_df["ClmDiagnosisCode_2"], fraud_provider_op_df["ClmDiagnosisCode_3"], fraud_provider_op_df["ClmDiagnosisCode_4"], fraud_provider_op_df["ClmDiagnosisCode_5"], fraud_provider_op_df["ClmDiagnosisCode_6"], fraud_provider_op_df["ClmDiagnosisCode_7"],  fraud_provider_op_df["ClmDiagnosisCode_8"], fraud_provider_op_df["ClmDiagnosisCode_9"], fraud_provider_op_df["ClmDiagnosisCode_10"]], axis=0, sort=True).dropna()
df_diagnosis_op2['Diagnosis'].head()


# In[46]:


grouped_F_diagnosis_op_df = df_diagnosis2['Diagnosis'].value_counts()
grouped_F_diagnosis_op_df.head()


# In[47]:


grouped_F_diagnosis_opdf2 = grouped_F_diagnosis_op_df.to_frame()
grouped_F_diagnosis_opdf2.columns = ['count']
grouped_F_diagnosis_opdf2['Diagnosis'] = grouped_F_diagnosis_opdf2.index
grouped_F_diagnosis_opdf2['Percentage'] = (grouped_F_diagnosis_opdf2['count']/sum(grouped_F_diagnosis_opdf2['count']))*100
grouped_F_diagnosis_opdf2['Percentage'].head()


# In[48]:


plot_F_procedure_opdf1 = grouped_F_procedure_opdf2.head(20)

plot_F_diagnosis_opdf1 = grouped_F_diagnosis_opdf2.head(20)


# In[49]:


plot_F_procedure_opdf1.plot(x ='Procedure', y='Percentage', kind = 'bar', color ='g', figsize=(15,7))


# In[50]:


plot_F_diagnosis_opdf1.plot(x ='Diagnosis', y='Percentage', kind = 'bar', color ='y', figsize=(15,7))


# In[51]:


beneficiary.head()


# In[52]:


fraud_beneficiary_ip_op_df = pd.merge(beneficiary, fraud_provider_ip_df, how='inner', on='BeneID')
fraud_beneficiary_ip_op_df.head()


# In[53]:


Train_F_Beneficiary_grouped = fraud_beneficiary_ip_op_df['State'].value_counts()
Train_F_Beneficiary_grouped.head()


# In[54]:


Train_F_Beneficiary_grouped1 = Train_F_Beneficiary_grouped.to_frame()
Train_F_Beneficiary_grouped1['Count'] =  Train_F_Beneficiary_grouped1['State']
Train_F_Beneficiary_grouped1['STATE'] = Train_F_Beneficiary_grouped1.index
Train_F_Beneficiary_grouped1 = Train_F_Beneficiary_grouped1.drop(['State'], axis = 1)
Train_F_Beneficiary_grouped1 = Train_F_Beneficiary_grouped1.head(20)
Train_F_Beneficiary_grouped1


# In[55]:


Train_F_Beneficiary_grouped1.plot(x ='STATE', y='Count', kind = 'bar', figsize= (15,7));


# In[56]:


fraud_beneficiary_ip_op_df['DOB'] =  pd.to_datetime(fraud_beneficiary_ip_op_df['DOB'], format='%Y-%m-%d')  
now = pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') # Assuming this is 2009 data as the last recorded death is for 2009
fraud_beneficiary_ip_op_df['DOB'] = fraud_beneficiary_ip_op_df['DOB'].where(fraud_beneficiary_ip_op_df['DOB'] < now) 
fraud_beneficiary_ip_op_df['age'] = (now - fraud_beneficiary_ip_op_df['DOB']).astype('<m8[Y]')  
ax = fraud_beneficiary_ip_op_df['age'].plot.hist(bins=20, alpha=0.5, figsize=(8, 6), edgecolor='b')


# In[57]:


beneficiary['DOB'] =  pd.to_datetime(beneficiary['DOB'], format='%Y-%m-%d')  
now = pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') # Assuming this is 2009 data as the last recorded death is for 2009
beneficiary['DOB'] = beneficiary['DOB'].where(beneficiary['DOB'] < now)
beneficiary['age'] = (now - beneficiary['DOB']).astype('<m8[Y]')
ax = beneficiary['age'].plot.hist(bins=20, alpha=0.5, figsize=(8, 6), edgecolor='b')


# In[58]:


beneficiary['DOB'] =  pd.to_datetime(beneficiary['DOB'], format='%Y-%m-%d')  
now = pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') # Assuming this is 2009 data as the last recorded death is for 2009
beneficiary['DOB'] = beneficiary['DOB'].where(beneficiary['DOB'] < now)
beneficiary['age'] = (now - beneficiary['DOB']).astype('<m8[Y]')
ax = beneficiary['age'].plot.hist(bins=20, alpha=0.5, figsize=(8, 6), edgecolor='b')


# In[59]:


ax = inpatient['InscClaimAmtReimbursed'].plot.hist(bins=20, alpha=0.5, figsize=(8, 6), facecolor='g', edgecolor='g')
# Insurance Claim amount reimbursed.


# In[60]:


import seaborn as sns
inpatient_1 = pd.merge(inpatient, train, how='inner', on='Provider')
g = sns.FacetGrid(inpatient_1, col='PotentialFraud', height=8)
g.map(plt.hist, 'InscClaimAmtReimbursed', bins=20, color = 'g')


# In[61]:


inpatient_1 = inpatient_1.loc[(inpatient_1['PotentialFraud'] == 'Yes')]
Total = inpatient_1['InscClaimAmtReimbursed'].sum()
print(Total)


# In[62]:


ax = outpatient['InscClaimAmtReimbursed'].plot.hist(bins=100,range=[0,5000], alpha=0.5, figsize=(8, 6), facecolor='c', edgecolor='k')


# In[63]:


outpatient_1 = pd.merge(outpatient, train, how='inner', on='Provider')
g = sns.FacetGrid(outpatient_1, col='PotentialFraud', height=8)
g.map(plt.hist, 'InscClaimAmtReimbursed', bins=20, range=[0, 5000], color ='c')


# In[64]:


beneficiary.isna().sum()


# In[65]:


beneficiary['DOB'] = pd.to_datetime(beneficiary['DOB'] , format = '%Y-%m-%d')
beneficiary['DOD'] = pd.to_datetime(beneficiary['DOD'],format = '%Y-%m-%d',errors='ignore')
beneficiary['Age'] = round(((beneficiary['DOD'] - beneficiary['DOB']).dt.days)/365)

## As we see that last DOD value is 2009-12-01 ,which means Beneficiary Details data is of year 2009.
## so we will calculate age of other benficiaries for year 2009.


# In[66]:


beneficiary.Age.fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - beneficiary['DOB']).dt.days)/365),
                                 inplace=True)


# In[67]:


beneficiary.head()


# In[68]:


## Creating the master DF
inpatient['EncounterType'] = 0
outpatient['EncounterType'] = 1
frames = [inpatient, outpatient]
TrainInAndOut = pd.concat(frames)
TrainInAndOutBenf = pd.merge(TrainInAndOut, beneficiary, how='inner', on='BeneID')
Master_df = pd.merge(TrainInAndOutBenf, train, how='inner', on='Provider')


# In[69]:


Master_df.head()


# In[70]:


Master_df['PotentialFraud'].value_counts()


# In[71]:


Master_df.shape


# In[72]:


Master_df.isnull().sum()


# In[73]:


## removing the column DOD and DOB also creating a new column IsDead as we already have the age we do not need date of death and date of birth 

Master_df.loc[Master_df['DOD'].isnull(), 'IsDead'] = '0'
Master_df.loc[(Master_df['DOD'].notnull()), 'IsDead'] = '1'
Master_df = Master_df.drop(['DOD'], axis = 1)
Master_df = Master_df.drop(['DOB'], axis = 1)


# In[74]:


Master_df = Master_df.drop(['age'], axis = 1) 


# In[75]:


Master_df['AdmissionDt'] = pd.to_datetime(Master_df['AdmissionDt'] , format = '%Y-%m-%d')
Master_df['DischargeDt'] = pd.to_datetime(Master_df['DischargeDt'],format = '%Y-%m-%d')
Master_df['DaysAdmitted'] = ((Master_df['DischargeDt'] - Master_df['AdmissionDt']).dt.days)+1
Master_df.loc[Master_df['EncounterType'] == 1, 'DaysAdmitted'] = '0'
Master_df[['EncounterType','DaysAdmitted','DischargeDt','AdmissionDt']].head()
Master_df = Master_df.drop(['DischargeDt'], axis = 1)
Master_df = Master_df.drop(['AdmissionDt'], axis = 1)


# In[76]:


Master_df.loc[Master_df['DeductibleAmtPaid'].isnull(), 'DeductibleAmtPaid'] = '0'


# In[77]:


cols= ['ClmAdmitDiagnosisCode', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_10',
       'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
       'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
       'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6']


# In[78]:


import numpy as np


# In[79]:


Master_df[cols]= Master_df[cols].replace({np.nan:0})
Master_df


# In[80]:


for i in cols:
    Master_df[i][Master_df[i]!=0]= 1


# In[81]:


Master_df[cols]= Master_df[cols].astype(float)


# In[82]:


Master_df['TotalDiagnosis']= Master_df['ClmDiagnosisCode_1']+Master_df['ClmDiagnosisCode_10']+Master_df['ClmDiagnosisCode_2']+ Master_df['ClmDiagnosisCode_3']+ Master_df['ClmDiagnosisCode_4']+Master_df['ClmDiagnosisCode_5']+ Master_df['ClmDiagnosisCode_6']+ Master_df['ClmDiagnosisCode_7']+Master_df['ClmDiagnosisCode_8']+ Master_df['ClmDiagnosisCode_9']


# In[83]:


Master_df['TotalProcedure']= Master_df['ClmProcedureCode_1']+Master_df['ClmProcedureCode_2']+Master_df['ClmProcedureCode_3']+ Master_df['ClmProcedureCode_4']+ Master_df['ClmProcedureCode_5']+Master_df['ClmProcedureCode_6']


# In[84]:


Master_df.columns


# In[85]:


remove=['Provider','BeneID', 'ClaimID', 'ClaimStartDt','ClaimEndDt','AttendingPhysician',
       'OperatingPhysician', 'OtherPhysician', 'ClmDiagnosisCode_1',
       'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
       'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
       'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
       'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
       'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6',
       'ClmAdmitDiagnosisCode','DeductibleAmtPaid','NoOfMonths_PartACov',
        'NoOfMonths_PartBCov','DiagnosisGroupCode',
        'State', 'County']


# In[86]:


Master_df.drop(columns=remove, axis=1, inplace=True)


# In[87]:


Master_df.head()


# In[88]:


Master_df.shape


# In[89]:


Master_df['RenalDiseaseIndicator'].value_counts()


# In[90]:


Master_df['RenalDiseaseIndicator']= Master_df['RenalDiseaseIndicator'].replace({'Y':1,'0':0})


# In[91]:


Master_df['RenalDiseaseIndicator']=Master_df['RenalDiseaseIndicator'].astype(int)


# In[92]:


Master_df.describe(include='O')


# In[93]:


Master_df['IsDead']=Master_df['IsDead'].astype(float)
Master_df['DaysAdmitted']=Master_df['DaysAdmitted'].astype(float)


# In[94]:


Master_df['PotentialFraud']=Master_df['PotentialFraud'].replace({'Yes':1, 'No':0})


# In[95]:


Master_df['PotentialFraud']=Master_df['PotentialFraud'].astype(int)


# In[96]:


Master_df['PotentialFraud']


# In[97]:


x= Master_df.drop('PotentialFraud', axis=1)
y= Master_df.loc[:,'PotentialFraud']


# In[98]:


x.columns


# In[99]:


num_col= ['InscClaimAmtReimbursed',
       'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
       'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Age',
       'DaysAdmitted', 'TotalDiagnosis', 'TotalProcedure']


# In[100]:


numerical_columns= x.loc[:,num_col]
numerical_columns.describe()


# In[101]:


numerical_columns.head()


# In[102]:


cat_col= ['EncounterType', 'Gender', 'Race',
       'RenalDiseaseIndicator', 'ChronicCond_Alzheimer',
       'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
       'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
       'ChronicCond_Depression', 'ChronicCond_Diabetes',
       'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
       'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke','IsDead']


# In[103]:


x_cat= x.loc[:,cat_col]
x_cat


# In[104]:


from sklearn.preprocessing import StandardScaler


# In[105]:


scale= StandardScaler()
x_num= scale.fit_transform(x[num_col])


# In[106]:


x_num= pd.DataFrame(x_num, columns=['InscClaimAmtReimbursed','IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt','OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Age','DaysAdmitted', 'TotalDiagnosis', 'TotalProcedure'])


# In[107]:


x= pd.concat([x_num, x_cat], axis=1)
x


# In[108]:


x.columns


# In[109]:


y


# In[110]:


from sklearn.model_selection import train_test_split


# In[111]:


x_train,x_test, y_train, y_test= train_test_split(x,y, test_size=0.1, random_state=42)


# In[112]:


from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
x_train1,y_train1 = rus.fit_resample(x_train, y_train)


# In[113]:


'''from imblearn import over_sampling

ada = over_sampling.ADASYN(random_state=0)
x_train2, y_train2 = ada.fit_resample(x_train, y_train)'''


# In[114]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, auc, roc_curve


# In[115]:


from xgboost import plot_importance
from xgboost import XGBClassifier
xgb= XGBClassifier()
xgb.fit(x_train,y_train)
plot_importance(xgb)


# In[116]:


xgb= XGBClassifier()
xgb.fit(x_train1,y_train1)
plot_importance(xgb)


# In[117]:


'''xgb.fit(x_train2,y_train2)
plot_importance(xgb)'''


# In[118]:


acc_score=[]


# In[119]:


from sklearn.model_selection import GridSearchCV


# In[120]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
estimator=  DecisionTreeClassifier()
param_grid= {'criterion':['gini', 'entropy'],
             'max_depth':[3,4,5],
             'min_samples_split':[2,3,5]
             }
grid_search = GridSearchCV(estimator = estimator, param_grid = param_grid)
grid_search.fit(x_train1, y_train1)
print(grid_search.best_score_)
print(grid_search.best_params_)


# In[121]:


'''param_grid= {'criterion':['gini', 'entropy'],
             'max_depth':[3,4,5],
             'min_samples_split':[2,3,5]
             }
grid_search = GridSearchCV(estimator = estimator, param_grid = param_grid)
grid_search.fit(x_train2, y_train2)
print(grid_search.best_score_)
print(grid_search.best_params_)'''


# In[122]:


grid_search.fit(x_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)


# In[123]:


from sklearn.ensemble import RandomForestClassifier
estimator1= RandomForestClassifier()
'''estimator1.fit(x_train, y_train)
model_score= estimator1.predict(x_train)
accuracy= estimator1.predict(x_test)
print(accuracy_score(y_train, model_score))
print(accuracy_score(y_test, accuracy))'''


# In[124]:


'''estimator1.fit(x_train1, y_train1)
model_score= estimator1.predict(x_train1)
accuracy= estimator1.predict(x_test)
print(accuracy_score(y_train1, model_score))
print(accuracy_score(y_test, accuracy))'''


# In[125]:


'''estimator1.fit(x_train2, y_train2)
model_score= estimator1.predict(x_train2)
accuracy= estimator1.predict(x_test)
print(accuracy_score(y_train2, model_score))
print(accuracy_score(y_test, accuracy))'''


# In[126]:


from sklearn.naive_bayes import GaussianNB
bayes= GaussianNB()
bayes.fit(x_train, y_train)
train_pred= bayes.predict(x_train)
test_pred= bayes.predict(x_test)
print(accuracy_score(y_train,train_pred))
print(accuracy_score(y_test,test_pred))


# In[127]:


bayes.fit(x_train, y_train)
train_pred= bayes.predict(x_train1)
test_pred= bayes.predict(x_test)
print(accuracy_score(y_train1,train_pred))
print(accuracy_score(y_test,test_pred))


# In[128]:


'''bayes.fit(x_train, y_train)
train_pred= bayes.predict(x_train2)
test_pred= bayes.predict(x_test)
print(accuracy_score(y_train2,train_pred))
print(accuracy_score(y_test,test_pred))'''


# In[129]:


from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(x_train, y_train)
train_pred= lr.predict(x_train)
test_pred= lr.predict(x_test)
print(accuracy_score(y_train,train_pred))
print(accuracy_score(y_test,test_pred))


# In[130]:


lr.fit(x_train1, y_train1)
train_pred= lr.predict(x_train1)
test_pred= lr.predict(x_test)
print(accuracy_score(y_train1,train_pred))
print(accuracy_score(y_test,test_pred))


# In[131]:


'''lr.fit(x_train2, y_train2)
train_pred= lr.predict(x_train2)
test_pred= lr.predict(x_test)
print(accuracy_score(y_train2,train_pred))
print(accuracy_score(y_test,test_pred))'''


# In[132]:


import time
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
estimator=  DecisionTreeClassifier(criterion= 'gini', max_depth=5, min_samples_split= 2)
estimator.fit(x_train, y_train)
model_score= estimator.predict(x_train)
accuracy= estimator.predict(x_test)
start = time.time()
estimator.score(x_train, y_train)
acc_random_forest = round(accuracy_score(y_test,accuracy)*100, 2)
f1_random_forest = round(f1_score(y_test,accuracy,average = "binary")*100, 2)
f_beta_random_forest = round(fbeta_score(y_test,accuracy,average = "binary",beta=0.5)*100, 2)

end = time.time()

acc_score.append({'Model':'Decision Tree', 'Score': accuracy_score(y_train, model_score), 'Accuracy': accuracy_score(y_test, accuracy), 'Time_Taken':end - start})


# In[133]:


fn= ['InscClaimAmtReimbursed', 'IPAnnualReimbursementAmt',
    'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt',
    'OPAnnualDeductibleAmt', 'Age', 'DaysAdmitted',
    'TotalDiagnosis', 'TotalProcedure', 'EncounterType', 'Gender', 'Race',
    'RenalDiseaseIndicator', 'ChronicCond_Alzheimer',
    'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
    'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
    'ChronicCond_Depression', 'ChronicCond_Diabetes',
    'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
    'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke', 'IsDead']


# In[134]:


cl=['No','Yes']


# In[135]:


fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (15,10), dpi=300)
tree.plot_tree(estimator, feature_names= fn, class_names=cl);


# In[136]:


m1= Master_df['DaysAdmitted'].mean()
s1= Master_df['DaysAdmitted'].std()
print((0.007*s1)+m1)
print((9.57*s1)+m1)
print((9.135*s1)+m1)


# In[137]:


m2= Master_df['OPAnnualDeductibleAmt'].mean()
s2= Master_df['OPAnnualDeductibleAmt'].std()
print((10.514*s2)+m2)


# In[138]:


m3= Master_df['Age'].mean()
s3= Master_df['Age'].std()
print((-0.481*s3)+m3)


# In[139]:


m4= Master_df['InscClaimAmtReimbursed'].mean()
s4= Master_df['InscClaimAmtReimbursed'].std()
print((10.075*s4)+m4)


# In[140]:


confusion_matrix(y_test,accuracy)


# In[141]:


tn, fp, fn, tp = confusion_matrix(y_test,accuracy).ravel()
(tn, fp, fn, tp)  


# In[142]:


train_fpr, train_tpr, thresholds = roc_curve(y_train, estimator.predict_proba(x_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, estimator.predict_proba(x_test)[:,1])
plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC CURVE")
plt.show()


# In[143]:


from sklearn.ensemble import RandomForestClassifier
estimator1= RandomForestClassifier()
estimator1.fit(x_train, y_train)
model_score= estimator1.predict(x_train)
accuracy= estimator1.predict(x_test)
start = time.time()
estimator1.score(x_train, y_train)
acc_random_forest = round(accuracy_score(y_test,accuracy)*100, 2)
f1_random_forest = round(f1_score(y_test,accuracy,average = "binary")*100, 2)
f_beta_random_forest = round(fbeta_score(y_test,accuracy,average = "binary",beta=0.5)*100, 2)
end = time.time()
acc_score.append({'Model':'Random Forest', 'Score': accuracy_score(y_train, model_score), 'Accuracy': accuracy_score(y_test, accuracy),'Time_Taken':end - start})


# In[144]:


confusion_matrix(y_test,accuracy)


# In[145]:


tn, fp, fn, tp = confusion_matrix(y_test,accuracy).ravel()
(tn, fp, fn, tp)


# In[146]:


train_fpr, train_tpr, thresholds = roc_curve(y_train, estimator1.predict_proba(x_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, estimator1.predict_proba(x_test)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC CURVE")
plt.show()


# In[147]:


y_test_rf= y_test.reset_index()
y_test.head()


# In[148]:


x_test_rf=  x_test.reset_index()
x_test.head()


# In[149]:


y_test_rf.shape


# In[150]:


accuracy= accuracy.reshape(55822,1)


# In[151]:


accuracy.shape


# In[152]:


accuracy= pd.DataFrame(accuracy, columns= ['Predict'])
accuracy


# In[153]:


predictor= pd.concat([y_test_rf,accuracy], axis=1)
predictor


# In[154]:


Index_label = predictor[(predictor['PotentialFraud'] ==1) & (predictor['Predict']==1)]
indicies= Index_label['index']


# In[155]:


wrong_predictions= Master_df.iloc[indicies,:]
wrong_predictions


# In[156]:


print('Fraud Insurance Claims detected - ',wrong_predictions['InscClaimAmtReimbursed'].sum())
print('Fraud Insurance Claims for Inpatients detected - ',wrong_predictions['IPAnnualReimbursementAmt'].sum())
print('Fraud Insurance Claims for Outpatients detected - ',wrong_predictions['OPAnnualReimbursementAmt'].sum())


# In[157]:


fraud_index= y_test_rf[y_test_rf['PotentialFraud']==1]
indicies1= fraud_index['index']


# In[158]:


frauds= Master_df.iloc[indicies1,:]
frauds


# In[159]:


print('Fraud Insurance Claims without model - ',frauds['InscClaimAmtReimbursed'].sum())
print('Fraud Insurance Claims for Inpatients without model - ',frauds['IPAnnualReimbursementAmt'].sum())
print('Fraud Insurance Claims for Outpatients without model - ',frauds['OPAnnualReimbursementAmt'].sum())


# In[160]:


print('Insurance Claim Amount Saved - $', 30673230- 23525520)
print('Inpatient Insurance Claim Amount Saved - $',122946970- 78254980)
print('Outpatient insurance Claim Amount Saved - $',48642580- 27754480)


# In[161]:


from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(x_train, y_train)
train_pred= lr.predict(x_train)
test_pred= lr.predict(x_test)
start = time.time()
lr.score(x_train, y_train)
acc_random_forest = round(accuracy_score(y_test,test_pred)*100, 2)
f1_random_forest = round(f1_score(y_test,test_pred,average = "binary")*100, 2)
f_beta_random_forest = round(fbeta_score(y_test,test_pred,average = "binary",beta=0.5)*100, 2)

end = time.time()
acc_score.append({'Model': "Logistic Regression", 'Score': accuracy_score(y_train,train_pred), 'Accuracy': accuracy_score(y_test,test_pred), 'Time_Taken':end - start})


# In[162]:


confusion_matrix(y_test,test_pred)


# In[163]:


tn, fp, fn, tp = confusion_matrix(y_test,test_pred).ravel()
(tn, fp, fn, tp)


# In[164]:


train_fpr, train_tpr, thresholds = roc_curve(y_train, lr.predict_proba(x_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, lr.predict_proba(x_test)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC CURVE")
plt.show()


# In[165]:


xgb= XGBClassifier()
xgb.fit(x_train, y_train)
model_score= xgb.predict(x_train)
accuracy= xgb.predict(x_test)
start = time.time()
xgb.score(x_train, y_train)
acc_random_forest = round(accuracy_score(y_test,accuracy)*100, 2)
f1_random_forest = round(f1_score(y_test,accuracy,average = "binary")*100, 2)
f_beta_random_forest = round(fbeta_score(y_test,accuracy,average = "binary",beta=0.5)*100, 2)

end = time.time()
acc_score.append({'Model':'XG boost', 'Score': accuracy_score(y_train, model_score), 'Accuracy': accuracy_score(y_test, accuracy), 'Time_Taken':end - start})


# In[166]:


confusion_matrix(y_test,accuracy)


# In[167]:


tn, fp, fn, tp = confusion_matrix(y_test,accuracy).ravel()
(tn, fp, fn, tp)


# In[168]:


train_fpr, train_tpr, thresholds = roc_curve(y_train, xgb.predict_proba(x_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, xgb.predict_proba(x_test)[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC CURVE")
plt.show()


# In[169]:


accuracy= pd.DataFrame(acc_score, columns=['Model','Score','Accuracy','Time_Taken'])
accuracy.sort_values(by='Accuracy', ascending= False, inplace= True)
accuracy


# In[170]:


plt.figure(figsize=(15,8))
sns.barplot(x= accuracy.Model, y=accuracy.Accuracy);


# In[171]:


import pickle


# In[172]:


pickle.dump(estimator1, open('medical.pkl','wb'))

model = pickle.load(open('medical.pkl','rb'))


# In[173]:


print(model.predict([[6.542662,2.610838,2.234826,-0.571436,-0.578530,-0.519851,2.832646,2.446318,-0.190910,0,1,1,0,1,1,1,2,2,1,1,1,2,1,1,0.0]]))

