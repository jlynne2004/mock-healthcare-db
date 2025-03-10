#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries to use
from faker import Faker
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


# In[2]:


#Set seed for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

#Number of patients
num_patients = 250


# In[3]:


#Create patient base data
def generate_sleep_disorder_dataset():
    #Patient demographics
    patient_ids = [f'P{str(i).zfill(4)}' for i in range(1, num_patients + 1)]

    # Age with realistic distribution for sleep disorders
    ages_younger = np.random.normal(30, 5, int(num_patients * 0.15))
    ages_middle = np.random.normal(45, 10, int(num_patients * 0.45))
    # Ensure we have exactly num_patients by adjusting the last group
    remaining_count = num_patients - (len(ages_younger) + len(ages_middle))
    ages_older = np.random.normal(65, 8, remaining_count)
    
    ages = np.concatenate([ages_younger, ages_middle, ages_older])
    ages = [max(18, min(90, int(age))) for age in ages]  # Clamp between 18-90

    #Gender (slightly more common in males for some sleep disorders)
    genders = np.random.choice(['Male', 'Female'], num_patients, p=[0.55, 0.45])

    # BMI - tends to be higher in sleep apnea patients
    bmis_normal = np.random.normal(24, 3, int(num_patients * 0.3))
    bmis_overweight = np.random.normal(29, 4, int(num_patients * 0.4))
    # Ensure we have exactly num_patients by adjusting the last group
    remaining_count = num_patients - (len(bmis_normal) + len(bmis_overweight))
    bmis_obese = np.random.normal(34, 5, remaining_count)

    bmis = np.concatenate([bmis_normal, bmis_overweight, bmis_obese])
    bmis = [max(18.5, min(45, round(bmi, 1))) for bmi in bmis]

    # Verify lengths of arrays
    assert len(patient_ids) == num_patients
    assert len(ages) == num_patients
    assert len(genders) == num_patients
    assert len(bmis) == num_patients

    #Sleep disorder types with realistic distribution 
    disorder_types = np.random.choice(
        ['Sleep Apnea', 'Insomnia', 'Narcolepsy',' Restless Leg Syndrome', 'Circardian Rhythm Disorder'],
        num_patients,
        p=[0.45, 0.30, 0.08, 0.12, 0.05] #sleep apnea and insomnia are most common
    )

    #Severity scores (1-10)
    severity_scores = []
    for disorder in disorder_types:
        if disorder == 'Sleep Apnea':
            #Sleep apnea tends to have higher severity
            severity_scores.append(min(10, max(1, int(np.random.normal(7, 2)))))
        elif disorder == 'Insomnia':
            severity_scores.append(min(10, max(1, int(np.random.normal(6, 2)))))
        else:
            severity_scores.append(min(10, max(1, int(np.random.normal(5, 2)))))


    #Treatment types based on disorder
    treatment_types = []
    for i, disorder in enumerate(disorder_types):
        if disorder == 'Sleep Apnea':
            if severity_scores[i] >= 7:
                treatment_types.append(np.random.choice(['CPAP', 'BiPAP'], p=[0.8, 0.2]))
            else:
                treatment_types.append(np.random.choice(['CPAP', 'Oral Appliance', 'Lifestyle Changes'],
                                                        p=[0.6, 0.2, 0.2]))
        elif disorder == 'Insomnia':
            treatment_types.append(np.random.choice(['CBT-I', 'Sleep Medication', 'Lifestyle Changes'],
                                                    p=[0.4, 0.4, 0.2]))
        elif disorder == 'Narcolepsy':
            treatment_types.append(np.random.choice(['Stimulant Medication', 'Scheduled Naps', 'Lifestyle Changes'],
                                                    p=[0.7, 0.15, 0.15]))
        elif disorder == 'Restless Leg Syndrome':
            treatment_types.append(np.random.choice(['Iron Supplements', 'Dopaminergic Agents', 'Lifestyle Changes'],
                                                    p=[0.3, 0.5, 0.2]))
        else: #Circardian Rhythm Disorder
            treatment_types.append(np.random.choice(['Light Therapy', 'Melatonin', 'Sleep Schedule Adjustment'],
                                                    p=[0.3, 0.3, 0.4]))

    #Treatment duration (weeks)
    durations = []
    for i, treatment in enumerate(treatment_types):
        if treatment in ['CPAP', 'BiPAP']:
            #CPAP is ongoing treatment
            durations.append(np.random.randint(24, 52))
        elif treatment in ['Stimulant Medication', 'Dopaminergic Agents']:
            durations.append(np.random.randint(16, 36))
        else:
            durations.append(np.random.randint(8, 24))

    #Georgraphic regions
    regions = np.random.choice(
        ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West'],
        num_patients,
        p=[0.25, 0.20, 0.20, 0.15, 0.20]
    )

    #Insurance types
    insurance_types = np.random.choice(
        ['Medicare', 'Medicaid', 'Private', 'Self-Pay', 'Other'],
        num_patients,
        p=[0.30, 0.15, 0.40, 0.10, 0.05]
    )

    # Verify all arrays have the same length before creating DataFrame
    assert len(disorder_types) == num_patients
    assert len(severity_scores) == num_patients
    assert len(treatment_types) == num_patients
    assert len(durations) == num_patients
    assert len(regions) == num_patients
    assert len(insurance_types) == num_patients

    #Create the basic dataframe
    data = {
        'PatientID': patient_ids,
        'Age': ages,
        'Gender': genders,
        'BMI': bmis,
        'DisorderType': disorder_types,
        'SeverityScore': severity_scores,
        'TreatmentType': treatment_types,
        'TreatmentDurationWeeks': durations,
        'Region': regions,
        'InsuranceType': insurance_types
    }

    return pd.DataFrame(data)


# In[4]:


#Generate the base dataframe
df = generate_sleep_disorder_dataset()


# In[5]:


#Now add pre-treatment metrics
def add_pre_treatment_metrics(df):
    #Sleep Efficiency (%)
    df['Pre_SleepEfficiency'] = df.apply(
        lambda row: max(50, min(75, np.random.normal(65, 7))),
        axis=1
    )

    #Total Sleep Time (hours)
    df['Pre_TotalSleepTime'] = df.apply(
        lambda row: max(3, min(6, np.random.normal(4.5, 1))),
        axis=1
    )

    #Sleep Latency (minutes)
    df['Pre_SleepLatency'] = df.apply(
        lambda row: max(30, min(90, np.random.normal(60, 15))) if row['DisorderType'] == 'Insomnia'
        else max(15, min(60, np.random.normal(30, 15))),
        axis=1
    )

    #Apnea-Hypopnea Index (events per hour)
    df['Pre_AHI'] = df.apply(
        lambda row: max(5, min(50, np.random.normal(20, 10))) if row['DisorderType'] == 'Sleep Apnea'
        else np.random.normal(2, 1),
        axis=1
    )

    #Oxygen Saturation (lowest %)
    df['Pre_OxygenSaturation'] = df.apply(
        lambda row: max(70, min(88, np.random.normal(82, 5))) if row['DisorderType'] == 'Sleep Apnea'
        else max(85, min(95, np.random.normal(92, 2))),
        axis=1
    )

    #Epworth Sleepiness Scale (0-24)
    df['Pre_EpworthScale'] = df.apply(
        lambda row: max(10, min(20, np.random.normal(15, 3))),
        axis=1
    )

    #Pittsburgh Sleep Quality Index (0-21)
    df['Pre_PSQI'] = df.apply(
        lambda row: max(10, min(18, np.random.normal(14, 2))),
        axis=1
    )

    #Fatigue Severity Scale (1-7)
    df['Pre_FatigueSeverity'] = df.apply(
        lambda row: max(4, min(7, np.random.normal(5.5, 0.8))),
        axis=1
    )

    #Daytime Functioning Score (1-10)
    df['Pre_DaytimeFunctioning'] = df.apply(
        lambda row: max(3, min(6, np.random.normal(4.5, 0.8))),
        axis=1
    )

    #Mood Assessment (1-10)
    df['Pre_MoodScore'] = df.apply(
        lambda row: max(3, min(6, np.random.normal(4.5, 0.8))),
        axis=1
    )

    #Medication Usage (sleep aids per week)
    df['Pre_MedicationUsage'] = df.apply(
        lambda row: max(3, min(7, np.random.normal(5, 1))) if row['DisorderType'] == 'Insomnia'
        else max(0, min(4, np.random.normal(2, 1))),
        axis=1
    )

    return df


# In[6]:


#Add pre-treatment metrics
df = add_pre_treatment_metrics(df)


# In[7]:


#Add post-treatment metrics based on pre-treatment values and treatment effectiveness
def add_post_treatment_metrics(df):
    #Treatment effectiveness - some randomness but generally good results
    #Higher effectiveness for certain treatments
    df['TreatmentEffectiveness'] = df.apply(
        lambda row: np.random.uniform(0.6, 0.9) if row['TreatmentType'] in ['CPAP', 'BiPAP', 'CBT-I']
        else np.random.uniform(0.4, 0.8),
        axis=1
    )

    #Post-Treatment Sleep Efficiency
    df['Post_SleepEfficiency'] = df.apply(
        lambda row: min(95, row['Pre_SleepEfficiency'] +
                    (95 - row['Pre_SleepEfficiency']) * row['TreatmentEffectiveness']),
        axis=1
    )

    #Post-Treatment Total Sleep Time
    df['Post_TotalSleepTime'] = df.apply(
        lambda row: min(8, row['Pre_TotalSleepTime'] +
                    (7.5 - row['Pre_TotalSleepTime']) * row['TreatmentEffectiveness']),
        axis=1
    )

    #Post-Treatment Sleep Latency (lower is better)
    df['Post_SleepLatency'] = df.apply(
        lambda row: max(5, row['Pre_SleepLatency'] - 
                       (row['Pre_SleepLatency'] - 10) * row['TreatmentEffectiveness']),
        axis=1
    )

    #Post-Treatment AHI (lower is better)
    df['Post_AHI'] = df.apply(
        lambda row: max(0, row['Pre_AHI'] - 
                    (row['Pre_AHI'] - 2) * row['TreatmentEffectiveness']) if row['DisorderType'] == 'Sleep Apnea'
        else row['Pre_AHI'],
        axis=1
    )

    #Post-Treatment Oxygen Saturation (higher is better)
    df['Post_OxygenSaturation'] = df.apply(
        lambda row: min(96, row['Pre_OxygenSaturation'] +
                    (95 - row['Pre_OxygenSaturation']) * row['TreatmentEffectiveness']) if row['DisorderType'] == 'Sleep Apnea'
        else row['Pre_OxygenSaturation'],
        axis=1
    )

    #Post-Treatment Epworth Scale (lower is better)
    df['Post_EpworthScale'] = df.apply(
        lambda row: max(0, row['Pre_EpworthScale'] - 
                    (row['Pre_EpworthScale'] - 5) * row['TreatmentEffectiveness']),
        axis=1
    )

    #Post-Treatment PSQI (lower is better)
    df['Post_PSQI'] = df.apply(
        lambda row: max(0, row['Pre_PSQI'] - 
                       (row['Pre_PSQI'] - 4) * row['TreatmentEffectiveness']),
        axis=1
    )

    #Post-Treatment Fatigue Severity (lower is better)
    df['Post_FatigueSeverity'] = df.apply(
        lambda row: max(1, row['Pre_FatigueSeverity'] - 
                    (row['Pre_FatigueSeverity'] - 2) * row['TreatmentEffectiveness']),
        axis=1
    )

    #Post-Treatment Daytime Functioning (higher is better)
    df['Post_DaytimeFunctioning'] = df.apply(
        lambda row: min(10, row['Pre_DaytimeFunctioning'] +
                    (9 - row['Pre_DaytimeFunctioning']) * row['TreatmentEffectiveness']),
        axis=1
    )

    #Post-Treatment Mood Score (higher is better)
    df['Post_MoodScore'] = df.apply(
        lambda row: min(10, row['Pre_MoodScore'] +
                       (9 - row['Pre_MoodScore']) * row['TreatmentEffectiveness']),
        axis=1
    )

    #Post-Treatment Medication Usage (lower is better)
    df['Post_MedicationUsage'] = df.apply(
        lambda row: max(0, row['Pre_MedicationUsage'] + 
                       (row['Pre_MedicationUsage'] - 1) * row['TreatmentEffectiveness']),
        axis=1
    )

    #Round all metrics to 1 decimal place for readability
    metric_columns = [col for col in df.columns if col.startswith('Pre_') or col.startswith('Post_')]
    df[metric_columns] = df[metric_columns].round(1)

    return df


# In[8]:


#Add post-treatment metrics
df = add_post_treatment_metrics(df)


# In[9]:


#Add treatment compliance data
def add_treatment_compliance(df):
    #CPAP usage (hours per night)
    df['CPAP_HoursPerNight'] = df.apply(
        lambda row: round(np.random.normal(5, 1.5), 1) if row['TreatmentType'] in ['CPAP', 'BiPAP'] else None,
        axis=1
    )

    #Treatment compliance (% of prescribed days)
    df['TreatmentCompliance'] = df.apply(
        lambda row: min(100, max(40, np.random.normal(75, 15))),
        axis=1
    )

    #Success flag (based on overall improvement)
    df['TreatmentSuccess'] = df.apply(
        lambda row:
            'Excellent' if (row['Post_SleepEfficiency'] - row['Pre_SleepEfficiency'])/row['Pre_SleepEfficiency'] > 0.25
            else 'Good' if (row['Post_SleepEfficiency'] - row['Pre_SleepEfficiency'])/row['Pre_SleepEfficiency'] > 0.15
            else 'Moderate' if (row['Post_SleepEfficiency'] - row['Pre_SleepEfficiency'])/row['Pre_SleepEfficiency'] > 0.05
            else 'Poor',
        axis=1
    )

    return df


# In[10]:


#Add treatment compliance
df = add_treatment_compliance(df)


# In[11]:


#Add appointment data
def add_appointment_data(df):
    #Start date (between 6-12 months ago)
    today = datetime.now()
    start_dates = [(today - timedelta(days=random.randint(180, 365))).strftime('%Y-%m-%d') for _ in range(num_patients)]
    df['StartDate'] = start_dates

    #Number of scheduled appointments
    df['ScheduledAppointments'] = df.apply(
        lambda row: random.randint(4, 12),
        axis=1
    )

    #Number of completed appointments
    df['CompletedAppointments'] = df.apply(
        lambda row: int(row['ScheduledAppointments'] * random.uniform(0.7, 1.0)),
        axis=1
    )

    #Number of cancelled appointments
    df['CancelledAppointments'] = df.apply(
        lambda row: row['ScheduledAppointments'] - row['CompletedAppointments'],
        axis=1
    )

    #Next appointment date (for active patients)
    next_dates = []
    for i in range(num_patients):
        if random.random() < 0.8: #80% of patients have future appointments
            next_dates.append((today + timedelta(days=random.randint(1, 60))).strftime('%m/%d/%Y'))
        else:
            next_dates.append(None)
    df['NextAppointmentDate'] = next_dates

    return df


# In[12]:


#Add appointment data
df = add_appointment_data(df)

#Export the dataset
df.to_excel('sleep_disorder_dataset.xlsx', index=False)

print(f'Generated dataset with {len(df)} patient records and {len(df.columns)} variables.')
print('Sample of the data:')
print(df.head())


# In[ ]:




