#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas seaborn matplotlib numpy scipy scikit-learn')


# # Data Analysis and Preprocessing for Thermal Stability Dataset

# ### Importing all the dependencies

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


# ### Loading dataset from Github [It may take some time]

# In[3]:


#https://raw.githubusercontent.com/Shown246/CSE445_Datasets/refs/heads/main/thermal-stability.csv
df = pd.read_csv('https://raw.githubusercontent.com/Shown246/CSE445_Datasets/refs/heads/main/thermal-stability.csv')
df.head()


# ### Checking for missing values

# In[4]:


# Checking missing values
df.isnull().sum()


# ### Statistical features of the dataset

# In[5]:


df.describe()


# ### Showing Boxplot to view Outliers

# In[6]:


# List of columns for box plots
columns_for_boxplot = [
    "waveguide_width", "waveguide_height",  # Waveguide Dimensions
    "temperature_min", "temperature_max",   # Thermal Parameters
    "thermal_expansion_coefficient", "thermal_conductivity",
    "propagation_loss_min", "propagation_loss_max",  # Optical Performance
    "dn_dT", "thermal_tuning_efficiency",
    "thermal_stress", "strain_rate",  # Environmental and Operational
    "optical_power_input", "optical_power_output_min", "optical_power_output_max",
    "measurement_uncertainty"  # Simulation and Metadata
]

# Melt the dataframe to long format for seaborn
df_melted = df[columns_for_boxplot].melt(var_name="Feature", value_name="Value")

# Create a single box plot
plt.figure(figsize=(15, 8))
sns.boxplot(x="Feature", y="Value", data=df_melted)

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha="right")
plt.title("Box Plot of Thermal Stability Dataset Features", fontsize=14)
plt.xlabel("Feature")
plt.ylabel("Value")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# ## Analysis Statistical Outlier detection

# ### Z Score

# In[7]:


# Z-score outlier detection
# Select numerical columns
columns_to_check = df.select_dtypes(include=['float64', 'int64']).columns
# Compute Z-score
z_scores = np.abs(stats.zscore(df[columns_to_check]))

# Identify row indices containing outliers
outliers_indices = df[(z_scores > 3).any(axis=1)].index

# Total number of outliers
total_outliers = len(outliers_indices)
print(f"Total number of outliers: {total_outliers}")
# Remove outliers
df = df.drop(outliers_indices)
df.shape


# ### Interquantile Range

# In[8]:


# IQR calculation
Q1 = df[columns_to_check].quantile(0.25)  # 25th percentile (Q1)
Q3 = df[columns_to_check].quantile(0.75)  # 75th percentile (Q3)
IQR = Q3 - Q1  # Interquartile Range

# Identify outliers
# Outliers are values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
outliers_mask = (df[columns_to_check] < (Q1 - 1.5 * IQR)) | (df[columns_to_check] > (Q3 + 1.5 * IQR))

# Filter rows with at least one outlier
outliers_df = df[outliers_mask.any(axis=1)]

# Print results
print("Outliers detected using IQR:", outliers_df.shape[0])  # Number of rows with outliers
print("Outliers per column:")
print(outliers_mask.sum())  # Count of outliers in each column


# ## Machine Learning Method to detect Outlier

# ### Isolation Forest

# In[9]:


# Drop missing values for Isolation Forest analysis
df_clean = df[columns_to_check].dropna()

# Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df_clean['Outlier'] = iso_forest.fit_predict(df_clean)

# -1 indicates outliers
outliers_iso = df_clean[df_clean['Outlier'] == -1]

# Show the number of outliers detected
print("Outliers detected using Isolation Forest:", outliers_iso.shape[0])
outliers_iso.head()


# ### Local Outlier Factor

# In[10]:


# Clean dataset by dropping NaN values (if any)
df_clean = df[columns_to_check].dropna()

# Apply Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20)
outliers_lof = lof.fit_predict(df_clean)

# Create a new column for outliers (-1 means outlier, 1 means inlier)
df_clean['Outlier'] = outliers_lof

# -1 indicates outliers
df_lof_outliers = df_clean[df_clean['Outlier'] == -1]

# Show the number of outliers detected
print("Outliers detected using LOF:", df_lof_outliers.shape[0])

# Display first few detected outliers
df_lof_outliers.head()


# In[11]:


correlation = df[columns_to_check].corr()["thermal_tuning_efficiency"].sort_values(ascending=False)
print(correlation)


# In[12]:


# Drop the columns 
columns_to_drop = ["thermal_expansion_coefficient", "thermal_conductivity"]
df = df.drop(columns=columns_to_drop)


# In[13]:


# IQR calculation
# Columns to check for outliers
columns_to_check = df.select_dtypes(include=['float64','int64']).columns
Q1 = df[columns_to_check].quantile(0.25)
Q3 = df[columns_to_check].quantile(0.75)
IQR = Q3 - Q1

# Find outliers
outliers_mask = (df[columns_to_check] < (Q1 - 1.5 * IQR)) | (df[columns_to_check] > (Q3 + 1.5 * IQR))
outlier_IF = df[outliers_mask.any(axis=1)]

# Print results
print("Outliers detected using IQR:", outlier_IF.shape[0])
print("Outliers per column:")
print(outliers_mask.sum())
# Drop outliers
df = df.drop(outlier_IF.index)
df.shape


# ### Showing Box plot again after handling Outliers

# In[14]:


# Box plot of the cleaned dataset
columns_for_boxplot = [
    "waveguide_width", "waveguide_height",  # Waveguide Dimensions
    "temperature_min", "temperature_max",   # Thermal Parameters
    "propagation_loss_min", "propagation_loss_max",  # Optical Performance
    "dn_dT", "thermal_tuning_efficiency",
    "thermal_stress", "strain_rate",  # Environmental and Operational
    "optical_power_input", "optical_power_output_min", "optical_power_output_max",
    "measurement_uncertainty"  # Simulation and Metadata
]
df_melted1 = df[columns_for_boxplot].melt(var_name="Feature", value_name="Value")
plt.figure(figsize=(15, 8))
sns.boxplot(x="Feature", y="Value", data=df_melted1)
plt.xticks(rotation=45, ha="right")
plt.title("Box Plot of Cleaned Thermal Stability Dataset Features", fontsize=14)
plt.xlabel("Feature")
plt.ylabel("Value")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# ## Encoding

# In[15]:


# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_cols.tolist())

# Display unique values in categorical columns
for col in categorical_cols:
    print(f"\nUnique values in '{col}': {df[col].unique()}")


# ## Encoding Binary Categorical features
# ### Manually encoding to avoid Bias

# In[16]:


# Manually encode binary categorical features for consistency
df["cladding_material"] = df["cladding_material"].map({"Air": 0, "Silicon Dioxide (SiO₂)": 1})
df["simulation_model"] = df["simulation_model"].map({"Finite Element Method (FEM)": 0, "FDTD": 1})

# Apply One-Hot Encoding for `waveguide_material`
df = pd.get_dummies(df, columns=["waveguide_material"])
# Convert boolean to integer (0/1)
df[df.select_dtypes('bool').columns] = df.select_dtypes('bool').astype(int)
# Display encoded dataset
df.head()


# ## Scalling

# ### Plotting Histrograms to see data distrubustions

# In[17]:


# Plot histograms for all numerical features
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_features].hist(bins=30, figsize=(20, 15))
plt.suptitle('Feature Distributions', fontsize=20)
plt.show()


# In[18]:


# List of features to exclude (adjust if needed)
exclude_features = [
    'waveguide_material', 'cladding_material', 'simulation_model',
    'temperature_step'
]
# Select numerical features excluding the ones above
numerical_features = [col for col in df.columns if col not in exclude_features]
print(numerical_features)


# In[19]:


print(df.head())


# ### Applying Standard Scaling to Normal distrubution data

# In[20]:


scaler = StandardScaler()

# Fit and transform the numerical features
df_scaled = df.copy()
df_scaled[numerical_features] = scaler.fit_transform(df[numerical_features])

# Check the result
print(df_scaled.head())


# ### Before and After Scaling

# In[21]:


df.describe()


# In[22]:


df_scaled.describe()


# In[23]:


df=df_scaled


# ## # Exploratory Data Analysis

# In[24]:


sns.kdeplot(df["waveguide_width"], shade=True)
plt.show()


# ### Bar plots for binary and categorical features Vs Counts

# In[25]:


for col in ["cladding_material", "simulation_model", 
            "waveguide_material_Polymethyl Methacrylate (PMMA)", 
            "waveguide_material_SU-8", 
            "waveguide_material_Silicon Dioxide (SiO₂)", 
            "waveguide_material_Silicon Nitride (Si₃N₄)"]:
    if col in df.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=df[col])
        plt.xticks(rotation=45)
        plt.title(f"Distribution of {col}")
        plt.show()
    else:
        print(f"Column {col} does not exist in the dataframe.")


# ### Correlation Matrix

# In[26]:


# Correlation matrix
correlation_matrix = df[columns_to_check].corr()

# Plot the correlation matrix

plt.figure(figsize=(15,10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()


# # Data Analysis and Preprocessing for Credit Card Customer Churn Dataset

# ### Loading dataset from Github [It may take some time]

# In[28]:


# https://raw.githubusercontent.com/Shown246/CSE445_Datasets/refs/heads/main/Customer_Churn_Classification.csv
df = pd.read_csv('https://raw.githubusercontent.com/Shown246/CSE445_Datasets/refs/heads/main/Customer_Churn_Classification.csv')
df.head()


# ### In this dataset no missing value. Insted they are replaced with Zero. So we replaced the Zeros with NaN

# In[29]:


column_names = ["creditscore","age","tenure","balance","estimatedsalary","mem__no__products","cred_bal_sal","bal_sal","tenure_age","age_tenure_product"]
df[column_names] = df[column_names].replace(0, np.nan)


# In[30]:


# Check for missing values\
df.isnull().sum()


# ### Droped some unnecessary columns

# In[31]:


# drop the columns that are not needed
columns_to_drop = ["surname", "surname_tfidf_0" ,"surname_tfidf_1" ,"surname_tfidf_2", "surname_tfidf_3" ,"surname_tfidf_4"]
df = df.drop(columns=columns_to_drop)
df.describe()


# ### We also droped some columns that have no correlation with our target after seeing the Correlation matrix

# In[32]:


# Correlation matrix
corr = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()


# In[33]:


columns_to_drop = ["mem__no__products",	"cred_bal_sal",	"bal_sal", "balance"]
df = df.drop(columns=columns_to_drop)
df.isnull().sum()


# #### Droped the rows with missing values

# In[34]:


df = df.dropna()
df.shape


# ## Outlier Handling

# In[35]:


# Column to exclude
exclude = ["hascrcard", "isactivemember", "exited", "france","germany" ,"spain"]
# Columns to include
include = [col for col in df.columns if col not in exclude]

# Create boxplots
plt.figure(figsize=(20, 8))
sns.boxplot(data=df[include])
plt.xticks(rotation=45)
plt.title("Boxplot of Numerical Columns Customer_Churn_Classification")
plt.show()


# ## Analysis Statistical Outlier detection

# ### Z Score

# In[36]:


# Outlier detection using Z-score method
z_scores = stats.zscore(df[include])
abs_z_scores = abs(z_scores)
threshold = 3
outlier_indices = (abs_z_scores > threshold).any(axis=1)

# Number of outliers detected
print("Number of outliers detected:", outlier_indices.sum())

# Display the row indices containing outliers
outlier_rows = df[outlier_indices]
print("Row indices containing outliers:", outlier_rows.index.tolist())



# ### Interquantile Range

# In[37]:


# Calculate IQR
Q1 = df[include].quantile(0.25)
Q3 = df[include].quantile(0.75)
IQR = Q3 - Q1

# Detect outliers
outliers_mask = ((df[include] < (Q1 - 1.5 * IQR)) | (df[include] > (Q3 + 1.5 * IQR))).any(axis=1)
outliers_df = df[outliers_mask]

# Print results
print("Outliers detected using IQR:", outliers_df.shape[0])
# Drop the outliers
df = df[~outliers_mask]
print("After deleting Outliers detected column-wise:")
print(((df[include] < (Q1 - 1.5 * IQR)) | (df[include] > (Q3 + 1.5 * IQR))).sum())


# ## Machine Learning Method to detect Outlier

# In[38]:


# Outlier detection using LOF method
lof = LocalOutlierFactor(n_neighbors=20)
outlier_labels = lof.fit_predict(df[include])
outlier_indices_lof = df[outlier_labels == -1]

# Total number of outliers detected
print("Number of outliers detected using LOF:", outlier_indices_lof.shape[0])


# In[39]:


# Drop missing values for Isolation Forest analysis
df_clean = df[include]

# Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_preds = iso_forest.fit_predict(df_clean)
outliers_iso = df_clean[outlier_preds == -1]

# Total number of outliers detected
print("Number of outliers detected using Isolation Forest:", outliers_iso.shape[0])


# ## Seeing the Box plot again after handling outlier

# In[40]:


plt.figure(figsize=(20, 8))
sns.boxplot(data=df[include])
plt.xticks(rotation=45)
plt.title("Boxplot of Numerical Columns Customer_Churn_Classification")
plt.show()


# In[41]:


df.shape


# ## Scaling

# In[42]:


# Select only numerical columns
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

# Plot histograms
df[numerical_features].hist(figsize=(12, 12), bins=30)
plt.suptitle("Feature Distributions", fontsize=14)
plt.show()


# ### Applying Standard Scaling to Normal distrubution data and MinMax Scaler to Skewed data

# In[43]:


standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

# Apply StandardScaler for normally distributed features
df_scaled = df.copy()
df_scaled[['creditscore', 'age']] = standard_scaler.fit_transform(df[['creditscore', 'age']])

# Apply MinMaxScaler for skewed features
df_scaled[['estimatedsalary', 'tenure_age', 'age_tenure_product']] = minmax_scaler.fit_transform(
    df[['estimatedsalary', 'tenure_age', 'age_tenure_product']]
)


# In[44]:


print("Before Scaling:")
print(df[['creditscore', 'age', 'estimatedsalary', 'tenure_age', 'age_tenure_product']].describe())

print("\nAfter Scaling:")
print(df_scaled[['creditscore', 'age', 'estimatedsalary', 'tenure_age', 'age_tenure_product']].describe())


# In[45]:


#Check if Scaling worked

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.boxplot(df[['creditscore', 'age', 'estimatedsalary', 'tenure_age', 'age_tenure_product']])
plt.title("Before Scaling")

plt.subplot(1,2,2)
plt.boxplot(df_scaled[['creditscore', 'age', 'estimatedsalary', 'tenure_age', 'age_tenure_product']])
plt.title("After Scaling")

plt.show()


# # Exploratory Data Analysis

# In[46]:


df.head()


# ### Bar plots for binary and categorical features Vs Counts

# In[47]:


categorical_cols = ["numofproducts", "hascrcard", "isactivemember", "exited", "france", "germany", "spain", "female", "male"]

for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[col])
    plt.title(f"Distribution of {col}")
    plt.show()


# ### Visualizing some highly correlated features with our target features

# In[48]:


plt.figure(figsize=(8, 4))
sns.boxplot(x=df["exited"], y=df["creditscore"])
plt.title("Credit Score vs Exited")
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x=df["exited"], y=df["age"])
plt.title("Age vs Exited")
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(x=df["isactivemember"], hue=df["exited"])
plt.title("Active Members vs Exited")
plt.show()


# In[49]:


sns.pairplot(df[["creditscore", "age", "tenure", "numofproducts", "estimatedsalary", "exited"]], hue="exited")
plt.show()


# # Data Analysis and Preprocessing for Big Five Personality Dataset

# ### Importing Dependencies and loading the dataset
# #### It will take time as it's a big file [It can take upto 5 minitues]

# In[51]:


df = pd.read_csv('https://raw.githubusercontent.com/Shown246/CSE445_Datasets/refs/heads/main/Big_Five_Personality_Clustering.csv')
df.head()


# In[52]:


df.isnull().sum()


# In[53]:


df = df.drop(columns=['country'])
df.describe()


# In[54]:


numerical_cols = df.select_dtypes(include=['number']).columns

# Create boxplots for numerical columns
plt.figure(figsize=(20, 8))
sns.boxplot(data=df[numerical_cols])
plt.xticks(rotation=45)  # Rotate x-axis labels if necessary
plt.title("Boxplot of Numerical Columns")
plt.show()


# In[55]:


# Outlier detection using Z-score method
z_scores = stats.zscore(df[numerical_cols])
abs_z_scores = abs(z_scores)
threshold = 3
outlier_indices = (abs_z_scores > threshold).any(axis=1)

# Number of outliers detected
print("Number of outliers detected:", outlier_indices.sum())


# In[56]:


# Outlier detection using IQR method
Q1 = df[numerical_cols].quantile(0.25)
Q3 = df[numerical_cols].quantile(0.75)
IQR = Q3 - Q1

outliers_mask = (df[numerical_cols] < (Q1 - 1.5 * IQR)) | (df[numerical_cols] > (Q3 + 1.5 * IQR))
outliers_df = df[outliers_mask.any(axis=1)]

# Print IQR results
print("Outliers detected using IQR:", outliers_df.shape[0])
# print(outliers_mask.sum())  # Total number of outliers column-wise


# ### We won't remove the outliers
# #### These values are just answers to some questions ranged 1-6
# #### So, Statistically they are outlier but these are not make sense as outlier

# ### Same reason there's no need for Scaling

# # # Exploratory Data Analysis

# In[57]:


df.hist(figsize=(15, 10), bins=20)
sns.despine()
plt.tight_layout()
plt.show()


# In[58]:


plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

