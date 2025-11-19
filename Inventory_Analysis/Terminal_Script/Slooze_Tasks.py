#!/usr/bin/env python
# coding: utf-8

# ## Tasks performed: 1) ABC Analysis 2) Reorder Point Analysis 3) Lead Time Analysis
# ## Demand forecasting is performed in other file with streamlit hosting

# 

# # Importing required Libraries

# In[1]:


import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter('ignore')


# # Loading and Preprocessing Data

# In[2]:


sales=pd.read_csv(r'D:\Slooze Dataset\slooze_challenge\SalesFINAL12312016.csv')
purchases = pd.read_csv(r'D:\Slooze Dataset\slooze_challenge\PurchasesFINAL12312016.csv')


# In[3]:


sales.info()


# In[4]:


purchases.info()


# ## Converting Datatypes of Date columns

# In[5]:


#for sales data
sales['SalesDate'] = pd.to_datetime(sales['SalesDate'], errors='coerce')

#for purchase data
date_cols = ["PODate", "ReceivingDate", "InvoiceDate", "PayDate"]
for col in date_cols:
    purchases[date_cols] = purchases[date_cols].apply(pd.to_datetime, errors='coerce')


# In[ ]:





# # 1) ABC ANALYSIS (based on total revenue of each item)

# ## i) Grouping data on  "InventoryId", "Description"  to get aggregate sales...followed by sorting in decreasing order

# In[6]:


abc = sales.groupby(["InventoryId", "Description" ] , as_index=False).agg(
    total_revenue = ("SalesDollars", "sum"))


# In[7]:


# Sort by revenue
abc = abc.sort_values("total_revenue", ascending=False).reset_index(drop=True)


# In[8]:


abc.head()


# In[9]:


abc.count()


# ## ii) Creating cumulative revenue and cumulative percent for aggregated revenue

# In[10]:


# Cumulative percentage
abc["cum_revenue"] = abc["total_revenue"].cumsum()


# In[11]:


abc.head(5)


# In[ ]:





# In[12]:


#Cumulative percent
abc["cum_percent"] = abc["cum_revenue"] / abc["total_revenue"].sum() * 100


# In[13]:


abc.head()


# In[14]:


abc.tail()


# ## iii)Preparing and applying ABC criteria to classify product

# In[15]:


# ABC classification rule
def classify(p):
    if p <= 70:
        return "A"
    elif p <= 90:
        return "B"
    else:
        return "C"


# In[16]:


abc['ABC_class']=abc['cum_percent'].apply(classify)


# In[17]:


abc.head()


# In[18]:


num_A=len(abc[abc['ABC_class']=='A'])
num_B=len(abc[abc['ABC_class']=='B'])
num_C=len(abc[abc['ABC_class']=='C'])


# ## iv) ABC Classification summary and Visualisation

# In[19]:


print("ABC Classification Summary")
print("----------------------------")
print(f"Class A Products: {num_A}")
print(f"Class B Products: {num_B}")
print(f"Class C Products: {num_C}")


# In[20]:


labels = ['Class A', 'Class B', 'Class C']
sizes = [num_A, num_B, num_C]
colors = ['#FF6F91', '#9B59B6',  '#58D68D']  

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',textprops={'color':'black', 'fontsize':12}, startangle=140)
plt.title("ABC Classification Distribution", color='black', fontsize=16)
plt.tight_layout()
plt.show()


# # 2. Reorder Point Analysis

# ## i) Preparing daily demand data by aggregating sales quantity with grouping by 'InventoryId', 'Description', 'SalesDate'

# In[21]:


daily_demand = (
    sales.groupby(['InventoryId', 'Description', 'SalesDate'], as_index=False)['SalesQuantity']
         .sum()
         .rename(columns={'SalesQuantity': 'daily_demand'})
)


# In[22]:


daily_demand.head()


# In[23]:


len(daily_demand)


# In[ ]:





# ## ii) Preparing Statistical data for daily demand : Mean and std dev

# In[24]:


# Demand stats per product
demand_stats = (
    daily_demand.groupby(['InventoryId', 'Description'])
                .agg(
                    avg_daily_demand=('daily_demand', 'mean'),
                    std_daily_demand=('daily_demand', 'std')
                )
                .reset_index()
)


# In[25]:


len(demand_stats)


# In[26]:


demand_stats.head()


# In[27]:


demand_stats.isna().sum()


# In[ ]:





# In[28]:


demand_stats['std_daily_demand'].fillna(0,inplace=True)   #filling std deviation null values with 0


# In[29]:


demand_stats.head()


# ## iii) Using purchase data to get lead days 

# In[30]:


pur=purchases.copy()


# In[31]:


pur.head()


# In[32]:


pur['lead_days'] = (pur['ReceivingDate'] - pur['PODate']).dt.days


# In[33]:


pur.head()


# ## iv) Preparing statistical data for lead days

# In[34]:


lead_stats = (
    pur.groupby(['InventoryId', 'Description'])
             .agg(
                 avg_lead_time=('lead_days', 'mean'),
                 std_lead_time=('lead_days', 'std'),
                 total_orders=('lead_days', 'count')
             )
             .reset_index()
)


# In[35]:


lead_stats.head()


# In[36]:


lead_stats.isnull().sum()


# In[37]:


lead_stats['std_lead_time'].fillna(0,inplace=True)
lead_stats.isnull().sum()


# ## v) Merging Daily demand and Lead time data 

# In[38]:


#Merge Demand + Lead Time
rop_df = demand_stats.merge(lead_stats, on=['InventoryId', 'Description'], how='inner')
rop_df.head()


# In[39]:


len(rop_df)


# In[40]:


rop_df.isnull().sum()


# ## vi) Computing Safety Stock and ROP

# In[41]:


#Compute Safety Stock
Z = 1.65   # 95% service level

rop_df['safety_stock'] = (Z * rop_df['std_daily_demand'] * np.sqrt(rop_df['avg_lead_time']))


# In[42]:


# ROP calculation
rop_df['ROP'] = (rop_df['avg_daily_demand'] * rop_df['avg_lead_time'] + rop_df['safety_stock'])


# In[43]:


rop_df.head()


# In[44]:


# Round values
rop_df[['avg_daily_demand', 'avg_lead_time']] = \
    rop_df[['avg_daily_demand', 'avg_lead_time']].round(2)

rop_df[[ 'safety_stock', 'ROP']] = \
    rop_df[[ 'safety_stock', 'ROP']].round()



# In[45]:


rop_df.head()


# ## vii) REORDER POINT (ROP) SUMMARY

# In[46]:


print("====================================")
print("REORDER POINT (ROP) ANALYSIS SUMMARY")
print("====================================")

# 1. Total products analyzed
total_products = len(rop_df)
print(f"Total Products Analyzed: {total_products}")

# 2. Products requiring safety stock (std > 0)
products_with_variability = (rop_df['std_daily_demand'] > 0).sum()
print(f"Products With Demand Variability (Safety Stock > 0): {products_with_variability}")

# 3. Products with zero variability
products_no_variability = (rop_df['std_daily_demand'] == 0).sum()
print(f"Products With Zero Variability (Safety Stock = 0): {products_no_variability}")
print()
print("----------------------------------------------------")

print()

# TOP 5 HIGHEST ROP

top_rop = rop_df.nlargest(5, 'ROP')[['InventoryId', 'Description', 'avg_daily_demand', 'avg_lead_time', 'ROP']]
print("Top 5 Products With Highest Reorder Point:")
print("-----------------------------------------")
print(top_rop.to_string(index=False))
print()
print("----------------------------------------------------")
print()


# TOP 5 LOWEST ROP

bottom_rop = rop_df.nsmallest(5, 'ROP')[['InventoryId', 'Description', 'avg_daily_demand', 'avg_lead_time', 'ROP']]
print("Top 5 Products With Lowest Reorder Point:")
print("----------------------------------------")
print(bottom_rop.to_string(index=False))
print()
print("------------------------------------------------------")
print()


# AVERAGE VALUES SUMMARY

print("Average Metrics Across All Products:")
print("-------------------------------------")
print(f"Average Daily Demand: {rop_df['avg_daily_demand'].mean():.2f}")
print(f"Average Lead Time: {rop_df['avg_lead_time'].mean():.2f}")
print(f"Average Safety Stock: {rop_df['safety_stock'].mean():.2f}")
print(f"Average ROP: {rop_df['ROP'].mean():.2f}")
print()


# # 3. LEAD TIME ANALYSIS (purchase data)

# ## i) Using Purchase data for generating Lead days data 

# In[47]:


purchases["lead_days"] = (purchases["ReceivingDate"] - purchases["PODate"]).dt.days
purchases.head(1)


# ## ii) Preparing statistical data with respect to lead time for analysis

# In[48]:


lead = purchases.groupby(["InventoryId",'Description'], as_index=False).agg(
    avg_lead_time = ("lead_days","mean"),
    median_lead_time = ("lead_days","median"),
    total_orders = ("lead_days","count")
)


# In[49]:


lead['avg_lead_time']=lead['avg_lead_time'].round(2)


# In[50]:


lead.head()


# ### Lead Time Analysis Summary

# In[51]:


print(" ============================")
print("  Lead Time Analysis Summary ")
print(" ============================")


print(f"Total Products Analyzed: {lead.shape[0]}")

print()

print(f"Average Lead Time (Overall): {lead['avg_lead_time'].mean():.2f} days")
print()

print(f"Median Lead Time (Overall): {lead['median_lead_time'].median():.2f} days")
print()

# Products with longest lead time (top 5)
top_longest = lead.nlargest(5, 'avg_lead_time')[['InventoryId', 'Description', 'avg_lead_time']]
print("Top 5 Products with Longest Lead Time:")
print("------------------------------------")
print(top_longest.to_string(index=False))
print()

# Products with shortest lead time (top 5)
top_shortest = lead.nsmallest(5, 'avg_lead_time')[['InventoryId', 'Description', 'avg_lead_time']]
print("Top 5 Products with Shortest Lead Time:")
print("-------------------------------------")
print(top_shortest.to_string(index=False))
print()


# In[ ]:




