
# Read all types of file : 
import pandas as pd

def read_file(file_path):
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'csv':
        df = pd.read_csv(file_path)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file_path)
    elif file_extension == 'json':
        df = pd.read_json(file_path)
    elif file_extension == 'parquet':
        df = pd.read_parquet(file_path)
    elif file_extension == 'hdf':
        df = pd.read_hdf(file_path)
    elif file_extension == 'feather':
        df = pd.read_feather(file_path)
    elif file_extension == 'sql':
        # Make sure to replace 'table_name' with the actual table name in the SQL database.
        df = pd.read_sql_table('table_name', 'sqlite:///' + file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    return df

if __name__ == "__main__":
    file_path = "path/to/your/file.csv"  # Replace with the path to your file
    try:
        df = read_file(file_path)
        print(df.head())
    except ValueError as ve:
        print(ve)


##To read different types of files using pandas, you can use various pandas functions that support different file formats. Here's a Python code that demonstrates how to read different types of files using pandas

import pandas as pd

def write_file(df, file_path, file_format):
    file_extension = file_format.lower()
    
    if file_extension == 'csv':
        df.to_csv(file_path, index=False)
    elif file_extension in ['xls', 'xlsx']:
        df.to_excel(file_path, index=False)
    elif file_extension == 'json':
        df.to_json(file_path, orient='records')
    elif file_extension == 'parquet':
        df.to_parquet(file_path, index=False)
    elif file_extension == 'hdf':
        df.to_hdf(file_path, key='data', mode='w')
    elif file_extension == 'feather':
        df.to_feather(file_path)
    elif file_extension == 'sql':
        # Make sure to replace 'table_name' with the desired table name in the SQL database.
        engine = f'sqlite:///{file_path}'
        df.to_sql('table_name', engine, if_exists='replace', index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

if __name__ == "__main__":
    # Sample DataFrame for demonstration purposes
    data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 22],
        'City': ['New York', 'San Francisco', 'Los Angeles']
    }
    df = pd.DataFrame(data)

    file_path = "path/to/your/output_file"  # Replace with the desired output file path
    file_format = "csv"  # Replace with the desired file format (csv, xls, xlsx, json, parquet, hdf, feather, sql)

    try:
        write_file(df, f"{file_path}.{file_format}", file_format)
        print(f"Data written to {file_path}.{file_format}")
    except ValueError as ve:
        print(ve)




#!/usr/bin/env python
# coding: utf-8

# In[243]:


import pandas as pd
import numpy as np


# In[244]:


#loading raw dcm data
dcm_merged=pd.read_csv(r' T1.csv',header='infer')



# loading pre-processed direct buy data
pre_proc_directbuy=pd.read_csv(r'file.csv', header='infer')
pre_proc_directbuy=pre_proc_directbuy[['Site','Date','Campaign','Campaign ID','Placement Name','Placement ID','Placement','Impressions','Media Cost']]

pre_proc_directbuy.rename(columns={ 'Site':'Site (CM360)','Media Cost':'Cost'},inplace=True)

pre_proc_directbuy['Source']='Direct Buy'
pre_proc_directbuy['Designated Market Area (DMA)']='No DMA'

pre_proc_directbuy['Date']=pd.to_datetime(pre_proc_directbuy['Date'])


pre_proc_directbuy=pre_proc_directbuy.loc[pre_proc_directbuy['Date']>='2022-04-01']

pre_proc_directbuy.count()


# In[245]:


# Union

T1_combined=pd.concat([pre_proc_directbuy,dcm_merged])
T1_combined.rename(columns={ 'Site (CM360)':'Site DCM'},inplace=True)

# col casting 
T1_combined['Impressions']=T1_combined['Impressions'].astype(float)
T1_combined['Cost']=T1_combined['Cost'].astype(float)


T1_combined['Date']=pd.to_datetime(T1_combined['Date'])
T1_combined['Week']=T1_combined['Date'].dt.to_period('W').dt.start_time
T1_combined.count()



# In[246]:


# rolling upto weekly  "2021-09-27" "2022-03-28"
T1_combined=T1_combined.groupby(['Campaign','Campaign ID','Site DCM','Placement','Placement ID','Designated Market Area (DMA)', 'Source','Week'],as_index=False,dropna=False)['Impressions','Cost'].sum()
T1_combined.rename(columns={ 'Cost':'Spend'},inplace=True)

T1_combined.head()


# In[247]:


T1_combined['Comp_plac']=T1_combined['Campaign'] + T1_combined['Placement']


T1_combined['Source']=np.where(T1_combined['Source'].isnull(),'dcm',T1_combined['Source'])

T1_combined['Spend']=np.where(T1_combined['Source'].eq('dcm'), 0 ,  T1_combined['Spend'])
T1_combined.count() 


# In[248]:


# Filter to get the valid impressions
accepted=T1_combined.loc[~T1_combined['Comp_plac'].str.contains("Facebook|Instagram|CPO|Fleet|_FLEET-|-Fleet_|FLEET|fleet|Parts&Service|Parts&Services|DART Search|Parts & Service|Parts-&-Service| FB | IG |_FB_|_IG_|-FBIG-|-FBIG_|Military|Ariya",na=False)]

#discarded=T1_combined.loc[T1_combined['Comp_plac'].str.contains("Facebook|Instagram|CPO|Fleet|_FLEET-|-Fleet_|FLEET|fleet|Parts&Service|Parts&Services|DART Search|Parts & Service|Parts-&-Service| FB | IG |_FB_|_IG_|-FBIG-|-FBIG_|Military|Ariya",na=False)]
# removed comp_plac column from df
accepted=accepted[['Campaign', 'Campaign ID', 'Site DCM', 'Placement', 'Placement ID', 'Designated Market Area (DMA)', 'Impressions', 'Spend', 'Source', 'Week']]
#discarded=discarded[['Campaign', 'Campaign ID', 'Site DCM', 'Placement', 'Date', 'Placement ID', 'Designated Market Area (DMA)', 'Impressions', 'Placement Name', 'Spend', 'Source', 'Week']]
accepted.count()
#print(discarded.count())


# In[249]:


####### categorization starts
#accepted.groupby(['campaign','Campaign ID','Site DCM','Placement','Placement ID','Designated Market Area (DMA)', 'Source','Week']'])['impressions','spend'].sum()
categorization=accepted                                      
categorization_fu=accepted


# In[250]:


#categorization=categorization.loc[~categorization['Site DCM'].str.contains("Facebook|Military|snapchat.com|Twitter|Reddit|TikTok|Twitter-Official",na=False)]
#categorization_fu=categorization.loc[~categorization['Site DCM'].str.contains("Facebook|Military|snapchat.com|Twitter|Reddit|TikTok|Twitter-Official",na=False)]
#categorization.columns


# In[251]:


categorization=categorization.groupby(['Campaign','Site DCM','Placement', 'Source','Week'],as_index=False,dropna=False)['Impressions','Spend'].sum()
#categorization=categorization[['Campaign','Site DCM','Placement', 'Source','Week','Impressions','Spend']]
categorization.count()


# In[252]:


categorization.head()


# In[253]:


#Filter
categorization1=categorization.loc[categorization['Site DCM'].str.contains('DV360|The Trade Desk',na=False)]

categorization2=categorization.loc[~categorization['Site DCM'].str.contains('DV360|The Trade Desk',na=False)]
print(categorization1.count())
print(categorization2.count())


# In[254]:


categorization1['Placement_new']=categorization1['Placement'].str.replace('_',' ')

categorization1['Site']=categorization1['Placement_new'].apply(lambda x: str(x).replace("  ",' ').split(' ')[4])
categorization1['Site']=categorization1['Site'].apply(lambda x: str(x).replace("-",' '))
#categorization1.head()
#categorization1[['site1','site2']]=categorization1['Site'].apply(lambda x: pd.Series(str(x).split("-",1)))

categorization1['Site1']=categorization1['Site'].apply(lambda x: x if len(x.split(' '))==1 else x.split(' ',1)[0])
categorization1['Site2']=categorization1['Site'].apply(lambda x: x if len(x.split(' '))==1 else x.split(' ',1)[1])

categorization1['Site3']=np.where(categorization1['Site2'].isnull(),categorization1['Site1'],categorization1['Site2'])

categorization1=categorization1.drop(['Site','Site1','Site2'],axis=1)
categorization1.rename(columns={ 'Site3':'Site'},inplace=True)


categorization1=categorization1[['Campaign','Site DCM','Placement','Source','Week','Impressions','Spend','Site']]
print(categorization.columns)
#discarded.rename(columns={'Site (CM360)':'Site'},inplace=True)
#print(discarded.columns)



#categorization2

categorization2['Site']=categorization2['Site DCM']



#Union
union=pd.concat([categorization2,categorization1])
union.count()


# In[255]:


union.head()


# In[256]:




categorization=union

categorization['U_Placement']=categorization['Placement'].str.upper()
categorization['U_Campaign']=categorization['Campaign'].str.upper()

categorization.head()


# In[257]:


categorization['U_Campaign']=categorization['U_Campaign'].str.strip()
categorization['U_Placement']=categorization['U_Placement'].str.strip()
categorization['Campaign']=categorization['Campaign'].str.strip()
categorization['Placement']=categorization['Placement'].str.strip()


# In[258]:




## Nameplate categorization by pattern matching from campaign and placement column
conditions=[categorization['U_Campaign'].str.contains("ALTIMA",na=False, case=False) |categorization['U_Placement'].str.contains("ALTIMA|_AS_",na=False, case=False),
          categorization['U_Campaign'].str.contains('SENTRA',na=False, case=False)|categorization['U_Placement'].str.contains("SENTRA|_SE_",na=False, case=False),
          categorization['U_Campaign'].str.contains('MAXIMA',na=False, case=False)| categorization['U_Placement'].str.contains('MAXIMA',na=False, case=False),
          categorization['U_Campaign'].str.contains("LEAF",na=False, case=False)|categorization['U_Placement'].str.contains("LEAF|_LF_",na=False, case=False),
          categorization['U_Campaign'].str.contains("VERSA|VERSASEDAN|VERSA NOTE",na=False, case=False)| categorization['U_Placement'].str.contains("VERSA|VERSASEDAN|VERSA NOTE",na=False, case=False),
          categorization['Campaign'].str.contains('Rogue Sport|RogueSport',na=False, case=False)| categorization['Placement'].str.contains("Rogue Sport|RogueSport",na=False, case=False),
          categorization['Campaign'].str.contains('Rogue',na=False, case=False)| categorization['Placement'].str.contains('Rogue|_RG_',na=False, case=False),
          categorization['Campaign'].str.contains('Murano',na=False, case=False)| categorization['Placement'].str.contains('Murano',na=False, case=False),
           categorization['Campaign'].str.contains('Pathfinder',na=False, case=False)| categorization['Placement'].str.contains('Pathfinder',na=False, case=False),
          categorization['Campaign'].str.contains('Armada',na=False, case=False)| categorization['Placement'].str.contains('Armada',na=False, case=False),
          categorization['Campaign'].str.contains('Kicks',na=False, case=False)| categorization['Placement'].str.contains('Kicks|_KI_|_KIcks_'),
          categorization['Campaign'].str.contains('Titan|Titan XD|TitanXD',na=False, case=False)| categorization['Placement'].str.contains("Titan|Titan XD|TitanXD|_TN_",na=False, case=False),
          categorization['Campaign'].str.contains('Frontier',na=False, case=False)| categorization['Placement'].str.contains('Frontier',na=False, case=False),
          categorization['Campaign'].str.contains("NCV| NV |NV200Cargo|NVCargo|NVPassenger|NVP|NV200",na=False, case=False)| categorization['Placement'].str.contains("NCV| NV |NV200Cargo|NVCargo|NVPassenger|NVP|NV200",na=False, case=False),
          categorization['Campaign'].str.contains("GT-R|GTR| Z |ZRoadster|370Z|_Z_",na=False, case=False)| categorization['Placement'].str.contains("GT-R|GTR| Z |ZRoadster|370Z|_Z_",na=False, case=False),
          categorization['Campaign'].str.contains("MultiModel|Multi Model|Multi-Model|_Brand_|-BRAND-",na=False, case=False)| categorization['Placement'].str.contains( "MultiModel|Multi Model|Mutli Model|Multi-Model|_BR_|_BRAND |_Brand |_Brand_|-BRAND-",na=False, case=False),
          categorization['Campaign'].str.contains("THRILLCAP",na=False, case=False)|categorization['Placement'].str.contains("THRILLCAP|Performance Capability",na=False, case=False),
          categorization['Campaign'].str.contains("THRILL BON|THRLL BON|Best of Nissan",na=False, case=False)|categorization['Placement'].str.contains("THRILL BON|THRLL BON|Best of Nissan",na=False, case=False),
           categorization['Campaign'].str.contains("THRLL ELEC|THRILL ELEC_|Electrification",na=False, case=False)|categorization['Placement'].str.contains("THRLL ELEC|THRILL ELEC_|Electrification",na=False, case=False),
            categorization['Campaign'].str.contains('Nissan @ Home |_Thrill N@H_|_Thrill N@H | Nissan At Home|_S@Home|Nissan Home',na=False, case=False)|categorization['Placement'].str.contains('Nissan @ Home |Nissan At home|_Thrill N@H_|_Thrill N@H ',na=False, case=False),
           categorization['Campaign'].str.contains("_THRILL_P&C",na=False, case=False),
           categorization['U_Campaign'].str.contains('SUPERBOWL|_NCAA|_BET|LGBTQ|NIL_FM',na=False, case=False)|categorization['U_Placement'].str.contains("SUPERBOWL|_NCAA|HEISMAN|_BET|LGBTQ|NIL_FM")
           
           ]
choices=['ALT','SEN','MAX','LEF','VER','RGS','RGE','MUR','PTH','ARM','KCS','TTN','FRO','NV','GTZ','MBR','Thrill Capability','BON','thrill_elec','thrill_nh','thrill_pc','sponsorship' ]
categorization['Car']=np.select(conditions,choices,default='unknown')
categorization.count()                              


# In[259]:


## channel categorization 
l=list(range(10,100))
conditions=[categorization['Campaign'].str.contains('Audio| Aud |audio|aud| AUD |_ Audio')| categorization['Placement'].str.contains('Audio|_Aud_|audio|_AUD_'),
           categorization['Campaign'].str.contains('Streaming|OLV|Video|VID|STREAMING')| categorization['Placement'].str.contains('Streaming|OLV|Video|VID|YouTube-Non-Skippable|YouTube-Skippable|YouTube-Reserve')|categorization['Site'].str.contains('teads.tv'),
           categorization['Placement'].str.contains('Pre-Roll|Preroll|PreRoll|Pre-roll') ,
           categorization['Placement'].str.contains('_Immersive|_:15_|_:06_|_:30_|_:60_|_Snapchat_|_In-Screen |_:15s') ,
           categorization['Campaign'].str.contains('Display|_DIS|_DISP_|_BAN_| Banner| Banner_') | categorization['Placement'].str.contains('Display|_DIS|_DISP_|-Image |_Twitter_|_HearstAutos_|_BAN_| Banner| Banner_'),
           categorization['Placement'].isin(l) | categorization['Placement'].isin(l)
           
           ]
choices=["Audio","Streaming","Streaming","Streaming","Display","Display"]
categorization['Channel']=np.select(conditions,choices,default="unknown")                                                                                      
categorization.count()                                                                                                 


# In[260]:


## funnel categorization
conditions=[categorization['Campaign'].str.contains('Future-Market|Future Market|_FM | FM' ) |categorization['Placement'].str.contains('Future-Market|Future Market|FUTURE MARKET|_FM_|_FM' ),
           categorization['Campaign'].str.contains('_THRILL|_N@H|_S@Home|THRILL|_Thrill_')| categorization['Placement'].str.contains('_THRILL|_N@H|_S@Home|THRILL|_Thrill_'),
           categorization['Campaign'].str.contains('Near-Market|Near Market|_NearMarket')|categorization['Placement'].str.contains('Near-Market|Near Market|NearMarket|_NM_|_NM |_Near') ,
           categorization['Campaign'].str.contains('In-Market|In Market') | categorization['Placement'].str.contains('In-Market|In Market|Inmarket|InMarket|_IM_'),
            categorization['Placement'].str.contains('NAT_NA_NA_NA|NA_NA_NA_NA')
           
           ]
choices=["Future-Market","Future-Market","Near-Market", "In-Market","unknown"]
categorization['Funnel']=np.select(conditions,choices,default="unknown") 
categorization.count()


# In[261]:


## market categorization
conditions=[categorization['Campaign'].str.contains("_GM_|_GM| GM | GM|_LGBTQ")|categorization['Placement'].str.contains("_GM_|_GM| GM | GM|_LGBTQ"),
           categorization['Campaign'].str.contains("_HM_|_HIS_|_HIS |Hisp|HM|Hispanic")|categorization['Placement'].str.contains("_HM_|_HIS_|_HIS |Hispanic"),
           categorization['Campaign'].str.contains("_AA_| AA |_AA|-BET-")|categorization['Placement'].str.contains("_AA_"),
           categorization['Campaign'].str.contains( "_ASA_|Asian")|categorization['Placement'].str.contains("_ASA_")
            ]
choices=["GM","HM", "AA","ASA"]
categorization['Market']=np.select(conditions,choices,default="unknown") 
categorization.count()


# In[262]:


categorization['Buy']="Programmatic"
#categorization.count()


# In[263]:


conditions=[categorization['Campaign'].str.contains( "Social|Snap|Snapchat|Twitter|Pinterest|Twiter",na=False)|categorization['Placement'].str.contains( "Social|Snap|Snapchat|Twitter|Pinterest|Twiter",na=False)|categorization['Site'].str.contains("Twitter|Snapchat",na=False),
            categorization['Campaign'].str.contains("-superbowl",na=False)|categorization['Placement'].str.contains("_Superbowl|_Superbowl |superbowl|Superbowl",na=False),
            categorization['Campaign'].str.contains("_NCAA ",na=False)|categorization['Placement'].str.contains( "_NCAA ",na=False),
             categorization['Campaign'].str.contains("_Heisman_| Heisman",na=False)|categorization['Placement'].str.contains("_Heisman_| Heisman",na=False),
             categorization['Campaign'].str.contains("_BET Digital_|_BET ",na=False)|categorization['Placement'].str.contains("_BET_|_BET Digital_",na=False)]

choices=["Social","superbowl", "ncaa","heisman","bet"]
categorization['Platform']=np.select(conditions,choices,default="others")
categorization.count()           


# In[264]:


conditions=[categorization['Campaign'].str.contains('THRLL ELEC|THRILL ELEC_|Electrification',na=False) | categorization['Placement'].str.contains('Placement',na=False),
           categorization['Campaign'].str.contains('Nissan @ Home |_Thrill N@H_|_Thrill N@H | Nissan Home|_S@Home',na=False)|categorization['Placement'].str.contains('Nissan @ Home |_Thrill N@H_|_Thrill N@H ',na=False) ,
            categorization['Campaign'].str.contains('_THRILL P&C_',na=False),
           categorization['Campaign'].str.contains('-superbowl',na=False) |categorization['Placement'].str.contains('_Superbowl|_Superbowl |superbowl|Superbowl',na=False) ,
            categorization['Campaign'].str.contains('_NCAA',na=False) | categorization['Placement'].str.contains('_NCAA',na=False),
             categorization['Campaign'].str.contains('_Heisman_| Heisman',na=False) | categorization['Placement'].str.contains('_Heisman_| Heisman',na=False),
             categorization['Campaign'].str.contains('_BET Digital_|_BET ',na=False) |categorization['Placement'].str.contains('_BET_|_BET Digital_',na=False)                               
           ]

choices=["thrill_elec","thrill_nh","thrill_pc","sponsorship","sponsorship","sponsorship","sponsorship",]
categorization['Car']=np.select(conditions,choices,default=categorization['Car'])
categorization.count()


# In[265]:


categorization['Campaign']=np.where(categorization['Campaign'].isnull(),'Not Available',categorization['Campaign'])

categorization['Placement']=np.where(categorization['Placement'].isnull(),'Not Available',categorization['Placement'])

conditions=[categorization['Site'].str.contains("Youtube|Trueview|Vevo|Studio71|WarnerMusic|YouTube.com",na=False),
           categorization['Site'].str.contains("Amazon",na=False),
           categorization['Site'].str.contains("Roku|roku.com",na=False),
           categorization['Site'].str.contains("Samsung Electronics|Samsung Adhub|Samsung|Samsung Electronics Samsung Adhub",na=False),
           categorization['Site'].str.contains("Hulu|ESPNRON|ESPN.com|DisneyXP|ESPN|ESPNDeportes|ESPNLive",na=False),
           categorization['Site'].str.contains("Teads|teads.tv",na=False)]
choices=["Youtube","Amazon","Roku", "Samsung","Hulu","Teads"]
categorization['Site']=np.select(conditions,choices,default="others")


categorization.count()


# In[266]:


categorization['Key']=categorization['Campaign']+categorization['Placement']+categorization['Site DCM']+categorization['Source']

categorization=categorization.loc[categorization['Campaign']!='Not Available']



categorization.count()


# In[270]:


df3=pd.read_excel(r'file.xlsx',sheet_name='Car_Unknown')
df4=pd.read_excel(r'filexlsx',sheet_name='Channel_Unknown')
df5=pd.read_excel(r'file.xlsx',sheet_name='Funnel_Unknown')
df6=pd.read_excel(r'file.xlsx',sheet_name='Market_Unknown')
df7=pd.concat([df3,df4,df5,df6])

del [df3,df4,df5,df6] 

df7['Key']=df7['campaign']+ df7['placement']+df7['site (cm360)']+ df7['source']



#df7.to_csv(r'file.csv')

df7.count()


# In[271]:


df7.columns=[x.capitalize() for x in df7.columns]

#categorization=categorization.loc[categorization['Campaign']!='Not Available']
categorization=categorization.drop(['U_Placement','U_Campaign'],axis=1)
df7.head()


# In[272]:


#Filter :col car ,channel,market='unknown'

categorization_unknown=categorization.loc[categorization['Car'].eq('unknown')|categorization['Channel'].eq('unknown') |categorization['Funnel'].eq('unknown') | categorization['Market'].eq('unknown')]
#categorization=categorization.loc[categorization['Car'].ne('unknown')|categorization['Channel'].ne('unknown') |categorization['Funnel'].ne('unknown') | categorization['Market'].ne('unknown')]
categorization=categorization[~categorization.isin(categorization_unknown)]
print(categorization_unknown.count())
print(categorization.count())


# In[273]:


categorization_unknown['Car2']=''
categorization_unknown['Channel2']=''
categorization_unknown['Funnel2']=''
categorization_unknown['Market2']=''


if categorization_unknown['Key'].equals(df7['Key']):
    categorization_unknown['Car2']=df7['Car']
    
if categorization_unknown['Key'].equals(df7['Key']):
    categorization_unknown['Channel2']=df7['Channel']    

if categorization_unknown['Key'].equals(df7['Key']):
    categorization_unknown['Funnel2']=df7['Funnel']   

if categorization_unknown['Key'].equals(df7['Key']):
    categorization_unknown['Market2']=df7['Market']   
        
categorization_unknown.count()


# In[274]:


categorization_unknown['Car']=np.where(categorization_unknown['Car'] == 'unknown',categorization_unknown['Car2'],categorization_unknown['Car'])

categorization_unknown['Channel']=np.where(categorization_unknown['Channel'] == 'unknown',categorization_unknown['Channel2'],categorization_unknown['Channel'])

categorization_unknown['Market']=np.where(categorization_unknown['Market'] == 'unknown',categorization_unknown['Market2'],categorization_unknown['Market'])

categorization_unknown['Funnel']=np.where(categorization_unknown['Funnel'] == 'unknown',categorization_unknown['Funnel2'],categorization_unknown['Funnel'])

categorization_unknown['Site DCM']=np.where(categorization_unknown['Site DCM'] == 'VIZIO','Vizio',(np.where(categorization_unknown['Site DCM'] == 'Conde Nast','CondeNast' ,categorization_unknown['Site DCM'])))

categorization_unknown=categorization_unknown.drop(['Car2','Channel2','Market2','Funnel2'],axis=1)
categorization_unknown.head(2)


# In[275]:


# Union 
union1=pd.concat([categorization,categorization_unknown])
categorization=union1
categorization.count()


# In[276]:




categorization_dcm=categorization.loc[categorization['Source']=='dcm']
categorization_ndcm=categorization.loc[categorization['Source']!='dcm']

categorization_dcm=categorization_dcm.drop_duplicates(subset=['Campaign','Site DCM'])
categorization_ndcm=categorization_ndcm.drop_duplicates(subset=['Campaign','Site DCM'])

categorization_inner=pd.merge(categorization_dcm, categorization_ndcm, on=['Campaign','Site DCM'], how='inner')

categorization_left=pd.merge(categorization, categorization_inner, on=['Campaign','Site DCM'], how='left')

#categorization_left=pd.merge(categorization, categorization_inner, on=['Campaign','Site DCM'], how='left',indicator=True)
#categorization_left=categorization_left.loc['_merge'=='left_only',['Campaign','Site DCM']]

#categorization_inner1=categorization_inner1.loc[categorization_inner1['Source']=='dcm']


#categorization_union=pd.concat([categorization_inner,categorization_left])

categorization_union=categorization_left

categorization_union=categorization_union[['Campaign','Site DCM','Placement','Source','Week','Impressions','Spend','Site','Car','Channel','Funnel','Market','Buy','Platform','Key']]

categorization_union.count()


# In[277]:


categorization_union=categorization_union.loc[~ categorization_union['Campaign'].str.contains("_FY17_|FY18|FY18_|FY19_|FY 19_|FY20_|FY20|FY21",na=False)]
categorization_union.count()


# In[278]:


categorization_union['Car']=np.where(categorization_union['Campaign'].str.contains('BHM',na=False),'MBR',(np.where(categorization_union['Car']=='superbowl','sponsorship',categorization_union['Car'])))
categorization_union.count()


# In[279]:


categorization_fu_T=categorization_fu.loc[categorization_fu['Source']=='dcm']
categorization_fu_F=categorization_fu.loc[categorization_fu['Source']!='dcm']

categorization_fu_T=categorization_fu_T.drop_duplicates(subset=['Campaign','Site DCM'])
categorization_fu_F=categorization_fu_F.drop_duplicates(subset=['Campaign','Site DCM'])

categorization_fu_inner=pd.merge(categorization_fu_T, categorization_fu_F, on=['Campaign','Site DCM'], how='inner')

categorization_fu_left1=pd.merge(categorization_fu_inner, categorization_fu, on=['Campaign','Site DCM'], how='left')

#categorization_fu_left=pd.merge(categorization_fu_inner, categorization_fu, on=['Campaign','Site DCM'], how='inner')

#categorization_fu_inner_ndcm=categorization_fu_inner1.loc[categorization_fu_inner1['Source']!='dcm']

categorization_fu_union=pd.concat([categorization_fu_inner,categorization_fu_left1])

categorization_fu_union=categorization_fu[['Campaign','Campaign ID','Site DCM','Placement','Placement ID','Designated Market Area (DMA)','Source','Week','Impressions','Spend']]
categorization_fu_union.count()


# In[280]:


categorization_union_inner=pd.merge(categorization_fu_union,categorization_union, on=['Campaign','Site DCM','Placement','Source','Week'], how='inner')
categorization_union_inner.count()


# In[281]:



categorization_union_inner.rename(columns={ 'Impressions_x':'Impressions','Spend_x':'Spend'},inplace=True)
categorization_union_inner=categorization_union_inner.drop(['Impressions_y','Spend_y'],axis=1)
categorization_union_inner.head(5)


# In[282]:




categorization_union_inner=categorization_union_inner.loc[~categorization_union_inner['Platform'].isin(["social"])]

categorization_union_inner=categorization_union_inner.loc[~categorization_union_inner['Channel'].isin(["Audio"])  | ~categorization_union_inner['Source'].isin(['dcm'])]

categorization_union_inner=categorization_union_inner.groupby(['Week','Campaign','Campaign ID','Site DCM','Placement','Placement ID','Designated Market Area (DMA)','Car','Channel','Platform','Funnel','Market','Site'],as_index=False,dropna=False)['Impressions'].sum()


df_dma=pd.read_excel(r'C:\Users\Prashant.Londhe\Desktop\Project\Data\2021-06-09 DMA to Region Map.xlsx',header='infer')
df_dma=df_dma[['DFA','DMA_CODE','P25-54','Five Region Pop','Five Region Names']]

#df_dma_inner=pd.merge(categorization_union_inner, df_dma, left_on='Designated Market Area (DMA)',right_on='DFA', how='inner')

df_dma_left=pd.merge(categorization_union_inner, df_dma, left_on='Designated Market Area (DMA)',right_on='DFA', how='left')

#df_dma_union=pd.concat([df_dma_inner,df_dma_left])

df_dma_union=df_dma_left

df_dma_union.count()


# In[283]:


#filter
df_dma_union['DMA_CODE']=np.where(df_dma_union['DMA_CODE'].isnull(),"No Metro",df_dma_union['DMA_CODE'])

df_dma_union['Tier']='T1'

date=pd.to_datetime('today').normalize()
date1=str(date).replace('-','')[0:8]

df_dma_union['Today']=date1

df_dma_union=df_dma_union[['Week','Campaign','Campaign ID','Site DCM','Placement','Placement ID','Designated Market Area (DMA)','Car','Channel','Platform','Funnel','Market','Site','Impressions','DFA','DMA_CODE','P25-54','Five Region Pop','Five Region Names','Tier','Today']]
df_dma_union.head()


# In[284]:


df_dma_union=df_dma_union.loc[~(df_dma_union.Week.isnull() | df_dma_union.Car.isnull() |df_dma_union.Channel.isnull() |df_dma_union['Channel'].str.contains('unknown') | df_dma_union['Campaign'].str.contains('22017 FY22 WER RMP Custom Campaign') )]

df_dma_union.count()


# In[170]:


# file1
df_dma_union1=df_dma_union
df_dma_union1=df_dma_union1.loc[df_dma_union1['Campaign'].str.contains("YTDMABRANDTEST")]
df_dma_union1.to_csv(r'C:\Users\Prashant.Londhe\Desktop\Project\Output\file1.csv')
df_dma_union1.count()


# In[171]:


#file2
df_dma_union2=df_dma_union
df_dma_union2=df_dma_union2.loc[df_dma_union2['Car'].isin(["BON", 'Thrill Capability'])]
df_dma_union2.to_csv(r'C:\Users\Prashant.Londhe\Desktop\Project\Output\file2.csv')
df_dma_union2.count()


# In[172]:


#file3
df_dma_union3=df_dma_union
df_dma_union3=df_dma_union3.loc[df_dma_union3['Car'].isin(["MBR", 'sponsorship'])]
df_dma_union3.to_csv(r'C:\Users\Prashant.Londhe\Desktop\Project\Output\file3.csv')
df_dma_union3.count()


# In[173]:


#file4
df_dma_union4=df_dma_union
df_dma_union4.to_csv(r'C:\Users\Prashant.Londhe\Desktop\Project\Output\file4.csv')
df_dma_union4.count()

df_dma_union1.to_csv(r'C:\Users\Prashant.Londhe\Desktop\Project\Output\file1.csv')
df_dma_union2.to_csv(r'C:\Users\Prashant.Londhe\Desktop\Project\Output\file2.csv')
df_dma_union3.to_csv(r'C:\Users\Prashant.Londhe\Desktop\Project\Output\file3.csv')
df_dma_union4.to_csv(r'C:\Users\Prashant.Londhe\Desktop\Project\Output\file4.csv')
# In[ ]:




