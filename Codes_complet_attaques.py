#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


import holoviews as hv


# In[ ]:


from holoviews import opts
hv.extension('bokeh')


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


import sklearn


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[ ]:


import os


# In[ ]:


get_ipython().system('pip uninstall -y holidays')
get_ipython().system('pip install holidays')


# In[ ]:


get_ipython().system('pip install --upgrade holidays')


# In[ ]:


get_ipython().system('pip install holidays==0.11.1')


# In[ ]:


pip install plotly


# In[ ]:


from fbprophet import Prophet


# In[ ]:


from fbprophet.plot import add_changepoints_to_plot


# In[ ]:


#!pip install seaborn


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# 2) Chargement des données
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv("E:/IOT-temp.csv")
print(f'IOT-temp.csv : {df.shape}')
df.head(3)


# In[ ]:


df[['outside','inside']]=pd.get_dummies(df['out/in'])
df.head(5)


# In[ ]:


#changer les noms de colonne pour comprendre facilement: 'out/in' devient 'place',...
df.rename(columns={'noted_date':'date', 'out/in':'place'}, inplace=True)
df.head(3)


# In[ ]:


print(df.columns)


# In[ ]:


# Graphique des temperatures selon la place : in et out
sns.barplot(x='place', y='temp', data=df)
plt.show()


# In[ ]:


# Quelques éléments qui serviront dans l'analyse des résultats des prédictions
# Compter le nombre des valeurs "X" dans chaque colonne
nombre_In=df['place'].value_counts()['In']
print("Les valeurs 'In' sont : ", nombre_In)
nombre_out=df['place'].value_counts()['Out']
print("Les valeurs 'Out' sont : ", nombre_out)
nombre_29=df['temp'].value_counts()[29]
print("Les valeurs '29' sont : ", nombre_29)


# In[ ]:


# 4) Tâches
arr = df['inside']
x=[]
y=[]
for i in arr:
    if i==1:
        x.append(i)
    else :
        y.append(i)
x=pd.Series(x)
y=pd.Series(y)
type(arr)


# In[ ]:


#Variance de température entre l'intérieur et l'extérieur de la pièce ?
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.violinplot(x='inside', y='temp', data=df, ax=axes[0], color='b').set_title("Inside v/s Temp")
sns.violinplot(x='outside', y='temp', data=df, ax=axes[1], color='r').set_title("Outside v/s Temp")
sns.violinplot(x='place', y='temp', data=df, ax=axes[2]).set_title("place v/s Temp")


# In[ ]:


# Compter le nombre des valeurs "X" dans chaque colonne (suite)
nombre_inside=df['inside'].value_counts()[1]
print("Les valeurs '1' dans la colonne 'inside'sont : ", nombre_inside)
nombre_outside=df['outside'].value_counts()[1]
print("Les valeurs '1' dans la colonne 'outside'sont : ", nombre_outside)


# In[ ]:


# Debut du script proprement dit: On peut soit entrer un nombre des neouds, soit laisser une selection aléatoire
# nombre_noeuds=int(input("Entrez le nombre des neouds à sélectionner"))


# In[ ]:


# Nombre aléatoire de nœuds à sélectionner
import random
nombre_noeuds = random.randint(1, len(df))


# In[ ]:


# Affichage du nombre de noeuds selectionnées
print("Les noeuds pris aléatoirement sont au nombre de : ",nombre_noeuds )


# In[ ]:


indices_noeuds = random.sample(range(len(df)), nombre_noeuds)


# In[ ]:


noeuds_selectionnes = df.iloc[indices_noeuds]


# In[ ]:


# 3) Pre-traitement
df['room_id/id'].value_counts()


# In[ ]:


# Nombre des noeuds à empoisonner
for i in df.loc[indices_noeuds]:
    
    nombre_29=df.loc[indices_noeuds, 'temp'].value_counts()[29]
    nombre_out=df.loc[indices_noeuds, 'place'].value_counts()['Out']
    nombre_in=df.loc[indices_noeuds, 'place'].value_counts()['In']
    nombre_outsid=df.loc[indices_noeuds, 'outside'].value_counts()[1]
    nombre_Insid=df.loc[indices_noeuds, 'inside'].value_counts()[1]
    
    
print("Les valeurs '29' dans la colonne 'temp'sont : ", nombre_29)
print("Les valeurs 'Out' dans la colonne 'place'sont : ", nombre_out)
print("Les valeurs 'In' dans la colonne 'Place'sont : ", nombre_in)
print("Les valeurs '1' dans la colonne 'outside'sont : ", nombre_outsid)
print("Les valeurs '1' dans la colonne 'outside'sont : ", nombre_Insid)


# In[ ]:


# Valeur que vous souhaitez rechercher pour la duplication
valeur_a_dupliquer = 29

# Facteur de multiplication
facteur_multiplicatif = 3
# updating the column value/data
for i in df.iloc[indices_noeuds]:
    #df['temp'] = df['temp'].replace({29: 27})
    #df.loc[indices_noeuds, 'outside'] = df.loc[indices_noeuds, 'inside'].replace()
    #df.loc[indices_noeuds, 'inside'] = df.loc[indices_noeuds, 'outside'].replace()
    #df.loc[indices_noeuds, 'room_id/id'] = df.loc[indices_noeuds, 'out/in'].replace()
    df.loc[indices_noeuds, 'temp'] = df.loc[indices_noeuds, 'temp'].replace({29: 15})
    #df.loc[indices_noeuds, 'temp'] = df.loc[indices_noeuds, 'temp'].replace({41: ''})
    #df.loc[indices_noeuds, 'out/in'] = df.loc[indices_noeuds, 'out/in'].replace({'In': 'Out'})
    #df['temp'] = df['temp'].apply(lambda x: x * facteur_multiplicatif if x == valeur_a_dupliquer else x)

# writing into the file
df.to_csv("IOT_temp.csv", index=False)
print(df)


# In[ ]:


df['room_id/id'].value_counts()


# In[ ]:


#la colonne 'room_id/id' n'a qu'une seule valeur (Room Admin), nous n'avons donc pas besoin de cette colonne pour l'analyse.
# Suppression de la colonne 'room_id/id'
df.drop('room_id/id', axis=1, inplace=True)
df.head(3)


# In[ ]:


#changer les noms de colonnes pour comprendre facilement
df.rename(columns={'noted_date':'date', 'out/in':'place'}, inplace=True)
df.head(3)


# In[ ]:


#Informations sur la date et l'heure
#La colonne datetime contient de nombreuses informations telles que l'année, le mois, le jour de la semaine, etc.
#Pour utiliser ces informations dans l'EDA et la phase de modélisation, nous devons les extraire de la colonne datetime.
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')
df['year'] = df['date'].apply(lambda x : x.year)
df['month'] = df['date'].apply(lambda x : x.month)
df['day'] = df['date'].apply(lambda x : x.day)
df['weekday'] = df['date'].apply(lambda x : x.day_name())
df['weekofyear'] = df['date'].apply(lambda x : x.weekofyear)
df['hour'] = df['date'].apply(lambda x : x.hour)
df['minute'] = df['date'].apply(lambda x : x.minute)
df.head(3)


# In[ ]:


#fonction pour convertir la variable mois en saisons
def month2seasons(x):
    if x in [12, 1, 2]:
        season = 'Winter'
    elif x in [3, 4, 5]:
        season = 'Summer'
    elif x in [6, 7, 8, 9]:
        season = 'Monsoon'
    elif x in [10, 11]:
        season = 'Post_Monsoon'
    return season


# In[ ]:


df['season'] = df['month'].apply(month2seasons)
df.head(3)


# In[ ]:


#Informations horaires
def hours2timing(x):
    if x in [22,23,0,1,2,3]:
        timing = 'Night'
    elif x in range(4, 12):
        timing = 'Morning'
    elif x in range(12, 17):
        timing = 'Afternoon'
    elif x in range(17, 22):
        timing = 'Evening'
    else:
        timing = 'X'
    return timing


# In[ ]:


df['timing'] = df['hour'].apply(hours2timing)
df.head(3)


# In[ ]:


#Vérification des doublons
#Après avoir vérifié si un enregistrement est en double, il s'est avéré qu'il y avait des enregistrements en double. 
# Nous devons donc mettre les enregistrements en double dans un seul enregistrement.
#df[df.duplicated()]


# In[ ]:


df[df['id']=='__export__.temp_log_196108_4a983c7e']


# In[ ]:


#df.drop_duplicates(inplace=True)
#df[df.duplicated()]


# In[ ]:


#Dans la même date/heure (2018-09-12 03:09:00), il existe de nombreux enregistrements et identifiants uniques.
df.loc[df['date']=='2018-09-12 03:09:00', ].sort_values(by='id').head(5)


# In[ ]:


#Le nombre de parties numériques dans 'id' a le même nombre que la longueur de l'ensemble des données, 
#sde sorte que les parties numériques indiquent l'unicité de chaque enregistrement.
df['id'].apply(lambda x : x.split('_')[6]).nunique() == len(df)


# In[ ]:


#Ajout de parties numériques dans 'id' comme nouvel identifiant.
df['id'] = df['id'].apply(lambda x : int(x.split('_')[6]))
df.head(3)


# In[ ]:


#Il y a des lacunes dans la colonne 'id'.
df.loc[df['date'] == '2018-09-12 03:09:00', ].sort_values(by ='id').head(5)


# In[ ]:


#Il y a un espace dans la colonne 'date' lorsqu'il est trié par 'id'.
df.loc[df['id'].isin(range(4000, 4011))].sort_values(by='id')


# In[ ]:


month_rd = np.round(df['date'].apply(lambda x : x.strftime("%Y-%m")).value_counts(normalize=True).sort_index() * 100,decimals=1)
month_rd_bar = hv.Bars(month_rd).opts(color="green")
month_rd_curve = hv.Curve(month_rd).opts(color="red")
(month_rd_bar * month_rd_curve).opts(title="Monthly Readings Count", xlabel="Month", ylabel="Percentage", yformatter='%d%%', width=700, height=300,tools=['hover'],show_grid=True)


# In[ ]:


# Temperature
#La température consiste clairement en plusieurs distributions.
hv.Distribution(df['temp']).opts(title="Temperature Distribution", color="green", xlabel="Temperature", ylabel="Density")\
                            .opts(opts.Distribution(width=700, height=300,tools=['hover'],show_grid=True))


# In[ ]:


#Place
pl_cnt = np.round(df['place'].value_counts(normalize=True) * 100)
hv.Bars(pl_cnt).opts(title="Readings Place Count", color="green", xlabel="Places", ylabel="Percentage", yformatter='%d%%')\
                .opts(opts.Bars(width=700, height=300,tools=['hover'],show_grid=True))


# In[ ]:


# Season
season_cnt = np.round(df['season'].value_counts(normalize=True) * 100)
hv.Bars(season_cnt).opts(title="Season Count", color="green", xlabel="Season", ylabel="Percentage", yformatter='%d%%')\
                .opts(opts.Bars(width=700, height=300,tools=['hover'],show_grid=True))


# In[ ]:


# Timing
timing_cnt = np.round(df['timing'].value_counts(normalize=True) * 100)
hv.Bars(timing_cnt).opts(title="Timing Count", color="green", xlabel="Timing", ylabel="Percentage", yformatter='%d%%')\
                .opts(opts.Bars(width=700, height=300,tools=['hover'],show_grid=True))


# In[ ]:


in_month = np.round(df[df['place']=='In']['date'].apply(lambda x : x.strftime("%Y-%m")).value_counts(normalize=True).sort_index() * 100, decimals=1)
out_month = np.round(df[df['place']=='Out']['date'].apply(lambda x : x.strftime("%Y-%m")).value_counts(normalize=True).sort_index() * 100, decimals=1)
in_out_month = pd.merge(in_month,out_month,right_index=True,left_index=True).rename(columns={'date_x':'In', 'date_y':'Out'})
in_out_month = pd.melt(in_out_month.reset_index(), ['index']).rename(columns={'index':'Month', 'variable':'Place'})
hv.Bars(in_out_month, ['Month', 'Place'], 'value').opts(opts.Bars(title="Monthly Readings by Place Count", width=700, height=400,tools=['hover'],show_grid=True, ylabel="Count"))


# In[ ]:


#Répartition de la température par lieu
(hv.Distribution(df[df['place']=='In']['temp'], label='In') * hv.Distribution(df[df['place']=='Out']['temp'], label='Out'))\
                                .opts(title="Temperature by Place Distribution", xlabel="Temperature", ylabel="Density")\
                                .opts(opts.Distribution(width=700, height=300,tools=['hover'],show_grid=True))


# In[ ]:


season_agg = df.groupby('season').agg({'temp': ['min', 'max']})
season_maxmin = pd.merge(season_agg['temp']['max'],season_agg['temp']['min'],right_index=True,left_index=True)
season_maxmin = pd.melt(season_maxmin.reset_index(), ['season']).rename(columns={'season':'Season', 'variable':'Max/Min'})
hv.Bars(season_maxmin, ['Season', 'Max/Min'], 'value').opts(title="Temperature by Season Max/Min", ylabel="Temperature")\
                                                                    .opts(opts.Bars(width=700, height=300,tools=['hover'],show_grid=True))


# In[ ]:


timing_agg = df.groupby('timing').agg({'temp': ['min', 'max']})
timing_maxmin = pd.merge(timing_agg['temp']['max'],timing_agg['temp']['min'],right_index=True,left_index=True)
timing_maxmin = pd.melt(timing_maxmin.reset_index(), ['timing']).rename(columns={'timing':'Timing', 'variable':'Max/Min'})
hv.Bars(timing_maxmin, ['Timing', 'Max/Min'], 'value').opts(title="Temperature by Timing Max/Min", ylabel="Temperature")\
                                                                    .opts(opts.Bars(width=700, height=300,tools=['hover'],show_grid=True))


# In[ ]:


#Analyse des séries chronologiques
#Prétraitement pour l'analyse de séries chronologiques
#Il est facile d'essayer l'analyse de séries chronologiques avec des données d'index temporel uniques. 
#Nous devons donc calculer les valeurs moyennes par colonne 'date' et supprimer la colonne 'id'.
tsdf = df.drop_duplicates(subset=['date','place']).sort_values('date').reset_index(drop=True)
tsdf['temp'] = df.groupby(['date','place'])['temp'].mean().values
tsdf.drop('id', axis=1, inplace=True)
tsdf.head(3)


# In[ ]:


in_month = tsdf[tsdf['place']=='In'].groupby('month').agg({'temp':['mean']})
in_month.columns = [f"{i[0]}_{i[1]}" for i in in_month.columns]
out_month = tsdf[tsdf['place']=='Out'].groupby('month').agg({'temp':['mean']})
out_month.columns = [f"{i[0]}_{i[1]}" for i in out_month.columns]
hv.Curve(in_month, label='In') * hv.Curve(out_month, label='Out').opts(title="Monthly Temperature Mean", ylabel="Temperature", xlabel='Month')\
                                                                    .opts(opts.Curve(width=700, height=300,tools=['hover'],show_grid=True))


# In[ ]:


tsdf['daily'] = tsdf['date'].apply(lambda x : pd.to_datetime(x.strftime('%Y-%m-%d')))
in_day = tsdf[tsdf['place']=='In'].groupby(['daily']).agg({'temp':['mean']})
in_day.columns = [f"{i[0]}_{i[1]}" for i in in_day.columns]
out_day = tsdf[tsdf['place']=='Out'].groupby(['daily']).agg({'temp':['mean']})
out_day.columns = [f"{i[0]}_{i[1]}" for i in out_day.columns]
(hv.Curve(in_day, label='In') * hv.Curve(out_day, label='Out')).opts(title="Daily Temperature Mean", ylabel="Temperature", xlabel='Day', shared_axes=False)\
                                                                    .opts(opts.Curve(width=700, height=300,tools=['hover'],show_grid=True))


# In[ ]:


in_wd = tsdf[tsdf['place']=='In'].groupby('weekday').agg({'temp':['mean']})
in_wd.columns = [f"{i[0]}_{i[1]}" for i in in_wd.columns]
in_wd['week_num'] = [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(i) for i in in_wd.index]
in_wd.sort_values('week_num', inplace=True)
in_wd.drop('week_num', axis=1, inplace=True)
out_wd = tsdf[tsdf['place']=='Out'].groupby('weekday').agg({'temp':['mean']})
out_wd.columns = [f"{i[0]}_{i[1]}" for i in out_wd.columns]
out_wd['week_num'] = [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(i) for i in out_wd.index]
out_wd.sort_values('week_num', inplace=True)
out_wd.drop('week_num', axis=1, inplace=True)
hv.Curve(in_wd, label='In') * hv.Curve(out_wd, label='Out').opts(title="Weekday Temperature Mean", ylabel="Temperature", xlabel='Weekday')\
                                                                    .opts(opts.Curve(width=700, height=300,tools=['hover'],show_grid=True))


# In[ ]:


in_wof = tsdf[tsdf['place']=='In'].groupby('weekofyear').agg({'temp':['mean']})
in_wof.columns = [f"{i[0]}_{i[1]}" for i in in_wof.columns]
out_wof = tsdf[tsdf['place']=='Out'].groupby('weekofyear').agg({'temp':['mean']})
out_wof.columns = [f"{i[0]}_{i[1]}" for i in out_wof.columns]
hv.Curve(in_wof, label='In') * hv.Curve(out_wof, label='Out').opts(title="WeekofYear Temperature Mean", ylabel="Temperature", xlabel='WeekofYear')\
                                                                    .opts(opts.Curve(width=700, height=300,tools=['hover'],show_grid=True))


# In[ ]:


in_tsdf = tsdf[tsdf['place']=='In'].reset_index(drop=True)
in_tsdf.index = in_tsdf['date']
in_all = hv.Curve(in_tsdf['temp']).opts(title="[In] Temperature All", ylabel="Temperature", xlabel='Time', color='red')

out_tsdf = tsdf[tsdf['place']=='Out'].reset_index(drop=True)
out_tsdf.index = out_tsdf['date']
out_all = hv.Curve(out_tsdf['temp']).opts(title="[Out] Temperature All", ylabel="Temperature", xlabel='Time', color='blue')

in_tsdf_int = in_tsdf['temp'].resample('1min').interpolate(method='nearest')
in_tsdf_int_all = hv.Curve(in_tsdf_int).opts(title="[In] Temperature All Interpolated with 'nearest'", ylabel="Temperature", xlabel='Time', color='red', fontsize={'title':11})
out_tsdf_int = out_tsdf['temp'].resample('1min').interpolate(method='nearest')
out_tsdf_int_all = hv.Curve(out_tsdf_int).opts(title="[Out] Temperature All Interpolated with 'nearest'", ylabel="Temperature", xlabel='Time', color='blue', fontsize={'title':11})

(in_all + in_tsdf_int_all + out_all + out_tsdf_int_all).opts(opts.Curve(width=400, height=300,tools=['hover'],show_grid=True)).opts(shared_axes=False).cols(2)


# In[ ]:


in_d_org = hv.Curve(in_day).opts(title="[In] Daily Temperature Mean", ylabel="Temperature", xlabel='Time', color='red')
out_d_org = hv.Curve(out_day).opts(title="[Out] Daily Temperature Mean", ylabel="Temperature", xlabel='Time', color='blue')

inp_df = pd.DataFrame()
in_d_inp = in_day.resample('1D').interpolate('spline', order=5)
out_d_inp = out_day.resample('1D').interpolate('spline', order=5)
inp_df['In'] = in_d_inp.temp_mean
inp_df['Out'] = out_d_inp.temp_mean

in_d_inp_g = hv.Curve(inp_df['In']).opts(title="[In] Daily Temperature Mean Interpolated with 'spline'", ylabel="Temperature", xlabel='Time', color='red', fontsize={'title':10})
out_d_inp_g = hv.Curve(inp_df['Out']).opts(title="[Out] Daily Temperature Mean Interpolated with 'spline'", ylabel="Temperature", xlabel='Time', color='blue', fontsize={'title':10})

(in_d_org + in_d_inp_g + out_d_org + out_d_inp_g).opts(opts.Curve(width=400, height=300,tools=['hover'],show_grid=True)).opts(shared_axes=False).cols(2)


# In[ ]:


#6)La modélisation
# Préparation des données
#En plus des informations sur la température, j'ai ajouté des informations sur la saison, 
#qui est un facteur chronologique qui affecte la température (en particulier à l'extérieur).
org_df = inp_df.reset_index()
org_df['season'] = org_df['daily'].apply(lambda x : month2seasons(x.month))
org_df = pd.get_dummies(org_df, columns=['season'])
org_df.head(3)


# In[ ]:


def run_prophet(place, prediction_periods, plot_comp=True):
    # faire dataframe pour la formation
    prophet_df = pd.DataFrame()
    prophet_df["ds"] = pd.date_range(start=org_df['daily'][0], end=org_df['daily'][133])
    prophet_df['y'] = org_df[place]
    # add seasonal information
    prophet_df['monsoon'] = org_df['season_Monsoon']
    prophet_df['post_monsoon'] = org_df['season_Post_Monsoon']
    prophet_df['winter'] = org_df['season_Winter']

    # modèle d'entrainement par Prophet
    m = Prophet(changepoint_prior_scale=0.1, yearly_seasonality=2, weekly_seasonality=False)
    # include seasonal periodicity into the model
    m.add_seasonality(name='season_monsoon', period=124, fourier_order=5, prior_scale=0.1, condition_name='monsoon')
    m.add_seasonality(name='season_post_monsoon', period=62, fourier_order=5, prior_scale=0.1, condition_name='post_monsoon')
    m.add_seasonality(name='season_winter', period=93, fourier_order=5, prior_scale=0.1, condition_name='winter')
    m.fit(prophet_df)

    # créer une trame de données pour la prédiction
    future = m.make_future_dataframe(periods=prediction_periods)
    # ajouter des informations saisonnières
    future_season = pd.get_dummies(future['ds'].apply(lambda x : month2seasons(x.month)))
    future['monsoon'] = future_season['Monsoon']
    future['post_monsoon'] = future_season['Monsoon']
    future['winter'] = future_season['Winter']

    # prédire la température future
    prophe_result = m.predict(future)

    # prédiction de tracé
    fig1 = m.plot(prophe_result)
    ax = fig1.gca()
    ax.set_title(f"{place} Prediction", size=25)
    ax.set_xlabel("Période", size=15)
    ax.set_ylabel("Température", size=15)
    a = add_changepoints_to_plot(ax, m, prophe_result)
    fig1.show()
    # tracer les composants décomposés de la série temporelle
    if plot_comp:
        fig2 = m.plot_components(prophe_result)
        fig2.show()



# In[ ]:


#Variance de température entre l'intérieur et l'extérieur de la pièce ?
# Outcome : The temperature outside has larger variance than inside temperature.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.violinplot(x='inside', y='temp', data=df, ax=axes[0], color='b').set_title("Inside v/s Temp")
sns.violinplot(x='outside', y='temp', data=df, ax=axes[1], color='r').set_title("Outside v/s Temp")
sns.violinplot(x='place', y='temp', data=df, ax=axes[2]).set_title("Place v/s Temp")


# In[ ]:


run_prophet('In',30)


# In[ ]:


run_prophet('Out',30)


# In[ ]:


dist = (hv.Distribution(df[df['place']=='In']['temp'], label='In') * hv.Distribution(df[df['place']=='Out']['temp'], label='Out'))\
                                .opts(title="Temperature by Place Distribution", xlabel="Temperature", ylabel="Density",tools=['hover'],show_grid=True, fontsize={'title':11})
tsdf['daily'] = tsdf['date'].apply(lambda x : pd.to_datetime(x.strftime('%Y-%m-%d')))
in_day = tsdf[tsdf['place']=='In'].groupby(['daily']).agg({'temp':['mean']})
in_day.columns = [f"{i[0]}_{i[1]}" for i in in_day.columns]
out_day = tsdf[tsdf['place']=='Out'].groupby(['daily']).agg({'temp':['mean']})
out_day.columns = [f"{i[0]}_{i[1]}" for i in out_day.columns]
curve = (hv.Curve(in_day, label='In') * hv.Curve(out_day, label='Out')).opts(title="Daily Temperature Mean", ylabel="Temperature", xlabel='Day', shared_axes=False,tools=['hover'],show_grid=True)
(dist + curve).opts(width=400, height=300)


# In[ ]:


in_var = hv.Violin(org_df['In'].values, vdims='Temperature').opts(title="In Temperature Variance", box_color='red')
out_var = hv.Violin(org_df['Out'].values, vdims='Temperature').opts(title="Out Temperature Variance", box_color='blue')
(in_var + out_var).opts(opts.Violin(width=400, height=300,show_grid=True))


# In[ ]:


run_prophet('In',30, False)
run_prophet('Out',30, False)


# In[ ]:




