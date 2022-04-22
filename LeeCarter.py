import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# preparing the data

# data downloded from the human mortality database

def mort_rates_db(data_file):

    with open(data_file) as f:
        lines = f.readlines()

    data = []

    for line in lines:
        data.append(line.split())

    death_rates = {}
    ln_death_rates = {}

    for l in data[3:]:
        death_rates[l[0]] = []
        ln_death_rates[l[0]] = []

    np.seterr(divide='ignore')
    for l in data[3:]:
        if l[2] == '.':
            death_rates[l[0]].append(1)
            ln_death_rates[l[0]].append(np.log(death_rates[l[0]][-1]))
        else:
            death_rates[l[0]].append(float(l[2]))
            ln_death_rates[l[0]].append(np.log(death_rates[l[0]][-1]))

    return [death_rates, ln_death_rates]

# custom data in the same format of the "Both.txt" file 

def mort_rates(data_file):

    with open(data_file) as f:
        lines = f.readlines()

    data = []

    for line in lines:
        data.append(line.split())

    death_rates = {}
    ln_death_rates = {}

    for e in data[0]:
        death_rates[e] = []
        ln_death_rates[e]= []
    np.seterr(divide='ignore')

    for row in data[1:]:
        for i in range(len(row)):
            death_rates[data[0][i]].append(float(row[i]))
            ln_death_rates[data[0][i]].append(np.log(death_rates[data[0][i]][-1]))

    ln_death_rates.pop("x")
    death_rates.pop("x")

    return [death_rates, ln_death_rates]

# calculate ax

def cal_ax(log_mt):

    n = len(log_mt[0])
    a = [0] * n

    for i in range(n):
        for coh in log_mt:
            a[i] += coh[i]
        a[i] = a[i] / n

    return a

# calculate Ax

def cal_hax(log_mt,ax):

    h = []

    for coh in log_mt:
        h.append(coh[:])

    for i in range(len(h)):
        for j in range(len(h[0])):
            h[i][j] = h[i][j] - ax[j]

    return h

# calculating bx, kt and s1 using the svd

def cal_bx_kt_s(hax):

    svd = np.linalg.svd(hax)
    k = np.matrix.transpose(svd[0])[0]
    s1 = svd[1][0]
    bx = svd[2][0]
    kt = [e * s1 for e in k]

    return [bx, kt]

# the fitted lee carter model

def lee_carter(log_mt):

    ax = cal_ax(log_mt)
    hax = cal_hax(log_mt, ax)
    par = cal_bx_kt_s(hax)

    return [ax] + par

# forecasting kt parameter using random walk with drift model

def rand_walk_drift_forc(kt,n):

    c = (kt[-1] - kt[0]) / (len(kt)-1)
    forecast = []

    for i in range(n):
        forecast.append(kt[-1] + c*(i+1))

    return forecast

# forecasting mortality rates

def forecast_mt(ax,bx,f_kt,start):

    mt_forc = {}
    year = start
    years = []

    for k in f_kt:
        years.append(year)
        mt_forc[year] = []
        for i in range(len(ax)):
            mt_forc[year].append(np.exp(ax[i] + bx[i]*k))
        year += 1

    return [mt_forc,years]

# forecat using the lee carter method

def forc_lee_carter(log_mt,start,n):

    par = lee_carter(log_mt)
    f_kt = rand_walk_drift_forc(par[2],n)
    forecasts = forecast_mt(par[0],par[1],f_kt,start)

    return forecasts

# calculating the difference vector

def diff_vect(test, forc):
    diff = {}
    for year in forc:
        if year in test:
            diff[year] = np.subtract(test[year],forc[year])
    return diff

# calculate MAE error

def mae(test, forc, d):

    diff = diff_vect(test, forc)
    error = {}

    for year in diff:

        pos = d
        error[year] = {}
        error[year]["g"] = 0
        error[year]["p"] = [0]

        for i in range(len(diff[year])):
            error[year]["g"] += np.absolute(diff[year][i])
            if i == pos:
                error[year]["p"][-1] = error[year]["p"][-1] / d
                pos = pos + d
                error[year]["p"].append(np.absolute(diff[year][i]))
            else:
                error[year]["p"][-1] += np.absolute(diff[year][i])

        error[year]["g"] = error[year]["g"] / len(diff[year])
        error[year]["p"][-1] = error[year]["p"][-1] / (len(diff[year]) - pos + d)

    return error

# calculate RMSE error

def rmse(test, forc, d):

    diff = diff_vect(test, forc)
    error = {}

    for year in diff:

        error[year] = {}
        error[year]["g"] = 0
        error[year]["p"] = [0]
        pos = d

        for i in range(len(diff[year])):
            error[year]["g"] += diff[year][i]**2
            if i == pos:
                error[year]["p"][-1] = np.sqrt(error[year]["p"][-1] / d)
                pos = pos + d
                error[year]["p"].append(diff[year][i]**2)
            else:
                error[year]["p"][-1] += diff[year][i]**2

        error[year]["g"] = np.sqrt(error[year]["g"] / len(diff[year]))
        error[year]["p"][-1] = np.sqrt(error[year]["p"][-1] / (len(diff[year]) - pos + d))

    return error

# calculate MAPE error

def mape(test, forc, d):

    diff = diff_vect(test, forc)
    error = {}

    for year in diff:

        pos = d
        error[year] = {}
        error[year]["g"] = 0
        error[year]["p"] = [0]

        for i in range(len(diff[year])):
            error[year]["g"] += np.absolute(100 * diff[year][i] / forc[year][i])
            if i == pos:
                error[year]["p"][-1] = error[year]["p"][-1] / d
                pos = pos + d
                error[year]["p"].append(np.absolute(100 * diff[year][i] / forc[year][i]))
            else:
                error[year]["p"][-1] += np.absolute(100 * diff[year][i] / forc[year][i])

        error[year]["g"] = error[year]["g"] / len(diff[year])
        error[year]["p"][-1] = error[year]["p"][-1] / (len(diff[year]) - pos + d)

    return error

# generating a set for a given period

def gen_set(data,start_y,t):

    new_data = []
    years = []

    for e in range(start_y,start_y+t):
        new_data.append(data[e])
        years.append(e)

    return [new_data, years]

# remove old age

def gen_set_without_old(data,age_limit):

    test = []
    limit_test = {}

    for year in data:
        test.append([year,data[year]])
    for year in test:
        limit_test[int(year[0])] = year[1][:age_limit]

    return limit_test

# generate the training sets for cross-validating forecasts horizon

def gen_train_set_forc_horizon(data,size,ya,yz):

    train = []

    for i in range(ya,yz-size+2):
        train.append(gen_set(data,i,size))

    return train

# generate the training sets for cross-validating data availability

def gen_train_set_data_av(data,min_size,ya,yf):

    train = []

    for i in range(min_size,yf-ya+1):
        train.append(gen_set(data,yf-i,i))

    return train

# cross validate the model in term of data availability

def cross_valid_data_av(data,ya,yf,min_size,hor,age_limit,d):

     test = gen_set_without_old(data[0],age_limit)
     forc_data = gen_set_without_old(data[1],age_limit)
     train = gen_train_set_data_av(forc_data,min_size,ya,yf)
     data_error = {
         "mae": {},
         "rmse": {},
         "mape": {}
     }

     for set in train:

         forc = forc_lee_carter(set[0], yf, hor)[0]
         data_error["mae"][len(set[1])] = mae(test, forc, d)
         data_error["rmse"][len(set[1])] = rmse(test, forc, d)
         data_error["mape"][len(set[1])] = mape(test, forc, d)

     return data_error


# cross validate the model in term of forecasts horizon

def cross_valid_forc_horizon(data,ya,yz,size,hor,age_limit,d):

     test = gen_set_without_old(data[0],age_limit)
     forc_data = gen_set_without_old(data[1],age_limit)
     train = gen_train_set_forc_horizon(forc_data,size,ya,yz)
     data_error = {
         "mae": {},
         "rmse": {},
         "mape": {}
     }

     for set in train:

         forc = forc_lee_carter(set[0], set[1][-1]+1, hor)[0]

         data_error["mae"][set[1][-1]] = mae(test, forc, d)
         data_error["rmse"][set[1][-1]] = rmse(test, forc, d)
         data_error["mape"][set[1][-1]] = mape(test, forc, d)

     return data_error

# getting tables from data_av:

def tab_data_av(models,size_min,ya,yf,hor,meth,g,p):

    years = []
    tab = []
    for y in range(hor):
        years.append(y+yf)

    for i in range(size_min,yf-ya+1):
        raw = [yf-i]
        for y in years:
            if g:
                raw.append(float("{:.2f}".format(models[meth][i][y]["g"])))
            else:
                raw.append(float("{:.2f}".format(models[meth][i][y]["p"][p])))
        tab.append(raw)

    years.insert(0,"1st year")
    t = np.matrix.transpose(np.array(tab))

    return [years, t]

# getting tables from forc_horizon:

def tab_forc_horizon(models,size,ya,yz,meth,g,p,year):

    years = []
    tab = []
    col = []

    for y in range(ya+size-1,yz+1):
        years.append(y)

    for y in range(year,yz+1):
        raw = []
        col.append(y)
        for i in years:
            if y in models[meth][i]:
                if g:
                    raw.append(float("{:.2f}".format(models[meth][i][y]["g"])))
                else:
                    raw.append(float("{:.2f}".format(models[meth][i][y]["p"][p])))
            else:
                raw.append(0)
        tab.append(raw)
    col.insert(0,"last year")
    tab.insert(0,years)

    return [col, tab]

# print table

def tabl(tab):
    fig = go.Figure(data=[go.Table(
        header=dict(values=tab[0],
                    line_color='black',
                    fill_color='white',
                    align='center'),
        cells=dict(values=tab[1],
                   line_color='black',
                   fill_color='white',
                   align='center'))
    ])
    fig.update_layout(width=1200, height=1600)
    fig.show()
    return 0