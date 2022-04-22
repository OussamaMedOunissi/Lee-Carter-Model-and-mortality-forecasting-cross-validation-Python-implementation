import LeeCarter as lc

# application

data = lc.mort_rates("Both.txt")
t = 15
m_data = lc.gen_set_without_old(data[0],80)


e0 = lc.cross_valid_forc_horizon(data,1977,2014,20,10,80,20)
e1 = lc.cross_valid_forc_horizon(data,1977,2014,18,10,80,20)
e2 = lc.cross_valid_forc_horizon(data,1977,2014,15,10,80,20)
t1 = lc.tab_forc_horizon(e0,20,1977,2014,"mape",True,-1,2010)
t2 = lc.tab_forc_horizon(e1,18,1977,2014,"mape",True,-1,2010)
t3 = lc.tab_forc_horizon(e2,15,1977,2014,"mape",True,-1,2010)

swed = lc.mort_rates_db("sweden.txt")
yf = [2010,2005,2000,1995]
ya = [1940] * 4
hor = [9,14,19,24]
size = [2,2,2,2]
v = True

table5 = lc.cross_valid_data_av(swed,ya[0],yf[0],size[0],hor[0],80,20)
table10 = lc.cross_valid_data_av(swed,ya[1],yf[1],size[1],hor[1],80,20)
table15 = lc.cross_valid_data_av(swed,ya[2],yf[2],size[2],hor[2],80,20)
table20 = lc.cross_valid_data_av(swed,ya[3],yf[3],size[3],hor[3],80,20)

f0 = lc.tab_data_av(table5,size[0],ya[0],yf[0],hor[0],"mape",v,-1)
f1 = lc.tab_data_av(table10,size[1],ya[1],yf[1],hor[1],"mape",v,-1)
f2 = lc.tab_data_av(table15,size[2],ya[2],yf[2],hor[2],"mape",v,-1)
f3 = lc.tab_data_av(table20,size[3],ya[3],yf[3],hor[3],"mape",v,-1)

lc.tabl(f3)