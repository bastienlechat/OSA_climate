
dp$dependant_var <- dp$t2m_mean

dp$idfactor <- do.call(paste, c(dp[,c('userid','year','month')], sep="-"))
dp$idfactor <- as.factor(dp$idfactor)

dp$idfactor2 <- do.call(paste, c(dp[,c('userid','year','week')], sep="-"))
dp$idfactor2 <- as.factor(dp$idfactor2)

#######################################################
## Cross-sectional models (between individuals)

varknots <- equalknots(dp$dependant_var,fun="ns",df=4)
lagknots <- logknots(4, 2)
cbdv<- crossbasis(dp$dependant_var, lag=4, argvar=list(fun="ns", knots=varknots),
                  arglag=list(knots=lagknots))

rh <- onebasis(dp$relative_humidity, "ns", df=4)
tcc <- onebasis(dp$tcc_mean, "ns", df=4)
sp <- onebasis(dp$sp_mean, "ns", df=4)
pm <- onebasis(dp$pm2p5_mean, "ns", df=4)
doy <- onebasis(dp$day_of_year, "ns", df=4)
daylight <- onebasis(dp$daylight, "ns", df=4)
tp <- onebasis(dp$total_precipitation_mean, "ns", df=4)

t2m <- onebasis(dp$dependant_var,fun="ns",df=4)

t2m_lin <- onebasis(dp$dependant_var,fun="lin")

dftrend <- round(as.numeric(diff(range(dp$day))/365.25 * 6))
btrend <- ns(dp$day, knots=19)


lagknots <- logknots(4, 2)
cbdv_lin<- crossbasis(dp$dependant_var, lag=4, argvar=list(fun="lin"),
                      arglag=list(knots=lagknots))

# Store predictions for each country
all_predictions <- list()


####################################
## Global -- continuous
## m0 -- main model
cen_value = quantile(dp$dependant_var, c(0.5), na.rm=TRUE)

m0 <- gnm(TST ~ cbdv + weekday + rh + tcc + sp + pm + doy + tp, data=dp, eliminate=factor(dp$idfactor2), trace=TRUE, verbose=TRUE)
m0_pred <- crosspred(cbdv, m0, cen=cen_value, at=seq(-10,30,by=0.05))
rm(m0)
gc()

##m1 - daylight adjustement instead of doy
m1 <- gnm(TST ~ cbdv + weekday + rh + tcc + sp + pm + daylight+tp, data=dp, eliminate=factor(dp$idfactor2), trace=TRUE, verbose=TRUE)
m1_pred <- crosspred(cbdv, m1, cen=cen_value, at=seq(-10,30,by=0.05))
rm(m1)
gc()

##m1b -- both daylight and doy (global model only)
m1b <- gnm(TST ~ cbdv + weekday + rh + tcc + sp + pm + daylight+doy+tp, data=dp, eliminate=factor(dp$idfactor2), trace=TRUE, verbose=TRUE)
m1b_pred <- crosspred(cbdv, m1b, cen=cen_value, at=seq(-10,30,by=0.05))
rm(m1b)
gc()

##m2 - trend adjustement instead of doy
m2 <- gnm(TST ~ cbdv + weekday + rh + tcc + sp + pm + btrend+tp, data=dp, eliminate=factor(dp$idfactor2), trace=TRUE, verbose=TRUE)
m2_pred <- crosspred(cbdv, m2, cen=cen_value, at=seq(-10,30,by=0.05))
rm(m2)
gc()

##m3 - year-month intercept instead of year-week
m3 <- gnm(TST ~ cbdv + weekday + rh + tcc + sp + pm + doy+tp, data=dp, eliminate=factor(dp$idfactor), trace=TRUE, verbose=TRUE)
m3_pred <- crosspred(cbdv, m3, cen=cen_value, at=seq(-10,30,by=0.05))
rm(m3)
gc()


###### Alternative exposure modelling
## m5 - spline no lag
m5 <- gnm(TST ~ t2m + weekday + rh + tcc + sp + pm +tp+doy, data=dp, eliminate=factor(dp$idfactor2), trace=TRUE, verbose=TRUE)
m5_pred <- crosspred(t2m, m5, cen=cen_value, at=seq(-10,30,by=0.05))

rm(m5)
gc()

## m6 - linear t2m - no lag
m6 <- gnm(TST ~ t2m_lin + weekday + rh + tcc + sp + pm +tp+doy+daylight, data=dp, eliminate=factor(dp$idfactor2), trace=TRUE, verbose=TRUE)
m6_pred <- crosspred(t2m_lin, m6, cen=cen_value, at=seq(-10,30,by=0.05))
rm(m6)
gc()

## m7 - linear t2m - 4 lag
m7 <- gnm(TST ~ cbdv_lin + weekday + rh + tcc + sp + pm +tp+doy+daylight, data=dp, eliminate=factor(dp$idfactor2), trace=TRUE, verbose=TRUE)
m7_pred <- crosspred(cbdv_lin, m7, cen=cen_value, at=seq(-10,30,by=0.05))
rm(m7)
gc()


par(mfrow=c(1,2))
plot(m0_pred, "overall", xlab="Temperature", ylab="Total sleep time", col='blue', ylim=c(-40,20))
lines(m1_pred, ci="l", col='red', lty=2, lwd=1.5)
lines(all_predictions[['global']]$m1b_pred, ci="l", col='#fc8d62', lty=2, lwd=3)
lines(m2_pred, ci="l", col='green', lty=2, lwd=2)
lines(m3_pred, ci="l", col='black', lty=2, lwd=2)
legend(x='topright', legend=c("Main", "m1","m1b","m2","m3"),fill = c("blue","red",'#fc8d62',"green",'black'))

plot(m0_pred, "overall", xlab="Temperature", ylab="Total sleep time", col='blue', ylim=c(-40,20))
lines(m5_pred, ci="l", col='yellow', lty=2, lwd=2)
lines(m6_pred, ci="l", col='purple', lty=2, lwd=2)
lines(m7_pred, ci="l", col='orange', lty=2, lwd=2)
legend(x='topright', legend=c("Main", "Spline","Lin","Lin+lag"),fill = c("blue","yellow","purple",'orange'))

#legend(x='topright', legend=c("Main", "MS1","MS2","MS3"),fill = c("blue","red","green",'black'))

# Store all predictions for this country
all_predictions[['global']] <- list(
  m0_pred = m0_pred,
  m1_pred = m1_pred,
  m1b_pred = m1b_pred,
  m2_pred = m2_pred,
  m3_pred = m3_pred,
  m5_pred = m5_pred,
  m6_pred = m6_pred,
  m7_pred = m7_pred
  
)

qsl = quantile(dp$dependant_var,c(0.01, 0.99), na.rm=TRUE)
lbt = as.numeric(qsl[1])
hbt = as.numeric(qsl[2])
med = quantile(dp$dependant_var,c(0.5), na.rm=TRUE)
pob = sum(dp$TSTbin)/length(dp$TSTbin)
preds_global <- data.frame(xvar=as.numeric(m0_pred$predvar), 
                           m0_est = as.numeric(m0_pred$allfit), 
                           m0_lb =as.numeric(m0_pred$alllow) , 
                           m0_hb =as.numeric(m0_pred$allhigh),
                           m1_est = as.numeric(m1_pred$allfit), 
                           m1_lb = as.numeric(m1_pred$alllow), 
                           m1_hb = as.numeric(m1_pred$allhigh),
                           m1b_est = as.numeric(m1b_pred$allfit), 
                           m1b_lb = as.numeric(m1b_pred$alllow), 
                           m1b_hb = as.numeric(m1b_pred$allhigh),
                           m2_est = as.numeric(m2_pred$allfit), 
                           m2_lb = as.numeric(m2_pred$alllow), 
                           m2_hb = as.numeric(m2_pred$allhigh),
                           m3_est = as.numeric(m3_pred$allfit), 
                           m3_lb = as.numeric(m3_pred$alllow), 
                           m3_hb = as.numeric(m3_pred$allhigh),
                           m5_est = as.numeric(m5_pred$allfit), 
                           m5_lb = as.numeric(m5_pred$alllow), 
                           m5_hb = as.numeric(m5_pred$allhigh),
                           m6_est = as.numeric(m6_pred$allfit), 
                           m6_lb = as.numeric(m6_pred$alllow), 
                           m6_hb = as.numeric(m6_pred$allhigh),
                           m7_est = as.numeric(m7_pred$allfit), 
                           m7_lb = as.numeric(m7_pred$alllow), 
                           m7_hb = as.numeric(m7_pred$allhigh),
                           p0 = pob,
                           device=device,
                           dependant_var='t2m_mean',
                           medt=med,
                           lbt=lbt,
                           hbt=hbt)


####################################
## Global -- binary
## m0 -- main model
m0 <- gnm(TSTbin ~ cbdv + weekday + rh + tcc + sp + pm + doy + tp, data=dp, eliminate=factor(dp$idfactor2), trace=TRUE, verbose=TRUE, family=binomial)
m0_pred <- crosspred(cbdv, m0, cen=cen_value, at=seq(-10,30,by=0.05))
rm(m0)
gc()

##m1 - daylight adjustement instead of doy
m1 <- gnm(TSTbin ~ cbdv + weekday + rh + tcc + sp + pm + daylight+tp, data=dp, eliminate=factor(dp$idfactor2), trace=TRUE, verbose=TRUE, family=binomial)
m1_pred <- crosspred(cbdv, m1, cen=cen_value, at=seq(-10,30,by=0.05))
rm(m1)
gc()

##m1b -- both daylight and doy (global model only)
m1b <- gnm(TSTbin ~ cbdv + weekday + rh + tcc + sp + pm + daylight+doy+tp, data=dp, eliminate=factor(dp$idfactor2), trace=TRUE, verbose=TRUE, family=binomial)
m1b_pred <- crosspred(cbdv, m1b, cen=cen_value, at=seq(-10,30,by=0.05))
rm(m1b)
gc()

##m2 - trend adjustement instead of doy
m2 <- gnm(TSTbin ~ cbdv + weekday + rh + tcc + sp + pm + btrend+tp, data=dp, eliminate=factor(dp$idfactor2), trace=TRUE, verbose=TRUE, family=binomial)
m2_pred <- crosspred(cbdv, m2, cen=cen_value, at=seq(-10,30,by=0.05))
rm(m2)
gc()

##m3 - year-month intercept instead of year-week
m3 <- gnm(TSTbin ~ cbdv + weekday + rh + tcc + sp + pm + doy+tp, data=dp, eliminate=factor(dp$idfactor), trace=TRUE, verbose=TRUE, family=binomial)
m3_pred <- crosspred(cbdv, m3, cen=cen_value, at=seq(-10,30,by=0.05))
rm(m3)
gc()


###### Alternative exposure modelling
## m5 - spline no lag
m5 <- gnm(TSTbin ~ t2m + weekday + rh + tcc + sp + pm +tp+doy, data=dp, eliminate=factor(dp$idfactor2), trace=TRUE, verbose=TRUE, family=binomial)
m5_pred <- crosspred(t2m, m5, cen=cen_value, at=seq(-10,30,by=0.05))

rm(m5)
gc()

## m6 - linear t2m - no lag
m6 <- gnm(TSTbin ~ t2m_lin + weekday + rh + tcc + sp + pm +tp+doy+daylight, data=dp, eliminate=factor(dp$idfactor2), trace=TRUE, verbose=TRUE, family=binomial)
m6_pred <- crosspred(t2m_lin, m6, cen=cen_value, at=seq(-10,30,by=0.05))
rm(m6)
gc()

## m7 - linear t2m - 4 lag
m7 <- gnm(TSTbin ~ cbdv_lin + weekday + rh + tcc + sp + pm +tp+doy+daylight, data=dp, eliminate=factor(dp$idfactor2), trace=TRUE, verbose=TRUE, family=binomial)
m7_pred <- crosspred(cbdv_lin, m7, cen=cen_value, at=seq(-10,30,by=0.05))
rm(m7)
gc()


par(mfrow=c(1,2))
plot(m0_pred, "overall", xlab="Temperature", ylab="Risk Ratio", col='blue', ylim=c(0.7,3))
lines(m1_pred, ci="l", col='red', lty=2, lwd=1.5)
lines(m1b_pred, ci="l", col='#fc8d62', lty=2, lwd=3)
lines(m2_pred, ci="l", col='green', lty=2, lwd=2)
lines(m3_pred, ci="l", col='black', lty=2, lwd=2)
legend(x='topright', legend=c("Main", "m1","m1b","m2","m3"),fill = c("blue","red",'#fc8d62',"green",'black'))

plot(m0_pred, "overall", xlab="Temperature", ylab="Risk Ratio", col='blue', ylim=c(0.7,3))
lines(m5_pred, ci="l", col='yellow', lty=2, lwd=2)
lines(m6_pred, ci="l", col='purple', lty=2, lwd=2)
lines(m7_pred, ci="l", col='orange', lty=2, lwd=2)
legend(x='topright', legend=c("Main", "Spline","Lin","Lin+lag"),fill = c("blue","yellow","purple",'orange'))

#legend(x='topright', legend=c("Main", "MS1","MS2","MS3"),fill = c("blue","red","green",'black'))

# Store all predictions for this country
all_predictions[['global']] <- list(
  m0_pred = m0_pred,
  m1_pred = m1_pred,
  m1b_pred = m1b_pred,
  m2_pred = m2_pred,
  m3_pred = m3_pred,
  m5_pred = m5_pred,
  m6_pred = m6_pred,
  m7_pred = m7_pred
  
)

preds_global <- data.frame(xvar=as.numeric(m0_pred$predvar), 
                           m0_est = as.numeric(m0_pred$allRRfit), 
                           m0_lb =as.numeric(m0_pred$allRRlow) , 
                           m0_hb =as.numeric(m0_pred$allRRhigh),
                           m1_est = as.numeric(m1_pred$allRRfit), 
                           m1_lb = as.numeric(m1_pred$allRRlow), 
                           m1_hb = as.numeric(m1_pred$allRRhigh),
                           m1b_est = as.numeric(m1b_pred$allRRfit), 
                           m1b_lb = as.numeric(m1b_pred$allRRlow), 
                           m1b_hb = as.numeric(m1b_pred$allRRhigh),
                           m2_est = as.numeric(m2_pred$allRRfit), 
                           m2_lb = as.numeric(m2_pred$allRRlow), 
                           m2_hb = as.numeric(m2_pred$allRRhigh),
                           m3_est = as.numeric(m3_pred$allRRfit), 
                           m3_lb = as.numeric(m3_pred$allRRlow), 
                           m3_hb = as.numeric(m3_pred$allRRhigh),
                           m5_est = as.numeric(m5_pred$allRRfit), 
                           m5_lb = as.numeric(m5_pred$allRRlow), 
                           m5_hb = as.numeric(m5_pred$allRRhigh),
                           m6_est = as.numeric(m6_pred$allRRfit), 
                           m6_lb = as.numeric(m6_pred$allRRlow), 
                           m6_hb = as.numeric(m6_pred$allRRhigh),
                           m7_est = as.numeric(m7_pred$allRRfit), 
                           m7_lb = as.numeric(m7_pred$allRRlow), 
                           m7_hb = as.numeric(m7_pred$allRRhigh),
                           p0 = pob,
                           device=device,
                           dependant_var='t2m_mean',
                           medt=med,
                           lbt=lbt,
                           hbt=hbt)


##############

dpall <- dp

#######################################################
## Cross-sectional models (between individuals)

unique_cities <- unique(dpall$main_city)
for (city in unique_cities){
  print('computing')
  print(city)
  combined_df <- data.frame()
  tryCatch(
    expr = {
      dp <- dpall[dpall$main_city==city,]
      if (length(unique(dp$userid)) >= 50) {
        output_dir <- file.path(paste("C:/CODE/R/WITHINGS/BIG_DATASET/CLIMATE_CHANGE/review/alternative/countries_v2/",city,sep=""))
        print(output_dir)
        if (!dir.exists(output_dir)){
          dir.create(output_dir)
        } else {
          print("Dir already exists!")
        }
        
        #dp$dependant_var <- rescale(dp$dependant_var, to = c(0, 100), from = quantile(dp$dependant_var,c(0.001, 0.999), na.rm=TRUE))
        qs = quantile(dp$dependant_var,c(0.001, 0.999), na.rm=TRUE)
        dp <- dp[(dp$dependant_var>qs[1])&(dp$dependant_var<qs[2]), ]
        
        qs = quantile(dp$dependant_var,c(0.001, 0.999), na.rm=TRUE)
        qsl = quantile(dp$dependant_var,c(0.01, 0.99), na.rm=TRUE)
        lbt = as.numeric(qsl[1])
        hbt = as.numeric(qsl[2])
        med = quantile(dp$dependant_var,c(0.5), na.rm=TRUE)
        
        varknots <- equalknots(dp$dependant_var,fun="ns",df=4)
        lagknots <- logknots(4, 2)
        cbdv<- crossbasis(dp$dependant_var, lag=4, argvar=list(fun="ns", knots=varknots),
                          arglag=list(knots=lagknots))
        
        rh <- onebasis(dp$relative_humidity, "ns", df=4)
        tcc <- onebasis(dp$tcc_mean, "ns", df=4)
        sp <- onebasis(dp$sp_mean, "ns", df=4)
        daylight <- onebasis(dp$daylight, "ns", df=4)
        pm <- onebasis(dp$pm2p5_mean, "ns", df=4)
        doy <- onebasis(dp$day_of_year, "ns", df=4)
        tp <- onebasis(dp$total_precipitation_mean, "ns", df=4)
        
        
        ###
        t2m <- onebasis(dp$dependant_var,fun="ns",df=4)
        t2m_lin <- onebasis(dp$dependant_var,fun="lin")
        
        dftrend <- round(as.numeric(diff(range(dp$day))/365.25 * 8))
        btrend <- ns(dp$day, knots=equalknots(dp$day, dftrend-1))
        
        lagknots <- logknots(4, 2)
        cbdv_lin<- crossbasis(dp$dependant_var, lag=4, argvar=list(fun="lin"),
                              arglag=list(knots=lagknots))
        
        xpred <- seq(qs[1],qs[2],0.05)
        ####################################
        ## Model 0 #Main model
        m0c <- gnm(TST ~ cbdv + weekday + rh + tcc + sp + pm + doy + tp, data=dp, eliminate=factor(dp$idfactor2))
        m0b <- gnm(TSTbin ~ cbdv + weekday + rh + tcc + sp +  pm + doy + tp, data=dp, eliminate=factor(dp$idfactor2),
                   family=binomial)
        
        m0c_pred <- crosspred(cbdv, m0c, cen=med, at=xpred)
        m0b_pred <- crosspred(cbdv, m0b, cen=med, at=xpred)
        
        
        pob = sum(dp$TSTbin)/length(dp$TSTbin)
        pred_m0 <- data.frame(xvar=as.numeric(unname(m0c_pred$predvar)), 
                              est = as.numeric(unname(m0c_pred$allfit)), 
                              lb =as.numeric(unname(m0c_pred$alllow)) , 
                              hb =as.numeric(unname(m0c_pred$allhigh)),
                              est_b = as.numeric(unname(m0b_pred$allRRfit)), 
                              lb_b =as.numeric(unname(m0b_pred$allRRlow)), 
                              hb_b =as.numeric(unname(m0b_pred$allRRhigh)),
                              model='m0',
                              p0 = pob,
                              device=device,
                              dependant_var='t2m_mean',
                              medt=med,
                              lbt=lbt,
                              hbt=hbt)
        rm(m0c)
        rm(m0b)
        gc()
        print("here")
        ####################################
        ## Model 1 #daylight adjustements
        m1c <- gnm(TST ~ cbdv + weekday + rh + tcc + sp + pm + daylight + tp, data=dp, eliminate=factor(dp$idfactor2))
        m1b <- gnm(TSTbin ~ cbdv + weekday + rh + tcc + sp +  pm + daylight + tp, data=dp, eliminate=factor(dp$idfactor2),
                   family=binomial)
        
        m1c_pred <- crosspred(cbdv, m1c, cen=med, at=xpred)
        m1b_pred <- crosspred(cbdv, m1b, cen=med, at=xpred)
        pred_m1 <- data.frame(xvar=as.numeric(unname(m1c_pred$predvar)), 
                              est = as.numeric(unname(m1c_pred$allfit)), 
                              lb =as.numeric(unname(m1c_pred$alllow)) , 
                              hb =as.numeric(unname(m1c_pred$allhigh)),
                              est_b = as.numeric(unname(m1b_pred$allRRfit)), 
                              lb_b =as.numeric(unname(m1b_pred$allRRlow)), 
                              hb_b =as.numeric(unname(m1b_pred$allRRhigh)),
                              model='m1',
                              p0 = pob,
                              device=device,
                              dependant_var='t2m_mean',
                              medt=med,
                              lbt=lbt,
                              hbt=hbt)
        rm(m1c)
        rm(m1b)
        gc()
        print("here - m1")
        ####################################
        ## Model 2 #long smooth trend (8df per year)
        m2c <- gnm(TST ~ cbdv + weekday + rh + tcc + sp + pm + btrend + tp, data=dp, eliminate=factor(dp$idfactor2))
        m2b <- gnm(TSTbin ~ cbdv + weekday + rh + tcc + sp +  pm + btrend + tp, data=dp, eliminate=factor(dp$idfactor2),
                   family=binomial)
        
        m2c_pred <- crosspred(cbdv, m2c, cen=med, at=xpred)
        m2b_pred <- crosspred(cbdv, m2b, cen=med, at=xpred)
        
        
        pred_m2 <- data.frame(xvar=as.numeric(unname(m2c_pred$predvar)), 
                              est = as.numeric(unname(m2c_pred$allfit)), 
                              lb =as.numeric(unname(m2c_pred$alllow)) , 
                              hb =as.numeric(unname(m2c_pred$allhigh)),
                              est_b = as.numeric(unname(m2b_pred$allRRfit)), 
                              lb_b =as.numeric(unname(m2b_pred$allRRlow)), 
                              hb_b =as.numeric(unname(m2b_pred$allRRhigh)),
                              model='m2',
                              p0 = pob,
                              device=device,
                              dependant_var='t2m_mean',
                              medt=med,
                              lbt=lbt,
                              hbt=hbt)
        rm(m2c)
        rm(m2b)
        gc()
        print('here - m2')
        ####################################
        ## Model 3 # intercept year-month
        m3c <- gnm(TST ~ cbdv + weekday + rh + tcc + sp + pm + doy + tp, data=dp, eliminate=factor(dp$idfactor))
        m3b <- gnm(TSTbin ~ cbdv + weekday + rh + tcc + sp +  pm + doy + tp, data=dp, eliminate=factor(dp$idfactor),
                   family=binomial)
        
        m3c_pred <- crosspred(cbdv, m3c, cen=med, at=xpred)
        m3b_pred <- crosspred(cbdv, m3b, cen=med, at=xpred)
        
        
        pred_m3 <- data.frame(xvar=as.numeric(unname(m3c_pred$predvar)), 
                              est = as.numeric(unname(m3c_pred$allfit)), 
                              lb =as.numeric(unname(m3c_pred$alllow)) , 
                              hb =as.numeric(unname(m3c_pred$allhigh)),
                              est_b = as.numeric(unname(m3b_pred$allRRfit)), 
                              lb_b =as.numeric(unname(m3b_pred$allRRlow)), 
                              hb_b =as.numeric(unname(m3b_pred$allRRhigh)),
                              model='m3',
                              p0 = pob,
                              device=device,
                              dependant_var='t2m_mean',
                              medt=med,
                              lbt=lbt,
                              hbt=hbt)
        rm(m3c)
        rm(m3b)
        gc()
        print('here - m3')
        ####################################
        ## Model 4 #Exposure model using spline (no lag)
        m5c <- gnm(TST ~ t2m + weekday + rh + tcc + sp + pm + doy + tp, data=dp, eliminate=factor(dp$idfactor2))
        m5b <- gnm(TSTbin ~ t2m + weekday + rh + tcc + sp +  pm + doy + tp, data=dp, eliminate=factor(dp$idfactor2),
                   family=binomial)
        
        m5c_pred <- crosspred(t2m, m5c, cen=med, at=xpred)
        m5b_pred <- crosspred(t2m, m5b, cen=med, at=xpred)
        rm(m5c)
        rm(m5b)
        gc()
        pred_m5 <- data.frame(xvar=as.numeric(m5c_pred$predvar), 
                              est = as.numeric(m5c_pred$allfit), 
                              lb =as.numeric(m5c_pred$alllow) , 
                              hb =as.numeric(m5c_pred$allhigh),
                              est_b = as.numeric(m5b_pred$allRRfit), 
                              lb_b =as.numeric(m5b_pred$allRRlow) , 
                              hb_b =as.numeric(m5b_pred$allRRhigh),
                              model='m5',
                              p0 = pob,
                              device=device,
                              dependant_var='t2m_mean',
                              medt=med,
                              lbt=lbt,
                              hbt=hbt)
        
        print('here - m3')
        ####################################
        ## Model 4 #Exposure model using spline (no lag)
        m6c <- gnm(TST ~ t2m_lin + weekday + rh + tcc + sp + pm + doy + tp, data=dp, eliminate=factor(dp$idfactor2))
        m6b <- gnm(TSTbin ~ t2m_lin + weekday + rh + tcc + sp +  pm + doy + tp, data=dp, eliminate=factor(dp$idfactor2),
                   family=binomial)
        
        m6c_pred <- crosspred(t2m_lin, m6c, cen=med, at=xpred)
        m6b_pred <- crosspred(t2m_lin, m6b, cen=med, at=xpred)
        rm(m6c)
        rm(m6b)
        gc()
        pred_m6 <- data.frame(xvar=as.numeric(m6c_pred$predvar), 
                              est = as.numeric(m6c_pred$allfit), 
                              lb =as.numeric(m6c_pred$alllow) , 
                              hb =as.numeric(m6c_pred$allhigh),
                              est_b = as.numeric(m6b_pred$allRRfit), 
                              lb_b =as.numeric(m6b_pred$allRRlow) , 
                              hb_b =as.numeric(m6b_pred$allRRhigh),
                              model='m6',
                              p0 = pob,
                              device=device,
                              dependant_var='t2m_mean',
                              medt=med,
                              lbt=lbt,
                              hbt=hbt)
        print('here - m6')
        ####################################
        ## Model 4 #Exposure model using spline (no lag)
        m7c <- gnm(TST ~ cbdv_lin + weekday + rh + tcc + sp + pm + doy + tp, data=dp, eliminate=factor(dp$idfactor2))
        m7b <- gnm(TSTbin ~ cbdv_lin + weekday + rh + tcc + sp +  pm + doy + tp, data=dp, eliminate=factor(dp$idfactor2),
                   family=binomial)
        
        m7c_pred <- crosspred(cbdv_lin, m7c, cen=med, at=xpred)
        m7b_pred <- crosspred(cbdv_lin, m7b, cen=med, at=xpred)
        
        pred_m7 <- data.frame(xvar=as.numeric(m7c_pred$predvar), 
                              est = as.numeric(m7c_pred$allfit), 
                              lb =as.numeric(m7c_pred$alllow) , 
                              hb =as.numeric(m7c_pred$allhigh),
                              est_b = as.numeric(m7b_pred$allRRfit), 
                              lb_b =as.numeric(m7b_pred$allRRlow) , 
                              hb_b =as.numeric(m7b_pred$allRRhigh),
                              model='m7',
                              p0 = pob,
                              device=device,
                              dependant_var='t2m_mean',
                              medt=med,
                              lbt=lbt,
                              hbt=hbt)
        print('here - m7')
        rm(m7c)
        rm(m7b)
        gc()

        par(mfrow=c(1,2))
        plot(m0c_pred, "overall", xlab="Temperature", ylab="Total sleep time", col='blue', ylim=c(-40,20))
        lines(m1c_pred, ci="l", col='red', lty=2, lwd=1.5)
        lines(m2c_pred, ci="l", col='green', lty=2, lwd=2)
        lines(m3c_pred, ci="l", col='black', lty=2, lwd=2)
        legend(x='topright', legend=c("Main", "m1","m2","m3"),fill = c("blue","red","green",'black'))
        
        plot(m0c_pred, "overall", xlab="Temperature", ylab="Total sleep time", col='blue', ylim=c(-40,20))
        lines(m5c_pred, ci="l", col='yellow', lty=2, lwd=2)
        lines(m6c_pred, ci="l", col='purple', lty=2, lwd=2)
        lines(m7c_pred, ci="l", col='orange', lty=2, lwd=2)
        legend(x='topright', legend=c("Main", "Spline","Lin","Lin+lag"),fill = c("blue","yellow","purple",'orange'))
        dev.off()
        
        pdf(paste(output_dir,"/binary_alternative_models_",device,".pdf", sep=""))
        par(mfrow=c(1,2))
        plot(m0b_pred, "overall", xlab="Temperature", ylab="Risk Ratio", col='blue', ylim=c(0.5,4))
        lines(m1b_pred, ci="l", col='red', lty=2, lwd=1.5)
        lines(m2b_pred, ci="l", col='green', lty=2, lwd=2)
        lines(m3b_pred, ci="l", col='black', lty=2, lwd=2)
        legend(x='topright', legend=c("Main", "m1","m2","m3"),fill = c("blue","red","green",'black'))
        
        plot(m0b_pred, "overall", xlab="Temperature", ylab="Risk Ratio", col='blue', ylim=c(0.5,4))
        lines(m5b_pred, ci="l", col='yellow', lty=2, lwd=2)
        lines(m6b_pred, ci="l", col='purple', lty=2, lwd=2)
        lines(m7b_pred, ci="l", col='orange', lty=2, lwd=2)
        legend(x='topright', legend=c("Main", "Spline","Lin","Lin+lag"),fill = c("blue","yellow","purple",'orange'))
        dev.off()
        
        combined_df <- rbind(pred_m0, pred_m1, pred_m2, pred_m3, pred_m5, pred_m6, pred_m7)

      }
      
    }, 
    error = function(e){ 
      message(e)
    },
    finally = {
    })
}