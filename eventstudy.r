# Replication file for 
# Kurt Schmidheiny and Sebastian Siegloch, 
# "On Event Studies and Distributed-Lags in Two-Way Fixed Effects Models: 
# Identification, Equivalence, and Generalization", 
# Journal of Applied Econometrics, 2023.
# 
# (c) Kurt Schmidheiny, Sebastian Siegloch, and Jakob Miethe (2020, 2023)
# 
# Version 1 (May 27, 2020)
# Version 2 (Jan 30, 2023)

# This file was constructed using RStudio [2022.12.0+353] with R [4.2.2 (2022-10-31)]

# uncomment to install packages needed for this file:
# install.packages("data.table")
# install.packages("ggplot2")
# install.packages("zoo")
# install.packages("plm")
# install.packages("lmtest")
# install.packages("haven")

library("data.table")  # the basic data structure used here (nests data.frame)
library("ggplot2")     # plotting package
library("zoo")         # for time variables
library("plm")         # a panel package
library("lmtest")      # functions for covariance matrices
library("haven")       # to import Stata .dta files

# clear environment
rm(list=ls(all=T))

# set local directory
# setwd("[yourpath]")

# ------------------------------------------------------------------------------
# New functions
# ------------------------------------------------------------------------------	

# function for reverse cumulative sum of vector
# summing up from end to start
revcumsum <- function(x){
  x <- rev(cumsum(rev(x)))
}

# function to calculate standard errors of cumulative sum
# b: a coefficient vector
# vcov: a variance covariance matrix
secumsum <- function(vcov){
  L <- dim(vcov)[1]
  # result vector with standard errors
  se <- c()
  # loop over elements
  for (i in c(1:L)){
    # Variance of cumulative sum from 1 to i
    # V[ax] = a*V[x]*a', a=[1, ..., 1]
    # create vector for summation
    a <- matrix(rep(1,i), nrow = 1)
    V <- a %*% vcov[1:i,1:i] %*% t(a)
    se[i] <- sqrt(V)
  }
  return(se)
}

# function to calculate standard errors of reverse cumulative sum
# summing up from end to start
# b: a coefficient vector
# vcov: a variance covariance matrix
serevcumsum <- function(vcov){
  L <- dim(vcov)[1]
  # result vector with standard errors
  se <- c()
  # loop over elements
  for (i in c(L:1)){
    # Variance of cumulative sum from i to L
    # V[ax] = a*V[x]*a', a=[1, ..., 1]
    a <- matrix(rep(1,L-i+1), nrow = 1)
    V <- a %*% vcov[i:L,i:L] %*% t(a)
    se[i] <- sqrt(V)
  }
  return(se)
}

# ------------------------------------------------------------------------------
# Data from Baker and Fradkin, Table 8, col (3)
# ------------------------------------------------------------------------------	

# open data from Baker and Fradkin (2017, henceforth BF2017)
bf2017 <- read_dta("C:/Users/tormo/Jupyter for econometrics/Other files/bf2017.dta")
print(colnames(bf2017))


# create period variable from year and month in R 'yearmon' format
bf2017$yearmonth <- as.yearmon(paste(bf2017$year, bf2017$month), "%Y%m")
setcolorder(bf2017, c("state", "state_abb", "year", "month", "yearmonth"))
print(colnames(bf2017))



# create covariate used in Baker and Fradkin (2017)
bf2017$frac_total_ui <- (bf2017$cont_claims+bf2017$init_claims)/bf2017$population

# define data as panel data (plm package)
# states are cross-section, yearmonth are periods
bf2017 <- pdata.frame(bf2017, index = c("state", "yearmonth"))
colnames(bf2017)



# ------------------------------------------------------------------------------
# Figure B.1, Panel A, left
# multiple treatments of identical intensities
# replicates Baker and Fradkin, Table 8, col (3)
# ------------------------------------------------------------------------------	

# Baker and Fradkin use whole sample of dependent variable
# they assume that there was no treatment before and after observed sample

# create treatment adoption indicator
# increase by at least 13 weeks
bf2017$D_PBD_incr <- ifelse(diff(bf2017$PBD, 1L) >= 13, 1, 0)
# the diff functions of the plm package creates by default the 
# difference between current (t) and the previous time period (t-1)
# as defined by the time index of the pdata.frame

# replace missing first treatment with zero (implicitly assumed by BF2017)
bf2017$D_PBD_incr[is.na(bf2017$D_PBD_incr)] <- 0

# create 3 leads 
bf2017$F3_D_PBD_incr <- lead(bf2017$D_PBD_incr, 3L) 
bf2017$F2_D_PBD_incr <- lead(bf2017$D_PBD_incr, 2L) 
bf2017$F1_D_PBD_incr <- lead(bf2017$D_PBD_incr, 1L) 
# the lead functions of the plm package use by default the 
# subsequent time period (t+1) as defined by the time index of the pdata.frame

# create 4 lags
bf2017$L1_D_PBD_incr <- lag(bf2017$D_PBD_incr, 1L) 
bf2017$L2_D_PBD_incr <- lag(bf2017$D_PBD_incr, 2L) 
bf2017$L3_D_PBD_incr <- lag(bf2017$D_PBD_incr, 3L) 
bf2017$L4_D_PBD_incr <- lag(bf2017$D_PBD_incr, 4L) 
# the lag functions of the plm package use by default the 
# previous time period (t-1) as defined by the time index of the pdata.frame

# replace missing lags with zero (implicitly assumed by BF2017)
bf2017$L1_D_PBD_incr[is.na(bf2017$L1_D_PBD_incr)] <- 0 
bf2017$L2_D_PBD_incr[is.na(bf2017$L2_D_PBD_incr)] <- 0 
bf2017$L3_D_PBD_incr[is.na(bf2017$L3_D_PBD_incr)] <- 0 
bf2017$L4_D_PBD_incr[is.na(bf2017$L4_D_PBD_incr)] <- 0 

# view resulting design matrix
View(bf2017[, c("state", "year", "month", "yearmonth",
            names(bf2017)[grepl("D_PBD_", names(bf2017))])])

# estimate with fixed effects
estim_incr_bf <- plm(log(GJSI) ~ 
                       F3_D_PBD_incr + F2_D_PBD_incr + F1_D_PBD_incr + 
                       D_PBD_incr + 
                       L1_D_PBD_incr + L2_D_PBD_incr + L3_D_PBD_incr + L4_D_PBD_incr + 
                       frac_total_ui + as.factor(yearmonth), 
                     data = bf2017,
                     subset = (year <= 2011),
                     effect = "individual",
                     model = "within")
summary(estim_incr_bf)

# coefficients with standard errors
results_incr_bf <- coeftest(estim_incr_bf, vcov = vcovHC(estim_incr_bf, type = "sss", cluster = "group"))
# we use Stata standard errors for comparability with the Stata results
# note that the standard errors only conform to the Stata standard errors
# up to the 5th digit. This is due to conventions in Stata concerning small sample
# adjustments. You can use different standard errors with the "type" argument 
# of vcovHC(). Also note that we could leave out the "cluster = "group"" argument
# since the covariance matrix we chose already calculates cluster robust standard 
# errors on the i-level (here, the "state")

# beta coefficients and standard errors
beta_incr_bf <- data.table(month_to_reform = c(-3:4), 
                           coef = results_incr_bf[1:8,1],
                           se = results_incr_bf[1:8,2])

# plot beta coefficients
plot_incr_bf <- ggplot(beta_incr_bf, aes(x = month_to_reform, y = coef)) +
  geom_line(color = "darkblue") +
  geom_point(color = "darkblue") +
  geom_errorbar(aes(ymin = coef - 1.96*se, ymax = coef + 1.96*se), width=.1,
                color = "darkblue") +
  geom_hline(yintercept = 0) + 
  geom_vline(xintercept = -0.5, linetype = "dashed") +
  scale_x_continuous(breaks = c(-3:4)) +
  xlab(paste0("Months relative to reform \n Observations: ", nobs(estim_incr_bf), 
              ", states: ", length(unique(attributes(estim_incr_bf[["residuals"]])$index[,"state"])),
              ", periods: ", length(unique(attributes(estim_incr_bf[["residuals"]])$index[,"yearmonth"])),
              " (", sort(index(estim_incr_bf$model)[,2])[1], " - ",
              sort(index(estim_incr_bf$model)[,2])[nobs(estim_incr_bf)],").")) +
  scale_y_continuous(limits=c(-0.02, 0.04), breaks = seq(-0.02, 0.04, by = 0.02)) +
  ylab("Effect on log search intensity") +
  theme_bw() 

plot_incr_bf

pdf("./Fig_B1_A_left.pdf", height = 4, width = 8)
plot_incr_bf
dev.off()

# ------------------------------------------------------------------------------
# Figure B.1, Panel A, right
# multiple treatments of identical intensities
# replicates Baker and Fradkin, Table 8, col (5)
# ------------------------------------------------------------------------------	

# create treatment adoption indicator
# decrease by at least 7 weeks

bf2017$D_PBD_decr <- ifelse(diff(bf2017$PBD, 1L) <= -7, 1, 0)

# create 3 leads 
bf2017$F3_D_PBD_decr <- lead(bf2017$D_PBD_decr, 3L) 
bf2017$F2_D_PBD_decr <- lead(bf2017$D_PBD_decr, 2L) 
bf2017$F1_D_PBD_decr <- lead(bf2017$D_PBD_decr, 1L) 

# create 4 lags
bf2017$L1_D_PBD_decr <- lag(bf2017$D_PBD_decr, 1L) 
bf2017$L2_D_PBD_decr <- lag(bf2017$D_PBD_decr, 2L) 
bf2017$L3_D_PBD_decr <- lag(bf2017$D_PBD_decr, 3L) 
bf2017$L4_D_PBD_decr <- lag(bf2017$D_PBD_decr, 4L) 

# estimate
estim_decr_bf <- plm(log(GJSI) ~
                       F3_D_PBD_decr + F2_D_PBD_decr + F1_D_PBD_decr + 
                       D_PBD_decr + 
                       L1_D_PBD_decr + L2_D_PBD_decr + L3_D_PBD_decr + L4_D_PBD_decr + 
                       frac_total_ui + as.factor(yearmonth), 
                     data = bf2017,
                     subset = (year >= 2012),
                     effect = "individual",
                     model = "within")
summary(estim_decr_bf)

# coefficients with standard errors
results_decr_bf <- coeftest(estim_decr_bf, vcov = vcovHC(estim_decr_bf, type = "sss", cluster = "group"))

# beta coefficients and standard errors
beta_decr_bf <- data.table(month_to_reform = c(-3:4), 
                           coef = results_decr_bf[1:8,1],
                           se = results_decr_bf[1:8,2])

# plot beta coefficients
plot_decr_bf <- ggplot(beta_decr_bf, aes(x = month_to_reform, y = coef)) +
  geom_line(color = "darkblue") +
  geom_point(color = "darkblue") +
  geom_errorbar(aes(ymin = coef - 1.96*se, ymax = coef + 1.96*se), width=.1,
                color = "darkblue") +
  geom_hline(yintercept = 0) + 
  geom_vline(xintercept = -0.5, linetype = "dashed") +
  scale_x_continuous(breaks = c(-3:4)) +
  xlab(paste0("Months relative to reform \n Observations: ", nobs(estim_decr_bf), 
              ", states: ", length(unique(attributes(estim_decr_bf[["residuals"]])$index[,"state"])),
              ", periods: ", length(unique(attributes(estim_decr_bf[["residuals"]])$index[,"yearmonth"])),
              " (", sort(index(estim_decr_bf$model)[,2])[1], " - ",
              sort(index(estim_decr_bf$model)[,2])[nobs(estim_decr_bf)],").")) +
  scale_y_continuous(limits=c(-0.025, 0.04), breaks = seq(-0.02, 0.04, by = 0.02)) +
  ylab("Effect on log search intensity") +
  theme_bw() 

plot_decr_bf

pdf("./Fig_B1_A_right.pdf", height = 4, width = 8)
plot_decr_bf
dev.off()

# ------------------------------------------------------------------------------
# Figure B.1, Panel B, left
# multiple treatments of identical intensities (Schmidheiny/Siegloch case 2)
# effect window from -3 to +4
# estimation with distributed-lags in levels
# ------------------------------------------------------------------------------

# generate treatment status from treatment adoption indicator for increases
# cumulative sum, arbitrary starting value set to zero as missing values 
# are set to zero above
bf2017$PBD_incr <- ave(bf2017$D_PBD_incr, bf2017$state, FUN = cumsum)
summary(bf2017$PBD_incr)

# estimate distributed-lag model in levels with 2 leads and 4 lags
# note crisis subsample must be defined through subset not in data
# otherwise observations of leads and lags are missing
estim_incr_dl_fe <- plm(log(GJSI) ~ 
                          lead(PBD_incr, k=2) + lead(PBD_incr, k=1) + 
                          PBD_incr + 
                          lag(PBD_incr, k=1) + lag(PBD_incr, k=2) + lag(PBD_incr, k=3) + lag(PBD_incr, k=4) +
                          frac_total_ui + as.factor(yearmonth), 
                        data = bf2017,
                        subset = (year <= 2011),
                        effect = "individual",
                        model = "within")
summary(estim_incr_dl_fe)

# gamma coefficients
gamma_incr_dl_fe <- estim_incr_dl_fe$coefficients[1:7]

# variance-covariance matrix
vcov_incr_dl_fe <- vcovHC(estim_incr_dl_fe, type = "sss", cluster = "group")[1:7,1:7]

# beta coefficients and standard errors
# cumulative sum of gamma coefficients starting at zero in period -1
beta_incr_dl_fe <- data.table(
  # effect window
  month_to_reform = c(-3:4), 
  # point estimates
  coef = c(-revcumsum(gamma_incr_dl_fe[1:2]),  # leads
            0,                                 # reference period
            cumsum(gamma_incr_dl_fe[3:7])),    # lags
  # standard errors using the formula of linear combinations
  se = c(-serevcumsum(vcov_incr_dl_fe[1:2, 1:2]), # leads
         0,                                       # reference period
         secumsum(vcov_incr_dl_fe[3:7, 3:7])))    # lags
beta_incr_dl_fe

# plot beta coefficients
plot_incr_dl_fe <- ggplot(beta_incr_dl_fe, aes(x = month_to_reform, y = coef)) +
  geom_line(color = "darkblue") +
  geom_point(color = "darkblue") +
  geom_errorbar(aes(ymin = coef - 1.96*se, ymax = coef + 1.96*se), width=.1,
                color = "darkblue") +
  geom_hline(yintercept = 0) + 
  geom_vline(xintercept = -0.5, linetype = "dashed") +
  scale_x_continuous(breaks = c(-3:4)) +
  xlab(paste0("Months relative to reform \n Observations: ", nobs(estim_incr_dl_fe), 
              ", states: ", length(unique(attributes(estim_incr_dl_fe[["residuals"]])$index[,"state"])),
              ", periods: ", length(unique(attributes(estim_incr_dl_fe[["residuals"]])$index[,"yearmonth"])),
              " (", sort(index(estim_incr_dl_fe$model)[,2])[1], " - ",
              sort(index(estim_incr_dl_fe$model)[,2])[nobs(estim_incr_dl_fe)],").")) +
  ylab("Effect on log search intensity") +
  scale_y_continuous(limits=c(-0.06, 0.025), breaks = seq(-0.06, 0.02, by = 0.02)) +
  theme_bw() 
  
plot_incr_dl_fe

pdf("./Fig_B1_B_left.pdf", height = 4, width = 8)
plot_incr_dl_fe
dev.off()

# ------------------------------------------------------------------------------
# Figure B.1, Panel B, right
# multiple treatments of identical intensities (Schmidheiny/Siegloch case 2)
# effect window from -3 to +4
# estimation with distributed-lags in levels
# ------------------------------------------------------------------------------

# generate treatment status from treatment adoption indicator for decreases
# cumulative sum
# set arbitrary starting value set to zero 
bf2017$D_PBD_decr[is.na(bf2017$D_PBD_decr)] <- 0
bf2017$PBD_decr <- ave(bf2017$D_PBD_decr, bf2017$state, FUN = cumsum)
summary(bf2017$PBD_decr)

# estimate distributed-lag model in levels with 2 leads and 4 lags
estim_decr_dl_fe <- plm(log(GJSI) ~ 
                          lead(PBD_decr, k=2) + lead(PBD_decr, k=1) + 
                          PBD_decr + 
                          lag(PBD_decr, k=1) + lag(PBD_decr, k=2) + lag(PBD_decr, k=3) + lag(PBD_decr, k=4) +
                          frac_total_ui + as.factor(yearmonth), 
                        data = bf2017,
                        subset = (year >= 2012),
                        effect = "individual",
                        model = "within")
summary(estim_decr_dl_fe)

# gamma coefficients
gamma_decr_dl_fe <- estim_decr_dl_fe$coefficients[1:7]

# store variance-covariance matrix
vcov_decr_dl_fe <- vcovHC(estim_decr_dl_fe, type = "sss", cluster = "group")[1:7,1:7]

# beta coefficients and standard errors
# cumulative sum of gamma coefficients starting at zero in period -1
beta_decr_dl_fe <- data.table(
  month_to_reform = c(-3:4), 
  coef = c(-revcumsum(gamma_decr_dl_fe[1:2]), # leads
            0,                                 # reference period
            cumsum(gamma_decr_dl_fe[3:7])),    # lags
  se = c(-serevcumsum(vcov_decr_dl_fe[1:2, 1:2]), # leads
         0,                                       # reference period
         secumsum(vcov_decr_dl_fe[3:7, 3:7])))    # lags
beta_decr_dl_fe

# plot beta coefficients
plot_decr_dl_fe <- ggplot(beta_decr_dl_fe, aes(x = month_to_reform, y = coef)) +
  geom_line(color = "darkblue") +
  geom_point(color = "darkblue") +
  geom_errorbar(aes(ymin = coef - 1.96*se, ymax = coef + 1.96*se), width=.1,
                color = "darkblue") +
  geom_hline(yintercept = 0) + 
  geom_vline(xintercept = -0.5, linetype = "dashed") +
  scale_x_continuous(breaks = c(-3:4)) +
  xlab(paste0("Months relative to reform \n Observations: ", nobs(estim_decr_dl_fe), 
              ", states: ", length(unique(attributes(estim_decr_dl_fe[["residuals"]])$index[,"state"])),
              ", periods: ", length(unique(attributes(estim_decr_dl_fe[["residuals"]])$index[,"yearmonth"])),
              " (", sort(index(estim_decr_dl_fe$model)[,2])[1], " - ",
              sort(index(estim_decr_dl_fe$model)[,2])[nobs(estim_decr_dl_fe)],").")) +
  ylab("Effect on log search intensity") +
  scale_y_continuous(limits=c(-0.045, 0.04), breaks = seq(-0.04, 0.04, by = 0.02)) +
  theme_bw() 

plot_decr_dl_fe

pdf("./Fig_B1_B_right.pdf", height = 4, width = 8)
plot_decr_dl_fe
dev.off()

# ------------------------------------------------------------------------------
# Figure B.1, Panel B, left 
# alternative estimation with event study specification
# ------------------------------------------------------------------------------

# overwrite treatment adoption indicators from above because of assumed zeros for NAs

# create treatment adoption indicator
# increase by at least 13 weeks
bf2017$D_PBD_incr <- ifelse(diff(bf2017$PBD, 1L) >= 13, 1, 0)

# create 3 leads 
bf2017$F3_D_PBD_incr <- lead(bf2017$D_PBD_incr, 3L) 
bf2017$F2_D_PBD_incr <- lead(bf2017$D_PBD_incr, 2L) 
bf2017$F1_D_PBD_incr <- lead(bf2017$D_PBD_incr, 1L) 

# create 4 lags
bf2017$L1_D_PBD_incr <- lag(bf2017$D_PBD_incr, 1L) 
bf2017$L2_D_PBD_incr <- lag(bf2017$D_PBD_incr, 2L) 
bf2017$L3_D_PBD_incr <- lag(bf2017$D_PBD_incr, 3L) 
bf2017$L4_D_PBD_incr <- lag(bf2017$D_PBD_incr, 4L) 

# The first observation of the treatment status in the data is 2006-01. The first
# treatment adoption indicator is observed in 2006-02. The first observation of
# the dependent variable that can be included in the estimation is 4-1=3 periods
# later, i.e. 2006-5 (Schmidheiny/Siegloch Remark 4). The unobserved 4th lag 
# of the treatment adopton indicator for 2006-5 can be set to an arbitrary value,
# e.g. zero. 
bf2017$L4_D_PBD_incr[as.yearmon(bf2017$yearmonth) == as.yearmon("2006-05")] <- 0
# Note that in unbalanced panel, this operation may be unit specific.

# The last observation of the dependent variable in the crisis sample is 2011-12.
# There are more than 3 leads observed as the treatment adoption indicator is 
# observed up to 2015-12. No need to set arbitrary starting value.

# define crisis sample: 2006-05 to 2011-12
bf2017$crisis <- as.yearmon(bf2017$yearmonth) >= as.yearmon("2006-05") &
  as.yearmon(bf2017$yearmonth) <= as.yearmon("2011-12")

# generate binned endpoints according to eq. (5)
# cumulative sum of lag 4 starting at 2006-05
bf2017$L4bin_D_PBD_incr[bf2017$crisis] <- 
  ave(bf2017$L4_D_PBD_incr[bf2017$crisis], 
      bf2017$state[bf2017$crisis], 
      FUN = cumsum)
summary(bf2017$L4bin_D_PBD_incr)

# generate binned endpoint at lead 3 according to eq. (5)
# downward cumulative sum of lead 3 starting at 2011-12
bf2017$F3bin_D_PBD_incr[bf2017$crisis] <- 
  ave(bf2017$F3_D_PBD_incr[bf2017$crisis], 
      bf2017$state[bf2017$crisis], 
      FUN = revcumsum)
summary(bf2017$L4bin_D_PBD_incr)

# view binned and not binned endpoints
# View(bf2017[, c("state", "yearmonth", "D_PBD_incr", "L4_D_PBD_incr", "L4bin_D_PBD_incr", "F3_D_PBD_incr", "F3bin_D_PBD_incr")])

# estimate event-study model in levels with fixed effects
# with 4 lags, 3 leads, and normalization at -1
estim_incr_es_fe <- plm(log(GJSI) ~
                          F3bin_D_PBD_incr + F2_D_PBD_incr  + 
                          D_PBD_incr + 
                          L1_D_PBD_incr + L2_D_PBD_incr + L3_D_PBD_incr + L4bin_D_PBD_incr + 
                          frac_total_ui + as.factor(yearmonth), 
                        data = bf2017,
                        subset = (year <= 2011),
                        index = c("state", "yearmonth"),
                        effect = "individual",
                        model = "within")
summary(estim_incr_es_fe)

# beta coefficients and standard errors
results_incr_es_fe <- coeftest(estim_incr_es_fe, 
                            vcov = vcovHC(estim_incr_es_fe, type = "sss", cluster = "group"))

beta_incr_es_fe <- data.table(
  month_to_reform = c(-3:4), 
  coef = c(results_incr_es_fe[1:2,1],   # leads
           0,                           # reference period
           results_incr_es_fe[3:7,1]),  # lags
  se = c(results_incr_es_fe[1:2,2],  # leads
         0,                          # reference period
         results_incr_es_fe[3:7,2])) # lags
beta_incr_es_fe

# plot beta coefficients
plot_incr_es_fe <- ggplot(beta_incr_es_fe, aes(x = month_to_reform, y = coef)) +
  geom_line(color = "darkblue") +
  geom_point(color = "darkblue") +
  geom_errorbar(aes(ymin = coef - 1.96*se, ymax = coef + 1.96*se), width=.1,
                color = "darkblue") +
  geom_hline(yintercept = 0) + 
  geom_vline(xintercept = -0.5, linetype = "dashed") +
  scale_x_continuous(breaks = c(-3:4)) +
  xlab(paste0("Months relative to reform \n Observations: ", nobs(estim_incr_es_fe), 
              ", states: ", length(unique(attributes(estim_incr_es_fe[["residuals"]])$index[,"state"])),
              ", periods: ", length(unique(attributes(estim_incr_es_fe[["residuals"]])$index[,"yearmonth"])),
              " (", sort(index(estim_incr_es_fe$model)[,2])[1], " - ",
              sort(index(estim_incr_es_fe$model)[,2])[nobs(estim_incr_es_fe)],").")) +
  scale_y_continuous(limits=c(-0.06, 0.025), breaks = seq(-0.06, 0.02, by = 0.02)) +
  ylab("Effect on log search intensity") +
  theme_bw() 

plot_incr_es_fe

# identical sample, point estimates and confidence bounds as in distributed-lag 
# specification in levels

# ------------------------------------------------------------------------------
# Figure B.2, left
# multiple treatments of varying intensities (Schmidheiny/Siegloch case 4)
# effect window from -3 to +4
# estimation with distributed-lags in levels
# full sample
# ------------------------------------------------------------------------------

# describe treatment status
summary(bf2017$PBD)

# estimate distributed-lag model in levels with fixed effects with 2 leads and 4 lags
estim_dl_fe_full <- plm(log(GJSI) ~ 
                          lead(PBD, k=c(2:1)) + 
                          PBD + 
                          lag(PBD, k=c(1:4)) +
                          frac_total_ui + as.factor(yearmonth), 
                        data = bf2017,
                        effect = "individual",
                        model = "within")
summary(estim_dl_fe_full)

# gamma coefficients
gamma_dl_fe_full <- estim_dl_fe_full$coefficients[1:7]

# store variance-covariance matrix
vcov_dl_fe_full <- vcovHC(estim_dl_fe_full, type = "sss", cluster = "group")[1:7,1:7]

# beta coefficients and standard errors
# cumulative sum of gamma coefficients starting at zero in period -1
beta_dl_fe_full <- data.table(
  month_to_reform = c(-3:4), 
  coef = c(-revcumsum(gamma_dl_fe_full[1:2]),  # leads
            0,                                 # reference period
            cumsum(gamma_dl_fe_full[3:7])),    # lags
  se = c(-serevcumsum(vcov_dl_fe_full[1:2, 1:2]), # leads
         0,                                       # reference period
         secumsum(vcov_dl_fe_full[3:7, 3:7])))    # lags
beta_dl_fe_full

# plot beta coefficients:
plot_dl_fe_full <- ggplot(beta_dl_fe_full, aes(x = month_to_reform, y = coef)) +
  geom_line(color = "darkblue") +
  geom_point(color = "darkblue") +
  geom_errorbar(aes(ymin = coef - 1.96*se, ymax = coef + 1.96*se), width=.1,
                color = "darkblue") +
  geom_hline(yintercept = 0) + 
  geom_vline(xintercept = -0.5, linetype = "dashed") +
  scale_x_continuous(breaks = c(-3:4)) +
  xlab(paste0("Months relative to reform \n Observations: ", nobs(estim_dl_fe_full), 
              ", states: ", length(unique(attributes(estim_dl_fe_full[["residuals"]])$index[,"state"])),
              ", periods: ", length(unique(attributes(estim_dl_fe_full[["residuals"]])$index[,"yearmonth"])),
              " (", sort(index(estim_dl_fe_full$model)[,2])[1], " - ",
              sort(index(estim_dl_fe_full$model)[,2])[nobs(estim_dl_fe_full)],").")) +
  scale_y_continuous(limits=c(-0.006, 0.002), breaks = seq(-0.006, 0.002, by = 0.002)) +
  ylab("Effect on log search intensity") +
  theme_bw() 

plot_dl_fe_full

pdf("./Fig_B2_left.pdf", height = 4, width = 8)
plot_dl_fe_full
dev.off()

# ------------------------------------------------------------------------------
# Figure B.2, right
# also reported in Figure B.4, left (FE)
# multiple treatments of varying intensities (Schmidheiny/Siegloch case 4)
# effect window from -3 to +4
# estimation with distributed-lags in levels
# crisis sample
# ------------------------------------------------------------------------------

# describe treatment status
summary(bf2017$PBD)

# estimate distributed-lag model in levels with fixed effects with 2 leads and 4 lags
estim_dl_fe_crisis <- plm(log(GJSI) ~ 
                            lead(PBD, k=c(2:1)) + 
                            PBD + 
                            lag(PBD, k=c(1:4)) +
                            frac_total_ui + as.factor(yearmonth), 
                          data = bf2017,
                          subset = (year <= 2011),
                          effect = "individual",
                          model = "within")
summary(estim_dl_fe_crisis)

# gamma coefficients
gamma_dl_fe_crisis <- estim_dl_fe_crisis$coefficients[1:7]

# store variance-covariance matrix
vcov_dl_fe_crisis <- vcovHC(estim_dl_fe_crisis, type = "sss", cluster = "group")[1:7,1:7]

# beta coefficients and standard errors
# cumulative sum of gamma coefficients starting at zero in period -1
beta_dl_fe_crisis <- data.table(
  month_to_reform = c(-3:4), 
  coef = c(-revcumsum(gamma_dl_fe_crisis[1:2]),  # leads
            0,                                   # reference period
            cumsum(gamma_dl_fe_crisis[3:7])),    # lags
  se = c(-serevcumsum(vcov_dl_fe_crisis[1:2, 1:2]), # leads
         0,                                         # reference period
         secumsum(vcov_dl_fe_crisis[3:7, 3:7])))    # lags
beta_dl_fe_crisis

# plot beta coefficients:
plot_dl_fe_crisis <- ggplot(beta_dl_fe_crisis, aes(x = month_to_reform, y = coef)) +
  geom_line(color = "darkblue") +
  geom_point(color = "darkblue") +
  geom_errorbar(aes(ymin = coef - 1.96*se, ymax = coef + 1.96*se), width=.1,
                color = "darkblue") +
  geom_hline(yintercept = 0) + 
  geom_vline(xintercept = -0.5, linetype = "dashed") +
  scale_x_continuous(breaks = c(-3:4)) +
  xlab(paste0("Months relative to reform \n Observations: ", nobs(estim_dl_fe_crisis), 
              ", states: ", length(unique(attributes(estim_dl_fe_crisis[["residuals"]])$index[,"state"])),
              ", periods: ", length(unique(attributes(estim_dl_fe_crisis[["residuals"]])$index[,"yearmonth"])),
              " (", sort(index(estim_dl_fe_crisis$model)[,2])[1], " - ",
              sort(index(estim_dl_fe_crisis$model)[,2])[nobs(estim_dl_fe_crisis)],").")) +
  scale_y_continuous(limits=c(-0.006, 0.002), breaks = seq(-0.006, 0.002, by = 0.002)) +
  ylab("Effect on log search intensity") +
  theme_bw() 

plot_dl_fe_crisis

pdf("./Fig_B2_right.pdf", height = 4, width = 8)
plot_dl_fe_crisis
dev.off()

# ------------------------------------------------------------------------------
# Figure B.3
# multiple treatments of varying intensities (Schmidheiny/Siegloch case 4)
# changing effect window from -3,+4 to -3,+18
# estimation with distributed-lags in levels
# crisis sample
# ------------------------------------------------------------------------------

# describe treatment status
summary(bf2017$PBD)

# loop from 4 to 18 lags
beta_loops <- data.table(month_to_reform = 1, coef = 1, loop = 0, color = 0)[-1]
for (l in c(4:18)){
  
  # estimate distributed-lag model in levels with l lags and 2 leads
  estim_loop <- plm(log(GJSI) ~ 
                      lead(PBD, k=c(2:1)) + 
                      PBD + 
                      lag(PBD, k=c(1:l)) +
                      frac_total_ui + as.factor(yearmonth),
                    data = bf2017,
                    subset = (year <= 2011),
                    effect = "individual",
                    model = "within")
  
  # gamma coefficients
  gamma_loop <- estim_loop$coefficients[1:(3+l)]
  
  # beta coefficients
  # cumulative sum of gamma coefficients starting at zero in period -1
  beta_loop <- rep(NA,4+l)
  beta_loop[1:2] <- -revcumsum(gamma_loop[1:2]) # leads
  beta_loop[3] <- 0 # reference period
  beta_loop[4:(4+l)] <- cumsum(estim_loop$coefficients[3:(3+l)]) # lags
  beta_loop
  
  # format in data table
  beta_loop <- data.table(
    month_to_reform = c(-3:l), 
    coef = c(-revcumsum(gamma_loop[1:2]),   # leads
              0,                            # reference period
              cumsum(gamma_loop[3:(3+l)])), # lags
    loop = l,
    color = l)
  
  # combine plot data from all loops
  beta_loops <- rbind(beta_loops, beta_loop)
  
  # plot beta coefficients for 4 to l lags
  plot_loop <- ggplot(beta_loops, aes(x = month_to_reform, y = coef, 
                                      group = loop, color = color)) +
    geom_line() +
    geom_point() +
    geom_hline(yintercept = 0) + 
    geom_vline(xintercept = -0.5, linetype = "dashed") +
    scale_x_continuous(limits=c(-3, 18), breaks = seq(-3, 18, by = 1)) +
    xlab(paste0("Months relative to reform \n Observations: ",
                nobs(estim_loop), 
                ", states: ", 
                length(unique(attributes(estim_loop[["residuals"]])$index[,"state"])),
                ", periods: ", 
                length(unique(attributes(estim_loop[["residuals"]])$index[,"yearmonth"])),
                " (", sort(index(estim_loop$model)[,2])[1], " - ",
                sort(index(estim_loop$model)[,2])[nobs(estim_loop)],").")) +
    scale_y_continuous(limits=c(-0.0052, 0.001), breaks = seq(-0.005, 0.001, by = 0.001)) +
    ylab("Effect on log search intensity") +
    theme_bw() +
    theme(legend.position = "none")
  
  print(plot_loop)
  
  pdf(paste0("./Fig_B3_",l,".pdf"), height = 4, width = 8)
  print(plot_loop)
  dev.off()
}

# save last iteration with 18 lags
estim_dl_18_fe_crisis <- estim_loop

# estimate distributed-lag model in levels with fixed effects with 2 leads and 4 lags
# short sample as with 18 lags
estim_dl_4short_fe_crisis <- plm(log(GJSI) ~ 
                                   lead(PBD, k=c(2:1)) + 
                                   PBD + 
                                   lag(PBD, k=c(1:4)) +
                                   frac_total_ui + as.factor(yearmonth), 
                                 data = bf2017,
                                 subset = rownames(index(estim_dl_18_fe_crisis$model)),
                                 effect = "individual",
                                 model = "within")
summary(estim_dl_4short_fe_crisis)

# gamma coefficients
gamma_dl_4short_fe_crisis <- estim_dl_4short_fe_crisis$coefficients[1:7]

# beta coefficients
# cumulative sum of gamma coefficients starting at zero in period -1
beta_dl_4short_fe_crisis <- data.table(
  month_to_reform = c(-3:4), 
  coef = c(-revcumsum(gamma_dl_4short_fe_crisis[1:2]),  # leads
            0,                                          # reference period
            cumsum(gamma_dl_4short_fe_crisis[3:7])),    # lags
  loop = 4,
  color = 4)
beta_dl_4short_fe_crisis

# plot beta coefficients for 4 to 18 lags plus 4 lags with short sample
plot_dl_4_18_fe_crisis <-  ggplot(beta_loops, aes(x = month_to_reform, y = coef, 
                                                  group = loop, color = color)) +
  geom_line() +
  geom_point() +
  geom_line(data = beta_dl_4short_fe_crisis, linetype = "dashed") + 
  geom_point(data = beta_dl_4short_fe_crisis) + 
  geom_hline(yintercept = 0) + 
  geom_vline(xintercept = -0.5, linetype = "dashed") +
  scale_x_continuous(breaks = seq(-3, 18, by = 1)) +
  xlab("Months relative to reform\n") +
  scale_y_continuous(limits=c(-0.0052, 0.001), breaks = seq(-0.005, 0.001, by = 0.001)) +
  ylab("Effect on log search intensity") +
  theme_bw() +
  theme(legend.position = "none")

plot_dl_4_18_fe_crisis

pdf(paste0("./Fig_B3.pdf"), height = 4, width = 8)
plot_dl_4_18_fe_crisis
dev.off()

# ------------------------------------------------------------------------------
# Figure B.4, left
# multiple treatments of varying intensities (Schmidheiny/Siegloch case 4)
# effect window from -3 to +4
# estimation with distributed-lags in first differences and in levels
# crisis sample
# ------------------------------------------------------------------------------

# describe treatment status in levels and first differences
summary(bf2017$PBD)
summary(diff(bf2017$PBD))

# estimate distributed-lag model in levels
# see Figure B.2 above
# stored in beta_dl_fe_crisis and estim_dl_fe_crisis

# estimate distributed-lag model in first differences with 2 leads and 4 lags
# no constant in first differences, i.e. no linear trend in levels
estim_dl_fd_crisis <- 
  plm(diff(log(GJSI)) ~ 
        lead(diff(PBD), k=c(2:1)) +
        diff(PBD) +
        lag(diff(PBD), k=c(1:4)) +
        diff(frac_total_ui) + as.factor(yearmonth) - 1,
      data = bf2017,
      subset = (year <= 2011),
      model = "pooling")
summary(estim_dl_fd_crisis)

# gamma coefficients
gamma_dl_fd_crisis <- estim_dl_fd_crisis$coefficients[1:7]

# store variance-covariance matrix
vcov_dl_fd_crisis <- vcovHC(estim_dl_fd_crisis, type = "sss", cluster = "group")[1:8,1:8]

# beta coefficients and standard errors
# cumulative sum of gamma coefficients starting at zero in period -1
beta_dl_fd_crisis <- data.table(
  month_to_reform = c(-3:4), 
  coef = c(-revcumsum(gamma_dl_fd_crisis[1:2]),  # leads
            0,                                   # reference period
            cumsum(gamma_dl_fd_crisis[3:7])),    # lags
  se = c(-serevcumsum(vcov_dl_fd_crisis[1:2, 1:2]), # leads
         0,                                         # reference period
         secumsum(vcov_dl_fd_crisis[3:7, 3:7])))    # lags
beta_dl_fd_crisis

#  gamma (combined FE and FD)
beta_dl_fe_crisis[, estimator := "Fixed effects"]
beta_dl_fd_crisis[, estimator := "First difference"]
beta_dl_fefd_crisis <- rbind(beta_dl_fe_crisis, beta_dl_fd_crisis)

# plot beta coefficients (combined FE and FD)
plot_dl_fefd_crisis <- ggplot(beta_dl_fefd_crisis, aes(x = month_to_reform, y = coef,
                                                       group = estimator, color = estimator,
                                                       shape = estimator)) +
  geom_line() +
  geom_point() +
  scale_shape_manual(values = c(17, 19)) +
  scale_color_manual(values = c("lightblue", "darkblue")) +
  geom_errorbar(aes(ymin = coef - 1.96*se, ymax = coef + 1.96*se), width=.1) +
  geom_hline(yintercept = 0) + 
  geom_vline(xintercept = -0.5, linetype = "dashed") +
  scale_x_continuous(breaks = c(-3:4)) +
  xlab(paste0("Months relative to reform \n Observations: ", nobs(estim_dl_fe_crisis), 
              ", states: ", length(unique(attributes(estim_dl_fe_crisis[["residuals"]])$index[,"state"])),
              ", periods: ", length(unique(attributes(estim_dl_fe_crisis[["residuals"]])$index[,"yearmonth"])),
              " (", sort(index(estim_dl_fe_crisis$model)[,2])[1], " - ",
              sort(index(estim_dl_fe_crisis$model)[,2])[nobs(estim_dl_fe_crisis)],").")) +
  scale_y_continuous(limits=c(-0.01, 0.002), breaks = seq(-0.01, 0.002, by = 0.002)) +
  ylab("Effect on log search intensity") +
  theme_bw() +
  theme(legend.position = "bottom",
        legend.title = element_blank())

plot_dl_fefd_crisis

pdf("./Fig_B4_left.pdf", height = 4, width = 8)
plot_dl_fefd_crisis
dev.off()

# ------------------------------------------------------------------------------
# Figure B.4, right
# multiple treatments of varying intensities (Schmidheiny/Siegloch case 4)
# effect window from -3 to +18
# estimation with distributed-lags in first differences and in levels
# crisis sample
# ------------------------------------------------------------------------------

# describe treatment status in levels and first differences
summary(bf2017$PBD)
summary(diff(bf2017$PBD))

# estimate distributed-lag model in levels with 2 leads and 18 lag
# estimate distributed-lag model in levels with fixed effects with 2 leads and 4 lags
estim_dl_18_fe_crisis <- plm(log(GJSI) ~ 
                            lead(PBD, k=c(2:1)) + 
                            PBD + 
                            lag(PBD, k=c(1:18)) +
                            frac_total_ui + as.factor(yearmonth), 
                          data = bf2017,
                          subset = (year <= 2011),
                          effect = "individual",
                          model = "within")
summary(estim_dl_18_fe_crisis)

# gamma coefficients
gamma_dl_18_fe_crisis <- estim_dl_18_fe_crisis$coefficients[1:21]

# store variance-covariance matrix
vcov_dl_18_fe_crisis <- vcovHC(estim_dl_18_fe_crisis, type = "sss", cluster = "group")[1:21,1:21]

# beta coefficients and standard errors
# cumulative sum of gamma coefficients starting at zero in period -1
beta_dl_18_fe_crisis <- data.table(
  month_to_reform = c(-3:18), 
  coef = c(-revcumsum(gamma_dl_18_fe_crisis[1:2]),  # leads
            0,                                      # reference period
            cumsum(gamma_dl_18_fe_crisis[3:21])),   # lags
  se = c(-serevcumsum(vcov_dl_18_fe_crisis[1:2, 1:2]), # leads
         0,                                            # reference period
         secumsum(vcov_dl_18_fe_crisis[3:21, 3:21])))  # lags
beta_dl_18_fe_crisis

# estimate distributed-lag model in first differences with 2 leads and 18 lags
# no constant in first differences, i.e. no linear trend in levels
estim_dl_18_fd_crisis <- 
  plm(diff(log(GJSI)) ~ 
        lead(diff(PBD), k=c(2:1)) +
        diff(PBD) +
        lag(diff(PBD), k=c(1:18)) +
        diff(frac_total_ui) + as.factor(yearmonth) - 1,
      data = bf2017,
      subset = (year <= 2011),
      model = "pooling")
summary(estim_dl_18_fd_crisis)

# gamma coefficients
gamma_dl_18_fd_crisis <- estim_dl_18_fd_crisis$coefficients[1:21]

# store variance-covariance matrix
vcov_dl_18_fd_crisis <- vcovHC(estim_dl_18_fd_crisis, type = "sss", cluster = "group")[1:22,1:22]

# gamma
# cumulative sum of gamma coefficients starting at zero in period -1
beta_dl_18_fd_crisis <- data.table(
  month_to_reform = c(-3:18), 
  coef = c(-revcumsum(gamma_dl_18_fd_crisis[1:2]),  # leads
            0,                                      # reference period
            cumsum(gamma_dl_18_fd_crisis[3:21])),   # lags
  se = c(-serevcumsum(vcov_dl_18_fd_crisis[1:2, 1:2]), # leads
         0,                                            # reference period
         secumsum(vcov_dl_18_fd_crisis[3:21, 3:21])))  # lags
beta_dl_18_fd_crisis

#  gamma (combined FE and FD)
beta_dl_18_fe_crisis[, estimator := "Fixed effects"]
beta_dl_18_fd_crisis[, estimator := "First difference"]
beta_dl_18_fefd_crisis <- rbind(beta_dl_18_fe_crisis, beta_dl_18_fd_crisis)

# plot beta coefficients (combined FE and FD)
plot_dl_18_fefd_crisis <- ggplot(beta_dl_18_fefd_crisis, aes(x = month_to_reform, y = coef,
                                                             group = estimator, color = estimator,
                                                             shape = estimator)) +
  geom_line() +
  geom_point() +
  scale_shape_manual(values = c(17, 19)) +
  scale_color_manual(values = c("lightblue", "darkblue")) +
  geom_errorbar(aes(ymin = coef - 1.96*se, ymax = coef + 1.96*se), width=.1) +
  geom_hline(yintercept = 0) + 
  geom_vline(xintercept = -0.5, linetype = "dashed") +
  scale_x_continuous(breaks = seq(-3, 18, by = 1)) +
  xlab(paste0("Months relative to reform \n Observations: ", nobs(estim_dl_18_fe_crisis), 
              ", states: ", length(unique(attributes(estim_dl_18_fe_crisis[["residuals"]])$index[,"state"])),
              ", periods: ", length(unique(attributes(estim_dl_18_fe_crisis[["residuals"]])$index[,"yearmonth"])),
              " (", sort(index(estim_dl_18_fe_crisis$model)[,2])[1], " - ",
              sort(index(estim_dl_18_fe_crisis$model)[,2])[nobs(estim_dl_18_fe_crisis)],").")) +
  scale_y_continuous(limits=c(-0.0103, 0.002), breaks = seq(-0.01, 0.002, by = 0.002)) +
  ylab("Effect on log search intensity") +
  theme_bw() +
  theme(legend.position = "bottom",
        legend.title = element_blank())

plot_dl_18_fefd_crisis

pdf("./Fig_B4_right.pdf", height = 4, width = 8)
plot_dl_18_fefd_crisis
dev.off()
