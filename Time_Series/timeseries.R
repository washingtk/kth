library(dplyr)
library(pracma)
library(ggplot2)
library(forecast)


# load data and eliminate the last column(ave)
rawdata <- scan("https://bolin.su.se/data/stockholm/files/stockholm-historical-weather-observations-2017/temperature/monthly/stockholm_monthly_mean_temperature_1756_2017_hom.txt",
          sep = "")
rawdata <- t(matrix(rawdata, nrow = 14))
# eliminate year and average temp
data <- rawdata[, c(-1,-14)]
# extracting temparature in recent 100 years from 1917to 2017
x <- data[162, ]
for(i in 163:224){x <- c(x, data[i, ])}
y<- data[225, ]
for(i in 226:262){y <- c(y, data[i, ])}
z<- data[255, ]
for(i in 256:262){z <- c(z, data[i, ])}

# decompose by function
original <- ts(c(x, y), frequency = 12, start = 1917, end = 2017)
o.stl <- stl(original, s.window = 'periodic')
plot(o.stl)
autoplot(original) + ggtitle("Original data") +
  theme(axis.title=element_text(size=16), axis.text=element_text(size=16, face="bold"),
        plot.title = element_text(size = 30, face = "bold"))


# encode data into tie series data
until80 <- ts(x, frequency = 12, start = 1917, end = 1979)
from80 <- ts(y, frequency = 12, start = 1980, end = 2017)
plot(until80)
plot(from80)

# eliminate trend component in only 1980~
regression80 <- lm(y ~ linspace(1, 456, 456))
print(regression80)
plot(y)
abline(regression80)
subtract_man <- 0.004559959 * linspace(1, 456, 456)
adjusted_y <- y - subtract_man
plot(adjusted_y)
autoplot(original) + geom_segment(aes(x = 1917, y = 5.9, xend = 1980, yend = 5.9), color = "red") +
  geom_segment(aes(x = 1980, y = 5.63, xend = 2017, yend = 5.63+subtract_man[456]), color = "red") +
  ylab("Temp") + theme(axis.title=element_text(size=16), axis.text=element_text(size=16, face="bold"),
                       plot.title = element_text(size = 30, face = "bold"))

#data combined and analyze
data_adjusted <- c(x, adjusted_y)
data_adjusted.ts <- ts(data_adjusted, frequency = 12, start = 1917, end = 2017)
autoplot(data_adjusted.ts) + ylab("Temp") + ggtitle("adjusted temperature") +
  theme(axis.title=element_text(size=16), axis.text=element_text(size=16, face="bold"),
        plot.title = element_text(size = 30, face = "bold"))

# seasonal component
season <- t(matrix(data_adjusted, nrow =12))
seasonal <- colMeans(season)
plot(as.ts(rep(seasonal, 4)))
seasonal_gg <- ts(rep(seasonal, 101), frequency = 12, start = 1917, end = 2017)
autoplot(seasonal_gg, color = "blue") + ylab("Temp") + ggtitle("seasonal component") +
  theme(axis.title=element_text(size=16), axis.text=element_text(size=16, face="bold"),
        plot.title = element_text(size = 30, face = "bold"))

# only remainder left
remainder.adjusted <- data_adjusted - seasonal
remainder.ts <- ts(remainder.adjusted, frequency = 12, start = 1917, end = 2017)

# analyze remainder
autoplot(remainder.ts, color = "green") + ylab("Temp") + ggtitle("Remainder with mean") + 
  geom_hline(yintercept = mean(remainder.ts)) +
  theme(axis.title=element_text(size=16), axis.text=element_text(size=16, face="bold"),
        plot.title = element_text(size = 30, face = "bold"))

# acf and pacf of remainder
acf.ts <- acf(remainder.ts)
pacf.ts <- pacf(remainder.ts)
autoplot(acf.ts) + ggtitle("ACF of remainder") +
  theme(axis.title=element_text(size=16), axis.text=element_text(size=16, face="bold"),
        plot.title = element_text(size = 30, face = "bold"))
autoplot(pacf.ts) + ggtitle("PACF of remainder") + 
  theme(axis.title=element_text(size=16), axis.text=element_text(size=16, face="bold"),
        plot.title = element_text(size = 30, face = "bold"))

# fitting remaindre with auto.arima
auto.fit <- auto.arima(remainder.ts, max.p = 3, max.q = 3, stationary = TRUE)
print(auto.fit)
sim <- arima.sim(n = 1200, list(ar = 0.3954),
                 sd = sqrt(3.319), start.innov = remainder.ts[1], n.start = 1)
sim.ts <- ts(sim, frequency = 12, start = 1917, end = 2017)
acf.model <- acf(sim.ts)
autoplot(acf.model) + ylab("Temp") + ggtitle("model ACF") +
  theme(axis.title=element_text(size=16), axis.text=element_text(size=16, face="bold"),
        plot.title = element_text(size = 30))

pacf.model <- pacf(sim.ts)
autoplot(pacf.model) + ylab("Temp") + ggtitle("model PACF") +
  theme(axis.title=element_text(size=16), axis.text=element_text(size=16, face="bold"),
        plot.title = element_text(size = 30))

# qqplot
qqplot(remainder.ts, sim.ts, xlab = "remainder", ylab = "AR(1) simulation", main = "Q-Q plot for AR(1) : data 1917-2017",
       cex.lab = 1.5, cex.axis = 1.2)
abline(0,1, col="red")
autoplot(sim.ts, main = "AR(1) coefficient 0.3954") + ylab("Temp") +
  theme(axis.title=element_text(size=16), axis.text=element_text(size=16, face="bold"),
        plot.title = element_text(size = 30))

# predict
pred <- forecast(auto.fit, h = 12)
autoplot(pred) + ggtitle("AR(1) phi = 0.3954") +
  theme(axis.title=element_text(size=16), axis.text=element_text(size=16, face="bold"),
        plot.title = element_text(size = 30, face = "bold"))
trend_pre <- 5.6284 + 0.00456*linspace(457, 458, 12)
predict <- trend_pre + seasonal + pred$mean[1:12]
predictl80 <- trend_pre + seasonal + pred$lower[,1][1:12]
predictl95 <- trend_pre + seasonal + pred$lower[,2][1:12]
predicth80 <- trend_pre + seasonal + pred$upper[,1][1:12]
predicth95 <- trend_pre + seasonal + pred$upper[,2][1:12]

data19 <- c(z, predict)
l80 <- c(z, predictl80)
h80 <- c(z, predicth80)
l95 <- c(z, predictl95)
h95 <- c(z, predicth95)
time <- linspace(2010, 2019, 108)
mat <- data.frame(time, data19, l80, h80, l95, h95)

predict2018 <- ggplot(mat, aes(x = time, y = data19)) + geom_line() +
  geom_ribbon(aes(ymin=l80, ymax=h80), alpha=0.5) + 
  geom_ribbon(aes(ymin=l95, ymax=h95), alpha=0.2) + ggtitle("original and prediction") + ylab("Temp") + xlab("Year")
predict2018 + theme(axis.title=element_text(size=16), axis.text=element_text(size=16,face="bold"),
                    plot.title = element_text(size = 20, face = "bold")) + 
  scale_x_continuous(breaks = c(2010, 2012, 2014, 2016, 2018))

test <- ggplot(data = prediction.matrix, aes(x=x)) + geom_line(aes(y=predict), color = "blue") +
  geom_ribbon(aes(ymin=predictl95, ymax=predicth95), alpha=0.3) +
  geom_ribbon(aes(ymin=predictl80, ymax=predicth80), alpha=0.5) + xlab("Time") + ylab("Temp") + ggtitle("prediction 2018")
test + theme(axis.title=element_text(size=16), axis.text=element_text(size=16, face="bold"),
             plot.title = element_text(size = 30, face = "bold"))

hw_p <- hw(ts(z, frequency = 12, start = 2010, end = 2018)) 
autoplot(hw_p) + ylab("Temp") + xlab("Year") + scale_x_continuous(breaks = c(2010, 2012, 2014, 2016, 2018)) + ggtitle("forecast HW") + 
  theme(axis.title=element_text(size=20), axis.text=element_text(size=16, face="bold"), plot.title = element_text(size = 30))
hw_whole <- hw(original)
qqplot(original, hw_whole$fitted, xlab = "Original data", ylab = "Haltwinter model",
       main = "Q-Q plot for HaltWinter : data 1917-2017", cex.lab = 1.5, cex.axis = 1.2)
abline(0,1, col="red")

time <- linspace(2018, 2019, 12)
mat <- data.frame(time, predict, predictl80, predicth80, predictl95, predicth95)
predict2018 <- ggplot(mat, aes(x = time, y = predict)) + geom_line() +
  geom_ribbon(aes(ymin=predictl80, ymax=predicth80), alpha=0.5) + 
  geom_ribbon(aes(ymin=predictl95, ymax=predicth95), alpha=0.2) + ggtitle("only prediction") + ylab("Temp") + xlab("Year")
predict2018 + theme(axis.title=element_text(size=16), axis.text=element_text(size=14,face="bold"),
                    plot.title = element_text(size = 40, face = "bold")) + 
  scale_x_continuous(breaks = c(2018, 2019))