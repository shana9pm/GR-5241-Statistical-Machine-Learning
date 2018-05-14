
library(quantmod)

#First I copied the chart in Wiki with all their symbols in Excel.
#I copy the symbol column and paste with transpose,
#then I get longstring here.
longstring="AAPL	AXP	BA	CAT	CSCO	CVX	DIS	DWDP	GE	GS	HD	IBM	INTC	JNJ	JPM	KO	MCD	MMM	MRK	MSFT	NKE	PFE	PG	TRV	UNH	UTX	V	VZ	WMT	XOM
"
DJIname <- strsplit(longstring,split = "\t")
DJIname <- DJIname[[1]]
DJIname[30] <- "XOM"#has \n

data <- getSymbols("AAPL", auto.assign = F, from = "2017-01-01", to = "2018-01-01")
data <- data[,4]
#initialize data to get number of columns

for(i in 2:30){
  datatemp <- getSymbols(DJIname[i], auto.assign = F,
              from = "2017-01-01", to = "2018-01-01")
  #since we only take use of close price,I think to get it first is better
  datatemp <- datatemp[,4]
  data <- cbind.data.frame(data,datatemp)
}


princomp(data,cor=F)
biplot(princomp(data,cor=F))
screeplot(princomp(data,cor=F))

princomp(data,cor=T)
biplot(princomp(data,cor=T))
screeplot(princomp(data,cor=T))

data <- as.matrix(data)
data.lag <- diff(data,lag=1,differences = 1)
data <- data[-1,]
dim(data)
data.r <- data.lag/data
princomp(data.r,cor=T)
biplot(princomp(data.r,cor=T))
screeplot(princomp(data.r,cor=T))




