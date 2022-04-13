#%%
"""
股票高频回测
"""
import sys
import copy
import json
import glob
import dask.dataframe as dskf
from  matplotlib import  pyplot as plt
plt.style.use('ggplot')
import bintrees
import time
bintrees.has_fast_tree_support()
from bintrees import FastRBTree
from collections import namedtuple
import os
import time
import lightgbm as lgb
import pandas as pd
import numpy as np
from collections import  defaultdict
from collections import deque
fromDate = "20210420"
toDate = "20210423"
BondPath = '/mnt/lustre/group/it/convartiablBond/tickLocalTime/'
szStockPath = '/mnt/lustre/group/it/stock/tickLocalTime/'
szStockPath = '/mnt/lustre/group/it/stock/fullTickData'
toPath = '/home/aether/sampleproject/projects/gourdSample/sdata/'
toPath = '/home/tlei/data/'
ins = "123030"
snapCol = ['code', 'trdTime', 'recvTime', 'mthPrc', 'cumVol', 'turnover', 'cumCnt',
       'bidPrc10', 'bidPrc9', 'bidPrc8', 'bidPrc7', 'bidPrc6', 'bidPrc5',
       'bidPrc4', 'bidPrc3', 'bidPrc2', 'bidPrc1', 'askPrc1', 'askPrc2',
       'askPrc3', 'askPrc4', 'askPrc5', 'askPrc6', 'askPrc7', 'askPrc8',
       'askPrc9', 'askPrc10', 'bidVol10', 'bidVol9', 'bidVol8', 'bidVol7',
       'bidVol6', 'bidVol5', 'bidVol4', 'bidVol3', 'bidVol2', 'bidVol1',
       'askVol1', 'askVol2', 'askVol3', 'askVol4', 'askVol5', 'askVol6',
       'askVol7', 'askVol8', 'askVol9', 'askVol10']
def cpData(qDate,fromPath,toPath,insList):
    dataTypes =["order","transaction","snapshot"]
    tmp =  os.system('cp %s/order_format %s/order_format'%(fromPath,toPath))
    tmp =  os.system('cp %s/snapshot_format %s/snapshot_format'%(fromPath,toPath))
    tmp =  os.system('cp %s/transaction_format %s/transaction_format'%(fromPath,toPath))
    #order_format/snapshot_format/transaction_format指的是order/snapshot/trans数据格式
    for ins in insList:
        for dti in dataTypes:
            sPath_i = fromPath+dti+"/%s/%s/%s/%s.csv"%(qDate[:4],qDate[4:6],qDate[6:8],ins)
            if os.path.exists(sPath_i):
                to_path_i = toPath+dti+"/%s/%s/%s/"%(qDate[:4],qDate[4:6],qDate[6:8])
                os.makedirs(to_path_i, exist_ok=True)
                file_path_i = to_path_i+"%s.csv"%ins
                if not  os.path.exists(file_path_i):
                    tmp = os.system(
                        'cp %s %s'%(sPath_i,file_path_i))

def readByID(ins,fromDate,toDate,dataType,reformPath):
    if dataType== "order":
        filePath = reformPath+"order"
        colName = pd.read_csv(reformPath+"order_format",header=None).iloc[:,0].values
    elif dataType== "trans":
        filePath = reformPath+"transaction"
        colName = pd.read_csv(reformPath+"transaction_format",header=None).iloc[:,0].values
    elif dataType== "snap":
        filePath = reformPath+"snapshot"
        colName = pd.read_csv(reformPath+"snapshot_format",header=None).iloc[:,0].values
    else:
        raise RuntimeError("dataType error")

    files = glob.glob(os.path.join(filePath, "*/*/*/%s.csv"%ins))
    fz = [x  for x in files if "".join(x.split("/")[-4:-1])>= fromDate]
    fz = [x  for x in fz if "".join(x.split("/")[-4:-1])<=toDate]

    if (len(fz)<1):
        return
    fz.sort()
    colName = [x.strip() for x in colName]

    qDf = pd.concat([pd.read_csv(xfile,delimiter="\t",header=None)for xfile in fz])
    qDf.columns = colName

    return qDf

def loadMktData(ins,fromDate,toDate ):
    fromStamp = pd.to_datetime(fromDate,format="%Y%m%d")
    endStamp = pd.to_datetime(toDate,format="%Y%m%d")
    for dateStamp in pd.date_range(fromStamp,endStamp,freq="d"):

        cpData(dateStamp.strftime("%Y%m%d"),BondPath,toPath,[ins])
        cpData(dateStamp.strftime("%Y%m%d"),szStockPath,toPath,[ins])


    orderDf =  readByID(ins,fromDate,toDate,"order",toPath)
    transDf =  readByID(ins,fromDate,toDate,"trans",toPath)

    snapDf =  readByID(ins,fromDate,toDate,"snap",toPath)

    if orderDf is None:
        return None,None,None

    if transDf is None:
        return None,None,None
    if snapDf is None:
        return None,None,None

    snapDf["code"] = int(ins)
    transDf["flag2"].value_counts()
    transDf["flag1"].replace("0", 1, inplace=True)
    transDf["flag1"].replace("C", 0, inplace=True)
    transDf["flag2"].fillna(0, inplace=True)
    transDf["flag2"].replace("S", -1, inplace=True)
    transDf["flag2"].replace("B", 1, inplace=True)

    orderDf.insert(0, "dateType", 0)
    transDf.insert(0, "dateType",1)
    snapDf.insert(0, "dateType", 2)
    return orderDf,transDf,snapDf


# qDate = "20210604"
# ins,fromDate,toDate ="000725",qDate,qDate
# ins,fromDate,toDate ="128071",qDate,qDate
# 000725股票，128071可转债
# orderDf,transDf,snapDf = loadMktData(ins,fromDate,toDate)

def getMkData(orderDf,transDf,snapDf):

    orderQ = np.insert(orderDf.values[:,:8].astype("int64"), 0, 0, axis=1)
    if len(str(orderQ[0,2]))<10:
        orderQ[:, 2] =orderQ[:, 2]*1e6
    if len(str(orderQ[0,3]))<10:
        orderQ[:, 3] =orderQ[:, 3]*1e6


    transQ = np.insert(transDf.values[:,:10 ].astype("int64"), 0, 1, axis=1)
    if len(str(transQ[0,2]))<10:
        transQ[:, 2] =transQ[:, 2]*1e6
    if len(str(transQ[0,3]))<10:
        transQ[:, 3] =transQ[:, 3]*1e6



    snapQ =  np.insert(snapDf.values.astype("int64"), 0, 2, axis=1)
    if len(snapQ>1):
        if len(str(snapQ[0,2]))<10:
            snapQ[:, 2] = snapQ[:, 2] * 1e6
    #
    # snapLoc = 0
    # lastID = min(orderQ[-1, -1],transQ[-1,-3])
    # maxOrderLen = len(orderQ)
    # maxTransLen = len(transQ)
    # maxSnapLen = len(snapQ)
    # timeLine = 0
    dataDict = {}
    for i in range(len(orderQ)):
        dataDict[orderQ[i,-1]] = 0,i

    for i in range(len(transQ)):
        dataDict[transQ[i, -3]] = 1,i
    sortedKeys = sorted(dataDict.keys())

    for i in sortedKeys:
        dtype_i,dLoc_i = dataDict[i]
        if dtype_i==0:
            yield orderQ[dLoc_i]
        if  dtype_i==1:
            yield transQ[dLoc_i]
    # while sysID < lastID:
    #     if (orderLoc< maxOrderLen)and(transLoc<maxTransLen)and(orderQ[orderLoc, -1] < transQ[transLoc, -3]):
    #         yield  orderQ[orderLoc]
    #         timeLine =orderQ[orderLoc, 2]
    #         sysID = orderQ[orderLoc, -1]
    #         orderLoc += 1
    #     else:
    #         if transLoc<maxTransLen:
    #             yield  transQ[transLoc]
    #             timeLine = transQ[transLoc, 2]
    #             sysID = transQ[transLoc, -3]
    #             transLoc += 1

        # while ( snapLoc<maxSnapLen)and(snapQ[snapLoc,2]<timeLine):
        #     yield snapQ[snapLoc]
        #     snapLoc +=1
def getMkDataMult(dataStreamMap:dict,sortBy="idOrd"):
    # dataStreamMap = {ins: getMkData(*readH5Cache(ins, qDate)) for ins in insSet}
    dataDict = {}
    reformData = {}
    orderSortBy,transSortBy = 8,8
    if sortBy=="recvTime":
        orderSortBy,transSortBy = 3,3
    if sortBy=="BizIndex":
        orderSortBy, transSortBy = 10, 12
    snapLocD = {}
    maxSnapLenD = {}

    for key,datas in dataStreamMap.items():

        orderDf, transDf, snapDf = datas
        if len(orderDf)<1:
            continue
        if len(transDf)<1:
            return

        if len(snapDf)>0:
            snapDf = snapDf.sort_values("trdTime")
            snapDf = snapDf[snapCol]
        orderQ = np.insert(orderDf.values.astype("int64"), 0, 0, axis=1)
        if len(str(orderQ[0,2]))<10: #trdTime
            orderQ[:, 2] =orderQ[:, 2]*1e6
        if len(str(orderQ[0,3]))<10: #recvTime
            orderQ[:, 3] =orderQ[:, 3]*1e6


        transQ = np.insert(transDf.values.astype("int64"), 0, 1, axis=1)
        if len(str(transQ[0,2]))<10:
            transQ[:, 2] =transQ[:, 2]*1e6
        if len(str(transQ[0,3]))<10:
            transQ[:, 3] =transQ[:, 3]*1e6



        snapQ =  np.insert(snapDf.values.astype("int64"), 0, 2, axis=1)

        if len(snapQ>1):
            if len(str(snapQ[0,2]))<10:
                snapQ[:, 2] = snapQ[:, 2] * 1e6
            if len(str(snapQ[0, 3])) < 10:
                snapQ[:, 3] = snapQ[:, 3] * 1e6

        for i in range(len(orderQ)):
            dataDict[(orderQ[i,orderSortBy],key)] = key,0, i

        for i in range(len(transQ)):
            dataDict[(transQ[i,transSortBy],key)]= key,1, i

        reformData[key] = orderQ,transQ,snapQ
        if len(snapQ)>0:
            snapLocD[key] = 0
            maxSnapLenD[key] = len(snapQ)


    #

    # lastID = min(orderQ[-1, -1],transQ[-1,-3])
    # maxOrderLen = len(orderQ)
    # maxTransLen = len(transQ)
    # maxSnapLen = len(snapQ)
    # timeLine = 0
    sortedKeys = sorted(dataDict.keys())

    for i in sortedKeys:

        key,dtype_i,dLoc_i = dataDict[i]

        if dtype_i==0:
            data_i = reformData[key][0][dLoc_i]
        if  dtype_i==1:
            data_i = reformData[key][1][dLoc_i]
        snapLoc = snapLocD.get(key)
        if snapLoc is not None:
            snapQ = reformData[key][2]
            timeLine = data_i[3]
            # print(timeLine)
            maxSnapLen = maxSnapLenD[key]
            while ( snapLoc<maxSnapLen)and(snapQ[snapLoc,3]<timeLine):
                yield snapQ[snapLoc]
                snapLoc+=1
                snapLocD[key] +=1
        yield data_i


    # while sysID < lastID:
    #     if (orderLoc< maxOrderLen)and(transLoc<maxTransLen)and(orderQ[orderLoc, -1] < transQ[transLoc, -3]):
    #         yield  orderQ[orderLoc]
    #         timeLine =orderQ[orderLoc, 2]
    #         sysID = orderQ[orderLoc, -1]
    #         orderLoc += 1
    #     else:
    #         if transLoc<maxTransLen:
    #             yield  transQ[transLoc]
    #             timeLine = transQ[transLoc, 2]
    #             sysID = transQ[transLoc, -3]
    #             transLoc += 1

        # while ( snapLoc<maxSnapLen)and(snapQ[snapLoc,2]<timeLine):
        #     yield snapQ[snapLoc]
        #     snapLoc +=1


# struct TransactionDataPayload
# {
#     int transid;              // 成交编号
#     char type;                // 成交'0'，撤单'C'
#     OrderDirection directory; // 买单1，卖单-1，注意：撤单(type == 'C')情况下，directory == 0;且type == 'C' 是 directory == 0的充分不必要条件
#     float price;              // 成交价
#     int volume;               // 成交量
#     int askId;                // 卖方订单号
#     int bidId;                // 买方订单号
# };
#
# struct OrderDataPayload
# {
#     float price;              // 委托价格
#     int volume;               // 委托数量
#     OrderDirection directory; // 委托方向
#     OrderPriceType type;      // 委托类别
#     int orderId;              // 委托编号
# };
# orderDf.iloc[41]
# ["dateType",]
# snapDf.iloc[0]

class OrderData(object):
    def __init__(self,*datas):
        self.code,self.time,self.localtime,self.price,self.vol,self.bsFlag,self.type,self.orderID=datas
    def __repr__(self):
        return "code:%s\ntime:%s\nlocaltime:%s\nprice:%s\nvol:%s\nbsFlag:%s\ntype:%s\norderID:%s"%(self.code,self.time,self.localtime,self.price,self.vol,self.bsFlag,self.type,self.orderID)

class TransData(object):
    def __init__(self, *datas):
        self.code, \
        self.time, \
        self.localtime, \
        self.price, \
        self.vol, \
        self.bsFlag, \
        self.type, \
        self.transID ,\
        self.bidID,\
        self.askID= datas
    def __repr__(self):

        return "code:%s\ntime:%s\nlocaltime:%s\nprice:%s\nvol:%s\nbsFlag:%s\ntype:%s\ntransID:%s\nbidID:%s\naskID:%s" % (self.code, self.time, self.localtime, self.price, self.vol, self.bsFlag, self.type,self.transID ,self.bidID,self.askID)



class Snap(object):
    def __init__(self):
        pass
#
# transRc = TransData(*transQ[1,1:])
# orderRc = OrderData(*orderQ[1,1:])
# orderRc
# orderDf["type"].value_counts()

# dir(ask)
# if None:
#     print(2)
#
# ask = FastRBTree()
# ask.iter_items()
#
# for key,value in ask.iter_items(reverse=True):
#     print(key)
# dir(ask)
# ask[1] = set([20,30])
# ask[12] = set([208,30])
# dir(set)
# s1 = ask[1]
# s1.discard(20)
# ask.remove(1)
# ask.min_key()
# ask.min_key()
#
# dir(ask)
# ask = {}
# ask[1] =orderRc
# %time ask.pop(1)
# ask[1] =orderRc
# %time del ask[1]
# orderRc.vol -=10
#
# ask[1].vol
# set([204,30]).pop()
# set().add()
# z = ask.get(1)
# z.append(20)
# 1 in ask
# %timeit ask.get(1)
# dir(ask)
# ask.setdefault([])
# ask[1].append(20)
class OrderQue(object):
    def __init__(self,instrument):
        self.instrument = instrument
        self.priceCage = False
        if self.instrument[:3]=="300":
            self.priceCage = True
        self.maxOrderID = 0
        self.lossDataC = []

        self.start_of_day()
    def start_of_day(self):
        self.ask = FastRBTree()
        self.bid = FastRBTree()
        self.askVol = {}
        self.bidVol = {}
        self.orderSaver = {}
        self.mktTime = -1
        self.locTime = -1
        self.sysID = -1 #最大编号（transID,ordID）
        self.date = -1
        self.code = -1
        self.lastPrice = -1
        self.cumVol = 0
        self.cumAmount = np.int64(0)
        self.tradeCount = 0
        self.toTradeID = 0
        self.hangupCount = 0
        self.hangupOrderIDs = []
        self.hangupOver = []
        self.maxDataSize = 0
        self.endFlag = False
        self.lossDataC = []

    def mkTrade(self,orderID,trdQty ):
        orderRc = self.orderSaver.get(orderID)
        if orderRc is None:

            self.lossDataC.append([self.mktTime,orderID])
            return
        orderRc[4] -= trdQty
        bsFlag = orderRc[-3]
        orderPrice  = orderRc[3]
        type_i = orderRc[-2]
        if type_i==1:
            if bsFlag > 0:
                if orderPrice in self.bidVol:
                    self.bidVol[orderPrice] -= trdQty
            else:
                if orderPrice in self.askVol:
                    self.askVol[orderPrice] -= trdQty

        if orderRc[4] <0:
            print("trdQty more then left ",  self.instrument )
        if  orderRc[4]  ==0:
            self.removeOrder(orderID)

    def removeOrder(self,orderID):

        # if orderID == 162667:
        #     raise RuntimeError("remove %s" % orderID)
        if orderID in self.orderSaver:
            orderRc = self.orderSaver.pop(orderID)
            type_i = orderRc[-2]
            bsFlag = orderRc[-3]
            price =  orderRc[3]
            vol_i =  orderRc[4]
            if type_i==1:
                if  bsFlag == -1:
                    orderSet = self.ask.get(price)
                    if orderSet is not None:
                        orderSet.discard(orderID)
                        self.askVol[price] -= vol_i
                        if len(orderSet)<1:
                            self.ask.remove( price)
                        #print("remove price %s"%orderRc.price)
                else:
                    orderSet = self.bid.get(price)
                    if orderSet is not None:
                        orderSet.discard(orderID)
                        self.bidVol[price] -= vol_i
                        if len(orderSet)<1:
                            self.bid.remove( price)
                            #rint("remove price %s"%orderRc.price)
        else:
            self.lossDataC.append([self.mktTime,orderID])

    def matcher(self,transRc):
        transID = transRc[-3]
        time_i = transRc[1]
        locTime = transRc[2]
        type_i = transRc[6]
        askID = transRc[-1]
        bidID = transRc[-2]
        vol =  transRc[4]
        price = transRc[3]


        if type_i==0:
            orderID = max(askID,bidID)#单边行为
            self.removeOrder(orderID)
        else:
            self.lastPrice =  price
            self.cumAmount +=  price*vol
            self.tradeCount += 1
            self.cumVol +=vol
            self.mkTrade(askID,vol)
            self.mkTrade(bidID,vol)
            if (time_i >= 93000 * 1e9) and (self.ask.__len__()>0) and (self.bid.__len__()>0):

                aminP = self.ask.min_key()
                bmaxP = self.bid.max_key()
                if (askID<bidID)and(len(self.ask)>0)and(aminP<price):
                    z = list(self.ask[aminP])
                    zStr = ""
                    for orderId_i in z:
                        zStr = zStr+","+str(orderId_i)
                    # print("hang up ask %s %s %s "%(transID,aminP,zStr))
                    rmPrice = []
                    for askprice, orderIDset in self.ask.iter_items():
                        if askprice < price:
                            rmPrice.append(askprice)
                            self.askVol[askprice] = 0
                        else:
                            break
                    [self.ask.remove(askprice) for askprice in rmPrice]

                if (askID>bidID)and(len(self.bid)>0)and(bmaxP>price):
                    z = list(self.bid[bmaxP])
                    zStr = ""
                    for orderId_i in z:
                        zStr = zStr+","+str(orderId_i)
                    # print("hang up ask %s %s %s"%(transID,bmaxP,zStr))

                    rmPrice = []
                    for bidprice, orderIDset in self.bid.iter_items(reverse=True):
                        if bidprice > price:
                            rmPrice.append(bidprice)
                            self.bidVol[bidprice] = 0
                        else:
                            break
                    [self.bid.remove(bidprice) for bidprice in rmPrice]


            # if( self.ask.min_key()>self.bid.max_key()):
            if (time_i>93000*1e9)and(  time_i<145700*1e9 )and(self.ask.__len__()>0)and(self.bid.__len__()>0)and( self.ask.min_key()>self.bid.max_key()):
                self.endFlag = True
            #     if  self.toTradeID != max(askID, bidID):
            #         if self.orderSaver[self.toTradeID][-2]==1:
            #             self.hangupOrderIDs.append(self.toTradeID)
                # if self.sysID > max(askID, bidID):
            #         self.hangupCount += 1
            #         self.hangupOrderIDs.append(transID)
        self.sysID = max(self.sysID, transID)
        self.locTime = max(self.locTime, locTime)
        self.mktTime = max(self.mktTime, time_i)
        return self.endFlag


    def parser(self,data_i):
        if data_i[0]==0:
            return OrderData(*data_i[1:])
        if data_i[0] == 1:
            return TransData(*data_i[1:])
        pass

    def push_order(self,orderRc:np.array):
        # orderRc= orderQ[20,1:]
        orderID = orderRc[-1]
        time_i = orderRc[1]  #trdTime
        locTime = orderRc[2] #recvTime
        #snap_i[1] = self.locTime
        #snap_i[2] = self.mktTime
        type_i = orderRc[-2]
        bsFlag = orderRc[-3]
        price =  orderRc[3]
        vol_i = orderRc[4]
        # if type_i == 2:
        #     orderRc[-2] = 1
        #     if bsFlag == -1:
        #         orderRc[3] = self.ask.min_key()
        #     else:
        #         orderRc[3] = self.bid.max_key()

        self.sysID = max(self.sysID, orderID)
        self.orderSaver[orderID] = orderRc

        self.maxDataSize = max(self.maxDataSize,len(self.orderSaver))
        self.mktTime = max(self.mktTime, time_i)
        self.locTime = max(self.locTime, locTime)



        if (type_i== 1)   :

            if bsFlag == -1:
                if self.bid.__len__() > 0:
                    if (time_i > 93000*1e9) and (time_i < 145700*1e9) and (price < self.bid.max_key() * 0.98):
                        #这是为啥 价格笼子
                        return


                pRes = self.ask.get(price)

                if pRes is   None:
                    self.ask[price] = set([orderID])
                    self.askVol[price] = vol_i
                else:
                    pRes.add(orderID)
                    self.askVol[price] += vol_i
            else:
                if self.ask.__len__() > 0:
                    if  (time_i>93000*1e9)and(  time_i<145700*1e9) and ( price>   self.ask.min_key()*1.02):
                        return

                pRes = self.bid.get(price)
                if pRes is   None:
                    self.bid[price] = set([orderID])
                    self.bidVol[price] = vol_i
                else:
                    pRes.add(orderID)
                    self.bidVol[price] += vol_i





    def handle(self,data_i):
        self.endFlag = False

        if data_i[0] == 0:
            return  self.push_order(data_i[1:][:8])
        elif  data_i[0]==1:
            return self.matcher(data_i[1:][:10])
        else:
            pass

    def hang_up(self,orderID):
        pass


    def getSnap(self,pctRank=False,minIgnore=0):
        level = 10
        info_n = 7
        snap_i = np.array([0]*(level*4+7),dtype="int")
        snap_i[0] = self.sysID
        snap_i[1] = self.mktTime
        snap_i[2] = self.locTime
        snap_i[3] = self.lastPrice
        snap_i[4] = self.cumVol
        snap_i[5] = self.cumAmount
        snap_i[6] = self.tradeCount
        a_i = 0
        jumpR = 0.01
        askp1 = self.ask.min_key()
        bidp1 = self.bid.max_key()
        for price,orderIDset in self.ask.iter_items():
            if pctRank:
                a_i = int(np.floor((price/askp1-1)/jumpR))
                if a_i >= 5:
                    break


            vol = self.askVol[price]
            if vol<=minIgnore:
                continue
            if not pctRank:
                snap_i[info_n + level + a_i] = price
            else:
                snap_i[info_n + level + a_i] =  askp1*(1+jumpR*a_i)
            # for QorderID in orderIDset:
            #    vol += self.orderSaver[QorderID][4]
            snap_i[info_n+3*level+a_i] += vol

            if not pctRank:
                a_i += 1

            if a_i>=level:
                break
        b_i = 0
        for price,orderIDset in self.bid.iter_items(reverse=True):
            if pctRank :
                b_i = int(np.floor((1-price / bidp1  ) / jumpR))

                if b_i >= level:
                    break

            vol = self.bidVol[price]
            if vol <= minIgnore:
                continue
            if not pctRank:
                snap_i[info_n + level - b_i - 1] = price
            else:
                snap_i[info_n + level - b_i - 1] = bidp1 * (1 - jumpR * b_i)

            # for QorderID in orderIDset:
            #    vol += self.orderSaver[QorderID][4]
            snap_i[info_n+3*level - b_i-1] += vol
            if not pctRank:
                b_i += 1
            if b_i >= level:
                break
        return snap_i




    def getLoc(self,price,orderID):
        vol = 0
        if price in self.ask:
            for QorderID in self.ask[price]:
                if QorderID<orderID:
                    vol += self.orderSaver[QorderID][4]

        if price in self.bid:
            for QorderID in self.bid[price]:
                if QorderID<orderID:
                    vol += self.orderSaver[QorderID][4]

        return vol

class OrderReq(object):
    def __init__(self, price,vol,dirc,orderRef):
        self.price, self.vol, self.dirc, self.orderRef  = price,vol,dirc,orderRef
        self.tradePrice = 0
        self.createTime = -1
        self.insertTime = -1
        self.closeTime = -1
        self.OrderID =  np.int64(0)
        self.tradeType = -1
        self.tradeVol = 0
        self.tradeVolAcc = 0

        self.closeSysID = []
        self.status = 0
        self.predy = 0




class CancelReq(object):
    def __init__(self, orderRef):
        self.orderRef = orderRef
        self.createTime = 0
        self.insertTime = 0
        self.status = 0



class OrderQueSH(object):
    def __init__(self,instrument):
        self.instrument = instrument
        self.start_of_day()
        self.maxOrderID = 0
        self.lossDataC = []
    def start_of_day(self):
        self.ask = FastRBTree()
        self.bid = FastRBTree()
        self.orderSaver = {}
        self.mktTime = -1
        self.sysID = -1
        self.date = -1
        self.code = -1
        self.lastPrice = -1
        self.cumVol = 0
        self.cumAmount = 0
        self.tradeCount = 0
        self.toTradeID = 0
        self.hangupCount = 0
        self.hangupOrderIDs = []
        self.hangupOver = []
        self.maxDataSize = 0
        self.orderIDMax = 0
        self.gSnaps = []
        self.recvTime = -1
        self.endFlag = False
        self.bidVol = {}
        self.askVol = {}


        self.orderTransMap = {}

    def handle(self,data_i,seq=False):
        self.recvTime = data_i[3]
        self.endFlag = False
        if data_i[0] == 0:
            self.push_order(data_i[1:])
        elif data_i[0] == 1:
            self.matcher(data_i[1:])
        else:
            pass

    def push_order(self,orderRc:np.array):
        self.mktTime = max(orderRc[1],  self.mktTime )
        # self.orderIDMax = max(orderRc[9], self.orderIDMax )
        orderNo = orderRc[11]
        dirc = orderRc[5]
        ordPrc = orderRc[3]
        if orderRc[6] ==4:
            self.removeOrder(orderNo)

            return

        haveTrans =  self.orderTransMap.get(orderNo)
        if haveTrans is not None:
            for transRc_i in haveTrans:
                if transRc_i[11]>  orderRc[9]:
                    qty = transRc_i[4]
                    orderRc[4] -= qty
                    # if dirc>0:
                    #     if self.bidVol.get(ordPrc):
                    #         self.bidVol[ordPrc] -=qty
                    #     else:
                    #         raise RuntimeError("bidvol Trade error")
                    # else:
                    #     if self.askVol.get(ordPrc):
                    #         self.askVol[ordPrc] -= qty
                    #     else:
                    #         raise RuntimeError("askVol Trade error")

            del  self.orderTransMap[orderNo]
        if  orderRc[4]>0:
            self.orderSaver[orderNo] = orderRc


            if dirc>0:
                if self.bid.get(ordPrc):
                    self.bid[ordPrc].add(orderNo)
                    if self.bidVol.get(ordPrc):
                        self.bidVol[ordPrc]+=orderRc[4]
                    else:
                        raise RuntimeError("bidVol cache error")
                else:
                    self.bid[ordPrc] = set([orderNo])
                    self.bidVol[ordPrc] = orderRc[4]
            else:
                if self.ask.get(ordPrc):
                    self.ask[ordPrc].add(orderNo)
                    if self.askVol.get(ordPrc):
                        self.askVol[ordPrc] += orderRc[4]
                    else:
                        raise RuntimeError("askVol cache error")
                else:
                    self.ask[ordPrc] = set([orderNo])
                    self.askVol[ordPrc] =  orderRc[4]

    def mkTrade(self, orderID, trdQty):

        orderRc = self.orderSaver.get(orderID)
        dirc = orderRc[5]
        ordPrc = orderRc[3]
        if dirc > 0:
            try:
                self.bidVol[ordPrc] -= trdQty
            except Exception:
                raise RuntimeError("bidvol Trade error")
        else:
            try:
                self.askVol[ordPrc] -= trdQty
            except Exception:
                raise RuntimeError("askVol Trade error")
        orderRc[4] -= trdQty
        if orderRc[4] < 0:
            print("trdQty more then left ",  self.instrument )
        if orderRc[4] == 0:
            self.removeOrder(orderID)

    def removeOrder(self, orderID):
        if orderID not in self.orderSaver:
            return
        orderRc = self.orderSaver.pop(orderID)
        bsFlag = orderRc[5]
        price = orderRc[3]
        qty=  orderRc[4]



        if bsFlag == -1:
            orderSet = self.ask.get(price)
            if orderSet is not None:
                orderSet.discard(orderID)
                self.askVol[price] -= qty
                if len(orderSet) < 1:
                    self.ask.remove(price)
                    self.askVol.pop(price)
                    # print("remove price %s"%orderRc.price)
        else:
            orderSet = self.bid.get(price)
            if orderSet is not None:
                orderSet.discard(orderID)
                self.bidVol[price] -= qty
                if len(orderSet) < 1:
                    self.bid.remove(price)
                    self.bidVol.pop(price)
    def matcher(self,transRc:np.array):
        self.mktTime = max(transRc[1],  self.mktTime )
        trdPrc = transRc[3]
        askID = transRc[9]
        bidID = transRc[8]
        bizID = transRc[11]
        qty =  transRc[4]
        self.cumAmount += qty*trdPrc
        self.tradeCount += 1
        self.lastPrice = trdPrc
        self.cumVol += qty

        askIn = self.orderSaver.get(askID)
        bidIn = self.orderSaver.get(bidID)
        if askIn is not None:
            if askIn[9]< bizID:
                self.mkTrade(askID, qty)
        else:
            if  self.orderTransMap.get(askID):
                self.orderTransMap[askID].append(transRc)
            else:
                self.orderTransMap[askID] =[transRc]

        if bidIn  is not None:
            if bidIn[9]< bizID:
                self.mkTrade(bidID, qty)
        else:
            if  self.orderTransMap.get(bidID):
                self.orderTransMap[bidID].append(transRc)
            else:
                self.orderTransMap[bidID] = [transRc]

    def getSnap(self):
        level = 10
        info_n = 7
        snap_i = np.array([0]*(level*4+7),dtype="int")
        snap_i[0] = self.code
        snap_i[1] = self.recvTime
        snap_i[2] = self.mktTime
        snap_i[3] = self.lastPrice
        snap_i[4] = self.cumVol
        snap_i[5] = self.cumAmount
        snap_i[6] = self.tradeCount
        a_i = 0

        for price,orderIDset in self.ask.iter_items():
            snap_i[info_n+level+a_i] = price
            vol = 0
            # for QorderID in orderIDset:
            #    vol += self.orderSaver[QorderID][4]
            vol = self.askVol[price]
            snap_i[info_n+3*level+a_i] = vol
            a_i+=1
            if a_i>=level:
                break
        b_i = 0
        for price,orderIDset in self.bid.iter_items(reverse=True):
            snap_i[info_n + level - b_i-1] = price
            vol = 0
            # for QorderID in orderIDset:
            #    vol += self.orderSaver[QorderID][4]
            vol = self.bidVol[price]
            snap_i[info_n+3*level - b_i-1] = vol
            b_i += 1
            if b_i >= level:
                break
        return snap_i




    def getLoc(self,price,orderID):
        vol = 0
        if price in self.ask:
            for QorderID in self.ask[price]:
                if QorderID<orderID:
                    vol += self.orderSaver[QorderID][4]
            return vol
        if price in self.bid:
            for QorderID in self.bid[price]:
                if QorderID<orderID:
                    vol += self.orderSaver[QorderID][4]
            return vol

def getshMkData(orderDf,transDf,snapDf):
    needc = ["code", "recvTime", "trdTime", "mthPrc", "cumVol", "turnover", "cumCnt"]

    orderNp = orderDf.values
    transNp = transDf.values[:,:-1]
    mktSnapReformat = snapDf[needc].values
    mktSnapReformat = np.insert(mktSnapReformat,0,2,axis=1)
    snapLoc = 0
    maxSnapLen = len(mktSnapReformat)
    qCc =  pd.DataFrame(np.vstack([ np.insert(transNp,0,1,axis=1),  np.insert(orderNp,0,0,axis=1)])).sort_values(3).values
    for i in qCc:
        while (snapLoc<maxSnapLen) and( mktSnapReformat[snapLoc,3]*1e6<=i[3]):

            yield mktSnapReformat[snapLoc]
            snapLoc+=1

        # print(i)
        yield i
# orderDf.iloc[0]
# orderDf.shape
# transDf.iloc[0]
# transDf.shape
def getshMkDataSeq(orderDf,transDf,snapDf):
    # needc = ["code", "recvTime", "trdTime", "mthPrc", "cumVol", "turnover", "cumCnt"]

    orderNp = orderDf.values
    transNp = transDf.values[:, :-1]
    # mktSnapReformat = snapDf[needc].values
    # mktSnapReformat = np.insert(mktSnapReformat, 0, 2, axis=1)
    # snapLoc = 0
    # maxSnapLen = len(mktSnapReformat)
    transQ = np.insert(transNp, 0, 1, axis=1)
    orderQ = np.insert(orderNp, 0, 0, axis=1)
    dataDict ={}
    for i in range(len(orderQ)):
        dataDict[orderQ[i,-3]] = 0,i

    for i in range(len(transQ)):
        dataDict[transQ[i, -1]] = 1,i
    sortedKeys = sorted(dataDict.keys())
    for i in sortedKeys:
        dtype_i,dLoc_i = dataDict[i]
        if dtype_i==0:
            yield orderQ[dLoc_i]
        if  dtype_i==1:
            yield transQ[dLoc_i]
    # while sysID < lastID:

    pass
#zzDataTmp = "/home/aether/sampleproject_myBak/projects/gourdSample/zzDataTmp/"
zzDataTmp = "/home/tlei/data.zhongtai/"

def saveH5cache(dataT, dname,dayInt,prefix=None):
    dataT.sort_values("code", inplace=True)
    dataT.index = range(len(dataT))
    code_or = dataT["code"]
    #code_ofirst/code_olast反映了每个code所在的index
    code_ofirst = code_or.reset_index().groupby("code").first()
    code_olast = code_or.reset_index().groupby("code").last()
    if prefix is None:
        dataT.to_hdf(zzDataTmp + "%s_%s.h5" % (dayInt, dname), "df", mode="w", complevel=0, index=False)
        code_ofirst.to_hdf(zzDataTmp + "%s_%s.h5" % (dayInt, dname), "first",mode="a", complevel=0, index=False)
        code_olast.to_hdf(zzDataTmp + "%s_%s.h5" % (dayInt, dname), "last",mode="a", complevel=0, index=False)
    else:
        dataT.to_hdf(zzDataTmp + "%s_%s_%s.h5" % (dayInt,prefix, dname), "df", mode="w", complevel=0, index=False)
        code_ofirst.to_hdf(zzDataTmp + "%s_%s_%s.h5" % (dayInt,prefix, dname), "first", mode="a", complevel=0, index=False)
        code_olast.to_hdf(zzDataTmp + "%s_%s_%s.h5" % (dayInt,prefix, dname), "last", mode="a", complevel=0, index=False)

def multread_csv(fileList):
    from multiprocessing import Pool

    with  Pool(processes=50) as pool:

        mul_res = [pool.apply_async(pd.read_csv, (f_name_i,)) for f_name_i in fileList]
        rstList = [res.get() for res in mul_res]
    rstDf =  pd.concat(rstList)
    return rstDf

def getAllByDay(dayInt,prefix=None,partCodes=None,mainPath = "/mnt/lustre/data/raw/stock/md.data/zhongtai.data/",replace= False):
    '''
    处理原始数据的函数
    partCodes的含义：
    '''
    if prefix is  None:
        saveOrderPath = zzDataTmp+"%s_%s.h5"%(dayInt,"order")
        saveTransPath = zzDataTmp+"%s_%s.h5"%(dayInt,"trans")
        saveSnapPath = zzDataTmp+"%s_%s.h5"%(dayInt,"snap")
    else:
        saveOrderPath = zzDataTmp + "%s_%s_%s.h5" % (dayInt,prefix, "order")
        saveTransPath = zzDataTmp + "%s_%s_%s.h5" % (dayInt,prefix, "trans")
        saveSnapPath = zzDataTmp + "%s_%s_%s.h5" % (dayInt,prefix, "snap")

    import glob
    orderPath = mainPath + "order.stock"
    transPath = mainPath + "trans.stock"
    snapPath = mainPath + "md.stock"
    bondPath = mainPath + "md.bond*"
    #为啥是"md.bond*"而不是"md.bond"
    orderFiles = glob.glob(os.path.join(orderPath, dayInt, "*.*"))
    transFiles = glob.glob(os.path.join(transPath, dayInt, "*.*"))
    snapFiles = glob.glob(os.path.join(snapPath, dayInt, "*.*"))
    bondFiles = glob.glob(os.path.join(bondPath, dayInt, "*.*"))
    if replace or (not os.path.exists(saveOrderPath)):
        ordersByDay = dskf.read_csv(sorted(orderFiles),dtype={'BizIndex': 'int64',
           'bsFlag': 'int64',
           'channel': 'int64',
           'idOrd': 'int64',
           'ordPrc': 'int64',
           'ordQty': 'int64',
           'orderNo': 'int64',
           'seq': 'int64',
           'type': 'int64'}).compute()
        if partCodes is not None:
            saveH5cache(ordersByDay.query("code in [%s]"%",".join(partCodes)), "order", dayInt,prefix)
        else:
            saveH5cache(ordersByDay, "order", dayInt,prefix)


    if replace or (not os.path.exists(saveTransPath)):
        transByDay = dskf.read_csv(sorted(transFiles),dtype={'BizIndex': 'float64',
       'Channel': 'float64',
       'bsFlag': 'float64',
       'idOrdBuy': 'float64',
       'idOrdSell': 'float64',
       'idTrans': 'float64',
       'seq': 'float64',
       'trdPrc': 'float64',
       'trdQty': 'float64',
       'type': 'float64'}).compute()
        if partCodes is not None:
            saveH5cache(transByDay.query("code in [%s]" % ",".join(partCodes)), "trans", dayInt,prefix)
        else:
            saveH5cache(transByDay, "trans", dayInt,prefix)
    if replace or (not os.path.exists(saveSnapPath)):
        snapByDay = dskf.read_csv(sorted(snapFiles)).compute()
        snapByDayBond = dskf.read_csv(sorted(bondFiles)).compute()
        snapByDay = pd.concat([snapByDay,snapByDayBond])
        if partCodes is not None:
            saveH5cache(snapByDay.query("code in [%s]" % ",".join(partCodes)), "snap", dayInt, prefix)
        else:

            saveH5cache(snapByDay, "snap", dayInt,prefix)
    return


def getAllByDayTest(dayInt,prefix=None,partCodes=None,mainPath = "/mnt/lustre/data/raw/stock/md.data/zhongtai.data/",replace= False):
    if prefix is  None:
        saveOrderPath = zzDataTmp+"%s_%s.h5"%(dayInt,"order")
        saveTransPath = zzDataTmp+"%s_%s.h5"%(dayInt,"trans")
        saveSnapPath = zzDataTmp+"%s_%s.h5"%(dayInt,"snap")
    else:
        saveOrderPath = zzDataTmp + "%s_%s_%s.h5" % (dayInt,prefix, "order")
        saveTransPath = zzDataTmp + "%s_%s_%s.h5" % (dayInt,prefix, "trans")
        saveSnapPath = zzDataTmp + "%s_%s_%s.h5" % (dayInt,prefix, "snap")

    import glob
    orderPath = mainPath + "order.stock"
    transPath = mainPath + "trans.stock"
    snapPath = mainPath + "md.stock"
    bondPath = mainPath+"md.bond*"
    #
    orderFiles = glob.glob(os.path.join(orderPath, dayInt, "*.*"))
    transFiles = glob.glob(os.path.join(transPath, dayInt, "*.*"))
    snapFiles = glob.glob(os.path.join(snapPath, dayInt, "*.*"))
    bondFiles = glob.glob(os.path.join(bondPath, dayInt, "*.*"))
    # totalset = list(set(orderFiles+transFiles+snapFiles+bondFiles))
    # %time zf = pd.read_csv(totalset[0],index_col=["code","recvTime"])
    # %time zf =  pd.read_csv(totalset[0])
    # %time zf.loc[1018083]
    # zf.index
    # from multiprocessing import Pool
    # with  Pool(processes=50) as pool:
    #     # mul_res = [pool.apply_async(myBKone,(qDate,stk_i,)) for stk_i in sz_stocks ]
    #     mul_res = [pool.apply_async( pd.read_csv, (fname_i,) )for fname_i in totalset]
    #     rstList = [res.get() for res in mul_res]

    orderfList = []
    # for i in range(len(totalset)):
    #     if "order" in totalset[i]:
    #         orderfList.append(rstList[i])

    orderDf = pd.concat(orderfList)
    dataT = orderDf

    dataT.set_index(["code","recvTime"])
    # %time dataT.sort_values("code", inplace=True)
    dataT.index = range(len(dataT))
    code_or = dataT["code"]
    code_ofirst = code_or.reset_index().groupby("code").first()
    code_olast = code_or.reset_index().groupby("code").last()

    import psutil

    if replace or (not os.path.exists(saveOrderPath)):
        ordersByDay = dskf.read_csv(sorted(orderFiles),dtype={'BizIndex': 'int64',
           'bsFlag': 'int64',
           'channel': 'int64',
           'idOrd': 'int64',
           'ordPrc': 'int64',
           'ordQty': 'int64',
           'orderNo': 'int64',
           'seq': 'int64',
           'type': 'int64'}).compute()
        if partCodes is not None:
            saveH5cache(ordersByDay.query("code in [%s]"%",".join(partCodes)), "order", dayInt,prefix)
        else:
            saveH5cache(ordersByDay, "order", dayInt,prefix)


    if replace or (not os.path.exists(saveTransPath)):
        transByDay = dskf.read_csv(sorted(transFiles),dtype={'BizIndex': 'float64',
       'Channel': 'float64',
       'bsFlag': 'float64',
       'idOrdBuy': 'float64',
       'idOrdSell': 'float64',
       'idTrans': 'float64',
       'seq': 'float64',
       'trdPrc': 'float64',
       'trdQty': 'float64',
       'type': 'float64'}).compute()
        if partCodes is not None:
            saveH5cache(transByDay.query("code in [%s]" % ",".join(partCodes)), "trans", dayInt,prefix)
        else:
            saveH5cache(transByDay, "trans", dayInt,prefix)
    if replace or (not os.path.exists(saveSnapPath)):
        snapByDay = dskf.read_csv(sorted(snapFiles)).compute()
        snapByDayBond = dskf.read_csv(sorted(bondFiles)).compute()
        snapByDay = pd.concat([snapByDay,snapByDayBond])
        if partCodes is not None:
            saveH5cache(snapByDay.query("code in [%s]" % ",".join(partCodes)), "snap", dayInt, prefix)
        else:

            saveH5cache(snapByDay, "snap", dayInt,prefix)
    return



# prefix="zz"
# partCodes = ["1123030","1123047"]
#
# getAllByDay(dayInt,prefix,partCodes,"/mnt/lustre/data/raw/stock/md.data/caitong.data/",False)

def readH5CacheByName(ins,dayInt,dname,prefix=None):
    ins7 =str(ins)
    if len(str(ins))<7:
        ins7 = "1"+str(ins)
    refName = zzDataTmp+"%s_%s.h5"%(dayInt,dname)
    if prefix is not None:
        refName = zzDataTmp + "%s_%s_%s.h5" % (dayInt,prefix, dname)
    if not os.path.exists(refName):
        return pd.DataFrame()
    dFisrt = pd.read_hdf(refName,"first").iloc[:,0]
    dlast = pd.read_hdf(refName,"last").iloc[:,0]
    startRow = dFisrt.get(int(ins7))
    endRow = dlast.get(int(ins7))
    #和直接写成dFisrt.loc[int(ins7)]的区别在哪
    dDf = pd.DataFrame()
    if (startRow is not None) and(endRow is not None):
        dDf =  pd.read_hdf(refName,"df",start=startRow,stop=endRow+1)
    return dDf

def readH5Cache(ins,dayInt,prefix=None):
    orderDf  = readH5CacheByName(ins, dayInt, "order",prefix)
    transDf  = readH5CacheByName(ins, dayInt, "trans",prefix)
    snapDf  = readH5CacheByName(ins, dayInt, "snap",prefix)
    return orderDf,transDf,snapDf

def codeExchange(scode):
    if ((int(str(scode)[-6]) == 6) or (int(str(scode)[-6:-4]) == 11)):
        return "SH"
    return "SZ"


class DataController(object):
    def __init__(self,ins,refinsList, qDate,prefix=None):
        #ins和refinsList的含义：一个交易的对象，一个参考对象
        self.ins = str(ins)[-6:]
        self.fromDate = fromDate
        self.toDate = toDate
        self.trdingDate = fromDate
        self.pnlSeries = []
        self.positionSeries = []
        self.closePnl = []
        self.holdPnl = []
        self.timeList = []
        self.position = 0
        self.cpnl = 0
        self.pnl = 0
        self.bestPrice = {}
        self.amt = 0
        self.qDate = qDate
        self.lastPrice = 0

        self.orderCount = 0
        insSet = set([ins]+refinsList)
        self.dataStreamMap = {ins: readH5Cache(ins, qDate,prefix) for ins in insSet}
        self.orderStr = np.zeros((len(self.dataStreamMap[ins][1])*2, 12)) * np.nan
        # self.orderDf,self.transDf,self.snapDf = loadMktData(ins,fromDate,toDate)
        # self.orderDf,self.transDf,self.snapDf =orderDf,transDf,snapDf
        # self.orderQueRef =  {1000000 + int(str(insRef)[-6:]):  OrderQue(insRef) for insRef in refinsList if ((int(str(insRef)[-6])==6) or (int(str(insRef)[-6:-4])==11 )) else  OrderQue(insRef) }
        self.orderQueRef = {}
        for insRef in refinsList :
            n7int = 1000000 + int(str(insRef)[-6:])

            if codeExchange(insRef) == "SZ":
                self.orderQueRef[n7int] = OrderQue(insRef)
            else:
                self.orderQueRef[n7int] = OrderQueSH(insRef)
        if  codeExchange(ins) == "SZ":
            self.orderQue  = OrderQue(ins)
        else:
            self.orderQue  = OrderQueSH(ins)
        self.strOrderQueSh  = OrderQueSH(ins)

        self.waitOrderList = {}
        self.sysID =  np.int64(0)
        self.sysIDQ = []
        self.cancelReqList = []
        self.timeDelay = 2000000
        self.mktTime = -1
        self.localTime = -1
        self.doneRc ={}
        self.exchangeID = "SZ"
        self.maxOrderID = 0

    def react(self ):
        if(self.orderQue.ask.__len__()<1) or(self.orderQue.bid.__len__()<1) :
            return
        values  = list(self.waitOrderList.values())
        for orderReq  in values:
            ## 拿到的是orderq的引用
            self.actOrder(orderReq)

        self.cancelReqList =[ self.actCancel(cancelReq) for cancelReq in self.cancelReqList ]
        self.cancelReqList = [x for x in self.cancelReqList if x is not None]

    def actCancel(self,cancelReq: CancelReq):
        if cancelReq.createTime <0:
            cancelReq.createTime = self.mktTime
            cancelReq.insertTime = self.mktTime +100e6
            return cancelReq
        else:
            if cancelReq.insertTime < self.mktTime:
                orderReq_i:OrderReq
                orderReq_i =  self.waitOrderList.get(cancelReq.orderRef)
                if orderReq_i is not None:
                    if orderReq_i.status == "INSERTED":
                        orderReq_i.status="CANCELED"
                        orderReq_i.closeTime = self.mktTime
                    elif orderReq_i.status == "TRADED":
                        orderReq_i.status = "CANCELFAIL"
                    else:
                        print("yanhou")
                        return cancelReq

                    self.player.handle_order(copy.copy(orderReq_i))
                    self.doneRc[orderReq_i.orderRef] = orderReq_i
                    del self.waitOrderList[orderReq_i.orderRef]
                else:
                    orderReq_i = self.doneRc.get(cancelReq.orderRef)
                    if orderReq_i is not None:
                        if orderReq_i.status == "INSERTED":
                            orderReq_i.status = "CANCELED"
                            orderReq_i.closeTime =  self.mktTime
                            self.player.handle_order(copy.copy(orderReq_i))
                        elif orderReq_i.status == "TRADED":
                            orderReq_i.status = "CANCELFAIL"
                            self.player.handle_order(copy.copy(orderReq_i))



            else:
                return cancelReq
    def actOrder(self,orderReq:OrderReq):
        """
        为了处理 策略对手价 以及 之前挂单，后来有mktOrder 的价格合适
        :param orderReq:
        :return:
        """
        if orderReq.createTime < 0:
            orderReq.createTime = self.mktTime
            orderReq.insertTime = self.mktTime + self.timeDelay
            orderReq.createTime = self.mktTime
            orderReq.insertTime = self.mktTime + self.timeDelay

        if self.exchangeID == "SZ":
            if orderReq.insertTime>self.mktTime:
                orderReq.OrderID = self.maxOrderID#todo : check完要修正赋值方式
                orderReq.status = "INSERTED"
                self.player.handle_order(copy.copy(orderReq))
        else:
            if orderReq.insertTime<self.mktTime:
                orderReq.OrderID = self.maxOrderID#todo : check完要修正赋值方式
                orderReq.status = "INSERTED"
                self.player.handle_order(copy.copy(orderReq))

        minAskPrice = self.orderQue.ask.min_key()
        maxBidPrice = self.orderQue.bid.max_key()
        if maxBidPrice<minAskPrice:
            cpPrice = minAskPrice  if orderReq.dirc>0 else maxBidPrice
            if(((orderReq.price -cpPrice)*orderReq.dirc)>=0)and( orderReq.status == "INSERTED") :
                if orderReq.dirc>0:
                    for price, orderIDset in self.orderQue.ask.iter_items():
                        if price>orderReq.price:
                           break
                        mktVol = self.orderQue.askVol[price]
                        oTrdPrice = price
                        if orderReq.OrderID < self.maxOrderID:
                            oTrdPrice = orderReq.price


                        oTrdvol = max(min(mktVol, orderReq.vol - orderReq.tradeVolAcc), 0)
                        ctime = self.mktTime
                        self.oMKTrade(oTrdPrice, oTrdvol, ctime, 0, orderReq)
                        if  orderReq.status == "TRADED":
                            break
                else:
                    for price, orderIDset in  self.orderQue.bid.iter_items(reverse=True):
                        if price <orderReq.price:
                            break
                        mktVol = self.orderQue.bidVol[price]
                        oTrdPrice = price
                        if orderReq.OrderID < self.maxOrderID:
                            oTrdPrice = orderReq.price
                        oTrdvol = max(min(mktVol, orderReq.vol - orderReq.tradeVolAcc), 0)
                        ctime = self.mktTime
                        self.oMKTrade(oTrdPrice, oTrdvol, ctime, 0, orderReq)
                        if  orderReq.status == "TRADED":
                            break
    def mkTrade(self,price,closeTime,):
            pass
    def reactTrans(self,data_i):
        """
        为了check过价成交，以及排队位置合适
        :param data_i:
        :return:
        """
        if(self.orderQue.ask.__len__()<1) or(self.orderQue.bid.__len__()<1) :
            return

        if data_i[0]!=1:
            return
        mktVol = data_i[5]
        dmktTime = data_i[2]
        transPrice = data_i[4]
        tBidID =  data_i[9]
        tAskID = data_i[10]
        # transRc = TransData(*transQ[1,1:])
        # orderRc = OrderData(*orderQ[1,1:])
        orderReq: OrderReq
        keys = list(self.waitOrderList.keys())
        for key  in keys:
            ## 拿到的是orderq的引用
            orderReq=  self.waitOrderList.get(key)
            if (orderReq.insertTime<dmktTime)and( orderReq.status == "INSERTED") :
                price = transPrice
                if  (( orderReq.dirc == -1)and((price > orderReq.price) or (( price==orderReq.price) and( orderReq.OrderID  < tAskID)))) \
                        or((orderReq.dirc == 1) and((price<orderReq.price) or (( price==orderReq.price) and( orderReq.OrderID <tBidID)))):
                    oTrdPrice = orderReq.price
                    oTrdvol = max(min(mktVol, orderReq.vol - orderReq.tradeVolAcc), 0)
                    ctime =  dmktTime
                    self.oMKTrade(oTrdPrice,oTrdvol,ctime,1,orderReq)
                else:
                    cCnt = self.orderCount
                    self.pnl = self.cpnl + self.position * self.lastPrice
                    self.orderStr[cCnt, 0] = self.pnl / 1e4  # pnl
                    self.orderStr[cCnt, 1] = self.position  # position
                    self.orderStr[cCnt, 2] = self.mktTime # newesttime
                    self.orderStr[cCnt, 3] = orderReq.orderRef  # orderRef
                    self.orderStr[cCnt, 4] = orderReq.OrderID  # OrderID
                    self.orderStr[cCnt, 5] = orderReq.price  # price
                    self.orderStr[cCnt, 6] = self.sysID  # sysID
                    self.orderStr[cCnt, 7] = np.nan # tradePrice
                    self.orderStr[cCnt, 8] = orderReq.dirc  # dirc
                    self.orderStr[cCnt, 9] = 0  # tradeVol
                    self.orderStr[cCnt, 10] = orderReq.insertTime  # tradeVol
                    self.orderStr[cCnt, 11] = orderReq.createTime  # tradeVol
                    self.orderCount += 1


    def oMKTrade(self,oTrdprice,oTrdvol,ctime,trdType,orderReq: OrderReq):
        orderReq.tradePrice = oTrdprice
        orderReq.closeTime = ctime
        orderReq.tradeVol = oTrdvol
        orderReq.tradeVolAcc += orderReq.tradeVol
        orderReq.tradeType = trdType
        orderReq.closeSysID.append(self.sysID)

        self.position += orderReq.dirc * orderReq.tradeVol
        self.cpnl -= orderReq.dirc * orderReq.tradeVol * orderReq.tradePrice
        self.pnl = self.cpnl + self.position * orderReq.tradePrice
        cCnt = self.orderCount
        self.orderStr[cCnt,0] = self.pnl / 1e4 # pnl
        self.orderStr[cCnt,1] = self.position # position
        self.orderStr[cCnt,2] = orderReq.closeTime # closeTime
        self.orderStr[cCnt,3] = orderReq.orderRef # orderRef
        self.orderStr[cCnt,4] = orderReq.OrderID # OrderID
        self.orderStr[cCnt,5] = orderReq.price # price
        self.orderStr[cCnt,6] = self.sysID # sysID
        self.orderStr[cCnt,7] = orderReq.tradePrice # tradePrice
        self.orderStr[cCnt,8] = orderReq.dirc # dirc
        self.orderStr[cCnt,9] = orderReq.tradeVol # tradeVol
        self.orderStr[cCnt,10] = orderReq.insertTime # tradeVol
        self.orderStr[cCnt,11] = orderReq.createTime # tradeVol
        self.orderCount+=1

        # self.pnlSeries.append(self.pnl / 1e4)
        # self.positionSeries.append(self.position)
        # self.timeList.append(orderReq.closeTime)
        #
        # self.refList.append(orderReq.orderRef)
        # self.orderIDList.append(orderReq.OrderID)
        # self.oPriceList.append(orderReq.price)
        # self.transIDList.append(self.sysID)
        #
        # self.trdpList.append( orderReq.tradePrice)
        # self.dircList.append(orderReq.dirc)
        # self.trdVList.append(orderReq.tradeVol)

        if (orderReq.vol - orderReq.tradeVolAcc) <= 0:
            orderReq.status = "TRADED"
            self.doneRc[orderReq.orderRef] = orderReq
            del self.waitOrderList[orderReq.orderRef]
        self.player.handle_order(copy.copy(orderReq))
        # 为下次成交重置成交量
        orderReq.tradeVol = 0
    def playSHlx(self):
        transCache = []
        orderCache = []

        for data_i_m in getMkDataMult(self.dataStreamMap,"BizIndex"):
            if int(data_i_m[1]) == 1000000 + int(self.ins[-6:]):
                data_i = data_i_m
                if (data_i[0] != 2):
                    if self.mktTime != data_i[2]:
                        self.react()
                        self.mktTime = data_i[2]
                self.orderQue.handle(data_i.copy())
                if data_i[0] == 0:
                    orderCache.append(data_i)
                    self.maxOrderID = data_i[12]
                if (data_i[0] == 1) :
                    transCache.append(data_i)
                    self.tradeCount += 1
                    self.reactTrans(data_i)
                strRc = data_i.copy()
                self.strOrderQueSh.handle(strRc)
                orderReq, cancelReq = self.player.handle_data(strRc, self.strOrderQueSh, self.orderQueRef)
                self.waitOrderList.update(orderReq)
                self.cancelReqList.extend(cancelReq)


                # while (len(transCache) > 0) and (len(orderCache) > 0) and (transCache[0][3] < orderCache[0][3]):
                #     strRc = transCache.pop(0).copy()
                #     self.strOrderQueSh.handle(strRc)
                #     orderReq, cancelReq = self.player.handle_data(strRc, self.strOrderQueSh, self.orderQueRef)
                #     self.waitOrderList.update(orderReq)
                #     self.cancelReqList.extend(cancelReq)
                # while (len(orderCache) > 0) and (len(transCache) > 0) and (orderCache[0][3] < transCache[0][3]):
                #     strRc = orderCache.pop(0)
                #     self.strOrderQueSh.handle(strRc.copy())
                #     orderReq, cancelReq = self.player.handle_data(strRc, self.strOrderQueSh, self.orderQueRef)
                #     self.waitOrderList.update(orderReq)
                #     self.cancelReqList.extend(cancelReq)


            else:

                self.orderQueRef[int(data_i_m[1])].handle(data_i_m)
    def playSH(self):
        transCache = []
        orderCache = []

        for data_i_m in getMkDataMult(self.dataStreamMap,"BizIndex"):
            if int(data_i_m[1]) == 1000000 + int(self.ins[-6:]):
                data_i = data_i_m
                if (data_i[0] != 2):
                    if self.mktTime != data_i[2]:
                        self.react()
                        self.mktTime = data_i[2]
                self.orderQue.handle(data_i.copy())
                if data_i[0] == 0:
                    orderCache.append(data_i)
                    self.maxOrderID = data_i[12]
                if (data_i[0] == 1) :
                    transCache.append(data_i)
                    self.tradeCount += 1
                    self.reactTrans(data_i)
                # strRc = data_i.copy()
                # self.strOrderQueSh.handle(strRc)
                if (data_i[0] == 2):
                    strRc = data_i.copy()
                    orderReq, cancelReq = self.player.handle_data(strRc, self.strOrderQueSh, self.orderQueRef)
                    self.waitOrderList.update(orderReq)
                    self.cancelReqList.extend(cancelReq)


                while (len(transCache) > 0) and (len(orderCache) > 0) and (transCache[0][3] < orderCache[0][3]):
                    strRc = transCache.pop(0).copy()
                    self.strOrderQueSh.handle(strRc)
                    orderReq, cancelReq = self.player.handle_data(strRc, self.strOrderQueSh, self.orderQueRef)
                    self.waitOrderList.update(orderReq)
                    self.cancelReqList.extend(cancelReq)
                while (len(orderCache) > 0) and (len(transCache) > 0) and (orderCache[0][3] < transCache[0][3]):
                    strRc = orderCache.pop(0)
                    self.strOrderQueSh.handle(strRc.copy())
                    orderReq, cancelReq = self.player.handle_data(strRc, self.strOrderQueSh, self.orderQueRef)
                    self.waitOrderList.update(orderReq)
                    self.cancelReqList.extend(cancelReq)


            else:

                self.orderQueRef[int(data_i_m[1])].handle(data_i_m)
    def playSZ(self):
        self.orderSet = []
        self.transSet = []
        for data_i_m in getMkDataMult(self.dataStreamMap):
            #单只股票按照按照Ordid排序，多只股票按照recvTime排序
            if int(data_i_m[1]) == 1000000 + int(self.ins[-6:]):
                data_i = data_i_m
                self.localTime = data_i[3]
                if (data_i[0] != 2):
                    if self.mktTime != data_i[2]:
                        self.react()
                        self.mktTime = data_i[2]
                self.orderQue.handle(data_i)
                if (data_i[0]==0):
                    self.maxOrderID = data_i[8]
                    self.orderSet.append(data_i)

                if (data_i[0] != 2):
                    sysID = data_i[8]
                    if sysID > self.sysID:
                        self.sysID = sysID

                if (data_i[0] == 1) and (data_i[7] == 1):
                    self.tradeCount += 1
                    self.lastPrice = data_i[4]
                    self.reactTrans(data_i)
                    self.transSet.append(data_i)
                orderReq, cancelReq = self.player.handle_data(data_i, self.orderQue, self.orderQueRef)
                self.waitOrderList.update(orderReq)
                self.cancelReqList.extend(cancelReq)
            else:
                self.orderQueRef[int(data_i_m[1])].handle(data_i_m)

    def play(self,ply ):
        # if self.orderDf is None:
        #     return
        # if self.snapDf is None:
        #     return
        # if self.transDf is None:
        #     return

        # dates = sorted(list(set(self.orderDf.date)))
        self.player = ply
        self.tradeCount = 1
        ins = str(self.ins)
        if (ins[-6:-3] == "123") or (ins[-6:-3] == "127") or (ins[-6:-3] == "128"):
            self.player.ifzz = True
        # for qDate in dates:

        self.mktTime = -1
        self.localTime = -1
        self.orderQue.start_of_day()
        if len(self.orderQueRef)>0:
            [self.orderQueRef[insRef].start_of_day() for insRef in self.orderQueRef.keys()]
        if (str(ins).zfill(6)[-6:][0] == "6") or(str(ins).zfill(6)[-6:][:3] == "113"):

            self.exchangeID = "SH"
            self.timeDelay = 200000000
            self.playSH()

        else:
            self.playSZ()


def getMys(rst):
    rst =  rst.sort_values(0)
    qtr = rst[0].rank(pct=True)
    lowerI = np.array([0,0.0005,0.005,0.05,0.5])
    upperI =  1-lowerI
    sd = {}
    for i in range(1,len(lowerI-1)):
        sd["up%s_%s"%(upperI[i],upperI[i-1])] = rst[(qtr<upperI[i-1])&(qtr>=upperI[i])].mean()
        sd["low%s_%s"%(lowerI[i-1],lowerI[i])] =rst[(qtr>lowerI[i-1])&(qtr<=lowerI[i])].mean()
    for i in range(1,len(lowerI-1)):
        sd["up%s_%s"%(upperI[i],upperI[i-1])][">0"] = (rst[(qtr<upperI[i-1])&(qtr>=upperI[i])]>0)[1].mean()
        sd["low%s_%s"%(lowerI[i-1],lowerI[i])][">0"] =(rst[(qtr>lowerI[i-1])&(qtr<=lowerI[i])]>0)[1].mean()
    for i in range(1, len(lowerI - 1)):
        sd["up%s_%s" % (upperI[i], upperI[i - 1])]["<0"] = (rst[(qtr < upperI[i - 1]) & (qtr >= upperI[i])] < 0)[
            1].mean()
        sd["low%s_%s" % (lowerI[i - 1], lowerI[i])]["<0"] = (rst[(qtr > lowerI[i - 1]) & (qtr <= lowerI[i])] < 0)[
            1].mean()
    for i in range(1, len(lowerI - 1)):
        sd["up%s_%s" % (upperI[i], upperI[i - 1])]["corr"] = (rst[(qtr < upperI[i - 1]) & (qtr >= upperI[i])] ).corr()[0][1]
        sd["low%s_%s" % (lowerI[i - 1], lowerI[i])]["corr"] = (rst[(qtr > lowerI[i - 1]) & (qtr <= lowerI[i])]  ).corr()[0][1]


    sdf = pd.DataFrame(sd).T.sort_values(0)
    print("total corr:",rst.corr())
    print("self corr:",rst.sort_index()[0].autocorr(1))
    print(sdf)
    return sdf

#rtio = 3
#tillf = 16
def getRst(bkStream):
    feeRatio = 0.00126
    if bkStream.player.ifzz == True:
        feeRatio = 0.00012
    if len(bkStream.doneRc)<1:
        return None,None,None
    ss = {}


    for key, order_i in bkStream.doneRc.items():
        # ss[key] = [order_i.status, order_i.closeTime, order_i.insertTime, order_i.dirc, order_i.price / 1e4,
                   # order_i.tradePrice / 1e4, order_i.vol, order_i.predy, np.int32(order_i.OrderID),  np.int32(order_i.closeSysID)]
        ss[key] = [order_i.status, order_i.closeTime, order_i.insertTime,order_i.createTime, order_i.dirc, order_i.price / 1e4,
                   order_i.tradePrice / 1e4, order_i.tradeVolAcc, order_i.predy, np.int64(order_i.OrderID),  np.int64(order_i.closeSysID),order_i.tradeType ]
    for key, order_i in bkStream.waitOrderList.items():
        ss[key] = [order_i.status, order_i.closeTime, order_i.insertTime,order_i.createTime, order_i.dirc, order_i.price / 1e4,
                   order_i.tradePrice / 1e4, order_i.tradeVolAcc, order_i.predy, np.int64(order_i.OrderID),  np.int64(order_i.closeSysID),order_i.tradeType ]

    ssDf = pd.DataFrame(ss).T



    zff = ssDf[ssDf[6] !=0]
    if len(zff) < 1:
        return None, None, None
    bkStat = {}

    apnl = pd.DataFrame(bkStream.orderStr).dropna(how="all")

    apnl.columns = ["pnl", "netPosition", "closeTime", "orderRef", "OrderID", "oprice","sysID","trdPrice","dirc","trdVol","insertTime","createTime"]

    fee = (apnl["trdPrice"] * apnl["trdVol"] / 1e4).sum() * feeRatio / 2
    earing = apnl["pnl"].iloc[-1]
    maxRetrace = (apnl["pnl"] - apnl["pnl"].expanding().max()).min()

    position = apnl["netPosition"]
    bkStat["profit"] = earing
    bkStat["tradeAmount"] = (apnl["trdPrice"] * apnl["trdVol"] / 1e4).sum() /2
    bkStat["fee"] = fee
    bkStat["pvsf"] = bkStat["profit"] / bkStat["fee"]
    bkStat["netP"] = bkStat["profit"] - bkStat["fee"]
    bkStat["netP_bp"] = (bkStat["profit"] - bkStat["fee"]) / bkStat["tradeAmount"] * 10000
    bkStat["tradeRound"] = apnl["trdVol"].sum()/bkStream.player.baseVol/2
    bkStat["longMax"] = position.max()
    bkStat["shortMax"] = position.min()
    bkStat["closeNetPosition"] = position.iloc[-1]
    bkStat["meanPrice"] =apnl["trdPrice"].replace(0, np.nan).mean()/1e4
    bkStat["maxRetrace"] =maxRetrace
    pdQ = pd.DataFrame(np.vstack(bkStream.player.predAndQ)).iloc[20:]



    pj =160
    pdQ["yd0"] = pdQ.loc[:, 3].pct_change(pj).shift(-(pj +1))
    pdQ["yd1"] = pdQ.loc[:, 4].pct_change(pj).shift(-(pj+1))
    pdQ["yd2"] = pdQ.loc[:, 5].pct_change(pj).shift(-(pj +1))

    c0 = pdQ.iloc[:, [0, -3]].corr().iloc[1, 0]
    c1 =pdQ.iloc[:, [1, -2]].corr().iloc[1, 0]
    c2 =pdQ.iloc[:, [2, -1]].corr().iloc[1, 0]
    pr0 = pdQ.iloc[:, [0, -3]]
    pr0[pr0<pr0.iloc[:,0].quantile(0.0005)].mean()
    pr0.corr()
    bkStat["c0"] = c0
    bkStat["c1"] = c1
    bkStat["c2"] = c2


    apnl["fee"]=(apnl["trdVol"] * apnl["trdPrice"]).cumsum() *feeRatio / 1e4/2
    apnl["aftpnl"]=     apnl["pnl"]-   apnl["fee"]
    apnl.index = pd.to_datetime(bkStream.qDate+(apnl["closeTime"]/1e9).astype("str"),format="%Y%m%d%H%M%S.%f")
    pnl1min = apnl["aftpnl"].resample("1min").last().dropna()
    pnl1minRet = pnl1min.diff()
    bkStat["sharpe"] = pnl1minRet.mean() / pnl1minRet.std()

    return ssDf,bkStat,  apnl
# bst = lgb.Booster(params={"nthreads":1},model_file="lgb_123047.txt")

#bst = lgb.Booster(params={"nthreads":1},model_file="/home/aether/lgb_zz_003.txt")
#tillf = 20
#bst = lgb.Booster(params={"nthreads":1},model_file="/home/aether/zzmodel20_rv.txt")

class STRATEGY(object):
    def __init__(self,ins,bst):
        pass
def getRst(bkStream):
    feeRatio = 0.00126
    if bkStream.player.ifzz == True:
        feeRatio = 0.00012
    if len(bkStream.doneRc)<1:
        return None,None,None
    ss = {}


    for key, order_i in bkStream.doneRc.items():
        # ss[key] = [order_i.status, order_i.closeTime, order_i.insertTime, order_i.dirc, order_i.price / 1e4,
                   # order_i.tradePrice / 1e4, order_i.vol, order_i.predy, np.int32(order_i.OrderID),  np.int32(order_i.closeSysID)]
        ss[key] = [order_i.status, order_i.closeTime, order_i.insertTime,order_i.createTime, order_i.dirc, order_i.price / 1e4,
                   order_i.tradePrice / 1e4, order_i.tradeVolAcc, order_i.predy, np.int64(order_i.OrderID),  np.int64(order_i.closeSysID),order_i.tradeType ]
    for key, order_i in bkStream.waitOrderList.items():
        ss[key] = [order_i.status, order_i.closeTime, order_i.insertTime,order_i.createTime, order_i.dirc, order_i.price / 1e4,
                   order_i.tradePrice / 1e4, order_i.tradeVolAcc, order_i.predy, np.int64(order_i.OrderID),  np.int64(order_i.closeSysID),order_i.tradeType ]

    ssDf = pd.DataFrame(ss).T



    zff = ssDf[ssDf[6] !=0]
    if len(zff) < 1:
        return None, None, None
    bkStat = {}

    apnl = pd.DataFrame(bkStream.orderStr).dropna(how="all")
    bkStream.orderQue.lossDataC

    apnl.columns = ["pnl", "netPosition", "closeTime", "orderRef", "OrderID", "oprice","sysID","trdPrice","dirc","trdVol","insertTime","createTime"]

    fee = (apnl["trdPrice"] * apnl["trdVol"] / 1e4).sum() * feeRatio / 2
    earing = apnl["pnl"].iloc[-1]
    maxRetrace = (apnl["pnl"] - apnl["pnl"].expanding().max()).min()

    position = apnl["netPosition"]
    bkStat["profit"] = earing
    bkStat["tradeAmount"] = (apnl["trdPrice"] * apnl["trdVol"] / 1e4).sum() /2
    bkStat["fee"] = fee
    bkStat["pvsf"] = bkStat["profit"] / bkStat["fee"]
    bkStat["netP"] = bkStat["profit"] - bkStat["fee"]
    bkStat["netP_bp"] = (bkStat["profit"] - bkStat["fee"]) / bkStat["tradeAmount"] * 10000
    bkStat["tradeRound"] = apnl["trdVol"].sum()/bkStream.player.baseVol/2
    bkStat["longMax"] = position.max()
    bkStat["shortMax"] = position.min()
    bkStat["closeNetPosition"] = position.iloc[-1]
    bkStat["meanPrice"] =apnl["trdPrice"].replace(0, np.nan).mean()/1e4
    bkStat["maxRetrace"] =maxRetrace
    pdQ = pd.DataFrame(np.vstack(bkStream.player.predAndQ)).iloc[20:]



    pj =160
    pdQ["yd0"] = pdQ.loc[:, 3].pct_change(pj).shift(-(pj +1))
    pdQ["yd1"] = pdQ.loc[:, 4].pct_change(pj).shift(-(pj+1))
    pdQ["yd2"] = pdQ.loc[:, 5].pct_change(pj).shift(-(pj +1))

    c0 = pdQ.iloc[:, [0, -3]].corr().iloc[1, 0]
    c1 =pdQ.iloc[:, [1, -2]].corr().iloc[1, 0]
    c2 =pdQ.iloc[:, [2, -1]].corr().iloc[1, 0]
    pr0 = pdQ.iloc[:, [0, -3]]
    pr0[pr0<pr0.iloc[:,0].quantile(0.0005)].mean()
    pr0.corr()
    bkStat["c0"] = c0
    bkStat["c1"] = c1
    bkStat["c2"] = c2


    apnl["fee"]=(apnl["trdVol"] * apnl["trdPrice"]).cumsum() *feeRatio / 1e4/2
    apnl["aftpnl"]=     apnl["pnl"]-   apnl["fee"]
    apnl.index = pd.to_datetime(bkStream.qDate+(apnl["closeTime"]/1e9).astype("str"),format="%Y%m%d%H%M%S.%f")
    pnl1min = apnl["aftpnl"].resample("1min").last().dropna()
    pnl1minRet = pnl1min.diff()
    bkStat["sharpe"] = pnl1minRet.mean() / pnl1minRet.std()

    return ssDf,bkStat,  apnl

def myBKone(qDate, stk_i,runParams,prefix=None):
    import os
    os.environ['OMP_NUM_THREADS']= "1"
    try:
        import time
        start = time.time()
        ins = str(stk_i)
        if len(str(stk_i))<6:
            ins = ins.zfill(6)
        ins = ins[-6:]
        fromDate, toDate = qDate, qDate
        refinsList =[]
        bkStream = DataController(ins, refinsList, qDate,prefix)
        ply = STRATEGY(ins,bst)
        ply.profitOpen = runParams["profitOpen"]
        ply.profitClose =  runParams["profitClose"]
        ply.profitCancel =  runParams["profitCancel"]
        ply.profitPending =  runParams["profitPending"]
        ply.baseVol = runParams["minV"]
        ply.maxPosition = runParams["maxV"]
        if runParams.get("ydPositionLong") is not None:
            print(runParams["ydPositionLong"])
            ply.ydPositionLong = runParams["ydPositionLong"]
        bkStream.play(ply)

        detailDf, Statdf, tpnl = getRst(bkStream)

        # print(pd.Series(Statdf))
        end = time.time()
        print(stk_i, "done in ", end - start)
        if len(bkStream.orderQue.lossDataC) > 0:
            print(" ", float(len(bkStream.orderQue.lossDataC)))
        return detailDf, Statdf, tpnl
    except Exception as e:
        print(stk_i, " ", e)
        return None, None, None

def multBK(allsz2,qDate):
    print(qDate)
    from multiprocessing import Pool
    import os
    os.environ['OMP_NUM_THREADS']= "1"
    # partCodes = list(allsz2.index)
    # prefix = "myTmp"
    # getAllByDay(qDate,prefix, partCodes, "/mnt/lustre/data/raw/stock/md.data/caitong.data/", False)

    with  Pool(processes=min(50,len(allsz2))) as pool:
        # mul_res = [pool.apply_async(myBKone,(qDate,stk_i,)) for stk_i in sz_stocks ]
        mul_res = [pool.apply_async(myBKone, (qDate, str(code_i),vol_i)) for code_i,vol_i in allsz2.iterrows()]
        rstList = [res.get() for res in mul_res]
    import psutil

    bkStatList = {}
    k = 0
    tpnltt = []
    for items in rstList:
        detailDf, Statdf, tpnl = items
        if Statdf is not None:
            bkStatList[allsz2.index[k]] = Statdf
            tpnl["scode"] = allsz2.index[k]


            tpnl.index = pd.to_datetime(qDate+(tpnl["closeTime"]/1e9).astype("str"),format="%Y%m%d%H%M%S.%f")

            tpnltt.append(   tpnl.resample("1min").last().ffill() )
        k += 1
    # saveOrderPath = zzDataTmp + "%s_%s_%s.h5" % (qDate,prefix, "order")
    # saveTransPath = zzDataTmp + "%s_%s_%s.h5" % (qDate,prefix, "trans")
    # saveSnapPath = zzDataTmp + "%s_%s_%s.h5" % (qDate,prefix, "snap")
    # tmpss = os.system("rm %s %s %s"%(saveSnapPath,saveTransPath,saveOrderPath))

    bkStatDf = pd.DataFrame(bkStatList).T
    tpnlDf = pd.concat(tpnltt)
    return bkStatDf,rstList,tpnlDf


def bkOneDay(qDate):
    import os
    os.chdir("/home/aether/config_json")
    jsConfig =[x for x in os.listdir("./") if "wParams_bond_for_"in x]
    dofConfig =[x for x in jsConfig if x.split(".")[0][-8:]<=qDate ][-1]
    # dofConfig= "/home/aether/szStock/projects/AetherSZStock/wParams_bond.json"
    print(dofConfig)
    with open(dofConfig, "r") as jf:
        jVol = json.load(jf)
    RunParamDf = pd.DataFrame(jVol["RunParam"]).set_index("instrument")
    theVol = RunParamDf
    # theVol["profitOpen"]  = 5/ 1e4
    # theVol["profitClose"]  =5/ 1e4
    # theVol["profitCancel"] =  4/ 1e4
    # theVol["minV"] = theVol["minV"]*2
    # theVol["maxV"] = theVol["maxV"]*2
    theVol = theVol.sample(frac=1)
    bkStatDf, rstList, tpnlDf = multBK(theVol, qDate)
    ttstat = (bkStatDf.sum()[["profit", "tradeAmount", "fee", "netP"]])
    ttstat["fr"] = bkStatDf["netP"].sum() / bkStatDf["fee"].sum()
    tpnlDf.index.name = 'timeGap'
    tpnlDfdd = tpnlDf.reset_index()
    pvPNLdf = tpnlDfdd[["timeGap", "aftpnl", "scode"]].pivot_table(index=['timeGap'], columns=['scode']).ffill()
    feeline = tpnlDfdd[["timeGap", "fee", "scode"]].pivot_table(index=['timeGap'], columns=['scode']).ffill()
    pvPNLdf = pd.Series(pvPNLdf.T.fillna(0).values.sum(axis=0), pvPNLdf.index)
    feeline = pd.Series(feeline.T.fillna(0).values.sum(axis=0), feeline.index)
    pvPNLdfDiff = pvPNLdf.diff()
    ttstat["sharpe"] = pvPNLdfDiff.mean() / pvPNLdfDiff.std()
    plt.cla()

    # pvPNLdf.plot()
    # feeline.plot()
    # print(ttstat)
    # plt.show()
    return bkStatDf, rstList, tpnlDf ,pvPNLdf,feeline,ttstat

def cal_time_diff(a,b):
    #x = (a-b)%(int(1e09))
    x = (a)%(int(1e09)) - (b)%(int(1e09))
    y = (pd.to_datetime(a // int(1e09),format='%H%M%S') - pd.to_datetime(b // int(1e09),format='%H%M%S')).total_seconds()
    return y + x/1e09

def cnt_sec(a):
    x = str(a // int(1e09))
    h = int(x[:-4])
    m = int(x[-4:-2])
    s = int(x[-2:])
    if h>=13:
        return  (h * 3600 + m * 60 + s - 90*60) * int(1e09) + a % int(1e09)
    return (h * 3600 + m * 60 + s) * int(1e09) + a % int(1e09)


#getAllByDay('20210607',replace=True)
#if __name__ == '__main__':
#%%
if False:
    zzDataTmp = "/home/tlei/data.zhongtai/"
    insList = ['1300142','1300059', '1002594', '1002537','1002432',
               '1123138','1128069','1128040', '1123073', '1127007',
               '1123072','1127013'
               ]
    ins = insList[7]
    OrderQue_test = OrderQue(ins)
    dataStreamMap = {int(ins): readH5Cache(int(ins), '20220322')}
    cnt, time = [], []
    for data_i in getMkDataMult(dataStreamMap):
        OrderQue_test.handle(data_i)
        if data_i[0] ** 2 == data_i[0] and OrderQue_test.mktTime >= 93000 * 1e9 and OrderQue_test.mktTime <= 145700 * 1e9:
            try:
                #cnt.append(cal_time_diff(data_i[3], data_i[2]))
                #cnt.append(cnt_sec(data_i[3]) - cnt_sec(data_i[2]))
                #time.append(cnt_sec(data_i[2]))
                timedelta = pd.to_datetime(data_i[3],format='%H%M%S%f')-pd.to_datetime(data_i[2],format='%H%M%S%f')
                cnt.append(timedelta.total_seconds())
                time.append(data_i[2])
                if len(cnt) > 1000:
                    break
            except:
                pass


