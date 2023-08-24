from datetime import date
from datetime import datetime
from datetime import timedelta


class Time():
    '''
    classdocs
    '''
    DATE0 = date(1900, 1, 1)
    
    @staticmethod
    def todayCoordinate():
        dt = datetime.today()
        strtoday = str(dt.year) + "/" + str(dt.month) + "/" + str(dt.day)
        coord = Time.dateToCoordinate(strtoday, "/")
        return coord
    
    @staticmethod
    def mlength(m, y):
        if(m == 1 or m == 3 or m == 5 or m == 7 or m == 8 or m == 10 or m == 12):
            return 31
        elif(m == 2):
            if(Time.leapYear(y)):
                return 29
            else:
                return 28
        else:
            return 30
    
    @staticmethod
    def ylength(y):
        if Time.leapYear(y):
            return 366
        else:
            return 365

    @staticmethod
    def leapYear(y):
        return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)

    @staticmethod
    def ycoord(y):
        if(y < 1900):
            return 0
        else:
            v = 0
            yy = 1900
            while(yy < y):
                v = v + Time.ylength(yy)
                yy = yy + 1
            return v
        
    @staticmethod
    def mcoord(m, y):
        if(m <= 1):
            return 0
        else:
            return sum(map(lambda m0: Time.mlength(m0, y) , range(1, m)))

    @staticmethod
    def dateToCoordinate(strdate, delim="/-"):
        try:
            ymd = [int(strdate.split("-")[0].split("/")[0]), -1, -1]
        except:
            ymd = [-1, -1, -1]
        for ch in delim:
            if ch in strdate:
                for i, elem in enumerate(list(map(lambda x: int(x), strdate.split(ch)))):
                    ymd[i] = elem
        [y,m,d] = ymd
        if(m < 0):
            m = 1
        if(d < 0):
            d = 1
        m = Time.mcoord(m, y)
        y = Time.ycoord(y)
        return y + m + d
    
    @staticmethod    
    def coordinateToDate(coord):
        dat = Time.DATE0 + timedelta(coord - 1)
        return dat

    @staticmethod
    def coordinateToStringDate(coord, dateformat='%m/%d/%Y'):
        dat = Time.DATE0 + timedelta(coord - 1)
        return dat.strftime(dateformat)
        