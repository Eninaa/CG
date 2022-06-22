import numpy as np
from PIL import Image
import math


s = 4000
o = 500

class Color3: 
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

class Image1:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.m_data = np.array([])
        self.z_buffer = np.ones((self.height, self.width))
        temp = []
        for i in range(0, self.width * self.height):  
            temp.append(Color3(0, 0, 0))
        self.m_data = np.append(self.m_data,temp)
        self.m_data = self.m_data.reshape((self.height, self.width))

    def getHeight(self):
        return self.height
    
    def getWidth(self):
        return self.width
    
    def setPixel(self, x, y, color):
        self.m_data[self.height-y, x] = color
        # self.m_data[x, y] = color
        
    def getPixel(self, x, y):
        return self.m_data[x, y]
    
    def getData(self):
        return self.m_data
    
    def setZBuffer(self, value):
        self.z_buffer *= value
    
    def save1(self, filename):
        temp = []
        for i in range(self.height):
            for j in range(self.width):
              tempColor3 = self.m_data[i,j]
              temp.append([tempColor3.r, tempColor3.g, tempColor3.b])
        arr = np.array([])
        arr = np.append(arr, temp)
        arr = arr.reshape((self.height, self.width, 3))
        img = Image.frombytes('RGB', (self.height, self.width), np.uint8(arr))
        im = "images/" + filename + '.png'
        img.save(im, "PNG")
      
    def line1(self, x0, y0, x1, y1, color):
        for t in np.arange(0, 1, 0.01):
            x = x0*(1-t) + x1*t
            y = y0*(1-t) + y1*t
            self.setPixel(int(x), int(y), color)

    def line2(self, x0, y0, x1, y1, color):
        for x in np.arange(x0, x1+1, 1):
            t = (x-x0)/(x1-x0)
            y = y0*(1.0 -t) + y1*t
            self.setPixel(int(x), int(y), color)

    def line3(self, x0, y0, x1, y1, color):
        steep = False
        if (abs(x0-x1) < abs(y0-y1)):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            steep = True
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        for x in np.arange(x0, x1+1):
            t = (x-x0)/(x1-x0)
            y = y0*(1.-t) + y1*t
            if steep:
                self.setPixel(int(y), int(x), color)
            else:
                self.setPixel(int(x), int(y), color)
         
    def line4(self, x0, y0, x1, y1, color):
        steep = False
        if (abs(x0-x1) < abs(y0-y1)):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            steep = True
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
            
        dx = x1 - x0
        dy = y1 - y0
        derror = abs(dy/dx)
        error = 0
        y = y0
        
        for x in np.arange(x0, x1+1):
            if steep:
                self.setPixel(int(y), int(x), color)
            else:
                self.setPixel(int(x), int(y), color)
            error = error + derror
            if error > 0.5:
                a = 1 if y1 > y0 else -1
                y = y + a
                error -= 1.0
 # ЛР2 Задание 8
    def barycentric(self, x, y, x0, y0, x1, y1, x2, y2):
        l0 = ((x1 - x2)*(y - y2) - (y1 - y2)*(x - x2)) / ((x1 - x2)*(y0 - y2) - (y1 - y2)*(x0 - x2))
        l1 = ((x2 - x0)*(y - y0) - (y2 - y0)*(x - x0)) / ((x2 - x0)*(y1 - y0) - (y2 - y0)*(x1 - x0))
        l2 = ((x0 - x1)*(y - y1) - (y0 - y1)*(x - x1)) / ((x0 - x1)*(y2 - y1) - (y0 - y1)*(x2 - x1))
        return np.array([l0, l1, l2])   
         
    # ЛР2 Задание 9
    def triangle(self, x0, y0, x1, y1, x2, y2, color):
        xmin = math.floor(min(x0, x1, x2))
        xmax = math.ceil(max(x0, x1, x2))
        ymin = math.floor(min(y0, y1, y2))
        ymax = math.ceil(max(y0, y1, y2))
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if ymax > self.height: ymax = self.height
        if xmax > self.width: xmax = self.width
          
        for y in np.arange(ymin, ymax):
            for x in np.arange(xmin, xmax):
                temp = self.barycentric(x, y, x0, y0, x1, y1, x2, y2)
                if (all([t > 0 for t in temp])):
                    self.setPixel(x, y, color)
       
    def triangleZ(self, x0, y0, z0, x1, y1, z1, x2, y2, z2, color): 
        xmin = min(x0, x1, x2)
        ymin = min(y0, y1, y2)
        xmax = max(x0, x1, x2)
        ymax = max(y0, y1, y2)
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if ymax > self.height: ymax = self.height
        if xmax > self.width: xmax = self.width
            
        for x in np.arange(math.ceil(xmin), math.floor(xmax)):
            for y in np.arange(math.ceil(ymin), math.floor(ymax)):
                temp = self.barycentric(x, y, x0, y0, x1, y1, x2, y2)
                if (temp[0] > 0 and temp[1] > 0 and temp[2] > 0 ):
                    z = temp[0]*z0 + temp[1]*z1 + temp[2]*z2
                    if(z < self.z_buffer[x,y]):
                        self.setPixel(x, y, color)
                        self.z_buffer[x,y] = z
        
    def light(self, norm0, norm1, norm2):
        l = np.array([0, 0, 1])
        n0 = np.array([norm0.x, norm0.y, norm0.z])
        n1 = np.array([norm1.x, norm1.y, norm1.z])
        n2 = np.array([norm2.x, norm1.y, norm2.z])

        l0 = np.dot(n0, l)/(np.linalg.norm(n0) * np.linalg.norm(l))
        l1 = np.dot(n1, l)/(np.linalg.norm(n1) * np.linalg.norm(l))
        l2 = np.dot(n2, l)/(np.linalg.norm(n2) * np.linalg.norm(l))

        return np.array([l0, l1, l2])
           
    def Gouraud(self, x0, y0, x1, y1, x2, y2, norm0, norm1, norm2):
        xmin = math.floor(min(x0, x1, x2))
        xmax = math.ceil(max(x0, x1, x2))
        ymin = math.floor(min(y0, y1, y2))
        ymax = math.ceil(max(y0, y1, y2))
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if ymax > self.height: ymax = self.height
        if xmax > self.width: xmax = self.width
          
        for y in np.arange(ymin, ymax):
            for x in np.arange(xmin, xmax):
                bar = self.barycentric(x, y, x0, y0, x1, y1, x2, y2)
                l = self.light(norm0, norm1, norm2)
                if ((bar > 0). all()):
                    color = 255*(bar[0]*l[0] + bar[1]*l[1] + bar[2]*l[2])
                    tempColor3 = Color3(color, color, color)
                    print(color)
                    self.setPixel(x, y, tempColor3)
    
    def normal(self, x0, x1, x2, y0, y1, y2, z0, z1, z2):
        A = np.array([x1 - x0, y1 - y0, z1 - z0])
        B = np.array([x1 - x2, y1 - y2, z1 - z2])
        n = np.cross(A, B)
        return n
    
    def cosinus(self, normal):
        l = np.array([0,0,1])
        scalar = np.dot(normal, l)
        n = (np.linalg.norm(normal) * np.linalg.norm(l))
        return scalar/n
    
    def proj(self, ax, ay, u0, v0, X, Y, Z, tz):
        Mat = np.array([[ax, 0, u0],
                        [0, ay, v0],
                        [0, 0, 1]])
        vec = np.array([X, Y, Z])
        res = Mat.dot(vec + np.array([0, 0, tz[0]]))
        return res/res[2]
    
    
    def rot(self, X, Y, Z, alpha, beta, gamma):
        
        A = np.array([[1, 0, 0],
                      [0, math.cos(math.radians(alpha)), math.sin(math.radians(alpha))],
                      [0, -math.sin(math.radians(alpha)), math.cos(math.radians(alpha))]])
                                    
        B = np.array([[math.cos(math.radians(beta)), 0, math.sin(math.radians(beta))],
                      [0, 1, 0],
                      [-math.sin(math.radians(beta)), 0, math.cos(math.radians(beta))]])
                       
        C = np.array([[math.cos(math.radians(gamma)), math.sin(math.radians(gamma)), 0],
                      [-math.sin(math.radians(gamma)), math.cos(math.radians(gamma)), 0],
                      [0, 0, 1]])
        res1 = A.dot(B)
        R = res1.dot(C)
        res = R.dot(np.array([X, Y, Z]))
        return res
 

class Vertex:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def getZ(self):
        return self.z
    
class Model:
    def __init__(self):
        self.ver = np.array([])
        self.pol = np.array([])
        self.norms = np.array([])
        self.normIndex = np.array([])
        
    def getData(self):
        return self.arr
        
    def load(self, filename):
        try:
            f = open(filename, 'r')
            for line in f:
                str = line.split()
                if str[0] == 'v':
                    temp = Vertex(float(str[1]), float(str[2]), float(str[3]))
                    self.ver = np.append(self.ver, temp)
                elif str[0] == 'vn':
                    temp3 = Vertex(float(str[1]), float(str[2]), float(str[3]))
                    # temp3 = np.array([float(str[1]), float(str[2]), float(str[3])])
                    self.norms = np.append(self.norms, temp3)
                elif str[0] == 'f': 
                    temp2 = np.array([])
                    temp4 = np.array([])
                    for i in range(1, 4):
                        a = str[i].split('/')
                        temp2 = np.append(temp2, int(a[0]))
                        temp4 = np.append(temp4, int(a[2]))
                    self.pol = np.append(self.pol,temp2)
                    self.normIndex = np.append(self.normIndex, temp4)
                
            self.pol = self.pol.astype(int)
            self.normIndex = self.normIndex.astype(int)
            self.normIndex = self.normIndex
            
        finally:
            f.close()

if __name__ == "__main__":

    # Задание 18
    
    H = 1000
    W = 1000
    test18 = Image1(H, W)
    model = Model()
    model.load('Test.obj')    
    for i in range(0, model.pol.size, 3):
        v1 = model.ver[model.pol[i] - 1]
        v2 = model.ver[model.pol[i + 1] - 1]
        v3 = model.ver[model.pol[i + 2] - 1]
        
        norm0 = model.norms[model.normIndex[i] - 1]
        norm1 = model.norms[model.normIndex[i + 1] - 1]
        norm2 = model.norms[model.normIndex[i + 2] - 1]        
        x0 = s * v1.x + o
        y0 = s * v1.y + o
        z0 = s * v1.z + o
        x1 = s * v2.x + o
        y1 = s * v2.y + o
        z1 = s * v2.z + o
        x2 = s * v3.x + o
        y2 = s * v3.y + o
        z2 = s * v3.z + o
        # def normal(self, x0, x1, x2, y0, y1, y2, z0, z1, z2):
        # def Gouraud(self, x0, y0, x1, y1, x2, y2, norm0, norm1, norm2):
        # normal = test18.normal(v1.x, v2.x, v3.x, v1.y, v2.y, v3.y, v1.z, v2.z, v3.z)
        normal = test18.normal(x0, x1, x2, y0, y1, y2, z0, z1, z2)
        cosinus = test18.cosinus(normal)
        if cosinus < 0:
            test18.Gouraud(x0, y0, x1, y1, x2, y2, norm0, norm1, norm2)
    test18.save1('task18')  
    
    

         
                 
