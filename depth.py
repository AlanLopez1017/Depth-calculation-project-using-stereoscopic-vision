# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:53:43 2022

@author: Alan Rodrigo López López
"""
import cv2
import numpy as np
from skimage import io, data, color
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
import time
import imutils
from knn import KNN
from data import *
import tkinter
from knn import KNN

def take_picture(camera):
    cam = cv2.VideoCapture(camera)
    result = True
    objects = np.zeros((480, 640, 3))
    while(result):
        for _ in range(20):
            _,frame = cam.read()
            objects += frame
        result = False
    cam.release()
    cv2.destroyAllWindows()
    
    return frame


def HSV_filter(frame):
    
    blur = cv2.GaussianBlur(frame, (5,5), 0) # Aplicación del filtro

    # conversión de RGB a HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    l_r = np.array([60, 110, 50]) # límite inferior para objeto rojo
    u_r = np.array([225, 225, 255]) # límite superior para objeto rojo
    
    mask = cv2.inRange(hsv, l_r, u_r)
    mask = cv2.erode(mask, None, iterations = 2) # Operación morfológica: Erosión
    mask = cv2.dilate(mask, None, iterations = 2) # Operación morfológica: Dilatación
    
    return mask


def find_object(frame, mask):
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    center = None
    
    # Si se encontró al menos un contorno
    if len(contours) > 0:
        
        c = max(contours, key=cv2.contourArea)
        #((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c) # enuentra el punto central
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
    
    return center

def calculate_depth(circle_left, circle_right, frame_right, frame_left, b, theta):
     
    # convierte la longitud focal [mm] a [pixel]
    width_left = frame_left.shape[1]
    width_right = frame_right.shape[1]
    
    
    if width_left == width_right:
        f_pixel = (width_right* 0.5) / np.tan(theta * 0.5 * np.pi/180)
    else:
        print("Los frame de las camaras no tienen el mismo ancho de pixel")
        
    x_left = circle_left[0]
    x_right = circle_right[0]
    substract = x_left - x_right
    print(f_pixel)
    print(x_left)
    print(x_right)
    depth = abs((b*f_pixel/substract))  # Fórmula de profundidad
    
    X = -(b/2)*(x_left+x_right)/(x_right-x_left) # Fórmula de posición en X
    
    return depth, X


def run():

    # número de camara
    camera = 1
    # parámetros
    b = 6
    theta = 62
    
    continuar = True
    res = []
    numbers = []
    while(continuar):
        
    # camara de izquierda
        input("Primera foto")
        frame_left = take_picture(camera)
        input("Segunda foto")
        # camara de derecha
        frame_right = take_picture(camera)
        
        
        # Aplicación del filtro HSV
        mask_left = HSV_filter(frame_left)
        mask_right = HSV_filter(frame_right)
        
        
        # Aplicación de las máscaras a los frames
        res_left = cv2.bitwise_and(frame_left, frame_left, mask = mask_left)
        res_right = cv2.bitwise_and(frame_right, frame_right, mask = mask_right)
       
        plt.figure(0)
        plt.imshow(res_left, cmap = 'gray')
        
        plt.figure(1)
        plt.imshow(res_right, cmap = 'gray')
    
        # Reconocimiento de la forma segmentada
        circles_left = find_object(frame_left, mask_left)
        circles_right = find_object(frame_right, mask_right)
        
        
        if np.all(circles_left) == None or np.all(circles_right) == None:
            print("No se encontró ningún objeto")
        else:
            depth, X_real = calculate_depth(circles_left, circles_right, frame_left, frame_right, b, theta)
        
        print(depth)
        print(X_real)
        res = np.array([[depth],[X_real]])
        
        X, C = data()
        clasificador = KNN()
        clasificador.learning(X, C)
        clasificar = clasificador.classification(res)
        print(clasificar)
        
        number = clasificar[0]
        numbers.append(number)
        
        continuar = int(input("¿Desea tomar otra profundidad? "))
        
    window = tkinter.Tk()
    window.geometry("500x100")
    label = tkinter.Label(window, text = "Calculadora", fg = "blue", font = ("Verdana", 16))
    
    if numbers[1] == 10:
        res = numbers[0] + numbers[2]
        label_depth= tkinter.Label(window, text = str(numbers[0]) + " + " + str(numbers[2]) + " = " + str(res), font = ("Verdana", 14))
    else:
        res = numbers[0] - numbers[2]
        label_depth= tkinter.Label(window, text = str(numbers[0]) + " - " + str(numbers[2]) + " = " + str(res), font = ("Verdana", 14))
    
    label.pack()
    label_depth.pack()
    window.mainloop()

        
        
        
    
if __name__ == '__main__':
    run()