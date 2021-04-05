from cv2 import cv2
import numpy as np 

from matplotlib import pyplot as plt 
from scipy.ndimage import interpolation



def count_islands(arr):
    
    """
    На вход принимает массив Boolean
    Возвращает количество непрерывных областей, где принимается значение True
    """
    
    nz = np.flatnonzero(arr)
    rng = np.arange(len(nz))
    return len(np.unique(nz - rng))


def find_plato(arr):
    
    """
    Принимает массив целоцисленных значений
    Возвращает индексы начала непрерывных областей, где принимается одинаковое значение 
    """
    
    idx = np.zeros_like(arr, dtype='int')
    
    for i, k in list(enumerate(idx))[1:]:
        
        if arr[i - 1] == arr[i]:
            idx[i] = idx[i - 1]
        else:
            idx[i] = i
            
    return np.unique(idx)
            

def find_separate_level(hist):
    
    """
    Ищет уровень высоты, по которому лучше всего разрезать гистограмму
    """
    
    rng = np.arange(0, np.max(hist))
    curve = np.array([count_islands(hist >=i) for i in rng])
    mean = np.sum(curve * np.arange(len(curve))) / np.sum(curve) 
    p = find_plato(curve[:int(mean)])
    argmax = np.argmax(p[1:] - p[:-1])
    
    return p[argmax], p[argmax + 1]


def smooth_hist(hist, m=20):
    
    """
    Сглаживание гистограммы 
    """
    
    w = len(hist)
    smooth = np.array([np.mean(hist[i: i + m]) for i in np.arange(w - m + 1)])
    return smooth    


def find_left_island_coast(surface):
    
    """
    Принимает массив Boolean
    Ищет индексы начала непрерывных областей, где принимается значение True
    """
    
    nz = np.flatnonzero(surface)
    p = nz - np.arange(len(nz)) + 1
    p[1:] -= p[:-1]
    left = nz[np.flatnonzero(p)]
    return left


def find_islands(surface):
    
    """
    Принимает массив Boolean 
    Ищет индексы начала и конца непрерывных областей, где принимается значение True
    """
    
    w = len(surface)
    left_coast = find_left_island_coast(surface)
    right_coast = find_left_island_coast(surface[::-1])
    
    right_coast = (w - 1 - right_coast)[::-1]
    
    return  np.stack([left_coast, right_coast], axis=1)


def island_lengths(surface):
    
    w = len(surface)
    left_coast = find_left_island_coast(surface)
    right_coast = find_left_island_coast(surface[::-1])
    
    right_coast = (w - 1 - right_coast)[::-1]
    
    return right_coast - left_coast 


def find_gulf(surface, from_begin=True):
    
    """
    Метод ищет первую слева "яму" в гистограмме 
    """
    
    w = len(surface)
    
    if not from_begin:
        surface = surface[::-1]
    
    a, b = w, w
    
    for i in np.arange(w):
        if surface[i]:
            break
    
    for a in np.arange(i, w):
        if not surface[a]:
            break
    
    for b in np.arange(a, w):
        if surface[b]:
            b = b - 1
            break
            
    if a == w or b == w:
        return w, w
    
    if not from_begin:
        a = w - a - 1
        b = w - b - 1
        a, b = b, a
        
    return a, b


def std_score(tail, i):
    
    indexes = np.arange(len(tail))
    
    i1, i2 = indexes[:i], indexes[i:]
    p1, p2 = tail[:i], tail[i:]
    
    
    mean1 = sum(i1 * p1) / (sum(p1) + 1) 
    mean2 = sum(i2 * p2) / (sum(p2) + 1)
    
    var1 = sum(((i1 - mean1) * p1) ** 2) / (sum(p1) + 1)
    var2 = sum(((i2 - mean2) * p2) ** 2) / (sum(p2) + 1)
    
    return np.sqrt(var1 + var2)
    
    
def mean_score(tail, i):
    
    return np.abs(np.mean(tail[i:]) - np.mean(tail[:i]))



def crop_vertical(img):
    hist = np.sum(255 - img, axis=0)
    smoothed = smooth_hist(hist)
    level = np.mean(find_separate_level(smoothed))
    
    
    surface = smoothed > level 
    w = len(smoothed)

    c, d = find_gulf(surface)
    a, b = find_gulf(surface, 0)
    
    #обработка правого края
    if a > w // 2 and b > w // 2 and a != w and b != w:
    
        l = max(0, a - (b - a))
        r = min(w - 1, b + (b - a))
    
        #scores = np.array([std_score(smoothed[l:r], i) for i in np.arange(r - l)])
        #argmin = np.argmin(scores)
        
        
        
        ##plt.figure()
        ##plt.bar(np.arange(len(r - l)), smoothed[l:r])
        ##plt.plot()

        #crop_pos = argmin + l + m

        crop_pos = (a + b) // 2
        
        img[:, crop_pos:] = 255
    
    
    #обработка левого края 
    if c < w // 2 and d < w // 2:
        
        #l = max(0, d - (d - c))
        #r = min(w - 1, d + (d - c))
        
        #scores = np.array([std_score(smoothed[l:r], i) for i in np.arange(r - l)])
        #argmin = np.argmin(scores)
        
        #crop_pos = argmin + l
        
        crop_pos = (c + d) // 2
        
        img[:, :crop_pos] = 255    
        
    
def find_local_maxim(arr):
    
    arrf = np.hstack([[arr[0]], arr + 1, [arr[-1]]])
    p = find_plato(arrf)
    
    
    cur = arrf[p[1:-1]]
    prev = arrf[p[:-2]]
    nxt = arrf[p[2:]]
    
    is_loc_m = np.logical_and( cur > prev, cur > nxt)
    
    return p[1:-1][is_loc_m] - 1





def find_score(img, angle):
    data = interpolation.rotate(img, angle, order=0, cval=255)
    hist = np.sum(255 - data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score



def find_best_angle(img): 
    angles = np.linspace(-10, 10, 100)
    angle_index = np.argmax([find_score(img, angle)[1] for angle in angles])
    return angles[angle_index]

    
