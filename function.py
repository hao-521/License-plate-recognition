import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import pytesseract
from PIL import Image
import os
import json


def imreadex(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


SZ = 20  # 训练图片长宽
PROVINCE_START = 1000
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积

def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks
# 根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves):
    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]:wave[1]])
    return part_cards
# 来自opencv的sample，用于svm训练
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img
# 来自opencv的sample，用于svm训练
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)
# 不能保证包括所有省份
provinces = [
    "zh_cuan", "川",
    "zh_e", "鄂",
    "zh_gan", "赣",
    "zh_gan1", "甘",
    "zh_gui", "贵",
    "zh_gui1", "桂",
    "zh_hei", "黑",
    "zh_hu", "沪",
    "zh_ji", "冀",
    "zh_jin", "津",
    "zh_jing", "京",
    "zh_jl", "吉",
    "zh_liao", "辽",
    "zh_lu", "鲁",
    "zh_meng", "蒙",
    "zh_min", "闽",
    "zh_ning", "宁",
    "zh_qing", "靑",
    "zh_qiong", "琼",
    "zh_shan", "陕",
    "zh_su", "苏",
    "zh_sx", "晋",
    "zh_wan", "皖",
    "zh_xiang", "湘",
    "zh_xin", "新",
    "zh_yu", "豫",
    "zh_yu1", "渝",
    "zh_yue", "粤",
    "zh_yun", "云",
    "zh_zang", "藏",
    "zh_zhe", "浙",
    "zh_gang", "港",
    "zh_ao", "澳",
    "zh_tai", "台"
]
class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)
class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()
def Color(image):
    green = yellow = blue = black = white = 0
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    row_num, col_num = img_hsv.shape[:2]
    img_count = row_num * col_num
    for i in range(row_num):
        for j in range(col_num):
            H = img_hsv.item(i, j, 0)
            S = img_hsv.item(i, j, 1)
            V = img_hsv.item(i, j, 2)
            if 11 < H <= 34 and S > 34:  # 图片分辨率调整
                yellow += 1
            elif 35 < H <= 99 and S > 34:  # 图片分辨率调整
                green += 1
            elif 99 < H <= 124 and S > 34:  # 图片分辨率调整
                blue += 1

            if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                black += 1
            elif 0 < H < 180 and 0 < S < 30 and 221 < V < 225:
                white += 1
    color = "black"
    if yellow * 2.5 >= img_count:
        color = "yellow"
    elif green * 2.5 >= img_count:
        color = "green"
    elif blue * 2.5 >= img_count:
        color = "blue"
    #print(blue,img_count)
    return color
def detect(image):
    # 定义分类器
    cascade_path = 'cascade.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    # 修改图片大小
    resize_h = 2580
    height = image.shape[0]
    scale = image.shape[1] / float(height)
    image = cv2.resize(image, (int(scale * resize_h), resize_h))
    #image=cv2.resize(image,(800,200))
    pic_hight, pic_width = image.shape[:2]

    # 转为灰度图
    oldimg = image
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 车牌定位
    car_plates = cascade.detectMultiScale(image_gray, 1.11, 8, minSize=(36, 9), maxSize=(36 * 40, 9 * 40))

    # 2500 1.11 8 10  50%
    # 2580 1.11 8 10  55%

    add=10
    print("检测到车牌数", len(car_plates))
    if len(car_plates) > 0:
        for car_plate in car_plates:
            x, y, w, h = car_plate
            #plate = image[y - 3: y + h + 3, x - 3: x + w + 3]

            plate = image[y -add: y + h + add, x - add: x + w + add*4]
            #cv2.rectangle(image, (x - 3, y - 3), (x + w + 3, y + h + 3), (255, 0, 0), 2)
            cv2.rectangle(image, (x - add, y - add), (x + w + add*4, y + h + add), (255, 0, 0), 2)

    #resize_h = 90
    #height = plate.shape[0]
    #scale = plate.shape[1] / float(height)
    #plate = cv2.resize(plate, (int(scale * resize_h), resize_h))
    return plate
    #cv2.imshow("image", image)
def Split(image):
    # image = cv2.resize(image, (500, 300))
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
    # cv2.imshow('gray', img_gray)  # 显示图片
    # cv2.waitKey(0)

    # 2、将灰度图像二值化，设定阈值是100
    ret, img_thre = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ret, img_thre = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)

    # cv2.imwrite('thre_res.png', img_thre)

    # 4、分割字符
    white = []  # 记录每一列的白色像素总和
    black = []  # ..........黑色.......
    height = img_thre.shape[0]
    width = img_thre.shape[1]
    white_max = 0
    black_max = 0
    # 计算每一列的黑白色像素总和
    for i in range(width):
        s = 0  # 这一列白色总数
        t = 0  # 这一列黑色总数
        for j in range(height):
            if img_thre[j][i] == 255:
                s += 1
            if img_thre[j][i] == 0:
                t += 1
        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)

    arg = False  # False表示白底黑字；True表示黑底白字
    if black_max > white_max:
        arg = True

    # 分割图像
    def find_end(start_):
        end_ = start_ + 1
        for m in range(start_ + 1, width - 1):
            if (black[m] if arg else white[m]) > (
                    0.95 * black_max if arg else 0.95 * white_max):  # 0.95这个参数请多调整，对应下面的0.05（针对像素分布调节）
                end_ = m
                break
        return end_

    n = 1
    start = 1
    end = 2
    word = []
    while n < width - 2:
        n += 1
        if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):
            # 上面这些判断用来辨别是白底黑字还是黑底白字
            # 0.05这个参数请多调整，对应上面的0.95
            start = n
            end = find_end(start)
            n = end
            if end - start > 5:
                cj = image[1:height, start:end]
                cj = cv2.resize(cj, (15, 30))
                word.append(cj)

    print(len(word))

    return word
class Predictor:
    def __init__(self):
        # 车牌识别的部分参数保存在js中，便于根据图片分辨率做调整
        f = open('config.js')
        j = json.load(f)
        for c in j["config"]:
            if c["open"]:
                self.cfg = c.copy()
                break
        else:
            raise RuntimeError('没有设置有效配置参数')

    def __del__(self):
        self.save_traindata()

    def train_svm(self):
        # 识别英文字母和数字
        self.model = SVM(C=1, gamma=0.5)
        # 识别中文
        self.modelchinese = SVM(C=1, gamma=0.5)
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        else:
            chars_train = []
            chars_label = []

            for root, dirs, files in os.walk("train\\chars2"):
                if len(os.path.basename(root)) > 1:
                    continue
                root_int = ord(os.path.basename(root))
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    # chars_label.append(1)
                    chars_label.append(root_int)

            chars_train = list(map(deskew, chars_train))
            chars_train = preprocess_hog(chars_train)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            print(chars_train.shape)
            self.model.train(chars_train, chars_label)
        if os.path.exists("svmchinese.dat"):
            self.modelchinese.load("svmchinese.dat")
        else:
            chars_train = []
            chars_label = []
            for root, dirs, files in os.walk("train\\charsChinese"):
                if not os.path.basename(root).startswith("zh_"):
                    continue
                pinyin = os.path.basename(root)
                index = provinces.index(pinyin) + PROVINCE_START + 1  # 1是拼音对应的汉字
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    # chars_label.append(1)
                    chars_label.append(index)
            chars_train = list(map(deskew, chars_train))
            chars_train = preprocess_hog(chars_train)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            print(chars_train.shape)
            self.modelchinese.train(chars_train, chars_label)

    def save_traindata(self):
        if not os.path.exists("svm.dat"):
            self.model.save("svm.dat")
        if not os.path.exists("svmchinese.dat"):
            self.modelchinese.save("svmchinese.dat")

    def predict(self, car_pic):

        # 以下为识别车牌中的字符
        predict_result = []
        color = Color(car_pic)
        if (color == 'yellow' or color == 'blue' or color == 'black'):
            limit = 7
        elif (color == 'green'):
            limit = 8
        if color in ("blue", "yellow", "green", "black"):
            card_img = car_pic
            gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
            if color == "green" or color == "yello":
                gray_img = cv2.bitwise_not(gray_img)
            ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 查找水平直方图波峰
            x_histogram = np.sum(gray_img, axis=1)
            x_min = np.min(x_histogram)
            x_average = np.sum(x_histogram) / x_histogram.shape[0]
            x_threshold = (x_min + x_average) / 2
            wave_peaks = find_waves(x_threshold, x_histogram)
            if len(wave_peaks) == 0:
                print("peak less 0:")

            # 认为水平方向，最大的波峰为车牌区域
            wave = max(wave_peaks, key=lambda x: x[1] - x[0])
            gray_img = gray_img[wave[0]:wave[1]]
            # 查找垂直直方图波峰
            row_num, col_num = gray_img.shape[:2]
            # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
            gray_img = gray_img[1:row_num - 1]
            y_histogram = np.sum(gray_img, axis=0)
            y_min = np.min(y_histogram)
            y_average = np.sum(y_histogram) / y_histogram.shape[0]
            y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半

            wave_peaks = find_waves(y_threshold, y_histogram)

            # for wave in wave_peaks:
            #	cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2)
            # 车牌字符数应大于6
            if len(wave_peaks) <= 6:
                print("peak less 1:", len(wave_peaks))

            wave = max(wave_peaks, key=lambda x: x[1] - x[0])
            max_wave_dis = wave[1] - wave[0]
            # 判断是否是左侧车牌边缘
            if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                wave_peaks.pop(0)

            # 组合分离汉字
            cur_dis = 0
            for i, wave in enumerate(wave_peaks):
                if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                    break
                else:
                    cur_dis += wave[1] - wave[0]
            if i > 0:
                wave = (wave_peaks[0][0], wave_peaks[i][1])
                wave_peaks = wave_peaks[i + 1:]
                wave_peaks.insert(0, wave)

            # 去除车牌上的分隔点
            point = wave_peaks[2]
            if point[1] - point[0] < max_wave_dis / 3:
                point_img = gray_img[:, point[0]:point[1]]
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)

            if len(wave_peaks) <= 6:
                print("peak less 2:", len(wave_peaks))

            part_cards = seperate_card(gray_img, wave_peaks)
            # part_cards=Split(gray_img)
            for i, part_card in enumerate(part_cards):
                # 可能是固定车牌的铆钉
                if np.mean(part_card) < 255 / 5:
                    print("a point")
                    continue
                part_card_old = part_card
                w = abs(part_card.shape[1] - SZ) // 2

                part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)

                # part_card = deskew(part_card)
                part_card = preprocess_hog([part_card])
                if i == 0:
                    resp = self.modelchinese.predict(part_card)
                    charactor = provinces[int(resp[0]) - PROVINCE_START]
                else:
                    resp = self.model.predict(part_card)
                    charactor = chr(resp[0])
                print(charactor)
                # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
                if charactor == "1" and i == len(part_cards) - 1:
                    if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                        continue
                predict_result.append(charactor)
                if(len(predict_result)==limit):
                    break
        return predict_result  # 识别到的字符


def all(image):
    c = Predictor()
    c.train_svm()

    image_plate = detect(image)
    #cv2.imshow('plate', image_plate)
    color = Color(image_plate)
    print(color)
    words = c.predict(image_plate)
    return words,image_plate,color

if __name__ == '__main__':
    c = Predictor()
    c.train_svm()
    image = cv2.imread('6.jpg')
    image_plate = detect(image)
    cv2.imshow('plate', image_plate)
    print(Color(image_plate))
    words = c.predict(image_plate)
    print(words)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
