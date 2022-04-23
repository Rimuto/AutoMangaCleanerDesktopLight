from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QAbstractItemView
import sys
import os
import detect
import cv2
import clean
import numpy as np

labelsPath = "YOLOv4/obj.names"
cfgpath = "YOLOv4/yolov4-obj.cfg"
wpath = "YOLOv4/yolov4-obj_final.weights"

CFG = detect.config(cfgpath)
Weights = detect.weights(wpath)
nets = detect.load_model(CFG, Weights)

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        self.setFixedSize(760, 615)
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi('auto-manga-cleaner-desktop-light.ui', self)
        self.pushButton.clicked.connect(self.select_files)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.pushButton_2.clicked.connect(self.process)
        self.pushButton_4.clicked.connect(self.delete_btn)
        self.pushButton_5.clicked.connect(self.delete_all)
        self.pushButton_3.clicked.connect(self.select_saving_directory)
        self.listView.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.listView_2.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.model = QtGui.QStandardItemModel()
        self.model.rowsInserted.connect(self.on_rowsInserted)
        self.listView.setModel(self.model)
        self.progressBar.setValue(0)
        self.model_2 = QtGui.QStandardItemModel()
        self.listView_2.setModel(self.model_2)
        self.label.setText(self.dir_path)

    def on_rowsInserted(self):
        self.pushButton_2.setEnabled(True)
        self.pushButton_3.setEnabled(True)


    def read_image(self, path):
        f = open(path, "rb")
        chunk = f.read()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        return img

    def save_image(self, img, path, name, ext):
        cv2.imencode(ext, img)[1].tofile(path + '/' + name + ext)

    def select_files(self):
        file, _ = QFileDialog.getOpenFileNames(self, 'Open File', './', "Image (*.png *.jpg *jpeg)")
        if file:
            for i in file:
                item = QtGui.QStandardItem(i)
                item.setData(i)
                self.model.appendRow(item)

    def process(self):
        for index in range(self.model.rowCount()):
            item = self.model.item(index)
            name = os.path.basename(item.data())
            img = self.read_image(item.data())
            res, bboxes = detect.detect(nets, img.copy())
            for j in bboxes:
                x = j[0]
                y = j[1]
                w = j[2]
                h = j[3]
                cropped = img[y:y + h, x:x + w]
                cleaned = clean.remove(cropped)
                img[y: y + h, x: x + w] = cleaned
            print(self.dir_path + "/" + name)
            name, ext = os.path.splitext(name)
            self.save_image(img, self.dir_path, name, ext)
            item = QtGui.QStandardItem(self.dir_path + "/" + name + ext)
            item.setData(self.dir_path + "/" + name + ext)
            self.model_2.appendRow(item)
            self.progressBar.setValue(index * int(100 / self.model.rowCount()))
        self.progressBar.setValue(100)

    def select_saving_directory(self):
        self.dir_path = QFileDialog.getExistingDirectory(self, "Choose Directory")
        self.label.setText(self.dir_path)
        print(self.dir_path)

    def delete_btn(self):
        index = self.listView.currentIndex()
        if index.row() > 0:
            self.model.removeRow(index.row())

    def delete_all(self):
        self.model.clear()

def window():
    app = QtWidgets.QApplication(sys.argv)
    win = Ui()
    win.show()
    sys.exit(app.exec_())

window()