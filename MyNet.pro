QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = untitled
TEMPLATE = app


SOURCES += main.cpp \
    BackPropagation-master/src/Pattern.cpp \
    BackPropagation-master/src/Neuron.cpp \
    BackPropagation-master/src/Layer.cpp \
    BackPropagation-master/src/Backpropagation.cpp \
    Main.cpp \
    bolzman/boltzmann.cpp \
    GUI/MainWindow.cpp \

HEADERS += \
    BackPropagation-master/src/Sigmoid.h \
    BackPropagation-master/src/Pattern.h \
    BackPropagation-master/src/Neuron.h \
    BackPropagation-master/src/Layer.h \
    BackPropagation-master/src/Backpropagation.h \
    bolzman/bolzman.h \
    GUI/MainWindow.h \
    
