#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H
#include <QWindow>
#include <QWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>

class myWindow : public QWidget
{
    Q_OBJECT
private:
    QVBoxLayout* vertLayout1;
    QVBoxLayout* vertLayout2;
    QHBoxLayout* mainLayout;
    QLabel      *image;
    QPushButton *buttonTrain;
    QPushButton *buttonConfigureNet;
    QPushButton *buttonSetTrainingData;
    QPushButton *buttonChangeTrainingData;


public :
    myWindow(QWidget *parent=0);



public
    slots:


signals:


};
#endif
