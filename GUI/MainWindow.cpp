//Implementation of MainWindow.h
#include "MainWindow.h"

myWindow::myWindow(QWidget *parent) //: QWidget(parent)
{
    this->resize(500,500);
    this->setLayout(this->mainLayout);

    this->mainLayout=new QHBoxLayout();
    mainLayout->addLayout(this->vertLayout1);
    mainLayout->addLayout(this->vertLayout2);

    this->vertLayout1=new QVBoxLayout();
    vertLayout1->addWidget(this->image);
    //vertLayout1->addWidget(this->fileList);
    this->image=new QLabel();

    this->vertLayout2=new QVBoxLayout();
    vertLayout2->addWidget(this->buttonTrain);
    vertLayout2->addWidget(this->buttonSetTrainingData);
    vertLayout2->addWidget(this->buttonConfigureNet);
    vertLayout2->addWidget(this->buttonChangeTrainingData);

    this->buttonTrain=new QPushButton();
    buttonTrain->setText("Train");

    this->buttonSetTrainingData=new QPushButton();
    buttonSetTrainingData->setText("Choose images for train");

    this->buttonConfigureNet=new QPushButton();
    buttonConfigureNet->setText("Configurate Net");

    this->buttonChangeTrainingData=new QPushButton();
    buttonChangeTrainingData->setText("Set Output for data");



}
