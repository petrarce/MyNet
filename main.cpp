#include <QCoreApplication>
#include "GUI/MainWindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    myWindow *newWindow=new myWindow();

    newWindow->show();

	return a.exec();
}
