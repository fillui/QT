#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QStringList>
#include <QProcess>
#include <QListWidgetItem>
#include <QDateTime>
#include <QJsonArray>
#include <QJsonObject>
#include <QMap>
#include <QSqlDatabase>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    // 标注模块
    void openFolder();
    void loadImageAt(int row);
    void prevImage();
    void nextImage();
    void saveCurrentAnnotations();
    void refreshAnnotationList();
    void deleteSelectedAnnotation();
    void onOpenImageDir();
    void onAddLabel();
    void onDeleteLabel();
    void onDeleteClass();
    void onChangeAnnotationClass();
    void onUndoAnnotation();
    void onRedoAnnotation();
    void onZoomInImage();
    void onZoomOutImage();
    void onResetImageView();
    void onRotateLeftImage();
    void onRotateRightImage();
    void onGenerateDataYaml();
    void onSplitTrainValDataset();
    void onSaveAnnotation();
    void onClearAnnotation();
    void onImageItemClicked(QListWidgetItem *item);


    // 训练模块
    void onSelectDataset();
    void onSelectPythonExe();
    void onSelectScript();
    void onStartTraining();
    void onStopTraining();
    void onTrainReadyReadStdOut();
    void onTrainReadyReadStdErr();
    void onTrainFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onSelectQuantModel();
    void onSelectCalibDir();
    void onSelectQuantOutput();
    void onStartQuant();
    void onStopQuant();
    void onQuantReadyReadStdOut();
    void onQuantReadyReadStdErr();
    void onQuantFinished(int exitCode, QProcess::ExitStatus exitStatus);

    // 量化模型推理模块
    void onSelectInferPtModelPath();
    void onLoadInferPtModel();
    void onSelectInferOnnxModelPath();
    void onLoadInferOnnxModel();
    void onSelectInferImage();
    void onStartInfer();
private:
    // 标注模块
    void createToolBar();
    void populateImageList(const QStringList& imagePaths);
    void loadAnnotationsForCurrentImage();
    QString currentImagePath() const;
    QString currentImageDir;
    void loadClassList();
    void saveClassList();
    void addClassIfMissing(const QString &className);
    int annotationCountForImage(const QString &imagePath) const;
    void updateAnnotationSummaryStatus();
    QString datasetRootForYaml() const;
    QString relativePathForYaml(const QString &baseDir, const QString &path) const;
    QStringList collectClassNamesForYaml() const;
    bool generateDataYaml(bool showMessage = false);
    bool splitTrainValDataset(bool showMessage = true);
    bool initDatabase();
    void saveTrainingResultToDatabase(bool success, int exitCode);
    void updateTrainingResultDisplay(bool success,
                                     int exitCode,
                                     const QString &resultDir,
                                     const QString &bestModelPath,
                                     const QString &lastModelPath,
                                     const QMap<QString, double> &metrics);
    void resetTrainingResultDisplay();
    void updateTrainingProgressFromLogLine(const QString &line);
    void saveInferenceResultsToDatabase(const QJsonObject &root,
                                        const QJsonArray &detections,
                                        const QString &resultImagePath);
    // 训练模块
    void appendTrainLog(const QString& text);
    bool checkTrainInputs();
    QProcess *m_quantProcess = nullptr;
    QString m_quantBuffer;

    QString m_inferPtModelPath;
    QString m_inferOnnxModelPath;
    QString m_inferModelPath;      // 当前已加载、用于推理的模型路径
    QString m_inferModelType;      // 当前已加载模型类型：pt 或 onnx
    QString m_inferImagePath;
    bool m_inferModelLoaded = false;
    QSqlDatabase m_database;
    QDateTime m_trainStartTime;
    QString m_trainLogCache;
private:
    Ui::MainWindow *ui = nullptr;

    QString m_currentFolder;
    QStringList m_imagePaths;
    int m_currentIndex = -1;
    void appendQuantLog(const QString &text);
    void handleQuantLine(const QString &line);
    QProcess* m_trainProcess = nullptr;
};

#endif // MAINWINDOW_H