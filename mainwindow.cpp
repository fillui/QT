#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "annotation.h"
#include "annotationio.h"
#include "imagecanvas.h"
#include <QRegularExpression>
#include <QAction>
#include <QAbstractItemView>
#include <QBrush>
#include <QColor>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QInputDialog>
#include <QImage>
#include <QHeaderView>
#include <QLineEdit>
#include <QMap>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QProcessEnvironment>
#include <QProgressBar>
#include <QPushButton>
#include <QPixmap>
#include <QTableWidget>
#include <QSignalBlocker>
#include <QStatusBar>
#include <QToolBar>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QTableWidgetItem>
#include <QTextStream>
#include <QRegularExpression>
#include <QSqlDatabase>
#include <QSqlError>
#include <QSqlQuery>
#include <QSqlRecord>
#include <QVariant>
#include <algorithm>
#include <random>
#include <cmath>

// 根据图片路径推断可能的 YOLO 标注 txt 路径。
// 兼容两种常见结构：
// 1）图片和 txt 同目录：img_001.jpg / img_001.txt
// 2）YOLO 结构：images/.../img_001.jpg -> labels/.../img_001.txt
static QStringList annotationFileCandidatesForImage(const QString &imagePath)
{
    QStringList candidates;

    auto addCandidate = [&](const QString &path) {
        QString clean = QDir::cleanPath(path);
        if (!clean.isEmpty() && !candidates.contains(clean)) {
            candidates << clean;
        }
    };

    QFileInfo imgInfo(imagePath);
    const QString baseName = imgInfo.completeBaseName() + ".txt";

    // 情况 1：txt 和图片在同一个目录
    addCandidate(imgInfo.dir().filePath(baseName));

    // 情况 2：路径中包含 /images/，替换成 /labels/
    QString normalizedFile = QDir::fromNativeSeparators(imgInfo.absoluteFilePath());
    QString labelFile = normalizedFile;

    if (labelFile.contains("/images/")) {
        labelFile.replace("/images/", "/labels/");
        QFileInfo labelInfo(labelFile);
        addCandidate(QDir(labelInfo.absolutePath()).filePath(baseName));
    }

    // 情况 3：图片目录名就是 images，则找同级 labels 目录
    QString normalizedDir = QDir::fromNativeSeparators(imgInfo.absolutePath());
    if (normalizedDir.endsWith("/images")) {
        QDir parentDir(imgInfo.absolutePath());
        parentDir.cdUp();
        addCandidate(QDir(parentDir.filePath("labels")).filePath(baseName));
    }

    return candidates;
}

static bool removeExistingAnnotationFiles(const QString &imagePath,
                                          QStringList *removedFiles,
                                          QString *errorMessage)
{
    const QStringList candidates = annotationFileCandidatesForImage(imagePath);

    bool ok = true;

    for (const QString &path : candidates) {
        if (!QFile::exists(path)) {
            continue;
        }

        if (QFile::remove(path)) {
            if (removedFiles) {
                removedFiles->append(path);
            }
        } else {
            ok = false;
            if (errorMessage) {
                *errorMessage += "无法删除标注文件：" + path + "\n";
            }
        }
    }

    return ok;
}
static QString classListFilePath()
{
    return QCoreApplication::applicationDirPath() + "/classes.txt";
}

// 根据 train.py 的位置推断工程根目录。
// 推荐结构：工程根目录/python/train.py，所以从 python/ 返回上一级。
static QString projectRootFromScriptPath(const QString &scriptPath)
{
    QFileInfo scriptInfo(scriptPath);

    if (scriptInfo.exists()) {
        QDir dir = scriptInfo.absoluteDir();

        if (dir.dirName().compare("python", Qt::CaseInsensitive) == 0) {
            if (dir.cdUp()) {
                return QDir::cleanPath(dir.absolutePath());
            }
        }

        return QDir::cleanPath(scriptInfo.absoluteDir().absolutePath());
    }

    return QDir::cleanPath(QCoreApplication::applicationDirPath());
}


static QVariant nullIfEmpty(const QString &value)
{
    const QString trimmed = value.trimmed();
    if (trimmed.isEmpty()) {
        return QVariant();
    }
    return trimmed;
}

static bool sqliteTableHasColumn(QSqlDatabase &db, const QString &tableName, const QString &columnName)
{
    QSqlQuery query(db);
    if (!query.exec("PRAGMA table_info(" + tableName + ")")) {
        return false;
    }

    while (query.next()) {
        if (query.value(1).toString().compare(columnName, Qt::CaseInsensitive) == 0) {
            return true;
        }
    }
    return false;
}

static bool sqliteAddColumnIfMissing(QSqlDatabase &db,
                                     const QString &tableName,
                                     const QString &columnName,
                                     const QString &columnDefinition,
                                     QString *errorMessage = nullptr)
{
    if (sqliteTableHasColumn(db, tableName, columnName)) {
        return true;
    }

    QSqlQuery query(db);
    const QString sql = "ALTER TABLE " + tableName + " ADD COLUMN " + columnName + " " + columnDefinition;
    if (!query.exec(sql)) {
        if (errorMessage) {
            *errorMessage = query.lastError().text();
        }
        return false;
    }
    return true;
}

static QString newestFileByName(const QStringList &rootDirs,
                                const QString &targetName,
                                const QDateTime &notBefore = QDateTime())
{
    QString newestPath;
    QDateTime newestTime;

    for (const QString &root : rootDirs) {
        if (root.trimmed().isEmpty() || !QDir(root).exists()) {
            continue;
        }

        QDirIterator it(root,
                        QStringList() << targetName,
                        QDir::Files,
                        QDirIterator::Subdirectories);

        while (it.hasNext()) {
            const QString filePath = it.next();
            QFileInfo info(filePath);
            const QDateTime modified = info.lastModified();

            if (notBefore.isValid() && modified < notBefore.addSecs(-60)) {
                continue;
            }

            if (newestPath.isEmpty() || modified > newestTime) {
                newestPath = info.absoluteFilePath();
                newestTime = modified;
            }
        }
    }

    return QDir::cleanPath(newestPath);
}

static QMap<QString, double> readLastYoloMetricsFromCsv(const QString &csvPath)
{
    QMap<QString, double> metrics;

    QFile file(csvPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return metrics;
    }

    QTextStream in(&file);
    QString headerLine;
    QString lastLine;

    if (!in.atEnd()) {
        headerLine = in.readLine();
    }

    while (!in.atEnd()) {
        const QString line = in.readLine().trimmed();
        if (!line.isEmpty()) {
            lastLine = line;
        }
    }

    const QStringList headers = headerLine.split(',', Qt::KeepEmptyParts);
    const QStringList values = lastLine.split(',', Qt::KeepEmptyParts);

    auto valueByHeader = [&](const QStringList &nameCandidates) -> QVariant {
        for (int i = 0; i < headers.size() && i < values.size(); ++i) {
            const QString h = headers.at(i).trimmed();
            for (const QString &candidate : nameCandidates) {
                if (h == candidate || h.endsWith(candidate)) {
                    bool ok = false;
                    const double v = values.at(i).trimmed().toDouble(&ok);
                    if (ok) {
                        return v;
                    }
                }
            }
        }
        return QVariant();
    };

    const QVariant precision = valueByHeader(QStringList() << "metrics/precision(B)" << "precision(B)" << "precision");
    const QVariant recall = valueByHeader(QStringList() << "metrics/recall(B)" << "recall(B)" << "recall");
    const QVariant map50 = valueByHeader(QStringList() << "metrics/mAP50(B)" << "mAP50(B)" << "mAP50");
    const QVariant map5095 = valueByHeader(QStringList() << "metrics/mAP50-95(B)" << "mAP50-95(B)" << "mAP50-95");

    if (precision.isValid()) metrics.insert("precision", precision.toDouble());
    if (recall.isValid()) metrics.insert("recall", recall.toDouble());
    if (map50.isValid()) metrics.insert("map50", map50.toDouble());
    if (map5095.isValid()) metrics.insert("map5095", map5095.toDouble());

    return metrics;
}


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , m_trainProcess(new QProcess(this))
    , m_quantProcess(new QProcess(this))
{
    ui->setupUi(this);
    initDatabase();
    resetTrainingResultDisplay();

    // 类别列表不再固定使用 ui 文件中的 person/vehicle/object，
    // 而是从 classes.txt 读取；没有文件时默认保留 object。
    loadClassList();

    // 类别下拉框允许手动输入，也允许从已有类别列表中选择。
    ui->comboBoxClass->setEditable(true);
    ui->comboBoxClass->setInsertPolicy(QComboBox::NoInsert);
    ui->comboBoxClass->setDuplicatesEnabled(false);

    createToolBar();
    // ===== 量化模块 =====
    connect(ui->btnSelectModelPath, &QPushButton::clicked,
            this, &MainWindow::onSelectQuantModel);

    connect(ui->btnSelectCalibDir, &QPushButton::clicked,
            this, &MainWindow::onSelectCalibDir);

    connect(ui->btnSelectOutputPath, &QPushButton::clicked,
            this, &MainWindow::onSelectQuantOutput);

    connect(ui->btnStartQuant, &QPushButton::clicked,
            this, &MainWindow::onStartQuant);

    connect(ui->btnStopQuant, &QPushButton::clicked,
            this, &MainWindow::onStopQuant);

    connect(m_quantProcess, &QProcess::readyReadStandardOutput,
            this, &MainWindow::onQuantReadyReadStdOut);

    connect(m_quantProcess, &QProcess::readyReadStandardError,
            this, &MainWindow::onQuantReadyReadStdErr);

    connect(m_quantProcess,
            QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this,
            &MainWindow::onQuantFinished);

    ui->btnStopQuant->setEnabled(false);
    ui->progressBarQuant->setValue(0);

    // ===== 量化模型推理模块 =====
    connect(ui->btnSelectInferPtModelPath, &QPushButton::clicked,
            this, &MainWindow::onSelectInferPtModelPath);

    connect(ui->btnLoadInferPtModel, &QPushButton::clicked,
            this, &MainWindow::onLoadInferPtModel);

    connect(ui->btnSelectInferOnnxModelPath, &QPushButton::clicked,
            this, &MainWindow::onSelectInferOnnxModelPath);

    connect(ui->btnLoadInferOnnxModel, &QPushButton::clicked,
            this, &MainWindow::onLoadInferOnnxModel);

    connect(ui->btnSelectInferImage, &QPushButton::clicked,
            this, &MainWindow::onSelectInferImage);

    connect(ui->btnStartInfer, &QPushButton::clicked,
            this, &MainWindow::onStartInfer);

    ui->labelInferStatus->setText("推理状态：未加载模型");
    ui->tableWidgetResult->setColumnCount(7);
    ui->tableWidgetResult->setHorizontalHeaderLabels(
        QStringList() << "序号" << "类别" << "置信度" << "x" << "y" << "w" << "h"
        );
    ui->tableWidgetResult->horizontalHeader()->setStretchLastSection(true);
    ui->tableWidgetResult->setEditTriggers(QAbstractItemView::NoEditTriggers);
    ui->tableWidgetResult->setSelectionBehavior(QAbstractItemView::SelectRows);
    ui->tableWidgetResult->setRowCount(0);

    // ===================== 标注模块连接 =====================
    connect(ui->btnOpenImageDir, &QPushButton::clicked,
            this, &MainWindow::onOpenImageDir);

    connect(ui->btnAddLabel, &QPushButton::clicked,
            this, &MainWindow::onAddLabel);

    connect(ui->btnDeleteLabel, &QPushButton::clicked,
            this, &MainWindow::onDeleteLabel);

    connect(ui->btnDeleteClass, &QPushButton::clicked,
            this, &MainWindow::onDeleteClass);

    connect(ui->btnChangeAnnotationClass, &QPushButton::clicked,
            this, &MainWindow::onChangeAnnotationClass);

    connect(ui->btnUndoAnnotation, &QPushButton::clicked,
            this, &MainWindow::onUndoAnnotation);

    connect(ui->btnRedoAnnotation, &QPushButton::clicked,
            this, &MainWindow::onRedoAnnotation);

    connect(ui->btnSaveAnnotation, &QPushButton::clicked,
            this, &MainWindow::onSaveAnnotation);

    connect(ui->btnClearAnnotation, &QPushButton::clicked,
            this, &MainWindow::onClearAnnotation);

    connect(ui->imageListWidget, &QListWidget::itemClicked,
            this, &MainWindow::onImageItemClicked);

    // comboBoxClass 只负责类别选择，不再使用 annotationListWidget 当类别列表
    if (ui->comboBoxClass->count() > 0) {
        ui->canvasWidget->setCurrentLabel(ui->comboBoxClass->currentText());
    }

    connect(ui->comboBoxClass, &QComboBox::currentTextChanged,
            this, [=](const QString &text) {
                ui->canvasWidget->setCurrentLabel(text.trimmed());
                statusBar()->showMessage("当前标注类别：" + text.trimmed(), 1500);
            });

    connect(ui->comboBoxClass->lineEdit(), &QLineEdit::editingFinished,
            this, [=]() {
                const QString label = ui->comboBoxClass->currentText().trimmed();
                if (label.isEmpty()) {
                    return;
                }
                addClassIfMissing(label);
                ui->comboBoxClass->setCurrentText(label);
                ui->canvasWidget->setCurrentLabel(label);
                saveClassList();
            });

    // annotationListWidget 只显示当前图片的所有标注框；点击某一行，选中对应框，
    // 同时把下拉框切换为该框的类别，便于直接点“更改类别”。
    connect(ui->annotationListWidget, &QListWidget::currentRowChanged,
            this, [=](int row) {
                ui->canvasWidget->selectAnnotation(row);

                if (row >= 0 && row < ui->canvasWidget->annotations.size()) {
                    const QString label = ui->canvasWidget->annotations[row].label;
                    addClassIfMissing(label);
                    ui->comboBoxClass->setCurrentText(label);
                }
            });

    // 画框、删除、清空后自动刷新右侧标注列表
    connect(ui->canvasWidget, &ImageCanvas::annotationsChanged,
            this, &MainWindow::refreshAnnotationList);

    // ===================== 训练模块连接 =====================
    connect(ui->btnSelectDataset, &QPushButton::clicked,
            this, &MainWindow::onSelectDataset);

    connect(ui->btnSelectPythonExe, &QPushButton::clicked,
            this, &MainWindow::onSelectPythonExe);

    connect(ui->btnSelectScript, &QPushButton::clicked,
            this, &MainWindow::onSelectScript);

    connect(ui->btnStartTrain, &QPushButton::clicked,
            this, &MainWindow::onStartTraining);

    connect(ui->btnStopTrain, &QPushButton::clicked,
            this, &MainWindow::onStopTraining);

    connect(m_trainProcess, &QProcess::readyReadStandardOutput,
            this, &MainWindow::onTrainReadyReadStdOut);

    connect(m_trainProcess, &QProcess::readyReadStandardError,
            this, &MainWindow::onTrainReadyReadStdErr);

    connect(m_trainProcess,
            QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this,
            &MainWindow::onTrainFinished);

    // ===================== 初始化 =====================
    ui->plainTextEditLog->setReadOnly(true);
    ui->labelTrainStatus->setText("状态：未开始");
    ui->btnStopTrain->setEnabled(false);

    ui->annotationListWidget->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    ui->annotationListWidget->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);

    if (ui->lineEditPythonExe->text().trimmed().isEmpty()) {
        ui->lineEditPythonExe->setText("python");
    }

    if (ui->lineEditScript->text().trimmed().isEmpty()) {
        ui->lineEditScript->setText(QCoreApplication::applicationDirPath() + "/python/train.py");
    }

    statusBar()->showMessage("就绪");
}


MainWindow::~MainWindow()
{
    saveCurrentAnnotations();

    if (m_trainProcess && m_trainProcess->state() != QProcess::NotRunning) {
        m_trainProcess->kill();
        m_trainProcess->waitForFinished(3000);
    }

    delete ui;
}


// ===================== 顶部/界面按钮：标注相关 =====================

void MainWindow::onAddLabel()
{
    bool ok = false;

    QString labelName = QInputDialog::getText(
                            this,
                            "新增类别",
                            "请输入类别名称：",
                            QLineEdit::Normal,
                            "",
                            &ok
                        ).trimmed();

    if (!ok || labelName.isEmpty()) {
        return;
    }

    // 已存在则直接切换
    for (int i = 0; i < ui->comboBoxClass->count(); ++i) {
        if (ui->comboBoxClass->itemText(i) == labelName) {
            ui->comboBoxClass->setCurrentIndex(i);
            ui->canvasWidget->setCurrentLabel(labelName);
            statusBar()->showMessage("类别已存在，已切换为：" + labelName, 1500);
            return;
        }
    }

    ui->comboBoxClass->addItem(labelName);
    ui->comboBoxClass->setCurrentText(labelName);
    ui->canvasWidget->setCurrentLabel(labelName);
    saveClassList();
    generateDataYaml(false);

    statusBar()->showMessage("已新增类别：" + labelName, 1500);
}


void MainWindow::onOpenImageDir()
{
    openFolder();
}


void MainWindow::onDeleteLabel()
{
    deleteSelectedAnnotation();
}


void MainWindow::onDeleteClass()
{
    const int index = ui->comboBoxClass->currentIndex();

    if (index < 0) {
        QMessageBox::information(this, "提示", "当前没有可删除的类别。");
        return;
    }

    if (ui->comboBoxClass->count() <= 1) {
        QMessageBox::information(this, "提示", "至少需要保留一个类别，不能继续删除。");
        return;
    }

    const QString className = ui->comboBoxClass->currentText().trimmed();

    int usedCount = 0;
    for (const Annotation &ann : ui->canvasWidget->annotations) {
        if (ann.label == className) {
            ++usedCount;
        }
    }

    QString message = "确定要从类别列表中删除：" + className + " 吗？";
    if (usedCount > 0) {
        message += QString("\n\n当前图片中有 %1 个标注框正在使用该类别。"
                           "\n删除类别只会从下拉框移除它，不会删除已经画好的框。")
                       .arg(usedCount);
    }

    const int ret = QMessageBox::question(
        this,
        "删除类别",
        message,
        QMessageBox::Yes | QMessageBox::No
    );

    if (ret != QMessageBox::Yes) {
        return;
    }

    ui->comboBoxClass->removeItem(index);
    saveClassList();
    generateDataYaml(false);

    if (ui->comboBoxClass->count() > 0) {
        const QString newClass = ui->comboBoxClass->currentText();
        ui->canvasWidget->setCurrentLabel(newClass);
        statusBar()->showMessage("已删除类别：" + className + "，当前类别：" + newClass, 2000);
    }
}


void MainWindow::onChangeAnnotationClass()
{
    int row = ui->annotationListWidget->currentRow();

    if (row >= 0 && row < ui->canvasWidget->annotations.size()) {
        ui->canvasWidget->selectAnnotation(row);
    }

    if (ui->canvasWidget->selectedIndex < 0 ||
        ui->canvasWidget->selectedIndex >= ui->canvasWidget->annotations.size()) {
        QMessageBox::information(this, "提示", "请先在标注列表或画布中选择要修改类别的标注框。");
        return;
    }

    const int selected = ui->canvasWidget->selectedIndex;
    const QString oldLabel = ui->canvasWidget->annotations[selected].label.trimmed();

    // 下拉候选来源：
    // 1）类别下拉框里已有的类别；
    // 2）当前图片标注列表里已经出现过的类别；
    // 3）当前选中框的原类别。
    QStringList choices;
    auto addChoice = [&](const QString &label) {
        const QString t = label.trimmed();
        if (!t.isEmpty() && !choices.contains(t)) {
            choices << t;
        }
    };

    for (int i = 0; i < ui->comboBoxClass->count(); ++i) {
        addChoice(ui->comboBoxClass->itemText(i));
    }

    for (const Annotation &ann : ui->canvasWidget->annotations) {
        addChoice(ann.label);
    }

    addChoice(oldLabel);

    int currentIndex = choices.indexOf(oldLabel);
    if (currentIndex < 0) {
        currentIndex = 0;
    }

    bool ok = false;
    const QString newLabel = QInputDialog::getItem(
                                 this,
                                 "更改类别",
                                 "请选择或输入新的类别名称：",
                                 choices,
                                 currentIndex,
                                 true,   // true 表示既可以下拉选择，也可以手动输入
                                 &ok
                             ).trimmed();

    if (!ok) {
        return;
    }

    if (newLabel.isEmpty()) {
        QMessageBox::information(this, "提示", "类别名称不能为空。");
        return;
    }

    if (newLabel == oldLabel) {
        statusBar()->showMessage("类别未改变。", 1500);
        return;
    }

    // 手动输入的新类别如果不存在，则自动加入类别下拉框并保存到 classes.txt。
    addClassIfMissing(newLabel);
    ui->comboBoxClass->setCurrentText(newLabel);
    ui->canvasWidget->setCurrentLabel(newLabel);
    saveClassList();
    generateDataYaml(false);

    ui->canvasWidget->changeSelectedAnnotationLabel(newLabel);
    refreshAnnotationList();
    saveCurrentAnnotations();
    statusBar()->showMessage("已将选中标注框类别由 " + oldLabel + " 改为：" + newLabel, 2000);
}

void MainWindow::onUndoAnnotation()
{
    ui->canvasWidget->undo();
    refreshAnnotationList();
    saveCurrentAnnotations();
}


void MainWindow::onRedoAnnotation()
{
    ui->canvasWidget->redo();
    refreshAnnotationList();
    saveCurrentAnnotations();
}

void MainWindow::onSaveAnnotation()
{
    saveCurrentAnnotations();
    refreshAnnotationList();
}


void MainWindow::onClearAnnotation()
{
    if (ui->canvasWidget->annotations.isEmpty()) {
        saveCurrentAnnotations();
        refreshAnnotationList();
        return;
    }

    int ret = QMessageBox::question(
        this,
        "确认清空",
        "确定要清空当前图片的所有标注吗？\n本地 txt 标注文件也会同步删除。",
        QMessageBox::Yes | QMessageBox::No
    );

    if (ret != QMessageBox::Yes) {
        return;
    }

    ui->canvasWidget->clearAnnotations();
    refreshAnnotationList();
    saveCurrentAnnotations();
}


void MainWindow::onImageItemClicked(QListWidgetItem *item)
{
    if (!item) {
        return;
    }

    int row = ui->imageListWidget->row(item);
    loadImageAt(row);
}



void MainWindow::loadClassList()
{
    QStringList classes;

    QFile file(classListFilePath());
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream in(&file);
        while (!in.atEnd()) {
            const QString line = in.readLine().trimmed();
            if (!line.isEmpty() && !classes.contains(line)) {
                classes << line;
            }
        }
    }

    // 不再固定加载 ui 文件中的 person/vehicle/object。
    // 如果 classes.txt 还没有生成，则保留一个 object，避免没有类别无法画框。
    if (classes.isEmpty()) {
        classes << "object";
    }

    QSignalBlocker blocker(ui->comboBoxClass);
    ui->comboBoxClass->clear();
    ui->comboBoxClass->addItems(classes);

    ui->canvasWidget->setCurrentLabel(ui->comboBoxClass->currentText());
}


void MainWindow::saveClassList()
{
    QFile file(classListFilePath());

    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        statusBar()->showMessage("类别文件保存失败：" + classListFilePath(), 3000);
        return;
    }

    QTextStream out(&file);

    for (int i = 0; i < ui->comboBoxClass->count(); ++i) {
        const QString className = ui->comboBoxClass->itemText(i).trimmed();
        if (!className.isEmpty()) {
            out << className << "\n";
        }
    }
}


void MainWindow::addClassIfMissing(const QString &className)
{
    const QString t = className.trimmed();

    if (t.isEmpty()) {
        return;
    }

    if (ui->comboBoxClass->findText(t) < 0) {
        ui->comboBoxClass->addItem(t);
    }
}


// ===================== 标注模块核心逻辑 =====================

void MainWindow::createToolBar()
{
    QToolBar* toolbar = addToolBar("工具栏");

    QAction* openAction = toolbar->addAction("打开文件夹");
    QAction* prevAction = toolbar->addAction("上一张");
    QAction* nextAction = toolbar->addAction("下一张");
    QAction* saveAction = toolbar->addAction("保存");
    QAction* deleteAction = toolbar->addAction("删除框");
    QAction* changeClassAction = toolbar->addAction("更改类别");
    QAction* undoAction = toolbar->addAction("撤销");
    QAction* redoAction = toolbar->addAction("重做");
    QAction* yamlAction = toolbar->addAction("生成YAML");
    QAction* splitAction = toolbar->addAction("划分训练/验证集");

    toolbar->addSeparator();
    QAction* zoomInAction = toolbar->addAction("放大");
    QAction* zoomOutAction = toolbar->addAction("缩小");
    QAction* resetViewAction = toolbar->addAction("复位视图");
    QAction* rotateLeftAction = toolbar->addAction("左旋转");
    QAction* rotateRightAction = toolbar->addAction("右旋转");

    connect(openAction, &QAction::triggered, this, &MainWindow::openFolder);
    connect(prevAction, &QAction::triggered, this, &MainWindow::prevImage);
    connect(nextAction, &QAction::triggered, this, &MainWindow::nextImage);
    connect(saveAction, &QAction::triggered, this, &MainWindow::saveCurrentAnnotations);
    connect(deleteAction, &QAction::triggered, this, &MainWindow::deleteSelectedAnnotation);
    connect(changeClassAction, &QAction::triggered, this, &MainWindow::onChangeAnnotationClass);
    connect(undoAction, &QAction::triggered, this, &MainWindow::onUndoAnnotation);
    connect(redoAction, &QAction::triggered, this, &MainWindow::onRedoAnnotation);
    connect(yamlAction, &QAction::triggered, this, &MainWindow::onGenerateDataYaml);
    connect(splitAction, &QAction::triggered, this, &MainWindow::onSplitTrainValDataset);
    connect(zoomInAction, &QAction::triggered, this, &MainWindow::onZoomInImage);
    connect(zoomOutAction, &QAction::triggered, this, &MainWindow::onZoomOutImage);
    connect(resetViewAction, &QAction::triggered, this, &MainWindow::onResetImageView);
    connect(rotateLeftAction, &QAction::triggered, this, &MainWindow::onRotateLeftImage);
    connect(rotateRightAction, &QAction::triggered, this, &MainWindow::onRotateRightImage);
}


void MainWindow::onZoomInImage()
{
    ui->canvasWidget->zoomIn();
    statusBar()->showMessage("图像已放大。也可以滚动鼠标滚轮缩放。", 1500);
}


void MainWindow::onZoomOutImage()
{
    ui->canvasWidget->zoomOut();
    statusBar()->showMessage("图像已缩小。也可以滚动鼠标滚轮缩放。", 1500);
}


void MainWindow::onResetImageView()
{
    ui->canvasWidget->resetView();
    statusBar()->showMessage("图像视图已复位。", 1500);
}


void MainWindow::onRotateLeftImage()
{
    ui->canvasWidget->rotateLeft();
    statusBar()->showMessage("图像已向左旋转 90°。", 1500);
}


void MainWindow::onRotateRightImage()
{
    ui->canvasWidget->rotateRight();
    statusBar()->showMessage("图像已向右旋转 90°。", 1500);
}


void MainWindow::onGenerateDataYaml()
{
    if (!generateDataYaml(true)) {
        QMessageBox::warning(this, "生成失败", "data.yaml 生成失败，请先打开图片文件夹并至少保留一个类别。行内标注 txt 仍然已保存。 ");
    }
}


void MainWindow::onSplitTrainValDataset()
{
    splitTrainValDataset(true);
}


void MainWindow::openFolder()
{
    QString folder = QFileDialog::getExistingDirectory(this, "选择图片文件夹");

    if (folder.isEmpty()) {
        return;
    }

    saveCurrentAnnotations();
    m_currentFolder = folder;

    QStringList filters;
    filters << "*.jpg" << "*.jpeg" << "*.png" << "*.bmp";

    QStringList fullPaths;
    QDirIterator it(folder, filters, QDir::Files, QDirIterator::Subdirectories);

    while (it.hasNext()) {
        fullPaths << it.next();
    }

    fullPaths.sort();

    if (fullPaths.isEmpty()) {
        QMessageBox::information(this, "提示", "该文件夹及其子文件夹下没有图片文件。");

        m_imagePaths.clear();
        m_currentIndex = -1;

        ui->imageListWidget->clear();
        ui->annotationListWidget->clear();
        ui->canvasWidget->clearImage();

        return;
    }

    populateImageList(fullPaths);
    loadImageAt(0);
    generateDataYaml(false);
}


void MainWindow::populateImageList(const QStringList& imagePaths)
{
    m_imagePaths = imagePaths;
    m_currentIndex = -1;

    QSignalBlocker blocker(ui->imageListWidget);
    ui->imageListWidget->clear();

    for (const QString& path : m_imagePaths) {
        QFileInfo info(path);

        QListWidgetItem *item = new QListWidgetItem(info.fileName());
        item->setData(Qt::UserRole, path);

        ui->imageListWidget->addItem(item);
    }
}


QString MainWindow::currentImagePath() const
{
    if (m_currentIndex < 0 || m_currentIndex >= m_imagePaths.size()) {
        return QString();
    }

    return m_imagePaths[m_currentIndex];
}


void MainWindow::loadImageAt(int row)
{
    if (row < 0 || row >= m_imagePaths.size()) {
        return;
    }

    if (m_currentIndex != -1 && m_currentIndex != row) {
        saveCurrentAnnotations();
    }

    m_currentIndex = row;

    {
        QSignalBlocker blocker(ui->imageListWidget);
        ui->imageListWidget->setCurrentRow(row);
    }

    const QString imagePath = m_imagePaths[row];

    ui->canvasWidget->setImage(imagePath);
    loadAnnotationsForCurrentImage();

    QFileInfo info(imagePath);
    statusBar()->showMessage(QString("当前图片：%1").arg(info.fileName()));
}


void MainWindow::loadAnnotationsForCurrentImage()
{
    const QString imagePath = currentImagePath();

    ui->canvasWidget->clearAnnotations(false);

    if (imagePath.isEmpty()) {
        refreshAnnotationList();
        return;
    }

    QString err;
    QVector<Annotation> anns;

    if (!AnnotationIO::load(imagePath, anns, &err)) {
        QMessageBox::warning(this, "读取标注失败", err);
        refreshAnnotationList();
        return;
    }

    bool classListChanged = false;
    for (const Annotation &ann : anns) {
        const int before = ui->comboBoxClass->count();
        addClassIfMissing(ann.label);
        if (ui->comboBoxClass->count() != before) {
            classListChanged = true;
        }
    }

    if (classListChanged) {
        saveClassList();
    }

    ui->canvasWidget->setAnnotations(anns);
    refreshAnnotationList();
}


void MainWindow::saveCurrentAnnotations()
{
    const QString imagePath = currentImagePath();

    if (imagePath.isEmpty()) {
        return;
    }

    const QVector<Annotation> anns = ui->canvasWidget->annotations;

    // 关键修复：如果当前图片已经没有标注，则删除本地 txt 文件。
    // 之前只清空了内存中的标注框，没有删除硬盘上的标注文件。
    if (anns.isEmpty()) {
        QStringList removedFiles;
        QString err;

        if (!removeExistingAnnotationFiles(imagePath, &removedFiles, &err)) {
            QMessageBox::warning(this, "删除失败", err.trimmed());
            return;
        }

        if (removedFiles.isEmpty()) {
            statusBar()->showMessage("当前图片无标注，本地标注文件不存在", 2000);
        } else {
            statusBar()->showMessage("已删除本地标注文件：" + QFileInfo(removedFiles.first()).fileName(), 2000);
        }

        generateDataYaml(false);
        return;
    }

    QString err;

    if (!AnnotationIO::save(imagePath, anns, &err)) {
        QMessageBox::warning(this, "保存失败", err);
        return;
    }

    QFileInfo info(imagePath);
    generateDataYaml(false);
    statusBar()->showMessage(QString("已保存：%1；data.yaml 已同步更新").arg(info.fileName()), 2000);
}


int MainWindow::annotationCountForImage(const QString &imagePath) const
{
    if (imagePath.isEmpty()) {
        return 0;
    }

    // 当前正在显示的图片以界面内存为准，因为它可能刚修改但尚未写入磁盘。
    if (imagePath == currentImagePath()) {
        return ui->canvasWidget->annotations.size();
    }

    QVector<Annotation> anns;
    QString err;
    if (!AnnotationIO::load(imagePath, anns, &err)) {
        return 0;
    }

    return anns.size();
}


void MainWindow::updateAnnotationSummaryStatus()
{
    const int currentCount = ui->canvasWidget->annotations.size();

    int labeledImageCount = 0;
    int totalBoxCount = 0;

    for (const QString &path : m_imagePaths) {
        const int count = annotationCountForImage(path);
        if (count > 0) {
            ++labeledImageCount;
            totalBoxCount += count;
        }
    }

    if (m_imagePaths.isEmpty()) {
        statusBar()->showMessage(QString("当前图片标注：%1").arg(currentCount), 2000);
        return;
    }

    statusBar()->showMessage(
        QString("当前图片标注：%1；已标注图片：%2/%3；标注框总数：%4")
            .arg(currentCount)
            .arg(labeledImageCount)
            .arg(m_imagePaths.size())
            .arg(totalBoxCount),
        3000
    );
}



QString MainWindow::datasetRootForYaml() const
{
    QString folder = !m_currentFolder.isEmpty() ? m_currentFolder : QFileInfo(currentImagePath()).absolutePath();
    if (folder.isEmpty()) {
        return QString();
    }

    QString normalized = QDir::fromNativeSeparators(QDir::cleanPath(folder));

    const QStringList markers = {
        "/images/train", "/images/val", "/images/test",
        "/train/images", "/val/images", "/test/images"
    };

    for (const QString &marker : markers) {
        const int pos = normalized.indexOf(marker, 0, Qt::CaseInsensitive);
        if (pos >= 0) {
            return QDir::cleanPath(normalized.left(pos));
        }
    }

    if (normalized.endsWith("/images", Qt::CaseInsensitive) ||
        normalized.endsWith("/labels", Qt::CaseInsensitive)) {
        QDir dir(normalized);
        dir.cdUp();
        return QDir::cleanPath(dir.absolutePath());
    }

    return QDir::cleanPath(normalized);
}


QString MainWindow::relativePathForYaml(const QString &baseDir, const QString &path) const
{
    QString rel = QDir(baseDir).relativeFilePath(path);
    rel = QDir::fromNativeSeparators(rel);
    if (rel.isEmpty()) {
        return ".";
    }
    return rel;
}


QStringList MainWindow::collectClassNamesForYaml() const
{
    QStringList classes;

    auto addClass = [&](const QString &name) {
        QString t = name.trimmed();
        t.replace(' ', '_');
        if (!t.isEmpty() && !classes.contains(t)) {
            classes << t;
        }
    };

    for (int i = 0; i < ui->comboBoxClass->count(); ++i) {
        addClass(ui->comboBoxClass->itemText(i));
    }

    for (const QString &imagePath : m_imagePaths) {
        QVector<Annotation> anns;
        QString err;
        if (!AnnotationIO::load(imagePath, anns, &err)) {
            continue;
        }
        for (const Annotation &ann : anns) {
            addClass(ann.label);
        }
    }

    for (const Annotation &ann : ui->canvasWidget->annotations) {
        addClass(ann.label);
    }

    if (classes.isEmpty()) {
        classes << "object";
    }

    return classes;
}


bool MainWindow::generateDataYaml(bool showMessage)
{
    const QString rootDir = datasetRootForYaml();
    if (rootDir.isEmpty()) {
        return false;
    }

    QDir root(rootDir);
    if (!root.exists()) {
        return false;
    }

    QString trainRel;
    QString valRel;

    if (root.exists("images/train")) {
        trainRel = "images/train";
    } else if (root.exists("train/images")) {
        trainRel = "train/images";
    } else if (!m_currentFolder.isEmpty()) {
        trainRel = relativePathForYaml(rootDir, m_currentFolder);
    } else {
        trainRel = ".";
    }

    if (root.exists("images/val")) {
        valRel = "images/val";
    } else if (root.exists("valid/images")) {
        valRel = "valid/images";
    } else if (root.exists("val/images")) {
        valRel = "val/images";
    } else {
        // 演示项目没有单独验证集时，先让 val 与 train 相同，保证 YOLO 可以启动。
        // 后续正式训练时建议单独划分 images/val。
        valRel = trainRel;
    }

    const QString yamlPath = root.filePath("data.yaml");
    QFile file(yamlPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        if (showMessage) {
            QMessageBox::warning(this, "生成失败", "无法写入：" + yamlPath);
        }
        return false;
    }

    const QStringList classes = collectClassNamesForYaml();

    QTextStream out(&file);
    out << "path: " << QDir::fromNativeSeparators(root.absolutePath()) << "\n";
    out << "train: " << trainRel << "\n";
    out << "val: " << valRel << "\n";
    out << "nc: " << classes.size() << "\n";
    out << "names:\n";
    for (int i = 0; i < classes.size(); ++i) {
        out << "  " << i << ": " << classes[i] << "\n";
    }
    file.close();

    ui->lineEditDataset->setText(QDir::toNativeSeparators(yamlPath));

    if (showMessage) {
        QMessageBox::information(this, "生成成功", "已生成 data.yaml：\n" + QDir::toNativeSeparators(yamlPath));
    }

    return true;
}


bool MainWindow::splitTrainValDataset(bool showMessage)
{
    if (m_currentFolder.isEmpty() || m_imagePaths.isEmpty()) {
        if (showMessage) {
            QMessageBox::information(this, "提示", "请先在“数据标注”页打开包含图片和 txt 标注的原始数据集文件夹。");
        }
        return false;
    }

    saveCurrentAnnotations();

    bool ok = false;
    const int trainPercent = QInputDialog::getInt(
        this,
        "划分训练集和验证集",
        "请输入训练集比例（验证集比例 = 100 - 训练集比例）：",
        80,
        50,
        95,
        1,
        &ok
    );
    if (!ok) {
        return false;
    }

    const QString defaultOut = QDir(m_currentFolder).filePath("yolo_dataset");
    const QString outDir = QFileDialog::getExistingDirectory(
        this,
        "选择 YOLO 数据集输出文件夹",
        defaultOut
    );
    if (outDir.isEmpty()) {
        return false;
    }

    QDir out(outDir);
    out.mkpath("images/train");
    out.mkpath("images/val");
    out.mkpath("labels/train");
    out.mkpath("labels/val");

    QStringList images = m_imagePaths;
    images.removeDuplicates();
    std::sort(images.begin(), images.end());

    std::mt19937 rng(static_cast<unsigned int>(QDateTime::currentMSecsSinceEpoch() & 0xffffffff));
    std::shuffle(images.begin(), images.end(), rng);

    const int trainCount = std::max(1, static_cast<int>(std::round(images.size() * trainPercent / 100.0)));

    QStringList classes = collectClassNamesForYaml();
    if (classes.isEmpty()) {
        classes << "object";
    }

    auto normalizeClass = [](QString name) {
        name = name.trimmed();
        name.replace(' ', '_');
        return name;
    };

    QMap<QString, int> classToId;
    for (int i = 0; i < classes.size(); ++i) {
        classToId.insert(normalizeClass(classes[i]), i);
    }

    int copiedTrain = 0;
    int copiedVal = 0;
    int boxTotal = 0;

    auto copyFileOverwrite = [](const QString &src, const QString &dst) -> bool {
        QFileInfo dstInfo(dst);
        QDir().mkpath(dstInfo.absolutePath());
        if (QFile::exists(dst)) {
            QFile::remove(dst);
        }
        return QFile::copy(src, dst);
    };

    for (int i = 0; i < images.size(); ++i) {
        const QString imagePath = images[i];
        const QString splitName = (i < trainCount) ? "train" : "val";

        QFileInfo imgInfo(imagePath);
        const QString imageDst = out.filePath("images/" + splitName + "/" + imgInfo.fileName());
        const QString labelDst = out.filePath("labels/" + splitName + "/" + imgInfo.completeBaseName() + ".txt");

        if (!copyFileOverwrite(imagePath, imageDst)) {
            if (showMessage) {
                QMessageBox::warning(this, "复制失败", "无法复制图片：\n" + imagePath);
            }
            continue;
        }

        QImage img(imagePath);
        const int imgW = img.width();
        const int imgH = img.height();

        QVector<Annotation> anns;
        QString err;
        AnnotationIO::load(imagePath, anns, &err);

        QFile labelFile(labelDst);
        if (labelFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream labelOut(&labelFile);

            if (imgW > 0 && imgH > 0) {
                for (const Annotation &ann : anns) {
                    const QString cls = normalizeClass(ann.label);
                    if (!classToId.contains(cls)) {
                        classToId.insert(cls, classes.size());
                        classes << cls;
                    }

                    QRect r = ann.rect.normalized().intersected(QRect(0, 0, imgW, imgH));
                    if (r.width() <= 0 || r.height() <= 0) {
                        continue;
                    }

                    const double xCenter = (r.x() + r.width() / 2.0) / imgW;
                    const double yCenter = (r.y() + r.height() / 2.0) / imgH;
                    const double width = r.width() / static_cast<double>(imgW);
                    const double height = r.height() / static_cast<double>(imgH);

                    labelOut << classToId.value(cls) << " "
                             << QString::number(xCenter, 'f', 6) << " "
                             << QString::number(yCenter, 'f', 6) << " "
                             << QString::number(width, 'f', 6) << " "
                             << QString::number(height, 'f', 6) << "\n";
                    ++boxTotal;
                }
            }
            labelFile.close();
        }

        if (splitName == "train") {
            ++copiedTrain;
        } else {
            ++copiedVal;
        }
    }

    const QString yamlPath = out.filePath("data.yaml");
    QFile yamlFile(yamlPath);
    if (!yamlFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        if (showMessage) {
            QMessageBox::warning(this, "生成失败", "无法写入：" + yamlPath);
        }
        return false;
    }

    QTextStream yaml(&yamlFile);
    yaml << "path: " << QDir::fromNativeSeparators(out.absolutePath()) << "\n";
    yaml << "train: images/train\n";
    yaml << "val: images/val\n";
    yaml << "nc: " << classes.size() << "\n";
    yaml << "names:\n";
    for (int i = 0; i < classes.size(); ++i) {
        yaml << "  " << i << ": " << classes[i] << "\n";
    }
    yamlFile.close();

    ui->lineEditDataset->setText(QDir::toNativeSeparators(yamlPath));

    if (showMessage) {
        QMessageBox::information(
            this,
            "划分完成",
            QString("已生成 YOLO 数据集：\n%1\n\n训练集图片：%2\n验证集图片：%3\n标注框总数：%4\n类别数：%5\n\ndata.yaml 已自动填入训练页。")
                .arg(QDir::toNativeSeparators(out.absolutePath()))
                .arg(copiedTrain)
                .arg(copiedVal)
                .arg(boxTotal)
                .arg(classes.size())
        );
    }

    return true;
}


void MainWindow::refreshAnnotationList()
{
    QSignalBlocker blocker(ui->annotationListWidget);
    ui->annotationListWidget->clear();

    const auto anns = ui->canvasWidget->annotations;

    for (int i = 0; i < anns.size(); ++i) {
        const auto& a = anns[i];

        QString text = QString("%1 | %2 | x:%3 y:%4 w:%5 h:%6")
                           .arg(i + 1)
                           .arg(a.label)
                           .arg(a.rect.x())
                           .arg(a.rect.y())
                           .arg(a.rect.width())
                           .arg(a.rect.height());

        QListWidgetItem *item = new QListWidgetItem(text);
        item->setData(Qt::UserRole, i);
        ui->annotationListWidget->addItem(item);
    }

    if (ui->canvasWidget->selectedIndex >= 0 &&
        ui->canvasWidget->selectedIndex < ui->annotationListWidget->count()) {
        ui->annotationListWidget->setCurrentRow(ui->canvasWidget->selectedIndex);
    }

    updateAnnotationSummaryStatus();
}


void MainWindow::deleteSelectedAnnotation()
{
    int row = ui->annotationListWidget->currentRow();

    if (row >= 0 && row < ui->canvasWidget->annotations.size()) {
        ui->canvasWidget->selectedIndex = row;
    }

    if (ui->canvasWidget->selectedIndex < 0 ||
        ui->canvasWidget->selectedIndex >= ui->canvasWidget->annotations.size()) {
        QMessageBox::information(this, "提示", "请先在标注列表或画布中选择要删除的标注框。");
        return;
    }

    ui->canvasWidget->deleteSelectedAnnotation();
    refreshAnnotationList();

    // 关键修复：删除框后立即同步到本地 txt。
    // 如果删的是最后一个框，saveCurrentAnnotations() 会删除 txt 文件。
    saveCurrentAnnotations();
}


void MainWindow::prevImage()
{
    if (m_imagePaths.isEmpty()) {
        return;
    }

    int row = m_currentIndex;

    if (row > 0) {
        loadImageAt(row - 1);
    }
}


void MainWindow::nextImage()
{
    if (m_imagePaths.isEmpty()) {
        return;
    }

    int row = m_currentIndex;

    if (row >= 0 && row < m_imagePaths.size() - 1) {
        loadImageAt(row + 1);
    }
}


// ===================== 训练模块 =====================

static QString cleanConsoleText(const QString &text)
{
    QString s = text;

    // 去掉 ANSI 转义控制符，例如 \x1b[K、\x1b[34m
    s.remove(QRegularExpression("\x1B\\[[0-9;?]*[A-Za-z]"));

    // 去掉回车刷新符
    s.replace('\r', '\n');

    // 去掉部分 tqdm 进度条乱码字符
    s.replace("鈹€", "-");
    s.replace("鈹佲攣", "=");
    s.replace("鈹佲攢", ">");
    s.replace("鈺糕攢", ">");

    return s;
}
void MainWindow::appendTrainLog(const QString& text)
{
    QString msg = text;
    if (msg.isEmpty())
        return;

    QStringList lines = msg.split('\n', Qt::SkipEmptyParts);
    for (const QString& line : lines)
    {
        const QString trimmed = line.trimmed();
        if (trimmed.isEmpty()) {
            continue;
        }

        QString time = QDateTime::currentDateTime().toString("[yyyy-MM-dd hh:mm:ss] ");
        ui->plainTextEditLog->appendPlainText(time + trimmed);

        m_trainLogCache += time + trimmed + "\n";
        if (m_trainLogCache.size() > 60000) {
            m_trainLogCache = m_trainLogCache.right(60000);
        }

        updateTrainingProgressFromLogLine(trimmed);
    }
}

bool MainWindow::checkTrainInputs()
{
    QString dataset = ui->lineEditDataset->text().trimmed();
    QString pythonExe = ui->lineEditPythonExe->text().trimmed();
    QString script = ui->lineEditScript->text().trimmed();

    if (dataset.isEmpty()) {
        QMessageBox::warning(this, "提示", "请选择 data.yaml 数据集配置文件。");
        return false;
    }

    if (!QFileInfo::exists(dataset)) {
        QMessageBox::warning(this, "提示", "data.yaml 文件不存在。");
        return false;
    }

    if (pythonExe.isEmpty()) {
        QMessageBox::warning(this, "提示", "请设置 Python 解释器路径。");
        return false;
    }

    if (!QFileInfo::exists(pythonExe)) {
        QMessageBox::warning(this, "提示", "Python 解释器不存在。");
        return false;
    }

    if (script.isEmpty()) {
        QMessageBox::warning(this, "提示", "请选择训练脚本。");
        return false;
    }

    if (!QFileInfo::exists(script)) {
        QMessageBox::warning(this, "提示", "训练脚本不存在。");
        return false;
    }

    return true;
}

void MainWindow::onSelectDataset()
{
    QString file = QFileDialog::getOpenFileName(
        this,
        "选择 data.yaml",
        m_currentFolder.isEmpty() ? QString() : datasetRootForYaml(),
        "YAML Files (*.yaml *.yml);;All Files (*)"
    );

    if (!file.isEmpty()) {
        ui->lineEditDataset->setText(QDir::toNativeSeparators(file));
    }
}

void MainWindow::onSelectPythonExe()
{
    QString file = QFileDialog::getOpenFileName(
        this,
        "选择 Python 解释器",
        "",
        "Python (*.exe);;All Files (*)"
        );

    if (!file.isEmpty())
        ui->lineEditPythonExe->setText(file);
}

void MainWindow::onSelectScript()
{
    QString file = QFileDialog::getOpenFileName(
        this,
        "选择训练脚本",
        "",
        "Python Files (*.py)"
        );

    if (!file.isEmpty())
        ui->lineEditScript->setText(file);
}

void MainWindow::onStartTraining()
{
    if (!checkTrainInputs())
        return;

    if (m_trainProcess->state() != QProcess::NotRunning) {
        QMessageBox::information(this, "提示", "训练已经在运行中。");
        return;
    }

    QString pythonExe = ui->lineEditPythonExe->text().trimmed();
    QString scriptPath = ui->lineEditScript->text().trimmed();
    QString datasetPath = ui->lineEditDataset->text().trimmed();

    m_trainStartTime = QDateTime::currentDateTime();
    m_trainLogCache.clear();

    const QString projectRoot = projectRootFromScriptPath(scriptPath);
    const QString trainProjectDir = QDir(projectRoot).filePath("runs");
    const QString fixedModelDir = QDir(projectRoot).filePath("models");

    QDir().mkpath(trainProjectDir);
    QDir().mkpath(fixedModelDir);

    QString modelName = ui->comboBoxModel->currentText().trimmed();
    if (!modelName.endsWith(".pt", Qt::CaseInsensitive) &&
        !modelName.endsWith(".yaml", Qt::CaseInsensitive) &&
        !QFileInfo(modelName).exists()) {
        modelName += ".pt";
    }

    QStringList args;
    args << "-u"
         << scriptPath
         << "--data" << datasetPath
         << "--epochs" << QString::number(ui->spinBoxEpoch->value())
         << "--batch-size" << QString::number(ui->spinBoxBatch->value())
         << "--lr" << QString::number(ui->doubleSpinBoxLr->value(), 'f', 6)
         << "--model" << modelName
         << "--imgsz" << "640"
         << "--project" << trainProjectDir
         << "--name" << "focus_detect"
         << "--save-dir" << fixedModelDir
         << "--device" << "cpu";

    ui->plainTextEditLog->clear();
    resetTrainingResultDisplay();
    ui->progressBarTrain->setValue(0);
    appendTrainLog("开始训练...");
    appendTrainLog("命令: " + pythonExe + " " + args.join(" "));
    appendTrainLog("工程根目录: " + projectRoot);
    appendTrainLog("训练输出目录: " + trainProjectDir);
    appendTrainLog("固定模型目录: " + fixedModelDir);

    ui->labelTrainStatus->setText("状态：训练中");
    ui->btnStartTrain->setEnabled(false);
    ui->btnStopTrain->setEnabled(true);

    // 先清理旧状态
    m_trainProcess->setProgram(QString());
    m_trainProcess->setArguments(QStringList());
    m_trainProcess->setWorkingDirectory(QString());

    // ===== 关键修复：为训练子进程单独设置环境，避免 OpenMP 冲突 =====
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();

    QStringList pathList = env.value("PATH").split(';', Qt::SkipEmptyParts);
    QStringList filteredPaths;

    for (const QString& p : pathList)
    {
        QString lower = QDir::toNativeSeparators(p).toLower();

        // 去掉 Qt / MinGW / LLVM 相关路径，避免把 libomp.dll 带进去
        if (lower.contains("\\qt\\") ||
            lower.contains("\\mingw") ||
            lower.contains("\\llvm"))
        {
            continue;
        }

        filteredPaths << p;
    }

    // 根据用户选择的 python.exe 推断 conda 环境根目录
    QFileInfo pyInfo(pythonExe);
    QString pyDir = pyInfo.absolutePath();          // ...\envs\python3.9
    QString envRoot = pyDir;

    // 把 conda 环境路径放到 PATH 前面
    filteredPaths.removeAll(envRoot);
    filteredPaths.removeAll(envRoot + "\\Scripts");
    filteredPaths.removeAll(envRoot + "\\Library\\bin");

    filteredPaths.prepend(envRoot + "\\Library\\bin");
    filteredPaths.prepend(envRoot + "\\Scripts");
    filteredPaths.prepend(envRoot);

    env.insert("PATH", filteredPaths.join(';'));

    // 临时兜底，先确保 Qt 内能稳定跑通
    env.insert("KMP_DUPLICATE_LIB_OK", "TRUE");
    env.insert("OMP_NUM_THREADS", "1");
    env.insert("PYTHONUNBUFFERED", "1");

    m_trainProcess->setProcessEnvironment(env);

    // 工作目录设为工程根目录，避免训练结果落到 python/runs 下面
    m_trainProcess->setWorkingDirectory(projectRoot);

    // 输出和错误分开，保留你原来的日志逻辑
    m_trainProcess->setProcessChannelMode(QProcess::SeparateChannels);

    m_trainProcess->setProgram(pythonExe);
    m_trainProcess->setArguments(args);
    m_trainProcess->start();

    if (!m_trainProcess->waitForStarted(3000)) {
        appendTrainLog("训练进程启动失败。");
        appendTrainLog("[ERR] " + m_trainProcess->errorString());
        ui->labelTrainStatus->setText("状态：启动失败");
        ui->btnStartTrain->setEnabled(true);
        ui->btnStopTrain->setEnabled(false);
        return;
    }

    appendTrainLog("训练进程已启动。");
}

void MainWindow::onStopTraining()
{
    if (m_trainProcess->state() == QProcess::NotRunning)
        return;

    appendTrainLog("正在停止训练...");
    m_trainProcess->kill();
}

void MainWindow::onTrainReadyReadStdOut()
{
    QByteArray data = m_trainProcess->readAllStandardOutput();

    // 用 UTF-8 解码，不要用 fromLocal8Bit
    QString text = QString::fromUtf8(data);

    text = cleanConsoleText(text);

    if (!text.trimmed().isEmpty()) {
        appendTrainLog(text);
    }
}

void MainWindow::onTrainReadyReadStdErr()
{
    QByteArray data = m_trainProcess->readAllStandardError();

    QString text = QString::fromUtf8(data);
    text = cleanConsoleText(text);

    if (!text.trimmed().isEmpty()) {
        appendTrainLog("[ERR] " + text);
    }
}

void MainWindow::onTrainFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    Q_UNUSED(exitStatus);

    if (exitCode == 0) {
        appendTrainLog("训练结束：成功。");
        ui->labelTrainStatus->setText("状态：训练完成");
    } else {
        appendTrainLog(QString("训练结束：失败，退出码 = %1").arg(exitCode));
        ui->labelTrainStatus->setText("状态：训练失败");
    }

    saveTrainingResultToDatabase(exitCode == 0, exitCode);
    if (exitCode == 0) {
        ui->progressBarTrain->setValue(100);
    }

    ui->btnStartTrain->setEnabled(true);
    ui->btnStopTrain->setEnabled(false);
}

void MainWindow::resetTrainingResultDisplay()
{
    if (!ui->tableWidgetTrainMetrics) {
        return;
    }

    ui->tableWidgetTrainMetrics->setColumnCount(2);
    ui->tableWidgetTrainMetrics->setRowCount(8);
    ui->tableWidgetTrainMetrics->setHorizontalHeaderLabels(QStringList() << "项目" << "结果");
    ui->tableWidgetTrainMetrics->horizontalHeader()->setStretchLastSection(true);
    ui->tableWidgetTrainMetrics->verticalHeader()->setVisible(false);
    ui->tableWidgetTrainMetrics->setEditTriggers(QAbstractItemView::NoEditTriggers);
    ui->tableWidgetTrainMetrics->setSelectionBehavior(QAbstractItemView::SelectRows);
    ui->tableWidgetTrainMetrics->setAlternatingRowColors(false);

    const QStringList names = {
        "训练状态",
        "结果目录",
        "best.pt",
        "last.pt",
        "Precision",
        "Recall",
        "mAP50",
        "mAP50-95"
    };

    for (int i = 0; i < names.size(); ++i) {
        auto *nameItem = new QTableWidgetItem(names.at(i));
        auto *valueItem = new QTableWidgetItem("待训练");

        nameItem->setBackground(QBrush(QColor("#eff6ff")));
        nameItem->setForeground(QBrush(QColor("#1e3a8a")));
        valueItem->setBackground(QBrush(QColor("#f8fafc")));
        valueItem->setForeground(QBrush(QColor("#64748b")));

        ui->tableWidgetTrainMetrics->setItem(i, 0, nameItem);
        ui->tableWidgetTrainMetrics->setItem(i, 1, valueItem);
    }

    ui->tableWidgetTrainMetrics->resizeColumnsToContents();

    auto initMetricBar = [](QProgressBar *bar) {
        if (!bar) {
            return;
        }
        bar->setRange(0, 100);
        bar->setValue(0);
        bar->setFormat("未获取");
        bar->setTextVisible(true);
        bar->setMinimumHeight(20);
        bar->setStyleSheet(
            "QProgressBar {"
            "border: 1px solid #94a3b8;"
            "border-radius: 6px;"
            "background-color: #e2e8f0;"
            "color: #0f172a;"
            "text-align: center;"
            "font-weight: 600;"
            "}"
            "QProgressBar::chunk {"
            "border-radius: 5px;"
            "background-color: #94a3b8;"
            "}"
        );
    };

    initMetricBar(ui->progressBarMetricPrecision);
    initMetricBar(ui->progressBarMetricRecall);
    initMetricBar(ui->progressBarMetricMap50);
    initMetricBar(ui->progressBarMetricMap5095);

    if (ui->labelMetricPrecisionValue) ui->labelMetricPrecisionValue->setText("未获取");
    if (ui->labelMetricRecallValue) ui->labelMetricRecallValue->setText("未获取");
    if (ui->labelMetricMap50Value) ui->labelMetricMap50Value->setText("未获取");
    if (ui->labelMetricMap5095Value) ui->labelMetricMap5095Value->setText("未获取");
}


void MainWindow::updateTrainingProgressFromLogLine(const QString &line)
{
    if (!ui->progressBarTrain) {
        return;
    }

    // 只识别 train.py 主动输出的专用进度标记，避免误把 Ultralytics 日志里的 1/1、640/640 等内容当成 epoch 进度。
    static const QRegularExpression progressRegex(
        R"(^__TRAIN_PROGRESS__\s+(\d+)\s*/\s*(\d+)\s*$)"
    );

    const QRegularExpressionMatch match = progressRegex.match(line.trimmed());
    if (!match.hasMatch()) {
        return;
    }

    bool ok1 = false;
    bool ok2 = false;
    const int current = match.captured(1).toInt(&ok1);
    const int total = match.captured(2).toInt(&ok2);

    if (!ok1 || !ok2 || current <= 0 || total <= 0) {
        return;
    }

    const int percent = qBound(
        0,
        static_cast<int>(std::round(current * 100.0 / total)),
        100
    );

    ui->progressBarTrain->setValue(percent);
}

void MainWindow::updateTrainingResultDisplay(bool success,
                                             int exitCode,
                                             const QString &resultDir,
                                             const QString &bestModelPath,
                                             const QString &lastModelPath,
                                             const QMap<QString, double> &metrics)
{
    if (!ui->tableWidgetTrainMetrics) {
        return;
    }

    auto metricColor = [](double value) -> QColor {
        if (value >= 0.75) return QColor("#16a34a");   // 绿色：较好
        if (value >= 0.50) return QColor("#ca8a04");   // 黄色：一般
        return QColor("#dc2626");                      // 红色：较低
    };

    auto setValue = [&](int row, const QString &value, const QColor &bg, const QColor &fg = QColor("#0f172a")) {
        if (!ui->tableWidgetTrainMetrics->item(row, 0)) {
            ui->tableWidgetTrainMetrics->setItem(row, 0, new QTableWidgetItem());
        }
        if (!ui->tableWidgetTrainMetrics->item(row, 1)) {
            ui->tableWidgetTrainMetrics->setItem(row, 1, new QTableWidgetItem());
        }
        const QString text = value.trimmed().isEmpty() ? "未获取" : value;
        ui->tableWidgetTrainMetrics->item(row, 1)->setText(text);
        ui->tableWidgetTrainMetrics->item(row, 1)->setBackground(QBrush(bg));
        ui->tableWidgetTrainMetrics->item(row, 1)->setForeground(QBrush(fg));
    };

    setValue(0,
             success ? "成功" : QString("失败，退出码=%1").arg(exitCode),
             success ? QColor("#dcfce7") : QColor("#fee2e2"),
             success ? QColor("#166534") : QColor("#991b1b"));

    setValue(1, resultDir, QColor("#eff6ff"), QColor("#1d4ed8"));
    setValue(2, bestModelPath, QColor("#f0f9ff"), QColor("#0369a1"));
    setValue(3, lastModelPath, QColor("#f0f9ff"), QColor("#0369a1"));

    auto setMetric = [&](int row, const QString &key) {
        if (!metrics.contains(key)) {
            setValue(row, "未获取", QColor("#f1f5f9"), QColor("#64748b"));
            return;
        }
        const double v = qBound(0.0, metrics.value(key), 1.0);
        const QColor c = metricColor(v);
        QColor bg = c;
        bg.setAlpha(40);
        setValue(row, QString::number(v, 'f', 4), bg, c);
    };

    setMetric(4, "precision");
    setMetric(5, "recall");
    setMetric(6, "map50");
    setMetric(7, "map5095");

    ui->tableWidgetTrainMetrics->resizeColumnsToContents();
    ui->tableWidgetTrainMetrics->horizontalHeader()->setStretchLastSection(true);

    auto updateMetricBar = [](QProgressBar *bar, QLabel *valueLabel, const QMap<QString, double> &metricMap, const QString &key) {
        if (!bar) {
            return;
        }

        if (!metricMap.contains(key)) {
            bar->setValue(0);
            bar->setFormat("未获取");
            bar->setStyleSheet(
                "QProgressBar {"
                "border: 1px solid #94a3b8;"
                "border-radius: 6px;"
                "background-color: #e2e8f0;"
                "color: #0f172a;"
                "text-align: center;"
                "font-weight: 600;"
                "}"
                "QProgressBar::chunk {"
                "border-radius: 5px;"
                "background-color: #94a3b8;"
                "}"
            );
            if (valueLabel) {
                valueLabel->setText("未获取");
                valueLabel->setStyleSheet("color:#64748b;font-weight:600;");
            }
            return;
        }

        const double value = qBound(0.0, metricMap.value(key), 1.0);
        const int percent = qBound(0, static_cast<int>(std::round(value * 100.0)), 100);
        QString color = "#dc2626";
        if (value >= 0.75) color = "#16a34a";
        else if (value >= 0.50) color = "#ca8a04";

        bar->setValue(percent);
        bar->setFormat(QString::number(value, 'f', 4));
        bar->setTextVisible(true);
        bar->setStyleSheet(QString(
            "QProgressBar {"
            "border: 1px solid #94a3b8;"
            "border-radius: 6px;"
            "background-color: #e2e8f0;"
            "color: #0f172a;"
            "text-align: center;"
            "font-weight: 600;"
            "}"
            "QProgressBar::chunk {"
            "border-radius: 5px;"
            "background-color: %1;"
            "}"
        ).arg(color));

        if (valueLabel) {
            valueLabel->setText(QString("%1%").arg(percent));
            valueLabel->setStyleSheet(QString("color:%1;font-weight:700;").arg(color));
        }
    };

    updateMetricBar(ui->progressBarMetricPrecision, ui->labelMetricPrecisionValue, metrics, "precision");
    updateMetricBar(ui->progressBarMetricRecall, ui->labelMetricRecallValue, metrics, "recall");
    updateMetricBar(ui->progressBarMetricMap50, ui->labelMetricMap50Value, metrics, "map50");
    updateMetricBar(ui->progressBarMetricMap5095, ui->labelMetricMap5095Value, metrics, "map5095");
}


bool MainWindow::initDatabase()
{
    QString connectionName = "training_records_connection";

    if (QSqlDatabase::contains(connectionName)) {
        m_database = QSqlDatabase::database(connectionName);
    } else {
        m_database = QSqlDatabase::addDatabase("QSQLITE", connectionName);
        m_database.setDatabaseName(QCoreApplication::applicationDirPath() + "/training_records.db");
    }

    if (!m_database.open()) {
        QMessageBox::warning(this,
                             "数据库错误",
                             "无法打开 SQLite 数据库：\n" + m_database.lastError().text());
        return false;
    }

    QSqlQuery query(m_database);

    const QString createTrainingTable = R"SQL(
        CREATE TABLE IF NOT EXISTS training_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            dataset_path TEXT,
            python_exe TEXT,
            script_path TEXT,
            model_name TEXT,
            epochs INTEGER,
            batch_size INTEGER,
            learning_rate REAL,
            success INTEGER,
            exit_code INTEGER,
            result_dir TEXT,
            best_model_path TEXT,
            last_model_path TEXT,
            precision_value REAL,
            recall_value REAL,
            map50_value REAL,
            map5095_value REAL,
            log_excerpt TEXT
        )
    )SQL";

    if (!query.exec(createTrainingTable)) {
        QMessageBox::warning(this, "数据库错误", "创建 training_results 表失败：\n" + query.lastError().text());
        return false;
    }

    const QString createInferenceTable = R"SQL(
        CREATE TABLE IF NOT EXISTS inference_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            created_at TEXT NOT NULL,
            model_path TEXT,
            image_path TEXT,
            result_image_path TEXT,
            class_name TEXT,
            confidence REAL,
            x REAL,
            y REAL,
            w REAL,
            h REAL
        )
    )SQL";

    if (!query.exec(createInferenceTable)) {
        QMessageBox::warning(this, "数据库错误", "创建 inference_results 表失败：\n" + query.lastError().text());
        return false;
    }

    const QString createInferenceRunTable = R"SQL(
        CREATE TABLE IF NOT EXISTS inference_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            model_path TEXT,
            image_path TEXT,
            result_image_path TEXT,
            result_image_blob BLOB,
            result_image_format TEXT
        )
    )SQL";

    if (!query.exec(createInferenceRunTable)) {
        QMessageBox::warning(this, "数据库错误", "创建 inference_runs 表失败：\n" + query.lastError().text());
        return false;
    }

    QString alterError;
    if (!sqliteAddColumnIfMissing(m_database, "inference_results", "run_id", "INTEGER", &alterError)) {
        QMessageBox::warning(this, "数据库错误", "升级 inference_results 表失败：\n" + alterError);
        return false;
    }

    if (!sqliteAddColumnIfMissing(m_database, "inference_results", "result_image_blob", "BLOB", &alterError)) {
        QMessageBox::warning(this, "数据库错误", "升级 inference_results 表失败：\n" + alterError);
        return false;
    }

    if (!sqliteAddColumnIfMissing(m_database, "inference_results", "result_image_format", "TEXT", &alterError)) {
        QMessageBox::warning(this, "数据库错误", "升级 inference_results 表失败：\n" + alterError);
        return false;
    }

    return true;
}

void MainWindow::saveTrainingResultToDatabase(bool success, int exitCode)
{
    if (!m_database.isOpen() && !initDatabase()) {
        appendTrainLog("[DB] 数据库未打开，训练结果未保存。");
        return;
    }

    const QString datasetPath = ui->lineEditDataset->text().trimmed();
    const QString pythonExe = ui->lineEditPythonExe->text().trimmed();
    const QString scriptPath = ui->lineEditScript->text().trimmed();
    const QString modelName = ui->comboBoxModel->currentText().trimmed();
    const QString projectRoot = projectRootFromScriptPath(scriptPath);

    // 优先从工程根目录/runs 查找本次训练的 best.pt / last.pt。
    // 不再优先搜索 python/runs，避免选到旧的嵌套目录：python/runs/detect/runs/...
    QStringList runSearchRoots;
    runSearchRoots << QDir(projectRoot).filePath("runs");
    runSearchRoots << QDir(projectRoot).filePath("runs/detect");

    const QFileInfo datasetInfo(datasetPath);
    if (datasetInfo.exists()) {
        QDir datasetDir = datasetInfo.isDir() ? QDir(datasetInfo.absoluteFilePath()) : datasetInfo.absoluteDir();
        datasetDir.cdUp();
        runSearchRoots << datasetDir.filePath("runs");
        runSearchRoots << datasetDir.filePath("runs/detect");
    }

    runSearchRoots << QCoreApplication::applicationDirPath() + "/runs";
    runSearchRoots << QCoreApplication::applicationDirPath() + "/runs/detect";
    runSearchRoots.removeDuplicates();

    QString bestModelPath = newestFileByName(runSearchRoots, "best.pt", m_trainStartTime);
    QString lastModelPath = newestFileByName(runSearchRoots, "last.pt", m_trainStartTime);

    // 如果 runs 里没有找到，退回固定模型目录 models/best.pt。
    const QString fixedBestModelPath = QDir(projectRoot).filePath("models/best.pt");
    if (bestModelPath.isEmpty() && QFileInfo::exists(fixedBestModelPath)) {
        bestModelPath = QDir::cleanPath(fixedBestModelPath);
    }

    QString resultDir;
    QString resultsCsvPath;
    if (!bestModelPath.isEmpty()) {
        QFileInfo bestInfo(bestModelPath);
        QDir dir = bestInfo.absoluteDir(); // weights
        dir.cdUp();                       // train / train2 / ...
        resultDir = dir.absolutePath();
        resultsCsvPath = dir.filePath("results.csv");
    } else if (!lastModelPath.isEmpty()) {
        QFileInfo lastInfo(lastModelPath);
        QDir dir = lastInfo.absoluteDir();
        dir.cdUp();
        resultDir = dir.absolutePath();
        resultsCsvPath = dir.filePath("results.csv");
    }

    const QMap<QString, double> metrics = readLastYoloMetricsFromCsv(resultsCsvPath);

    updateTrainingResultDisplay(success, exitCode, resultDir, bestModelPath, lastModelPath, metrics);

    QSqlQuery query(m_database);
    query.prepare(R"SQL(
        INSERT INTO training_results (
            created_at, dataset_path, python_exe, script_path, model_name,
            epochs, batch_size, learning_rate, success, exit_code,
            result_dir, best_model_path, last_model_path,
            precision_value, recall_value, map50_value, map5095_value,
            log_excerpt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    )SQL");

    query.addBindValue(QDateTime::currentDateTime().toString(Qt::ISODate));
    query.addBindValue(datasetPath);
    query.addBindValue(pythonExe);
    query.addBindValue(scriptPath);
    query.addBindValue(modelName);
    query.addBindValue(ui->spinBoxEpoch->value());
    query.addBindValue(ui->spinBoxBatch->value());
    query.addBindValue(ui->doubleSpinBoxLr->value());
    query.addBindValue(success ? 1 : 0);
    query.addBindValue(exitCode);
    query.addBindValue(nullIfEmpty(resultDir));
    query.addBindValue(nullIfEmpty(bestModelPath));
    query.addBindValue(nullIfEmpty(lastModelPath));
    query.addBindValue(metrics.contains("precision") ? QVariant(metrics.value("precision")) : QVariant());
    query.addBindValue(metrics.contains("recall") ? QVariant(metrics.value("recall")) : QVariant());
    query.addBindValue(metrics.contains("map50") ? QVariant(metrics.value("map50")) : QVariant());
    query.addBindValue(metrics.contains("map5095") ? QVariant(metrics.value("map5095")) : QVariant());
    query.addBindValue(m_trainLogCache.right(20000));

    if (!query.exec()) {
        appendTrainLog("[DB] 训练结果保存失败：" + query.lastError().text());
        return;
    }

    appendTrainLog("[DB] 训练结果已保存到：" + m_database.databaseName());
    if (!bestModelPath.isEmpty()) {
        appendTrainLog("[DB] best.pt：" + bestModelPath);
    }
    if (metrics.contains("map50")) {
        appendTrainLog(QString("[DB] mAP50：%1").arg(metrics.value("map50"), 0, 'f', 4));
    }
}

void MainWindow::saveInferenceResultsToDatabase(const QJsonObject &root,
                                                const QJsonArray &detections,
                                                const QString &resultImagePath)
{
    Q_UNUSED(root);

    if (!m_database.isOpen() && !initDatabase()) {
        appendQuantLog("[DB] 数据库未打开，推理结果未保存。");
        return;
    }

    QByteArray resultImageBytes;
    QString resultImageFormat;

    if (!resultImagePath.trimmed().isEmpty()) {
        QFile imageFile(resultImagePath);
        if (imageFile.open(QIODevice::ReadOnly)) {
            resultImageBytes = imageFile.readAll();
            imageFile.close();
            resultImageFormat = QFileInfo(resultImagePath).suffix().toLower();
        } else {
            appendQuantLog("[DB] 推理结果图片读取失败，仅保存图片路径：" + resultImagePath);
        }
    }

    if (!m_database.transaction()) {
        appendQuantLog("[DB] 开启事务失败：" + m_database.lastError().text());
        return;
    }

    QSqlQuery runQuery(m_database);
    runQuery.prepare(R"SQL(
        INSERT INTO inference_runs (
            created_at, model_path, image_path, result_image_path,
            result_image_blob, result_image_format
        ) VALUES (?, ?, ?, ?, ?, ?)
    )SQL");

    runQuery.addBindValue(QDateTime::currentDateTime().toString(Qt::ISODate));
    runQuery.addBindValue(m_inferModelPath);
    runQuery.addBindValue(m_inferImagePath);
    runQuery.addBindValue(resultImagePath);
    runQuery.addBindValue(resultImageBytes.isEmpty() ? QVariant() : QVariant(resultImageBytes));
    runQuery.addBindValue(resultImageFormat);

    if (!runQuery.exec()) {
        m_database.rollback();
        appendQuantLog("[DB] 推理图片保存失败：" + runQuery.lastError().text());
        return;
    }

    const QVariant runId = runQuery.lastInsertId();

    auto insertOne = [&](const QString &className,
                         const QVariant &confidence,
                         const QVariant &x,
                         const QVariant &y,
                         const QVariant &w,
                         const QVariant &h) -> bool {
        QSqlQuery query(m_database);
        query.prepare(R"SQL(
            INSERT INTO inference_results (
                run_id, created_at, model_path, image_path, result_image_path,
                result_image_blob, result_image_format,
                class_name, confidence, x, y, w, h
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        )SQL");

        query.addBindValue(runId);
        query.addBindValue(QDateTime::currentDateTime().toString(Qt::ISODate));
        query.addBindValue(m_inferModelPath);
        query.addBindValue(m_inferImagePath);
        query.addBindValue(resultImagePath);
        query.addBindValue(resultImageBytes.isEmpty() ? QVariant() : QVariant(resultImageBytes));
        query.addBindValue(resultImageFormat);
        query.addBindValue(className);
        query.addBindValue(confidence);
        query.addBindValue(x);
        query.addBindValue(y);
        query.addBindValue(w);
        query.addBindValue(h);

        if (!query.exec()) {
            appendQuantLog("[DB] 推理检测框保存失败：" + query.lastError().text());
            return false;
        }
        return true;
    };

    bool ok = true;
    if (detections.isEmpty()) {
        ok = insertOne(QString(), QVariant(), QVariant(), QVariant(), QVariant(), QVariant());
    } else {
        for (const QJsonValue &value : detections) {
            const QJsonObject det = value.toObject();
            ok = insertOne(det.value("class_name").toString(),
                           det.value("confidence").toDouble(),
                           det.value("x").toDouble(),
                           det.value("y").toDouble(),
                           det.value("w").toDouble(),
                           det.value("h").toDouble()) && ok;
        }
    }

    if (ok) {
        m_database.commit();
        appendQuantLog(QString("[DB] 推理结果已保存到：%1").arg(m_database.databaseName()));
        if (!resultImageBytes.isEmpty()) {
            appendQuantLog(QString("[DB] 推理结果图片已写入数据库 BLOB，大小：%1 KB")
                               .arg(resultImageBytes.size() / 1024.0, 0, 'f', 1));
        }
    } else {
        m_database.rollback();
    }
}


static QString cleanQuantConsoleText(const QString &text)
{
    QString s = text;

    // 去掉 ANSI 控制符
    s.remove(QRegularExpression("\x1B\\[[0-9;?]*[A-Za-z]"));

    // 回车转换行
    s.replace('\r', '\n');

    return s;
}


void MainWindow::appendQuantLog(const QString &text)
{
    QString msg = cleanQuantConsoleText(text);

    if (msg.trimmed().isEmpty()) {
        return;
    }

    QStringList lines = msg.split('\n', Qt::SkipEmptyParts);

    for (const QString &line : lines) {
        QString time = QDateTime::currentDateTime().toString("[yyyy-MM-dd hh:mm:ss] ");
        ui->textEditQuantLog->append(time + line.trimmed());
    }
}


void MainWindow::handleQuantLine(const QString &line)
{
    QString trimmed = line.trimmed();

    if (trimmed.isEmpty()) {
        return;
    }

    QJsonParseError err;
    QJsonDocument doc = QJsonDocument::fromJson(trimmed.toUtf8(), &err);

    if (err.error != QJsonParseError::NoError || !doc.isObject()) {
        appendQuantLog(trimmed);
        return;
    }

    QJsonObject obj = doc.object();

    QString type = obj.value("type").toString();

    if (type == "progress") {
        int value = obj.value("value").toInt();
        QString msg = obj.value("msg").toString();

        ui->progressBarQuant->setValue(value);
        appendQuantLog(msg);
    }
    else if (type == "error") {
        QString msg = obj.value("msg").toString();
        appendQuantLog("[错误] " + msg);
    }
    else {
        appendQuantLog(trimmed);
    }
}


void MainWindow::onSelectQuantModel()
{
    QString file = QFileDialog::getOpenFileName(
        this,
        "选择待量化模型",
        "",
        "Model Files (*.pt *.onnx);;All Files (*)"
        );

    if (file.isEmpty()) {
        return;
    }

    ui->lineEditModelPath->setText(file);

    QFileInfo info(file);
    QString outputPath = info.absolutePath() + "/" + info.completeBaseName() + "_int8.onnx";
    ui->lineEditOutputPath->setText(outputPath);
}


void MainWindow::onSelectCalibDir()
{
    QString dir = QFileDialog::getExistingDirectory(
        this,
        "选择校准图片文件夹"
        );

    if (dir.isEmpty()) {
        return;
    }

    ui->lineEditCalibDir->setText(dir);
}


void MainWindow::onSelectQuantOutput()
{
    QString file = QFileDialog::getSaveFileName(
        this,
        "选择量化模型输出路径",
        "",
        "ONNX Model (*.onnx)"
        );

    if (file.isEmpty()) {
        return;
    }

    if (!file.endsWith(".onnx", Qt::CaseInsensitive)) {
        file += ".onnx";
    }

    ui->lineEditOutputPath->setText(file);
}


void MainWindow::onStartQuant()
{
    if (m_quantProcess->state() != QProcess::NotRunning) {
        QMessageBox::information(this, "提示", "量化进程正在运行中。");
        return;
    }

    QString pythonExe = ui->lineEditPythonExe->text().trimmed();
    QString scriptPath = QCoreApplication::applicationDirPath() + "/python/quantize_model.py";

    // 如果你的 quantize_model.py 固定在项目源码目录，可直接写死：
    // QString scriptPath = "E:/QT-study/untitled2/python/quantize_model.py";

    if (!QFileInfo::exists(scriptPath)) {
        scriptPath = "E:/QT-study/untitled2/python/quantize_model.py";
    }

    QString modelPath = ui->lineEditModelPath->text().trimmed();
    QString calibDir = ui->lineEditCalibDir->text().trimmed();
    QString outputPath = ui->lineEditOutputPath->text().trimmed();

    if (pythonExe.isEmpty() || !QFileInfo::exists(pythonExe)) {
        QMessageBox::warning(this, "提示", "Python 解释器路径无效。");
        return;
    }

    if (scriptPath.isEmpty() || !QFileInfo::exists(scriptPath)) {
        QMessageBox::warning(this, "提示", "量化脚本不存在：\n" + scriptPath);
        return;
    }

    if (modelPath.isEmpty() || !QFileInfo::exists(modelPath)) {
        QMessageBox::warning(this, "提示", "待量化模型不存在。");
        return;
    }

    if (outputPath.isEmpty()) {
        QMessageBox::warning(this, "提示", "请设置量化模型输出路径。");
        return;
    }

    QString modeText = ui->comboBoxQuantMode->currentText();
    QString mode = "static";

    if (modeText.contains("dynamic", Qt::CaseInsensitive) ||
        modeText.contains("动态")) {
        mode = "dynamic";
    }

    if (mode == "static") {
        if (calibDir.isEmpty() || !QFileInfo::exists(calibDir)) {
            QMessageBox::warning(this, "提示", "静态量化需要选择校准图片文件夹。");
            return;
        }
    }

    QStringList args;
    args << "-u"
         << scriptPath
         << "--model" << modelPath
         << "--output" << outputPath
         << "--mode" << mode
         << "--img_size" << QString::number(ui->spinBoxImgSize->value())
         << "--max_calib" << QString::number(ui->spinBoxMaxCalib->value());

    if (mode == "static") {
        args << "--calib_dir" << calibDir;
    }

    if (ui->checkBoxPerChannel->isChecked()) {
        args << "--per_channel";
    }

    ui->textEditQuantLog->clear();
    ui->progressBarQuant->setValue(0);
    ui->btnStartQuant->setEnabled(false);
    ui->btnStopQuant->setEnabled(true);

    appendQuantLog("开始量化...");
    appendQuantLog("命令: " + pythonExe + " " + args.join(" "));

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();

    QStringList pathList = env.value("PATH").split(';', Qt::SkipEmptyParts);
    QStringList filteredPaths;

    for (const QString &p : pathList) {
        QString lower = QDir::toNativeSeparators(p).toLower();

        if (lower.contains("\\qt\\") ||
            lower.contains("\\mingw") ||
            lower.contains("\\llvm")) {
            continue;
        }

        filteredPaths << p;
    }

    QFileInfo pyInfo(pythonExe);
    QString envRoot = pyInfo.absolutePath();

    filteredPaths.removeAll(envRoot);
    filteredPaths.removeAll(envRoot + "\\Scripts");
    filteredPaths.removeAll(envRoot + "\\Library\\bin");

    filteredPaths.prepend(envRoot + "\\Library\\bin");
    filteredPaths.prepend(envRoot + "\\Scripts");
    filteredPaths.prepend(envRoot);

    env.insert("PATH", filteredPaths.join(';'));
    env.insert("PYTHONUNBUFFERED", "1");
    env.insert("PYTHONIOENCODING", "utf-8");
    env.insert("PYTHONUTF8", "1");
    env.insert("KMP_DUPLICATE_LIB_OK", "TRUE");
    env.insert("OMP_NUM_THREADS", "1");

    m_quantProcess->setProcessEnvironment(env);
    m_quantProcess->setWorkingDirectory(QFileInfo(scriptPath).absolutePath());
    m_quantProcess->setProcessChannelMode(QProcess::SeparateChannels);
    m_quantProcess->setProgram(pythonExe);
    m_quantProcess->setArguments(args);

    m_quantBuffer.clear();

    m_quantProcess->start();

    if (!m_quantProcess->waitForStarted(3000)) {
        appendQuantLog("量化进程启动失败。");
        appendQuantLog("[ERR] " + m_quantProcess->errorString());

        ui->btnStartQuant->setEnabled(true);
        ui->btnStopQuant->setEnabled(false);

        return;
    }

    appendQuantLog("量化进程已启动。");
}


void MainWindow::onStopQuant()
{
    if (m_quantProcess->state() == QProcess::NotRunning) {
        return;
    }

    appendQuantLog("正在停止量化进程...");
    m_quantProcess->kill();
}


void MainWindow::onQuantReadyReadStdOut()
{
    QByteArray data = m_quantProcess->readAllStandardOutput();
    QString text = QString::fromUtf8(data);

    m_quantBuffer += text;

    QStringList lines = m_quantBuffer.split('\n');

    if (!m_quantBuffer.endsWith('\n')) {
        m_quantBuffer = lines.takeLast();
    } else {
        m_quantBuffer.clear();
    }

    for (const QString &line : lines) {
        handleQuantLine(line);
    }
}


void MainWindow::onQuantReadyReadStdErr()
{
    QByteArray data = m_quantProcess->readAllStandardError();
    QString text = QString::fromUtf8(data);

    text = cleanQuantConsoleText(text);

    if (!text.trimmed().isEmpty()) {
        appendQuantLog("[ERR] " + text);
    }
}


void MainWindow::onQuantFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    Q_UNUSED(exitStatus);

    ui->btnStartQuant->setEnabled(true);
    ui->btnStopQuant->setEnabled(false);

    if (exitCode == 0) {
        ui->progressBarQuant->setValue(100);
        appendQuantLog("量化结束：成功。");

        QString outputPath = ui->lineEditOutputPath->text().trimmed();

        if (QFileInfo::exists(outputPath)) {
            appendQuantLog("量化模型已生成: " + outputPath);

            if (ui->lineEditInferOnnxModelPath) {
                ui->lineEditInferOnnxModelPath->setText(outputPath);
                m_inferOnnxModelPath = outputPath;
                m_inferModelLoaded = false;
                m_inferModelPath.clear();
                m_inferModelType.clear();
                ui->labelInferStatus->setText("推理状态：ONNX量化模型已生成，未加载");
            }
        }

        QMessageBox::information(this, "完成", "模型量化完成。");
    }
    else {
        appendQuantLog(QString("量化结束：失败，退出码 = %1").arg(exitCode));
        QMessageBox::warning(this, "失败", "模型量化失败，请查看日志。");
    }
}

// ===================== 量化模型推理模块 =====================

void MainWindow::onSelectInferPtModelPath()
{
    const QString projectRoot = projectRootFromScriptPath(ui->lineEditScript->text().trimmed());
    QString startDir = QDir(projectRoot).filePath("runs");
    if (!QDir(startDir).exists()) {
        startDir = QDir(projectRoot).filePath("models");
    }

    QString currentPath = ui->lineEditInferPtModelPath->text().trimmed();
    if (!currentPath.isEmpty()) {
        QFileInfo info(currentPath);
        if (info.exists()) {
            startDir = info.absolutePath();
        }
    }

    QString file = QFileDialog::getOpenFileName(
        this,
        "选择 PT 推理模型",
        startDir,
        "PyTorch Model (*.pt);;All Files (*)"
        );

    if (file.isEmpty()) {
        return;
    }

    m_inferPtModelPath = file;
    m_inferModelLoaded = false;
    m_inferModelPath.clear();
    m_inferModelType.clear();
    ui->lineEditInferPtModelPath->setText(file);
    ui->labelInferStatus->setText("推理状态：已选择PT模型，未加载");
}


void MainWindow::onLoadInferPtModel()
{
    m_inferPtModelPath = ui->lineEditInferPtModelPath->text().trimmed();

    if (m_inferPtModelPath.isEmpty()) {
        QMessageBox::warning(this, "提示", "请先选择 PT 模型文件。");
        return;
    }

    QFileInfo info(m_inferPtModelPath);
    if (!info.exists()) {
        QMessageBox::warning(this, "错误", "PT 模型文件不存在：\n" + m_inferPtModelPath);
        ui->labelInferStatus->setText("推理状态：PT模型文件不存在");
        m_inferModelLoaded = false;
        return;
    }

    if (info.suffix().toLower() != "pt") {
        QMessageBox::warning(this, "错误", "请选择 .pt 模型文件。");
        ui->labelInferStatus->setText("推理状态：PT模型格式不支持");
        m_inferModelLoaded = false;
        return;
    }

    // Qt 端只做路径和格式检查，真正加载在 Python 推理脚本中完成。
    m_inferModelPath = m_inferPtModelPath;
    m_inferModelType = "pt";
    m_inferModelLoaded = true;
    ui->labelInferStatus->setText("推理状态：PT模型已加载，当前推理将调用 infer_pt.py");
}


void MainWindow::onSelectInferOnnxModelPath()
{
    const QString projectRoot = projectRootFromScriptPath(ui->lineEditScript->text().trimmed());
    QString startDir = QDir(projectRoot).filePath("models");
    if (!QDir(startDir).exists()) {
        startDir = QDir(projectRoot).filePath("runs");
    }

    QString currentPath = ui->lineEditInferOnnxModelPath->text().trimmed();
    if (!currentPath.isEmpty()) {
        QFileInfo info(currentPath);
        if (info.exists()) {
            startDir = info.absolutePath();
        }
    }

    QString file = QFileDialog::getOpenFileName(
        this,
        "选择 ONNX 推理模型",
        startDir,
        "ONNX Model (*.onnx);;All Files (*)"
        );

    if (file.isEmpty()) {
        return;
    }

    m_inferOnnxModelPath = file;
    m_inferModelLoaded = false;
    m_inferModelPath.clear();
    m_inferModelType.clear();
    ui->lineEditInferOnnxModelPath->setText(file);
    ui->labelInferStatus->setText("推理状态：已选择ONNX模型，未加载");
}


void MainWindow::onLoadInferOnnxModel()
{
    m_inferOnnxModelPath = ui->lineEditInferOnnxModelPath->text().trimmed();

    if (m_inferOnnxModelPath.isEmpty()) {
        QMessageBox::warning(this, "提示", "请先选择 ONNX 模型文件。");
        return;
    }

    QFileInfo info(m_inferOnnxModelPath);
    if (!info.exists()) {
        QMessageBox::warning(this, "错误", "ONNX 模型文件不存在：\n" + m_inferOnnxModelPath);
        ui->labelInferStatus->setText("推理状态：ONNX模型文件不存在");
        m_inferModelLoaded = false;
        return;
    }

    if (info.suffix().toLower() != "onnx") {
        QMessageBox::warning(this, "错误", "请选择 .onnx 模型文件。");
        ui->labelInferStatus->setText("推理状态：ONNX模型格式不支持");
        m_inferModelLoaded = false;
        return;
    }

    // Qt 端只做路径和格式检查，真正加载在 Python 推理脚本中完成。
    m_inferModelPath = m_inferOnnxModelPath;
    m_inferModelType = "onnx";
    m_inferModelLoaded = true;
    ui->labelInferStatus->setText("推理状态：ONNX模型已加载，当前推理将调用 infer_onnx.py");
}


void MainWindow::onSelectInferImage()
{
    QString startDir = "E:/QT-study/untitled2/data/train";

    QString currentPath = ui->lineEditInferImagePath->text().trimmed();
    if (!currentPath.isEmpty()) {
        QFileInfo info(currentPath);
        if (info.exists()) {
            startDir = info.absolutePath();
        }
    }

    QString file = QFileDialog::getOpenFileName(
        this,
        "选择测试图片",
        startDir,
        "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
    );

    if (file.isEmpty()) {
        return;
    }

    m_inferImagePath = file;
    ui->lineEditInferImagePath->setText(file);
    ui->labelInferStatus->setText("推理状态：已选择测试图片");

    QPixmap pix(file);
    if (!pix.isNull()) {
        ui->labelImage->setPixmap(
            pix.scaled(ui->labelImage->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation)
        );
    }
}


void MainWindow::onStartInfer()
{
    m_inferImagePath = ui->lineEditInferImagePath->text().trimmed();

    if (!m_inferModelLoaded || m_inferModelPath.trimmed().isEmpty()) {
        QMessageBox::warning(this, "提示", "请先加载 PT 模型或 ONNX 模型。");
        return;
    }

    if (m_inferModelPath.isEmpty() || !QFileInfo::exists(m_inferModelPath)) {
        QMessageBox::warning(this, "错误", "当前已加载的模型文件不存在。\n请重新选择并加载模型。");
        m_inferModelLoaded = false;
        ui->labelInferStatus->setText("推理状态：模型文件不存在");
        return;
    }

    if (m_inferImagePath.isEmpty() || !QFileInfo::exists(m_inferImagePath)) {
        QMessageBox::warning(this, "提示", "请先选择测试图片。");
        return;
    }

    QFileInfo modelInfo(m_inferModelPath);
    const QString modelSuffix = modelInfo.suffix().toLower();
    if ((m_inferModelType != "pt" && m_inferModelType != "onnx") || modelSuffix != m_inferModelType) {
        QMessageBox::warning(this, "错误", "当前模型类型与文件后缀不一致，请重新加载 PT 或 ONNX 模型。");
        ui->labelInferStatus->setText("推理状态：模型类型不一致");
        m_inferModelLoaded = false;
        return;
    }

    QString pythonExe = ui->lineEditPythonExe->text().trimmed();
    if (pythonExe.isEmpty()) {
        pythonExe = "python";
    }

    const QString scriptName = (m_inferModelType == "pt") ? "infer_pt.py" : "infer_onnx.py";

    QStringList scriptCandidates;

    QString trainScript = ui->lineEditScript->text().trimmed();
    if (!trainScript.isEmpty()) {
        QFileInfo trainInfo(trainScript);
        scriptCandidates << trainInfo.dir().filePath(scriptName);
    }

    scriptCandidates << QCoreApplication::applicationDirPath() + "/python/" + scriptName;
    scriptCandidates << QCoreApplication::applicationDirPath() + "/" + scriptName;
    scriptCandidates << "E:/QT-study/untitled2/python/" + scriptName;

    QDir modelDir(modelInfo.absolutePath());
    for (int i = 0; i < 8; ++i) {
        scriptCandidates << modelDir.filePath(scriptName);
        scriptCandidates << modelDir.filePath("python/" + scriptName);
        if (!modelDir.cdUp()) {
            break;
        }
    }

    QString scriptPath;
    for (const QString &candidate : scriptCandidates) {
        QString cleanPath = QDir::cleanPath(candidate);
        if (QFileInfo::exists(cleanPath)) {
            scriptPath = cleanPath;
            break;
        }
    }

    if (scriptPath.isEmpty()) {
        QMessageBox::warning(
            this,
            "错误",
            "没有找到 " + scriptName + "。\n\n请把 " + scriptName + " 放到：\n"
            "E:/QT-study/untitled2/python/" + scriptName + "\n\n"
            "或者放到 train.py 同一个目录。"
            );
        ui->labelInferStatus->setText("推理状态：缺少 " + scriptName);
        return;
    }

    QString outDir = QFileInfo(m_inferImagePath).absoluteDir().filePath("infer_results");
    QDir().mkpath(outDir);

    QStringList args;
    args << "-u"
         << scriptPath
         << "--model" << m_inferModelPath
         << "--image" << m_inferImagePath
         << "--out-dir" << outDir
         << "--conf" << "0.2"
         << "--iou" << "0.3"
         << "--imgsz" << "640";

    ui->tableWidgetResult->setRowCount(0);
    ui->labelInferStatus->setText("推理状态：正在推理...");
    appendQuantLog((m_inferModelType == "pt") ? "开始 PT 模型推理..." : "开始 ONNX 模型推理...");
    appendQuantLog("命令: " + pythonExe + " " + args.join(" "));

    QProcess inferProcess;

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert("PYTHONUNBUFFERED", "1");
    env.insert("PYTHONIOENCODING", "utf-8");
    env.insert("PYTHONUTF8", "1");
    env.insert("KMP_DUPLICATE_LIB_OK", "TRUE");
    env.insert("OMP_NUM_THREADS", "1");

    QFileInfo pyInfo(pythonExe);
    if (pyInfo.exists()) {
        QStringList pathList = env.value("PATH").split(';', Qt::SkipEmptyParts);
        QStringList filteredPaths;

        for (const QString &p : pathList) {
            QString lower = QDir::toNativeSeparators(p).toLower();
            if (lower.contains("\\qt\\") ||
                lower.contains("\\mingw") ||
                lower.contains("\\llvm")) {
                continue;
            }
            filteredPaths << p;
        }

        QString envRoot = pyInfo.absolutePath();
        filteredPaths.removeAll(envRoot);
        filteredPaths.removeAll(envRoot + "\\Scripts");
        filteredPaths.removeAll(envRoot + "\\Library\\bin");
        filteredPaths.prepend(envRoot + "\\Library\\bin");
        filteredPaths.prepend(envRoot + "\\Scripts");
        filteredPaths.prepend(envRoot);
        env.insert("PATH", filteredPaths.join(';'));
    }

    inferProcess.setProcessEnvironment(env);
    inferProcess.setProgram(pythonExe);
    inferProcess.setArguments(args);
    inferProcess.setWorkingDirectory(QFileInfo(scriptPath).absolutePath());
    inferProcess.setProcessChannelMode(QProcess::MergedChannels);
    inferProcess.start();

    if (!inferProcess.waitForStarted(5000)) {
        ui->labelInferStatus->setText("推理状态：启动失败");
        QMessageBox::warning(this, "错误", "推理进程启动失败：\n" + inferProcess.errorString());
        return;
    }

    // 等待推理结束，同时保持界面响应。
    while (inferProcess.state() != QProcess::NotRunning) {
        inferProcess.waitForFinished(100);
        QCoreApplication::processEvents();
    }

    QByteArray outputBytes = inferProcess.readAllStandardOutput();
    QString outputText = QString::fromUtf8(outputBytes);
    if (outputText.trimmed().isEmpty()) {
        outputText = QString::fromLocal8Bit(outputBytes);
    }

    appendQuantLog(outputText);

    const QString beginMark = "__RESULT_JSON_START__";
    const QString endMark = "__RESULT_JSON_END__";
    int begin = outputText.indexOf(beginMark);
    int end = outputText.indexOf(endMark);

    if (begin < 0 || end < 0 || end <= begin) {
        ui->labelInferStatus->setText("推理状态：结果解析失败");
        if (inferProcess.exitCode() != 0) {
            QMessageBox::warning(this, "错误", "推理脚本执行失败，请查看日志输出。");
        } else {
            QMessageBox::warning(this, "错误", "没有从 " + scriptName + " 输出中找到 JSON 结果。\n请查看日志输出。");
        }
        return;
    }

    begin += beginMark.length();
    QString jsonText = outputText.mid(begin, end - begin).trimmed();

    QJsonParseError parseError;
    QJsonDocument doc = QJsonDocument::fromJson(jsonText.toUtf8(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        ui->labelInferStatus->setText("推理状态：JSON 解析失败");
        QMessageBox::warning(this, "错误", "JSON 解析失败：\n" + parseError.errorString());
        return;
    }

    QJsonObject root = doc.object();
    if (!root.value("success").toBool(false)) {
        QString err = root.value("error").toString("未知错误");
        ui->labelInferStatus->setText("推理状态：失败");
        QMessageBox::warning(this, "错误", "推理失败：\n" + err);
        return;
    }

    QString resultImagePath = root.value("result_image").toString();
    if (!resultImagePath.isEmpty() && QFileInfo::exists(resultImagePath)) {
        QPixmap pix(resultImagePath);
        if (!pix.isNull()) {
            ui->labelImage->setPixmap(
                pix.scaled(ui->labelImage->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation)
                );
        }
    }

    QJsonArray detections = root.value("detections").toArray();
    ui->tableWidgetResult->setRowCount(detections.size());

    for (int i = 0; i < detections.size(); ++i) {
        QJsonObject det = detections.at(i).toObject();

        auto setItem = [&](int col, const QString &text) {
            QTableWidgetItem *item = new QTableWidgetItem(text);
            item->setTextAlignment(Qt::AlignCenter);
            ui->tableWidgetResult->setItem(i, col, item);
        };

        setItem(0, QString::number(i + 1));
        setItem(1, det.value("class_name").toString());
        setItem(2, QString::number(det.value("confidence").toDouble(), 'f', 3));
        setItem(3, QString::number(det.value("x").toDouble(), 'f', 1));
        setItem(4, QString::number(det.value("y").toDouble(), 'f', 1));
        setItem(5, QString::number(det.value("w").toDouble(), 'f', 1));
        setItem(6, QString::number(det.value("h").toDouble(), 'f', 1));
    }

    ui->tableWidgetResult->resizeColumnsToContents();

    saveInferenceResultsToDatabase(root, detections, resultImagePath);

    ui->labelInferStatus->setText(QString("推理状态：完成，检测到 %1 个目标").arg(detections.size()));
}

