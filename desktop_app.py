import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit
)
from rag_pipeline import RagPipeline # Подключаем свою генеративную модель

class RAGDesktopApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.pipeline = RagPipeline()

    def init_ui(self):
        self.setWindowTitle('RAG Assistant')
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        self.input_label = QLabel('Введите ваш запрос:')
        self.input_field = QLineEdit()
        self.submit_button = QPushButton('Получить ответ')
        self.submit_button.clicked.connect(self.on_submit)

        self.output_label = QLabel('Ответ:')
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)

        layout.addWidget(self.input_label)
        layout.addWidget(self.input_field)
        layout.addWidget(self.submit_button)
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_area)

        self.setLayout(layout)

    def on_submit(self):
        query = self.input_field.text().strip()
        if query:
            self.output_area.setText("⏳ Генерация ответа...")
            QApplication.processEvents()
            try:
                answer, time = self.pipeline.generate(query)
                self.output_area.setText(answer)
                print(time)
            except Exception as e:
                self.output_area.setText(f"Ошибка при генерации ответа:\n{e}")
        else:
            self.output_area.setText("Введите запрос для начала.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RAGDesktopApp()
    window.show()
    sys.exit(app.exec())