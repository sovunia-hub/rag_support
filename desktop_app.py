import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from rag_pipeline import RagPipeline

class MessageBubble(QLabel):
    def __init__(self, text, is_user=True):
        super().__init__(text)
        self.setWordWrap(True)
        self.setMargin(12)
        self.setFont(QFont("Segoe UI", 11))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.setStyleSheet(f'''
            QLabel {{
                background-color: {'#cbeffd' if is_user else '#f0f0f0'};
                border-radius: 14px;
                padding: 10px 14px;
                color: #2c2c2c;
            }}
        ''')
        self.setAlignment(Qt.AlignmentFlag.AlignRight if is_user else Qt.AlignmentFlag.AlignLeft)

class RAGDesktopApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.pipeline = RagPipeline()

    def init_ui(self):
        self.setWindowTitle('RAG Assistant')
        self.setGeometry(100, 100, 700, 600)

        self.layout = QVBoxLayout(self)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: #ffffff;")

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.addStretch(1)
        self.scroll_area.setWidget(self.scroll_content)

        self.input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setFont(QFont("Segoe UI", 11))
        self.input_field.setPlaceholderText("Введите ваш запрос...")

        self.send_button = QPushButton('➤')
        self.send_button.setFixedWidth(60)
        self.send_button.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.send_button.clicked.connect(self.on_submit)
        self.input_field.returnPressed.connect(self.on_submit)

        self.clear_button = QPushButton('Очистить')
        self.clear_button.setFont(QFont("Segoe UI", 10))
        self.clear_button.clicked.connect(self.clear_history)

        self.input_layout.addWidget(self.input_field)
        self.input_layout.addWidget(self.send_button)
        self.input_layout.addWidget(self.clear_button)

        self.layout.addWidget(self.scroll_area)
        self.layout.addLayout(self.input_layout)

    def clear_history(self):
        reply = QMessageBox.question(
            self,
            "Очистка истории",
            "Вы уверены, что хотите очистить историю?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            for i in reversed(range(self.scroll_layout.count() - 1)):
                widget = self.scroll_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            self.pipeline.clear_history()

    def add_message(self, text, is_user=True):
        bubble = MessageBubble(text, is_user)
        wrapper = QHBoxLayout()
        wrapper.setAlignment(Qt.AlignmentFlag.AlignRight if is_user else Qt.AlignmentFlag.AlignLeft)
        wrapper.addWidget(bubble)

        container = QWidget()
        container.setLayout(wrapper)

        self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, container)
        QApplication.processEvents()
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    def on_submit(self):
        query = self.input_field.text().strip()
        if not query:
            return

        self.add_message(query, is_user=True)
        self.input_field.clear()
        self.add_message("⏳ Генерация ответа...", is_user=False)

        QApplication.processEvents()

        try:
            answer, time = self.pipeline.generate(query)
            self.scroll_layout.itemAt(self.scroll_layout.count() - 2).widget().deleteLater()
            self.add_message(answer, is_user=False)
            print(time)
        except Exception as e:
            self.scroll_layout.itemAt(self.scroll_layout.count() - 2).widget().deleteLater()
            self.add_message(f"Ошибка: {e}", is_user=False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RAGDesktopApp()
    window.show()
    sys.exit(app.exec())
