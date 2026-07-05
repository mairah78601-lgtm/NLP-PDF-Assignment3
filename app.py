import sys
import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QTextEdit, QPushButton, QFrame, QHBoxLayout
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from transformers import BertTokenizer, BertForSequenceClassification

# Load model
model = BertForSequenceClassification.from_pretrained("saved_bert_model")
tokenizer = BertTokenizer.from_pretrained("saved_bert_model")

labels = ["😡 Negative", "😐 Neutral", "😊 Positive"]


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits).item()
    return labels[pred]


class App(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Sentiment Analyzer")
        self.setGeometry(250, 150, 700, 550)

        self.setStyleSheet("""
            QWidget {
                background-color: #0f172a;
                color: white;
                font-family: Arial;
            }

            QFrame {
                background-color: #111827;
                border-radius: 15px;
            }

            QTextEdit {
                background-color: #1f2937;
                border: 2px solid #374151;
                border-radius: 12px;
                padding: 10px;
                font-size: 14px;
                color: white;
            }

            QPushButton {
                background-color: #3b82f6;
                border-radius: 12px;
                padding: 12px;
                font-size: 15px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #2563eb;
            }

            QLabel#title {
                font-size: 22px;
                font-weight: bold;
                color: #60a5fa;
            }

            QLabel#result {
                font-size: 20px;
                font-weight: bold;
                padding: 10px;
                border-radius: 10px;
                background-color: #1f2937;
            }
        """)

        main_layout = QVBoxLayout()

        # HEADER CARD
        header = QFrame()
        header_layout = QVBoxLayout()

        title = QLabel("🧠 AI SENTIMENT ANALYZER")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Powered by BERT Transformer Model")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #94a3b8; font-size: 13px;")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header.setLayout(header_layout)

        # INPUT CARD
        input_card = QFrame()
        input_layout = QVBoxLayout()

        self.textbox = QTextEdit()
        self.textbox.setPlaceholderText("Type your sentence here...")

        self.btn = QPushButton("🚀 Analyze Sentiment")
        self.btn.clicked.connect(self.run_prediction)

        input_layout.addWidget(self.textbox)
        input_layout.addWidget(self.btn)
        input_card.setLayout(input_layout)

        # RESULT CARD
        result_card = QFrame()
        result_layout = QVBoxLayout()

        self.result = QLabel("Prediction: Waiting...")
        self.result.setObjectName("result")
        self.result.setAlignment(Qt.AlignCenter)

        result_layout.addWidget(self.result)
        result_card.setLayout(result_layout)

        # Add all
        main_layout.addWidget(header)
        main_layout.addSpacing(10)
        main_layout.addWidget(input_card)
        main_layout.addSpacing(10)
        main_layout.addWidget(result_card)

        self.setLayout(main_layout)

    def run_prediction(self):
        text = self.textbox.toPlainText()

        if text.strip() == "":
            self.result.setText("⚠ Please enter text")
            return

        pred = predict_sentiment(text)
        self.result.setText(f"Prediction: {pred}")


app = QApplication(sys.argv)
window = App()
window.show()
sys.exit(app.exec_())