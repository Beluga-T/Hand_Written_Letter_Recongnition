import sys
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QComboBox
from PyQt5.QtGui import QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import  QPainter, QPen, QImage, QPixmap

# Load the models
model = tf.keras.models.load_model('trained_models/my_model_CNN.keras')
model_ann = tf.keras.models.load_model('trained_models/my_model_best_ANN.keras')
model_ann_softmax = tf.keras.Sequential([model_ann, tf.keras.layers.Softmax()])  # Wrap the ANN model once here
class_names = [chr(c) for c in range(ord('a'), ord('z') + 1)]

from PyQt5.QtGui import QPainterPath

class DrawingCanvas(QLabel):
    def __init__(self):
        super().__init__()
        pixmap = QImage(280, 280, QImage.Format_RGB32)
        pixmap.fill(Qt.white)
        self.setPixmap(QPixmap.fromImage(pixmap))
        self.last_x, self.last_y = None, None
        self.pen = QPen(Qt.black, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.path = QPainterPath()

    def mouseMoveEvent(self, event):
        if self.last_x is None:
            self.last_x = event.x()
            self.last_y = event.y()
            self.path.moveTo(self.last_x, self.last_y)
            return

        self.path.lineTo(event.x(), event.y())
        painter = QPainter(self.pixmap())
        painter.setPen(self.pen)
        painter.drawPath(self.path)
        painter.end()
        self.update()

        self.last_x = event.x()
        self.last_y = event.y()

    def mouseReleaseEvent(self, event):
        self.last_x = None
        self.last_y = None

    def clear(self):
        self.pixmap().fill(Qt.white)
        self.path = QPainterPath()  # Reset the path when clearing
        self.update()


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.canvas = DrawingCanvas()
        layout.addWidget(self.canvas)

        # Add a QComboBox for model selection
        self.modelChoiceComboBox = QComboBox(self)
        self.modelChoiceComboBox.addItem("CNN Model")
        self.modelChoiceComboBox.addItem("ANN Model")
        layout.addWidget(self.modelChoiceComboBox)

        self.predict_button = QPushButton('Predict')
        self.predict_button.clicked.connect(self.on_predict)
        layout.addWidget(self.predict_button)

        self.clear_button = QPushButton('Clear')
        self.clear_button.clicked.connect(self.on_clear)
        layout.addWidget(self.clear_button)

        self.result_label = QLabel('Prediction will appear here.')
        layout.addWidget(self.result_label)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.setGeometry(100, 100, 300, 400)
        self.setWindowTitle('Sketch2Letter Prediction')
        self.show()

    def on_predict(self):
        img = self.canvas.pixmap().toImage()
        img = img.scaled(28, 28)
        # Convert QImage to numpy array
        ptr = img.constBits()
        ptr.setsize(img.byteCount())
        arr = np.array(ptr).reshape(img.height(), img.width(), 4)  # Format: BGRA
        arr = arr[..., :3]  # Convert to RGB

        # Your prediction logic here
        predicted_class, confidence = self.predict_image(arr)
        self.result_label.setText(f"Predicted: {predicted_class} with confidence: {confidence:.2f}%")

    def on_clear(self):
        self.canvas.clear()
        self.result_label.setText('Prediction will appear here.')

    def predict_image(self, image):
        # Convert to grayscale
        grayscale_np = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        # Invert the grayscale values
        grayscale_np = 255.0 - grayscale_np
        # Normalize to [0, 1]
        image_np = grayscale_np.astype(np.float32) / 255.0
        image_np = image_np.reshape(-1, 28, 28, 1)  # Reshape to match model input shape

        # Choose the model based on the combo box selection
        if self.modelChoiceComboBox.currentText() == 'CNN Model':
            prediction = model.predict(image_np)
        else:  # 'ANN Model'
            prediction = model_ann_softmax.predict(image_np)  # Use the wrapped model directly

        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        print("Predicted class:", class_names[predicted_class])
        print("Confidence: {:2.0f}%".format(confidence))
        return class_names[predicted_class], confidence

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
