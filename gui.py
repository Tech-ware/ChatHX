import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLineEdit, QPushButton, QDialog, QLabel, QFrame, 
                             QFileDialog, QProgressDialog, QComboBox, 
                             QMessageBox, QSlider, QScrollArea)
from PyQt6.QtGui import QIcon, QFont
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
from llama_cpp import Llama
import openrag
import re
import os
import json

from PyQt6.QtCore import QThread, pyqtSignal
import traceback

class LLMInferenceThread(QThread):
    token_ready = pyqtSignal(str)
    inference_finished = pyqtSignal()

    def __init__(self, rag_llm, query, context_length, use_rag=True):
        super().__init__()
        self.rag_llm = rag_llm
        self.query = query
        self.context_length = int(context_length)
        self.use_rag = use_rag

    def run(self):
        try:
            print("LLMInferenceThread started")
            if not isinstance(self.context_length, int):
                raise TypeError(
                    f"context_length should be an integer, got {type(self.context_length)} instead."
                )

            print("Generating response...")
            if self.use_rag:
                response_generator = self.rag_llm.generate_response(self.query, context_length=self.context_length)
            else:
                response_generator = self.rag_llm.generate_response_without_rag(self.query, context_length=self.context_length)

            for token in response_generator:
                print(f"Emitting token: {token}")
                self.token_ready.emit(token)

            print("All tokens emitted")

        except Exception as e:
            print(f"Error in LLMInferenceThread: {str(e)}")
            print(traceback.format_exc())
            self.token_ready.emit(f"Error: {str(e)}")
        finally:
            self.inference_finished.emit() 
        print("LLMInferenceThread finished")

class DatasetLoadingThread(QThread):
    progress_update = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, dataset_path, search_engine):
        super().__init__()
        self.dataset_path = dataset_path
        self.search_engine = search_engine

    def run(self):
        try:
            self.search_engine.load_documents(self.dataset_path)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

class ChatMessageBox(QWidget):
    def __init__(self, text, is_user, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        self.label = QLabel(text)
        if is_user:
            layout.addWidget(self.label)
        else:
            layout.addStretch(1)
            layout.addWidget(self.label)
        self.setLayout(layout)

    def setText(self, text):
        self.label.setText(text)

    def text(self):
        return self.label.text()

class ChatHXApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChatHX")
        self.setGeometry(100, 100, 1200, 800)

        # Default GUI settings
        self.gui_scale = 1.15
        self.dark_mode = True

        self.setStyleSheet(self.get_stylesheet())
        
        self.llm_model = None
        self.search_engine = openrag.LocalSemanticSearchEngine()
        self.model_config = {}
        self.loading_thread = None
        self.loading_progress = None
        self.is_inferencing = False
        self.max_token_length = 1024
        self.context_length = 2048
        self.max_context_length = self.context_length 
        self.loaded_model_name = None 
        self.last_model_path = None
        self.last_dataset_path = None
        self.current_response_box = None
        self.use_rag = True

        self.model_history = []
        self.load_history()
        self.create_widgets()

        self.llm_thread = None
        self.rag_llm = None

    def create_widgets(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        # Top section (AI model and Dataset)
        top_layout = QHBoxLayout()

        # AI model section
        ai_model_frame = QFrame()
        ai_model_frame.setFrameShape(QFrame.Shape.StyledPanel)
        ai_model_layout = QVBoxLayout()

        ai_model_label = QLabel("AI model")
        ai_model_label.setFont(QFont("Arial", int(12 * self.gui_scale)))
        ai_model_layout.addWidget(ai_model_label)

        self.model_combo = QComboBox()
        self.model_combo.addItem("Select AI model")

        if self.model_history:
            for model_data in self.model_history: 
                model_name = model_data.get("model_name", "Unknown Model")
                self.model_combo.addItem(model_name) 
        ai_model_layout.addWidget(self.model_combo)

        self.add_model_button = QPushButton("Add new models")
        self.add_model_button.clicked.connect(self.browse_model)
        ai_model_layout.addWidget(self.add_model_button)

        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("Model Path")
        if self.last_model_path:
            self.model_path.setText(self.last_model_path)
        ai_model_layout.addWidget(self.model_path)

        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)
        ai_model_layout.addWidget(self.load_model_button)

        self.eject_model_button = QPushButton("Eject Model")
        self.eject_model_button.clicked.connect(self.eject_model)
        ai_model_layout.addWidget(self.eject_model_button)

        self.max_token_length_button = QPushButton(QIcon("./assets/edit.png"), "Max Token Length")
        self.max_token_length_button.clicked.connect(self.show_max_token_length_slider)
        ai_model_layout.addWidget(self.max_token_length_button)

        self.context_length_button = QPushButton(QIcon("./assets/edit.png"), "Context Length")
        self.context_length_button.clicked.connect(self.show_context_length_slider)
        ai_model_layout.addWidget(self.context_length_button)

        ai_model_frame.setLayout(ai_model_layout)
        top_layout.addWidget(ai_model_frame)

        # Dataset section
        dataset_frame = QFrame()
        dataset_frame.setFrameShape(QFrame.Shape.StyledPanel)
        dataset_layout = QVBoxLayout()

        dataset_label = QLabel("Dataset")
        dataset_label.setFont(QFont("Arial", int(12 * self.gui_scale)))
        dataset_layout.addWidget(dataset_label)

        dataset_info = QLabel("txt files supported")
        dataset_info.setFont(QFont("Arial", int(10 * self.gui_scale)))
        dataset_layout.addWidget(dataset_info)

        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("Folder Path")
        if self.last_dataset_path:
            self.dataset_path.setText(self.last_dataset_path)
        dataset_layout.addWidget(self.dataset_path)

        self.browse_dataset_button = QPushButton(QIcon("./assets/folder.png"), "Choose your folder")
        self.browse_dataset_button.setIconSize(QSize(int(20 * self.gui_scale), int(20 * self.gui_scale)))
        self.browse_dataset_button.clicked.connect(self.browse_dataset)
        dataset_layout.addWidget(self.browse_dataset_button)

        self.load_dataset_button = QPushButton("Load Dataset")
        self.load_dataset_button.clicked.connect(self.load_dataset)
        dataset_layout.addWidget(self.load_dataset_button)

        # Use Model Knowledge button
        self.use_model_knowledge_button = QPushButton("Use Model Knowledge")
        self.use_model_knowledge_button.clicked.connect(self.toggle_rag)
        dataset_layout.addWidget(self.use_model_knowledge_button)

        dataset_frame.setLayout(dataset_layout)
        top_layout.addWidget(dataset_frame)

        main_layout.addLayout(top_layout)

        # Add copy and stop buttons
        button_layout = QHBoxLayout()
        self.copy_button = QPushButton("Copy")
        self.copy_button.clicked.connect(self.copy_latest_response)
        button_layout.addWidget(self.copy_button)
    
        self.stop_button = QPushButton("STOP")
        self.stop_button.setStyleSheet("background-color: #808080; color: #ffffff;")
        self.stop_button.clicked.connect(self.stop_inference)
        self.stop_button.setFixedWidth(100)
        button_layout.addWidget(self.stop_button)
    
        button_layout.addStretch(1)
        main_layout.addLayout(button_layout)

        # Chat area
        chat_layout = QVBoxLayout()

        # Make chat history scrollable
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True) 

        self.chat_history_frame = QFrame()
        self.chat_history_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.chat_history_layout = QVBoxLayout()
        self.chat_history_layout.addStretch(1)
        self.chat_history_frame.setLayout(self.chat_history_layout)

        scroll_area.setWidget(self.chat_history_frame)
        chat_layout.addWidget(scroll_area)

        main_layout.addLayout(chat_layout)

        # Suggested questions
        self.questions_layout = QHBoxLayout() 
        self.suggested_questions = [
            "How fast is the new Snapdragon X Elite ?",
            "How many cores would the leaked Core Ultra 7 268V have ?",
            "What is the date for the upcoming Galaxy Unpacked event ?",
            "What are the expected specs for the upcoming RTX 5090 ?"
        ]
        for question in self.suggested_questions:
            question_button = QPushButton(question)
            question_button.setFont(QFont("Arial", int(9 * self.gui_scale)))
            question_button.clicked.connect(lambda _, q=question: self.set_question(q))
            self.questions_layout.addWidget(question_button)
        main_layout.addLayout(self.questions_layout)

        # Input area
        input_layout = QHBoxLayout()
        self.input_area = QLineEdit()
        self.input_area.setFont(QFont("Arial", int(10 * self.gui_scale)))
        self.input_area.setPlaceholderText("ChatHX: Type your question")
        self.input_area.setFixedHeight(int(30 * self.gui_scale))
        input_layout.addWidget(self.input_area)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_chat_history)
        self.clear_button.setFixedHeight(int(30 * self.gui_scale))
        input_layout.addWidget(self.clear_button)

        self.send_button = QPushButton("SEND")
        self.send_button.setStyleSheet("background-color: #76b900; color: #ffffff;")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setFixedHeight(int(30 * self.gui_scale))
        self.send_button.setFixedWidth(int(100 * self.gui_scale))
        input_layout.addWidget(self.send_button)

        main_layout.addLayout(input_layout)

        # Footer
        footer_label = QLabel("ChatHX response quality depends on the AI model's accuracy and the input dataset. Please verify important information.")
        footer_label.setFont(QFont("Arial", int(9 * self.gui_scale)))
        main_layout.addWidget(footer_label)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)


    def show_context_length_slider(self):
        slider_dialog = QDialog(self)
        slider_dialog.setWindowTitle("Set Context Length")
        layout = QVBoxLayout()

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(128)
        slider.setMaximum(2048)
        slider.setValue(self.context_length)
        slider.setTickInterval(64) 
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        value_label = QLabel(str(self.context_length))

        def update_value(value):
            self.context_length = round(value / 64) * 64
            slider.setValue(self.context_length)
            value_label.setText(str(self.context_length))

        slider.valueChanged.connect(update_value)

        layout.addWidget(slider)
        layout.addWidget(value_label)

        slider_dialog.setLayout(layout)
        slider_dialog.exec()

    def show_max_token_length_slider(self):
        slider_dialog = QDialog(self)
        slider_dialog.setWindowTitle("Set Max Token Length")
        layout = QVBoxLayout()

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(64)
        slider.setMaximum(4096)
        slider.setValue(self.max_token_length)
        slider.setTickInterval(64)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        value_label = QLabel(str(self.max_token_length))

        def update_value(value):
            self.max_token_length = round(value / 64) * 64
            slider.setValue(self.max_token_length)
            value_label.setText(str(self.max_token_length))

        slider.valueChanged.connect(update_value)

        layout.addWidget(slider)
        layout.addWidget(value_label)

        slider_dialog.setLayout(layout)
        slider_dialog.exec()


    def copy_latest_response(self):
        if self.chat_history_layout.count() > 1:
            latest_message = self.chat_history_layout.itemAt(self.chat_history_layout.count() - 1).widget()
            if isinstance(latest_message, ChatMessageBox) and not latest_message.is_user:
                clipboard = QApplication.clipboard()
                clipboard.setText(latest_message.text())
                QMessageBox.information(self, "Copied", "Latest response copied to clipboard.")

    def stop_inference(self):
        if self.is_inferencing:
            if self.llm_thread and self.llm_thread.isRunning():
                self.llm_thread.terminate()
                self.llm_thread.wait()
                self.send_button.setEnabled(True)
                self.is_inferencing = False
                self.stop_button.setStyleSheet("background-color: #808080; color: #ffffff;")
                QMessageBox.information(self, "Info", "Inference stopped.")
        else:
            QMessageBox.warning(self, "Info", "No inference running to stop.")

    def get_stylesheet(self):
        return f"""
            QMainWindow, QWidget {{
                background-color: #1e1e1e;
                color: #ffffff;
            }}
            QComboBox, QLineEdit, QPushButton, QTextEdit, QSpinBox {{
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                padding: 5px;
                font-size: {10 * self.gui_scale}px;
            }}
            QPushButton {{
                background-color: #3d3d3d;
                font-size: {10 * self.gui_scale}px;
            }}
            QPushButton:hover {{
                background-color: #4d4d4d;
            }}
            QLabel {{
                color: #ffffff;
                font-size: {10 * self.gui_scale}px;
            }}
            QFrame {{
                background-color: #282c34;
                border: 1px solid #3d3d3d;
                padding: 10px;
            }}
            QScrollArea {{
                background-color: #1e1e1e;
                border: none;
            }}
            QTextEdit {{
                font-size: {10 * self.gui_scale}px;
            }}
            QMenu {{
                background-color: #2d2d2d;
                color: #ffffff;
            }}
            QMenu::item {{
                background-color: transparent;
            }}
            QMenu::item:selected {{
                background-color: #3d3d3d;
            }}
            QMenu::separator {{
                height: 1px;
                background-color: #3d3d3d;
            }}
            {f"QComboBox {{ color: {'#ffffff' if self.dark_mode else '#000000'}; }}" if self.dark_mode else ""}
            {f"QLineEdit {{ color: {'#ffffff' if self.dark_mode else '#000000'}; }}" if self.dark_mode else ""}
        """
        
    def clear_chat_history(self):
        for i in reversed(range(self.chat_history_layout.count())):
            widget = self.chat_history_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def set_question(self, question):
        self.input_area.setText(question)

    def send_message(self):
        user_message = self.input_area.text().strip()
        if not user_message:
            return

        if not self.llm_model:
            QMessageBox.warning(self, "Error", "Please load an LLM model first.")
            return

        try:
            print("Sending message:", user_message)
            if not self.rag_llm:
                self.rag_llm = openrag.RAGEnabledLLM(self.llm_model, self.search_engine, self.max_token_length)

            # Display user message
            user_message_box = ChatMessageBox(f"You: {user_message}", is_user=True)
            self.chat_history_layout.addWidget(user_message_box)

            # Create a new message box for the AI response
            self.current_response_box = ChatMessageBox("AI model: ", is_user=False)
            self.chat_history_layout.addWidget(self.current_response_box)

            self.input_area.clear()
            self.send_button.setEnabled(False)
            self.is_inferencing = True
            self.stop_button.setStyleSheet("background-color: #FF0000; color: #ffffff;")

            # Start response processing
            self.full_response = ""
            prompt = self.create_prompt(user_message) if self.use_rag else self.create_prompt_without_rag(user_message)
            print("Created prompt:", prompt)
        
            self.llm_thread = LLMInferenceThread(self.rag_llm, prompt, self.context_length, use_rag=self.use_rag)
            self.llm_thread.token_ready.connect(self.handle_token)
            self.llm_thread.inference_finished.connect(self.process_full_response)
            self.llm_thread.start()
            print("Started LLM inference thread")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during processing: {str(e)}")
            print(f"Detailed error: {e}")

    def handle_token(self, token):
        print("Received token:", token)
        self.full_response += token
        self.update_chat_history(token, from_user=False) 

        # Check for completion 
        if "<|end_of_text|>" in self.full_response:
            self.llm_thread.inference_finished.connect(self.process_full_response) 

    def process_full_response(self):
        print("Processing full response")
        self.is_inferencing = False
        self.stop_button.setStyleSheet("background-color: #808080; color: #ffffff;") 
        self.send_button.setEnabled(True)
        self.llm_thread.inference_finished.disconnect(self.process_full_response)  # Disconnect the signal

        try:
            # Extract answer and follow-up questions
            answer, follow_up_questions = self.extract_answer_and_questions(self.full_response)

            # Update the final response (if needed)
            if self.current_response_box:
                self.current_response_box.setText(f"AI model: {answer}")
            print("Updating final response:", answer) 

            # Update suggested questions
            self.suggested_questions = follow_up_questions
            self.update_suggested_questions()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing response: {str(e)}")
        
        self.full_response = ""  # Reset for next response
        self.current_response_box = None

    def create_prompt(self, query):
        most_similar_file, similarity_score = self.search_engine.search(query)
        if most_similar_file and similarity_score > 0.4:
            with open(most_similar_file, 'r', encoding='utf-8') as file:
                context = file.read()
            # Ensure context_length is an integer when used for slicing
            context = context[:min(int(self.context_length), self.max_context_length - len(query) - 100)] 
        else:
            context = ""

        # Escape curly braces in context and query
        context = context.replace("{", "{{").replace("}", "}}")
        query = query.replace("{", "{{").replace("}", "}}")

        # The prompt passed to the model
        prompt = f"""
        You are a helpful and informative AI assistant. 
        Context: '{context}'

        Question: {query}

        Answer the question based on the context and your prior knowledge. 
        If you don't have enough information to answer the question, say that.
        After your answer, provide 3 follow-up questions that are relevant to the context and the user's original question.

        You MUST format your response as a valid JSON object EXACTLY like this:

        ```json
        {{{{
         "answer": "Your answer here.",
         "follow_up_questions": [
         "Question 1",
         "Question 2",
         "Question 3"
         ]
        }}}}
        ```
        Do NOT include anything else in your response.
        """
        return prompt

    def create_prompt_without_rag(self, query):
        """Prompt template without the context section for using the model's own knowledge."""
        query = query.replace("{", "{{").replace("}", "}}")  # Escape curly braces
        prompt = f"""
        You are a helpful and informative AI assistant.

        Question: {query}

        Answer the question based on your own knowledge.
        If you don't have enough information to answer the question, say that.
        After your answer, provide 3 follow-up questions that are relevant to the user's original question.

        You MUST format your response as a valid JSON object EXACTLY like this:

        ```json
        {{{{
         "answer": "Your answer here.",
         "follow_up_questions": [
         "Question 1",
         "Question 2",
         "Question 3"
         ]
        }}}}
        ```
        Do NOT include anything else in your response.
        """
        return prompt

    def _try_parse_json(self, response):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Attempt to fix common JSON formatting issues
            response = response.replace("'", "\"")
            response = response.replace("```json", "").replace("```", "")
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return None

    def extract_answer_and_questions(self, response):
        """
        Extracts answer and follow-up questions from the LLM response.

        Handles variations in formatting, including code block wrapping,
        and prioritizes answer extraction even with potential errors.
        """
        answer = "Error: Could not process model output. Please try rephrasing."
        follow_up_questions = []

        try:
            # 1. Extract JSON if present (handling code block variations):
            json_match = re.search(r"(?i)```json\s*(.*?)\s*```", response, re.DOTALL)
            if not json_match:
                json_match = re.search(r"({.*})", response, re.DOTALL)  # Try without code block

            if json_match:
                json_str = json_match.group(1).strip()
                try:
                    response_json = json.loads(json_str)
                    answer = response_json.get("answer", answer)
                    follow_up_questions = response_json.get("follow_up_questions", [])
                    return answer, follow_up_questions  # Return early if JSON parsing succeeds
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    # Proceed to the next steps if JSON parsing fails

            # 2. Fallback mechanisms:
            answer_match = re.search(r"(?i)answer\s*[:-]\s*(.*?)(?=\n\n|\Z)", response, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                potential_questions_start = re.search(r"(?i)\n[-•*]", response)
                if potential_questions_start:
                    answer = response[:potential_questions_start.start()].strip()
                else:
                    answer = response.strip()

            questions_match = re.findall(r"\n\s*[-•* ]\s*(.*?)\n", response)
            follow_up_questions = [q.strip() for q in questions_match]

        except Exception as e:
            print(f"Error extracting information from response: {e}")

        # Ensure answer is always a string
        if not isinstance(answer, str):
             answer = "Error: Could not process model output. Please try rephrasing."

        # Ensure follow_up_questions is always a list
        if not isinstance(follow_up_questions, list):
            follow_up_questions = []
        else:
            # Ensure all elements in the list are strings
            follow_up_questions = [str(q) for q in follow_up_questions]

        return answer, follow_up_questions

    def update_suggested_questions(self):
        # Clear existing buttons
        for i in reversed(range(self.questions_layout.count())):
            widget = self.questions_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        # Add new buttons
        for question in self.suggested_questions:
            question_button = QPushButton(question)
            question_button.setFont(QFont("Arial", int(9 * self.gui_scale)))
            question_button.clicked.connect(lambda _, q=question: self.set_question(q))
            self.questions_layout.addWidget(question_button)

    def process_response(self):
        if self.rag_llm and self.rag_llm.response:
            response = self.rag_llm.response.strip()
            self.rag_llm.response = ""
            self.update_chat_history(response, from_user=False)

    def browse_model(self):
        model_path = QFileDialog.getOpenFileName(self, "Select Model File")[0]
        if model_path:
            self.model_path.setText(model_path)

    def load_model(self):
        selected_index = self.model_combo.currentIndex()
        model_path = "" # Initialize model_path
        if selected_index > 0:  # 0 is "Select AI Model"
            selected_model_data = self.model_history[selected_index - 1]
            model_path = selected_model_data.get("model_path", "")
        else:
            # Handle case where "Add new models" is selected
            model_path = QFileDialog.getOpenFileName(self, "Select Model File")[0]
            if model_path:
                self.model_path.setText(model_path)
                model_name = os.path.basename(model_path)
                self.model_history.append({"model_name": model_name, "model_path": model_path})
                self.model_combo.addItem(model_name)  # Add to the combo box
        
        if model_path:
            try:
                self.llm_model = Llama(model_path=model_path, n_ctx=self.context_length) 
                self.loaded_model_name = os.path.basename(model_path)
                self.last_model_path = model_path
                self.save_history()
                QMessageBox.information(self, "Success", f"LLM model '{self.loaded_model_name}' loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")

    def eject_model(self):
        if self.llm_model:
            self.llm_model = None
            self.loaded_model_name = None
            self.model_path.clear()
            QMessageBox.information(self, "Success", "Model ejected successfully.")
        else:
            QMessageBox.warning(self, "Info", "No model loaded to eject.")

    def browse_dataset(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder_path:
            self.dataset_path.setText(folder_path)

    def load_dataset(self):
        dataset_path = self.dataset_path.text()
        if not dataset_path:
            QMessageBox.warning(self, "Error", "Please select a dataset folder.")
            return

        self.loading_thread = DatasetLoadingThread(dataset_path, self.search_engine)
        self.loading_thread.progress_update.connect(self.update_progress)
        self.loading_thread.finished.connect(self.loading_finished)
        self.loading_thread.error.connect(lambda err: QMessageBox.critical(self, "Error", f"Failed to load dataset: {err}"))
        self.loading_thread.start()

        self.loading_progress = QProgressDialog("Loading Dataset...", "Cancel", 0, 100, self)
        self.loading_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.loading_progress.show()

    def update_progress(self, value):
        self.loading_progress.setValue(value)

    def loading_finished(self):
        self.loading_progress.close()
        self.last_dataset_path = self.dataset_path.text()
        self.save_history()
        QMessageBox.information(self, "Success", "Dataset loaded successfully.")
    
    def toggle_rag(self):
        """Toggles between using RAG and the model's knowledge."""
        self.use_rag = not self.use_rag
        button_text = "Use Model Knowledge" if self.use_rag else "Use Dataset"
        self.use_model_knowledge_button.setText(button_text)

        if self.use_rag and not self.search_engine.corpus:
            QMessageBox.warning(self, "Info", "No dataset loaded. Please load a dataset or switch back to 'Use Model Knowledge'.")

    def load_history(self):
        history_file = os.path.join(os.path.dirname(__file__), "history.json")
        if os.path.exists(history_file) and os.stat(history_file).st_size > 0:  # Check file size
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)
                    if data:  # Check if loaded data is not empty
                        self.model_history = [
                            {"model_name": item, "model_path": item} if isinstance(item, str) else item
                            for item in data
                        ]
            except json.JSONDecodeError:
                print("Error decoding history.json. File might be corrupted.")
        else:
            print("No previous history found.")

    def save_history(self):
        history_file = os.path.join(os.path.dirname(__file__), "history.json")
        try:
            with open(history_file, "w") as f:
                json.dump(self.model_history, f, indent=4)
        except Exception as e:
            print(f"Error saving history: {e}")

    def stop_inference(self):
        if self.is_inferencing:
            if self.llm_thread and self.llm_thread.isRunning():
                self.llm_thread.terminate()
                self.llm_thread.wait()
                self.send_button.setEnabled(True)
                self.is_inferencing = False
                self.stop_button.setStyleSheet("background-color: #808080; color: #ffffff;")
                QMessageBox.information(self, "Info", "Inference stopped.")
        else:
            QMessageBox.warning(self, "Info", "No inference running to stop.")

        # Disconnect the signal in stop_inference 
        if self.llm_thread:
            self.llm_thread.inference_finished.disconnect(self.process_full_response)

    def update_chat_history(self, text, from_user=False):
        if not self.current_response_box or from_user:
            message_box = ChatMessageBox(text, is_user=from_user)
            self.chat_history_layout.addWidget(message_box)
        else:
            current_text = self.current_response_box.text()
            self.current_response_box.setText(current_text + text)
        
        # Scroll to the bottom of the chat history
        scroll_area = self.chat_history_frame.parent()
        if isinstance(scroll_area, QScrollArea):
            scroll_bar = scroll_area.verticalScrollBar()
            scroll_bar.setValue(scroll_bar.maximum())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatHXApp()
    window.show()
    sys.exit(app.exec())