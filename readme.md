
# ChatHX : LLM chat made easy and fast with RAG and wide LLM support

![ChatHX_Render](https://github.com/Tech-ware/ChatHX-Prototyping/assets/134525789/84e54c93-264b-4bd1-81d0-927fc01fea6f)


ChatHX is a Python-based chat application that leverages the power of Large Language Models (LLMs) and Retrieval Augmented Generation (RAG) to provide context-aware and informative responses.

## Features:

* **LLM Integration:** Utilizes the `llama-cpp-python` library to interact with LLMs like LLaMA.
* **Retrieval Augmented Generation (RAG):** Employs a local semantic search engine to retrieve relevant context from a dataset of text files, enhancing the LLM's responses.
* **Dataset Loading:** Supports loading datasets of `.txt`, `.pdf`, `.docx`, and `.pptx` files for context retrieval.
* **User-Friendly Interface:** Built with PyQt6, providing an intuitive and visually appealing graphical interface for interacting with the chat application.
* **Context Length and Max Token Control:**  Allows users to adjust the context length and maximum token length for fine-tuning the LLM's responses.
* **Model History:** Remembers previously loaded models for easy access.
* **Suggested Questions:** Offers follow-up questions based on the conversation history, encouraging further interaction. 
* **Error Handling and Robustness:** Includes mechanisms to handle potential errors during LLM inference and JSON parsing, ensuring a smoother user experience.

## Getting Started:

1. **Installation:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Running the Application:**
   ```bash
   python gui_S.py
   ```

3. **Loading an LLM:**
   - Click "Add new models" to browse and select your LLM model file (e.g., `.gguf` file for LLaMA models).
   - Click "Load Model".

4. **Loading a Dataset (Optional, for RAG):**
   - Click "Choose your folder" to select the directory containing your dataset files.
   - Click "Load Dataset".

5. **Start Chatting:**
   - Type your questions in the input area and click "SEND".

## Using the GUI:

1. **AI Model Section:**
   - **Model Dropdown:** Choose a previously loaded model or click "Add new models" to add a new one.
   - **Model Path:** Displays the path of the currently loaded model.
   - **Load/Eject Model:** Load or remove the selected model.
   - **Max Token Length/Context Length:** Adjust the maximum token length and context length using the provided buttons and sliders. 

2. **Dataset Section:**
   - **Dataset Path:** Displays the path to the loaded dataset.
   - **Choose your folder:** Browse and select the dataset folder. 
   - **Load Dataset:**  Load the dataset for RAG.
   - **Use Model Knowledge:** Toggles between using RAG (with the dataset) and the model's own knowledge.

3. **Chat Area:**
   - Displays the conversation history. 

4. **Input Area:**
    - Type your questions here.
    - **SEND:** Sends your question to the AI model.
    - **Clear:** Clears the chat history.
    - **Copy:** Copies the AI's latest response to the clipboard.
    - **STOP:** Stops the LLM inference process if it's running. 

5. **Suggested Questions:**
    - Click on a suggested question to automatically fill the input area.


## Contributing:

Contributions to ChatHX are welcome! If you have any ideas for new features, improvements, or bug fixes, please feel free to open an issue or submit a pull request.

## Credits:

This project was created by Tech-ware.

## License:

This project is licensed under the [MIT License](LICENSE).
