## GroqChatBot 

This repository contains a Groq chatbot that tutors students passionate in cybersecurity, under the supervision of Dr. Juan Li of North Dakota State University. It uses the sentence-transformers library to generate embeddings and stores them in a faiss index for efficient retrieval.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Requirements

Python 3.7+
Libaries: groq, sentence-transformers, faiss-cpu, PyPDF2, python-dotenv

### Installing

Follow these steps to set up the project locally:

Clone the repository to your local machine:

```
git clone https://github.com/edf3ng/GroqChatbot
cd GroqChatbot
```

OR

Open the project on VSCode and create a cloned repository from there

### Setup

Ensure that you have python downloaded.
Open a virtual environment to download libraries to prevent conflicts between different projects:

```
python -m venv .venv
```

Navigate to .\.venv\Scripts and install dependencies by running the following commands

```
activate
pip install -r requirements.txt
```

Deactivate the virtual environment when done:

```
deactivate
```

### Setting up the API key:

Make sure you obtain a valid Groq API key. They are free at: https://console.groq.com/keys.
After obtaining an API key, create secrets.env in the project directory and add your API key:

```
GROQ_API_KEY=api_key
```

### Prepare Documents:

Download documents that you want the chatbot to retrieve and provide to the user prompt as context. Make sure they are in the same project directory or provide the correct path in chatbot.py. There are already two provided documents for usage.
If you want to edit the file path locations:

```
file_paths = ['document1.pdf', 'document2.pdf']
```

### Run The Application:
```
python chatbot.py
```

After running the program, start typing your queries!

## .gitignore

The [.gitignore](.gitignore) file is a copy of the [Github C++.gitignore file](https://github.com/github/gitignore/blob/master/C%2B%2B.gitignore),
with the addition of ignoring the build directory (`build/`).

## Contributing

Please read [CONTRIBUTING.md] for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Edwin Feng** - *Initial work* - [edf3ng](https://github.com/edf3ng)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thank you to Dr. Juan Li for her guidance through all this process!
