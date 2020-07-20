PYTHON_INTERPRETER = python

requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

preprocess:
	$(PYTHON_INTERPRETER) src/preprocess.py

train:
	$(PYTHON_INTERPRETER) src/train.py

inference:
	$(PYTHON_INTERPRETER) src/inference.py