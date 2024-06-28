import os
import subprocess

# Путь к вашей виртуальной среде (измените на соответствующий вашей среде)
venv_path = r'F:\DS\flask_openai_chatbot\venv\Scripts'

# Установка пути к среде
os.environ['PATH'] = venv_path + os.pathsep + os.environ['PATH']

# Команда для запуска MLflow UI
command = [
    'mlflow', 'ui',
    '--host', 'oftu-ml-vm',  # используйте '127.0.0.1' для локального запуска
    '--port', '9010'
]

# Запуск команды
subprocess.run(command)

