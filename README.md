# Project Name

This is a UI for doggies and kittens classification model to distinguish cat and dog on image.
Project was made as part of artificial intelligence methods project on WUST.

Second project repo:
https://github.com/jakubiwaszkiewicz/mentsiv2

## Requirements

- Python 3.x
- Dependencies listed in `requirements.txt`

## Setup

### 1. Clone the Repository

```sh
git clone [https://github.com/yourusername/projectname.git](https://github.com/dawid31/doggies-and-kittens-classification
cd doggies-and-kittens-classification
```

### 2. Create and Activate Virtual Environment

```sh
python -m venv venv
# On Windows
venv\Scripts\activate
# On MacOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

### 4. Apply Migrations

```sh
python manage.py migrate
```

### 5. Create a Superuser

```sh
python manage.py createsuperuser
```

## Running the Project

Start the development server:

```sh
python manage.py runserver
```

Access the application at `http://127.0.0.1:8000/`.

