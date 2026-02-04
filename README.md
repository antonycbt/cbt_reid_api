# create virtual env
    python -m venv venv
    source venv/bin/activate  # windows: venv\Scripts\activate

# install deps
    pip install -r requirements.txt

# init alembic
    python -m alembic init alembic

# run app
    uvicorn app.main:app --reload    or  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload 

# alembic migration
    1. python -m alembic revision --autogenerate -m "migration message"
    2. python -m alembic upgrade head

# pip install -r requirements.txt   
# pip install wheel cython setuptools --upgrade
# pip install --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git
