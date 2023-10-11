FROM python:3.11
WORKDIR /usr/src/app
# 기존 Dockerfile 내용...



COPY requirements.txt ./
RUN pip install numpy pandas
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "djangoproject.wsgi:application"]

# collectstatic 실행
RUN python3 manage.py collectstatic --noinput