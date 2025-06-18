@echo off
echo Starting Celery Worker...

REM 确保在正确的目录
cd /d %~dp0

REM 启动 Celery Worker
python -m celery -A app.tasks.celery:celery_app worker -l INFO -c 4 -n worker1@%%h -E --pool=solo

pause 