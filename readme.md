后端启动
cd d:\文档\毕业设计\teeth\backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

前端启动
cd d:\文档\毕业设计\teeth\frontend
npm install
npm run dev