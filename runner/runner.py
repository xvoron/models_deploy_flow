"""Take zip and run docker compose from this zip file"""

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2

app = FastAPI()


@app.get("/")
def index():
    """Index"""
    return Response("Go to /live-stream")

@app.get("/live-stream")
def live_stream():
    """Live stream"""
    camera = cv2.VideoCapture(0)
    return StreamingResponse(, media_type="multipart/x-mixed-replace;boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
