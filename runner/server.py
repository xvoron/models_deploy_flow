from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from runner import Edge, Node


app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

PORT = 5123


class BuildRequest(BaseModel):
    nodes: list[Node]
    edges: list[Edge]


@app.post("/build")
async def builder(data: BuildRequest):
    try:
        print(f"Received {len(data.nodes)} nodes and {len(data.edges)} edges")
        return {"message": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT)
