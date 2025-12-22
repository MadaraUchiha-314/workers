import uvicorn

if __name__ == "__main__":
    uvicorn.run("workers.service:app", reload=True, port=8000)
