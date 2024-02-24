"""Service to process video files and extract text from them."""

import asyncio
import hashlib
import json
import os
import uuid
from typing import List

from fastapi import FastAPI, File
from fastapi import HTTPException
from fastapi import UploadFile
from pydantic import BaseModel
from redis import asyncio as aioredis

from ocr_on_video import VideoOCR

app = FastAPI()


class VideoOCRRequest(BaseModel):
    """Request to process video files and extract text from them."""

    sample_rate: int = 1
    debug_dir: str = "logs"
    annotate_only: bool = False


redis = aioredis.Redis()

UPLOAD_LOCATION = "uploads"

if not os.path.exists(UPLOAD_LOCATION):
    os.makedirs(UPLOAD_LOCATION)


def calculate_file_hash(file):
    """Calculate the hash of a file using SHA-256."""
    hash_obj = hashlib.sha256()
    for chunk in iter(lambda: file.read(4096), b""):
        hash_obj.update(chunk)
    file.seek(0)
    return hash_obj.hexdigest()


async def save_upload_file(upload_file: UploadFile, destination: str):
    """Save an uploaded file to the destination."""
    try:
        with open(destination, "wb") as buffer:
            chunk = await upload_file.read(8192)
            while chunk:
                buffer.write(chunk)
                chunk = await upload_file.read(8192)
    finally:
        await upload_file.close()


def callback(file_hash, task_id):
    asyncio.create_task(
        redis.set(file_hash, json.dumps({"status": "completed", "task_id": task_id}))
    )


@app.post("/process_videos")
async def process_videos(input_videos: List[UploadFile] = [File(...)]):
    """Process video files and extract text from them."""
    task_ids = []
    for input_video in input_videos:
        file_hash = calculate_file_hash(input_video.file)
        if await redis.get(file_hash):
            raise HTTPException(
                status_code=400, detail="Duplicate video processing request"
            )
        task_id = str(uuid.uuid4())
        output_video = f"logs/{task_id}/output_{task_id}.mp4"
        upload_file_path = os.path.join(UPLOAD_LOCATION, f"temp_{task_id}.mp4")
        await save_upload_file(input_video, upload_file_path)
        video_ocr = VideoOCR(
            upload_file_path,
            output_video,
            1,
            f"logs/{task_id}",
            False,
        )
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, video_ocr.run)
        ## pylint: disable=cell-var-from-loop
        await redis.set(
            file_hash, json.dumps({"status": "processing", "task_id": task_id})
        )
        ## pylint: disable=cell-var-from-loop
        task.add_done_callback(lambda x: callback(file_hash, task_id))
        task_ids.append(task_id)
    return {"task_ids": task_ids}


@app.get("/task_status/{task_id}")
async def task_status(task_id: str):
    """Get the status of a video processing task."""
    for key in await redis.keys("*"):
        task_info = json.loads(await redis.get(key))
        if task_info["task_id"] == task_id:
            return {"task_id": task_id, "status": task_info["status"]}
    raise HTTPException(status_code=404, detail="Task not found")
