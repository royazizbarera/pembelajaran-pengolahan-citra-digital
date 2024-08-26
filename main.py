import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    file_extension = file.filename.split(".")[-1]
    filename = f"{uuid4()}.{file_extension}"
    file_path = os.path.join("static", "uploads", filename)

    with open(file_path, "wb") as f:
        f.write(image_data)

    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)

    rgb_array = {"R": r.tolist(), "G": g.tolist(), "B": b.tolist()}

    return templates.TemplateResponse("display.html", {
        "request": request,
        "image_path": f"/static/uploads/{filename}",
        "rgb_array": rgb_array
    })


@app.post("/grayscale/", response_class=HTMLResponse)
async def convert_to_grayscale(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    file_extension = file.filename.split(".")[-1]
    filename = f"{uuid4()}_grayscale.{file_extension}"
    file_path = os.path.join("static", "uploads", filename)

    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Convert to grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Save the grayscale image
    cv2.imwrite(file_path, grayscale_img)

    return templates.TemplateResponse("display_grayscale.html", {
        "request": request,
        "image_path": f"/static/uploads/{filename}"
    })
