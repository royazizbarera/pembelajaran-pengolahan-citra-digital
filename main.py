import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from skimage.exposure import match_histograms  # pastikan paket scikit-image sudah terinstal

import numpy as np
import cv2
import matplotlib.pyplot as plt

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")

if not os.path.exists("static/histograms"):
    os.makedirs("static/histograms")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    file_path = save_image(img, "uploaded")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": file_path,
        "modified_image_path": file_path
    })

@app.post("/operation/", response_class=HTMLResponse)
async def perform_operation(
    request: Request,
    file: UploadFile = File(...),
    operation: str = Form(...),
    value: int = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    original_path = save_image(img, "original")

    if operation == "add":
        result_img = cv2.add(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "subtract":
        result_img = cv2.subtract(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "max":
        result_img = np.maximum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "min":
        result_img = np.minimum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "inverse":
        result_img = cv2.bitwise_not(img)

    modified_path = save_image(result_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })
@app.post("/logic_operation/", response_class=HTMLResponse)
async def perform_logic_operation(
    request: Request,
    file1: UploadFile = File(...),
    file2: UploadFile = File(None),
    operation: str = Form(...)
):
    image_data1 = await file1.read()
    np_array1 = np.frombuffer(image_data1, np.uint8)
    img1 = cv2.imdecode(np_array1, cv2.IMREAD_COLOR)

    if img1 is None:
        return HTMLResponse("Gambar pertama tidak valid.", status_code=400)

    original_path = save_image(img1, "original")

    if operation == "not":
        result_img = cv2.bitwise_not(img1)
    else:
        if file2 is None:
            return HTMLResponse("Operasi AND dan XOR memerlukan dua gambar.", status_code=400)
        
        image_data2 = await file2.read()
        np_array2 = np.frombuffer(image_data2, np.uint8)
        img2 = cv2.imdecode(np_array2, cv2.IMREAD_COLOR)

        if img2 is None:
            return HTMLResponse("Gambar kedua tidak valid.", status_code=400)

        # Resize img2 agar sesuai dengan ukuran img1
        img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        if operation == "and":
            result_img = cv2.bitwise_and(img1, img2_resized)
        elif operation == "xor":
            result_img = cv2.bitwise_xor(img1, img2_resized)
        else:
            return HTMLResponse("Operasi tidak dikenal.", status_code=400)

    modified_path = save_image(result_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })
    
@app.get("/grayscale/", response_class=HTMLResponse)
async def grayscale_form(request: Request):
    # Menampilkan form untuk upload gambar ke grayscale
    return templates.TemplateResponse("grayscale.html", {"request": request})

@app.post("/grayscale/", response_class=HTMLResponse)
async def convert_grayscale(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    original_path = save_image(img, "original")
    modified_path = save_image(gray_img, "grayscale")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/histogram/", response_class=HTMLResponse)
async def histogram_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk histogram
    return templates.TemplateResponse("histogram.html", {"request": request})

@app.post("/histogram/", response_class=HTMLResponse)
async def generate_histogram(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Pastikan gambar berhasil diimpor
    if img is None:
        return HTMLResponse("Tidak dapat membaca gambar yang diunggah", status_code=400)

    # Buat histogram grayscale dan berwarna
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_histogram_path = save_histogram(gray_img, "grayscale")

    color_histogram_path = save_color_histogram(img)

    return templates.TemplateResponse("histogram.html", {
        "request": request,
        "grayscale_histogram_path": grayscale_histogram_path,
        "color_histogram_path": color_histogram_path
    })



@app.get("/equalize/", response_class=HTMLResponse)
async def equalize_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk equalisasi histogram
    return templates.TemplateResponse("equalize.html", {"request": request})
@app.post("/equalize/", response_class=HTMLResponse)
async def equalize_histogram(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    
    # Decode gambar asli dalam format berwarna (BGR)
    img_color = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    # Jika decoding gagal, kembalikan pesan error
    if img_color is None:
        return HTMLResponse("Gagal membaca gambar yang diunggah.", status_code=400)
    
    # Simpan gambar asli (berwarna) tanpa modifikasi
    original_path = save_image(img_color, "original")
    
    # Konversi gambar ke grayscale untuk equalization
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Lakukan histogram equalization pada gambar grayscale
    equalized_img = cv2.equalizeHist(img_gray)

    # Simpan gambar hasil equalization
    modified_path = save_image(equalized_img, "equalized")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,  # Gambar asli berwarna
        "modified_image_path": modified_path  # Gambar hasil equalization (grayscale)
    })

@app.get("/specify/", response_class=HTMLResponse)
async def specify_form(request: Request):
    # Menampilkan halaman untuk upload gambar dan referensi untuk spesifikasi histogram
    return templates.TemplateResponse("specify.html", {"request": request})

@app.post("/specify/", response_class=HTMLResponse)
async def specify_histogram(request: Request, file: UploadFile = File(...), ref_file: UploadFile = File(...)):
    # Baca gambar yang diunggah dan gambar referensi
    image_data = await file.read()
    ref_image_data = await ref_file.read()

    np_array = np.frombuffer(image_data, np.uint8)
    ref_np_array = np.frombuffer(ref_image_data, np.uint8)
		
		#jika ingin grayscale
    #img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    #ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_GRAYSCALE)

    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Membaca gambar dalam format BGR
    ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_COLOR)  # Membaca gambar referensi dalam format BGR


    if img is None or ref_img is None:
        return HTMLResponse("Gambar utama atau gambar referensi tidak dapat dibaca.", status_code=400)

    # Spesifikasi histogram menggunakan match_histograms dari skimage #grayscale
    #specified_img = match_histograms(img, ref_img, multichannel=False)
		    # Spesifikasi histogram menggunakan match_histograms dari skimage untuk gambar berwarna
    specified_img = match_histograms(img, ref_img, channel_axis=-1)
    # Konversi kembali ke format uint8 jika diperlukan
    specified_img = np.clip(specified_img, 0, 255).astype('uint8')

    original_path = save_image(img, "original")
    modified_path = save_image(specified_img, "specified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.post("/statistics/", response_class=HTMLResponse)
async def calculate_statistics(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    mean_intensity = np.mean(img)
    std_deviation = np.std(img)

    image_path = save_image(img, "statistics")

    return templates.TemplateResponse("statistics.html", {
        "request": request,
        "mean_intensity": mean_intensity,
        "std_deviation": std_deviation,
        "image_path": image_path
    })

def save_image(image, prefix):
    filename = f"{prefix}_{uuid4()}.png"
    path = os.path.join("static/uploads", filename)
    cv2.imwrite(path, image)
    return f"/static/uploads/{filename}"

def save_histogram(image, prefix):
    histogram_path = f"static/histograms/{prefix}_{uuid4()}.png"
    plt.figure()
    plt.hist(image.ravel(), 256, [0, 256])
    plt.savefig(histogram_path)
    plt.close()
    return f"/{histogram_path}"

def save_color_histogram(image):
    color_histogram_path = f"static/histograms/color_{uuid4()}.png"
    plt.figure()
    for i, color in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.savefig(color_histogram_path)
    plt.close()
    return f"/{color_histogram_path}"


@app.post("/resize/")
async def resize_image(request: Request, file: UploadFile = File(...), scale_percent: int = Form(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
    
    original_path = save_image(img, "original")
    

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    

    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    

    resized_path = save_image(resized_img, "resized")
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": resized_path
    })

@app.post("/merge_images/", response_class=HTMLResponse)
async def merge_images(request: Request, 
                       file1: UploadFile = File(...), 
                       file2: UploadFile = File(...), 
                       scale_percent: int = Form(...)):
    
    image_data1 = await file1.read()
    np_array1 = np.frombuffer(image_data1, np.uint8)
    picture_frame = cv2.imdecode(np_array1, cv2.IMREAD_COLOR)
    
    original_path = save_image(picture_frame, "original")

    image_data2 = await file2.read()
    np_array2 = np.frombuffer(image_data2, np.uint8)
    logo_polban = cv2.imdecode(np_array2, cv2.IMREAD_UNCHANGED)

    width = int(logo_polban.shape[1] * scale_percent / 100)
    height = int(logo_polban.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_logo = cv2.resize(logo_polban, dim, interpolation=cv2.INTER_AREA)

    b, g, r, alpha = cv2.split(resized_logo)
    mask = alpha

    x_offset = picture_frame.shape[1] // 2 - resized_logo.shape[1] // 2
    y_offset = picture_frame.shape[0] // 2 - resized_logo.shape[0] // 2

    roi = picture_frame[y_offset:y_offset+resized_logo.shape[0], x_offset:x_offset+resized_logo.shape[1]]

    foreground = cv2.merge((b, g, r))
    background = roi

    blended = cv2.add(cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask)), 
                      cv2.bitwise_and(foreground, foreground, mask=mask))

    result_frame = picture_frame.copy()
    result_frame[y_offset:y_offset+resized_logo.shape[0], x_offset:x_offset+resized_logo.shape[1]] = blended

    output_path = save_image(result_frame, "logo_frame")
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": output_path
    })
    
    
    
    
@app.post("/add_text/", response_class=HTMLResponse)
async def add_text(request: Request, 
                   file: UploadFile = File(...), 
                   text: str = Form(...), 
                   x: int = Form(...), 
                   y: int = Form(...)):
    
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    original_path = save_image(img, "original")
    
    text_coordinates = (x, y)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255)
    line_type = 2

    cv2.putText(img, text, text_coordinates, font, font_scale, font_color, line_type)
    
    output_path = save_image(img, "final_text")
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": output_path
    })




'''
Penjelasan soal 2.1:
1. Konversi Citra ke Grayscale: Warna tidak begitu penting disini, yang penting intensitasnya.
2. Thresholding: Untuk memisahkan objek dari background.
3. Inversi: Agar objek menjadi putih dan background menjadi hitam.
4. Deteksi Kontur: Untuk menemukan objek-objek pada citra.
5. Perhitungan Luas: Menghitung luas dari setiap objek.
6. Menampilkan Hasil: Menampilkan jumlah objek dan luas masing-masing objek dalam piksel.
'''



@app.post("/count_objects/", response_class=HTMLResponse)
async def count_objects(request: Request, file: UploadFile = File(...)):

    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    original_path = save_image(image, "original")
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    binary_image = cv2.bitwise_not(binary_image)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_result = []
    areas = []

    for contour in contours:
        area = cv2.contourArea(contour)
        areas.append(area)

    num_objects = len(contours)
    
    return templates.TemplateResponse("result_objects.html", {
        "request": request,
        "original_image_path": original_path,
        "num_objects": num_objects,
        "areas": areas
    })
    
@app.post("/count_objects_by_color/", response_class=HTMLResponse)
async def count_objects_by_color(request: Request, file: UploadFile = File(...)):
    # Membaca file gambar yang diunggah
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    original_path = save_image(image, "original")
    
    # Konversi gambar ke grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Melakukan thresholding adaptif
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Membalik gambar biner agar objek menjadi putih di atas latar belakang hitam
    binary_image = cv2.bitwise_not(binary_image)

    # Mendeteksi kontur objek pada gambar
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dictionary untuk menyimpan ukuran objek berdasarkan warna
    color_areas = {}
    color_counts = {}

    for contour in contours:
        area = cv2.contourArea(contour)

        # Membuat mask untuk kontur saat ini
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Mengambil ROI dari gambar asli menggunakan mask
        masked_image = cv2.bitwise_and(image, mask)

        # Menghitung rata-rata warna dari area objek
        mean_color = cv2.mean(image, mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

        # Buat tuple dari warna BGR
        mean_color_tuple = (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))

        # Menyimpan atau menambahkan ukuran area berdasarkan warna
        if mean_color_tuple in color_areas:
            color_areas[mean_color_tuple] += area
            color_counts[mean_color_tuple] += 1
        else:
            color_areas[mean_color_tuple] = area
            color_counts[mean_color_tuple] = 1

    return templates.TemplateResponse("result_objects_by_color.html", {
        "request": request,
        "original_image_path": original_path,
        "color_areas": color_areas,
        "color_counts": color_counts
    })

    
    
    
def save_histogram(image, prefix):
    histogram_path = f"static/histograms/{prefix}_{uuid4()}.png"
    color = ('b', 'g', 'r')
    plt.figure(figsize=(10, 5))
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title('Histogram for RGB Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(histogram_path)
    plt.close()
    return f"/{histogram_path}"

@app.post("/process_image/", response_class=HTMLResponse)
async def process_image(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    original_path = save_image(image, "original")
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]
        border_color = roi[1, 1] 
        cv2.drawContours(image, [contour], -1, tuple(int(c) for c in border_color), thickness=cv2.FILLED)

    output_image_path = save_image(image, "processed_image")
    
    histogram_path = save_histogram(image, "rgb_histogram")

    return templates.TemplateResponse("result_histogram.html", {
        "request": request,
        "original_image_path": original_path,
        "processed_image_path": output_image_path,
        "histogram_path": histogram_path
    })