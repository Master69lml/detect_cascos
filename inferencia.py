from ultralytics import YOLO
import cv2
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import shutil

model = YOLO("models/best.pt")

RED = (0, 0, 255)      
GREEN = (0, 255, 0)   
BLUE = (255, 0, 0)  

def autenticar_google_drive():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/drive'])
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secrets.json', ['https://www.googleapis.com/auth/drive'])
            creds = flow.run_local_server(port=8080)
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return build('drive', 'v3', credentials=creds)


def descargar_imagenes_google_drive(folder_id, destino):
    service = autenticar_google_drive()
    
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])

    if not items:
        print('No se encontraron archivos en esta carpeta.')
    else:
        for item in items:
            file_id = item['id']
            file_name = item['name']
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                request = service.files().get_media(fileId=file_id)
                file_path = os.path.join(destino, file_name)
                with io.FileIO(file_path, 'wb') as f:
                    downloader = MediaIoBaseDownload(f, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
                        print(f"Descargando {file_name} ({int(status.progress() * 100)}%)")
                print(f"Archivo {file_name} descargado a {file_path}")

def procesar_imagenes_google_drive(folder_id, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    descargar_imagenes_google_drive(folder_id, output_folder)

    procesar_carpeta(output_folder, output_folder)

def detectar_personas_sin_casco(result):
    personas_sin_casco = []
    personas_con_casco = []
    cascos = []

    for detection in result.boxes.data:
        cls = int(detection[5].item())  
        conf = detection[4].item()     
        x1, y1, x2, y2 = map(int, detection[:4].tolist())  
        label = result.names[cls]
        
        if label == "head":
            personas_sin_casco.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf
            })
        elif label == "helmet":
            cascos.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf
            })

    for persona in personas_sin_casco[:]:  
        px1, py1, px2, py2 = persona["bbox"]
        area_persona = (px2 - px1) * (py2 - py1)
        
        for casco in cascos:
            cx1, cy1, cx2, cy2 = casco["bbox"]
            
            x_left = max(px1, cx1)
            y_top = max(py1, cy1)
            x_right = min(px2, cx2)
            y_bottom = min(py2, cy2)
            
            if x_right > x_left and y_bottom > y_top:
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                area_casco = (cx2 - cx1) * (cy2 - cy1)
                
                if intersection_area > 0.3 * area_casco:
                    personas_con_casco.append(persona)
                    personas_sin_casco.remove(persona)
                    break

    return personas_sin_casco, personas_con_casco

def inferir_imagen(image_path, output_path):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        return

    results = model.predict(image, conf=0.25)  
    
    personas_sin_casco, personas_con_casco = detectar_personas_sin_casco(results[0])

    for persona in personas_sin_casco:
        x1, y1, x2, y2 = persona["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), RED, 3)
        cv2.putText(image, f"Sin Casco ({persona['conf']:.2f})", 
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

    for persona in personas_con_casco:
        x1, y1, x2, y2 = persona["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), GREEN, 3)
        cv2.putText(image, f"Con Casco ({persona['conf']:.2f})", 
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

    cv2.imwrite(output_path, image)
    print(f"Resultado guardado en {output_path}")

def procesar_carpeta(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            output_filename = f"processed_{filename}"
            output_path = os.path.join(output_folder, output_filename)
            inferir_imagen(input_path, output_path)

def procesar_carpeta_local(input_folder, output_folder, descargadas_folder):
    # Crear carpetas necesarias
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(descargadas_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            output_filename = f"processed_{filename}"
            output_path = os.path.join(output_folder, output_filename)
            
            inferir_imagen(input_path, output_path)
            
            shutil.move(input_path, os.path.join(descargadas_folder, filename))

    print(f"Todas las imágenes de {input_folder} han sido procesadas.")

def subir_resultados_a_drive(carpeta_local, folder_id_destino):
    service = autenticar_google_drive()
    
    for filename in os.listdir(carpeta_local):
        if filename.startswith('processed_'):
            file_path = os.path.join(carpeta_local, filename)
            
            # Determinar tipo MIME
            extension = filename.split('.')[-1].lower()
            mime_types = {
                'png': 'image/png',
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'bmp': 'image/bmp',
                'tif': 'image/tiff',
                'tiff': 'image/tiff'
            }
            mime_type = mime_types.get(extension, 'application/octet-stream')
            
            file_metadata = {
                'name': filename,
                'parents': [folder_id_destino]
            }
            
            media = MediaFileUpload(file_path, mimetype=mime_type)
            
            try:
                file = service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                print(f"Archivo {filename} subido correctamente a Drive")
            except Exception as e:
                print(f"Error al subir {filename}: {str(e)}")

if __name__ == "__main__":
    # Configuración
    CARPETA_DRIVE_ENTRADA = '1SuxvzXCE8iyQnZ5m1ahswyCIGpNUIWyc'  
    CARPETA_DRIVE_SALIDA = '1bFqSENW1zlTcLIbLx0wovVCAaupbfuu5'  
    CARPETA_TEMPORAL_LOCAL = "temp_processing"                   
    
    try:
        # Paso 1: Procesar imágenes desde Drive
        procesar_imagenes_google_drive(CARPETA_DRIVE_ENTRADA, CARPETA_TEMPORAL_LOCAL)
        
        # Paso 2: Subir resultados a Drive
        subir_resultados_a_drive(CARPETA_TEMPORAL_LOCAL, CARPETA_DRIVE_SALIDA)
        
    except Exception as e:
        print(f"Error durante el proceso: {str(e)}")
    
    finally:
        # Paso 3: Limpieza final (se ejecuta siempre)
        shutil.rmtree(CARPETA_TEMPORAL_LOCAL, ignore_errors=True)
        print("Proceso completado. Archivos locales eliminados.")
    
    
    
  

