from ultralytics import YOLO
import cv2
import os

# Cargar el modelo
model = YOLO("models/best.pt")

# Colors in BGR format
RED = (0, 0, 255)      # For people without helmet
GREEN = (0, 255, 0)    # For people with helmet
BLUE = (255, 0, 0)     # For helmets

def detectar_personas_sin_casco(result):
    personas_sin_casco = []
    personas_con_casco = []
    cascos = []

    # First detect all heads and helmets
    for detection in result.boxes.data:
        cls = int(detection[5].item())  # Get class index
        conf = detection[4].item()      # Get confidence
        x1, y1, x2, y2 = map(int, detection[:4].tolist())  # Get coordinates
        label = result.names[cls]
        
        #print(f"DEBUG - Detectado: {label} (clase {cls}) con conf: {conf:.2f}")
        
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

    # Check for helmet overlaps with improved overlap detection
    for persona in personas_sin_casco[:]:  # Use copy to avoid modification during iteration
        px1, py1, px2, py2 = persona["bbox"]
        area_persona = (px2 - px1) * (py2 - py1)
        
        for casco in cascos:
            cx1, cy1, cx2, cy2 = casco["bbox"]
            
            # Calculate intersection area
            x_left = max(px1, cx1)
            y_top = max(py1, cy1)
            x_right = min(px2, cx2)
            y_bottom = min(py2, cy2)
            
            if x_right > x_left and y_bottom > y_top:
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                area_casco = (cx2 - cx1) * (cy2 - cy1)
                
                # If overlap is significant (>30% of helmet area)
                if intersection_area > 0.3 * area_casco:
                    #print(f"DEBUG - Encontrada superposición significativa casco-cabeza")
                    personas_con_casco.append(persona)
                    personas_sin_casco.remove(persona)
                    break

    #print(f"DEBUG - Total personas sin casco: {len(personas_sin_casco)}")
    #print(f"DEBUG - Total personas con casco: {len(personas_con_casco)}")

    return personas_sin_casco, personas_con_casco

def inferir_imagen(image_path, output_path):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        return

    # Make prediction
    results = model.predict(image, conf=0.25)  # Lower confidence threshold
    
    # Get detections
    personas_sin_casco, personas_con_casco = detectar_personas_sin_casco(results[0])

    # Draw detections - people without helmets (RED)
    for persona in personas_sin_casco:
        x1, y1, x2, y2 = persona["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), RED, 3)
        cv2.putText(image, f"Sin Casco ({persona['conf']:.2f})", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

    # Draw detections - people with helmets (GREEN)
    for persona in personas_con_casco:
        x1, y1, x2, y2 = persona["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), GREEN, 3)
        cv2.putText(image, f"Con Casco ({persona['conf']:.2f})", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

    # Save result
    cv2.imwrite(output_path, image)
    print(f"Resultado guardado en {output_path}")

def procesar_carpeta(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            output_filename = f"processed_{filename}"
            output_path = os.path.join(output_folder, output_filename)
            #print(f"Procesando {input_path} -> {output_path}")
            inferir_imagen(input_path, output_path)

if __name__ == "__main__":
    input_folder = "data/input"  # Carpeta de entrada
    output_folder = "data/output"  # Carpeta de salida
    
    # Procesar todas las imágenes en la carpeta
    procesar_carpeta(input_folder, output_folder)
