import cv2
import pytesseract
import imutils

# Defina o caminho para o executável do Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Lista de placas cadastradas
placas_cadastradas = ["ABC1234", "XYZ9876", "JKL5678"]


def processar_image(image):
    # Converta a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplique um filtro bilateral para remover ruídos e manter bordas nítidas
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Aumente o contraste usando equalização do histograma
    gray = cv2.equalizeHist(gray)

    # Aplique limiarização adaptativa
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)

    return gray


def detectar_placa(image):
    gray = processar_image(image)

    # Detecte bordas na imagem
    edged = cv2.Canny(gray, 30, 200)

    # Encontre contornos na imagem
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    # Encontre o contorno que é mais provável ser a placa do carro
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        return None, None

    # Extraia a placa e aplique uma transformação de perspectiva
    x, y, w, h = cv2.boundingRect(screenCnt)
    plate = image[y:y+h, x:x+w]

    # Inverter a imagem da placa para corrigir a leitura
    plate = cv2.flip(plate, 1)

    return plate, (x, y, w, h)


def ler_placa(plate):
    # Use o Tesseract para realizar OCR na placa extraída
    text = pytesseract.image_to_string(plate, config='--psm 8')
    return text.strip()


def main():
    # Capture vídeo da câmera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detecte a placa do carro na imagem capturada
        plate, bbox = detectar_placa(frame)
        if plate is not None:
            # Leia o texto da placa
            text = ler_placa(plate)
            if text:
                print(f'Placa detectada: {text}')

                # Verifique se a placa está cadastrada
                if text in placas_cadastradas:
                    color = (0, 255, 0)  # Verde
                else:
                    color = (0, 0, 255)  # Vermelho

                # Desenhe um retângulo em volta da placa
                if bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Mostre a imagem com a placa destacada
        cv2.imshow('Frame', frame)

        # Saia do loop pressionando 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
