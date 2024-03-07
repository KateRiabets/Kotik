from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import cv2
import numpy as np

API_TOKEN = '6964634961:AAG3_lWRxqkOgGeuARiIMdNCq1YlmqgAbJ0'
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

droidcam_ip = '192.168.0.101'
droidcam_port = '4747'
video_url = f'http://{droidcam_ip}:{droidcam_port}/video'

async def analyze_frame(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    cat_found = False

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 15:  # ID класса "cat" в YOLO
                cat_found = True
                center_x, center_y, width, height = detection[0:4] * np.array(
                    [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                x, y, w, h = int(center_x - width / 2), int(center_y - height / 2), int(width), int(height)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite('cat_detected.jpg', frame)  # збереження зображення
    return cat_found


@dp.message_handler(commands=['capture'])
async def capture(message: types.Message):
    cap = cv2.VideoCapture(video_url)
    ret, frame = cap.read()
    if not ret:
        return
    cat_found = await analyze_frame(frame)
    if cat_found:
        await message.reply("Неможна відкривати вікно, там котик!")
        with open('cat_detected.jpg', 'rb') as photo:
            await message.reply_photo(photo)
    else:
        await message.reply("Можна відкрити вікно!")


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
