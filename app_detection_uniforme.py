import av
import cv2
import time
import streamlit as st 

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase,WebRtcMode, VideoHTMLAttributes 

import torch
import numpy as np
import queue



# Loading model and threshold
model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path ='model_yolov5m_2/exp3/weights/best.pt', force_reload=False)
CONFIDENCE_THRESHOLD = 0.5


# Defining labels to detect and messages to print
labels_list = ['Casque','Absence de casque', 'Gilet', 'Absence de gilet'] 
message_list = ["Uniforme non vérifié", "Uniforme vérifié"]


# Defining style
font = cv2.FONT_HERSHEY_PLAIN
colors_square = np.array([[102.0, 128.0, 255.0], [167.0, 201.0, 0.0]])
color_text = (255, 255, 255)




class UniformDetector(VideoProcessorBase):

	def __init__(self) -> None:
		self.label = False
	
	def UniformReader(self, image: av.VideoFrame) :
		'''
		Method to predict the labels. 

		Args
			image : videoframe

		Return
			image : videoframe
			class_ids : list
		'''

		boxes = []
		class_ids = []
		confidences = []

		results = model_yolo(image)
		
		if not results:
			return image, False

		else:
			for i in range(0,len(results.pred[0])) :
				confidence_prediction = results.pred[0][i,4]
				label_prediction = int(results.pred[0][i,5])
				
				if confidence_prediction > CONFIDENCE_THRESHOLD :

					# get the coordinates of the prediction
					x = int(results.pred[0][i,0])
					y = int(results.pred[0][i,1])
					w = int(results.pred[0][i,2])
					h = int(results.pred[0][i,3])
					box = np.array([x, y, w, h])
					boxes.append(box)

					# get the label prediction
					class_ids.append(label_prediction)

					# get the confidence of prediction
					confidences.append(confidence_prediction)

			for box, classid, confidence in zip(boxes,class_ids, confidences):
				if classid == 1 or classid == 3:
					color = colors_square[0]
				else:
					color = colors_square[1]

				confidence = round(confidence.item(), 1)  

				#drawing in the image
				cv2.rectangle(image, box, color, 2)
				cv2.rectangle(image, (box[0], box[1] - 30), (box[0] + box[2], box[1]), color, -1)
				cv2.putText(image, f"{labels_list[classid]} (precision : {confidence})", (box[0] + 10, box[1] - 10), font, 0.8, color_text)
	
			return image, class_ids


	def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
		''' 
		Method to read the videoframe with the model

		Args
			image : videoframe
		Return
			image : videoframe
		'''

		image = frame.to_ndarray(format="bgr24")

		annotated_image, result = self.UniformReader(image)

		if result == False:
			return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")
		else:
			self.label = result
			return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")




if __name__ == "__main__":


	# Graphic User Interface

	st.title("Vérification de l'uniforme")

	message_title = st.empty()

	#muted = st.checkbox("mute") 

	stream = webrtc_streamer(
			key="barcode-detection",
			mode=WebRtcMode.SENDRECV,
			video_processor_factory=UniformDetector,
			media_stream_constraints={"video": True, "audio": False},
			async_processing=True,
		)




	while True:
			time.sleep(0.10)
			message_title.empty()

			if stream.video_processor :
				try:
					label_id = stream.video_processor.label
				except queue.Empty:
					label_id = None
				
				if stream.video_processor.label != False :
					for i in label_id:
						if i == 1 or i == 3:
							message = message_list[0]
						else:
							message = message_list[1]
					message_title.subheader(message)



