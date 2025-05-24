import cv2

try:
	from google.colab.patches import cv2_imshow						# Colab
except ModuleNotFoundError:
	def cv2_imshow(img, win='img'):
		cv2.imshow(win, img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()	