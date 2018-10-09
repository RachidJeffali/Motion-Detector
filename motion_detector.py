import cv2, time, pandas
from datetime import datetime

# Initialisation de la 1ère frame, qui va servir de base pour la détection
first_frame = None

# création de l'objet qui capture la vidéo via la webcam 0 
# (=1 si c'est la 2ème webcam, 2 si c'est la 3ème etc...)
video = cv2.VideoCapture(0)

# initialisation de la liste des états, pour éviter une erreur à la ligne 58, on l'initialise à None,None
status_list = [None,None]
# Création d'une liste vide où on stockera les heures de détections
times = []
# Création d'un dataframe pour le convertir après en csv
df = pandas.DataFrame(columns = ["Start","End"])


# Boucle while tant que nous n'appuyons pas sur la touche 'q' pour sortir
while True:
	check, frame = video.read()
	# Variable qui donne l'état de détection
	# status = 0 -> pas de changement avec la frame d'origine
	# status = 1 -> changement au niveau de la frame d'origine
	status = 0
	# Convertion de l'image en noir et blanc
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Convertion de l'image en flou gaussian
	gray = cv2.GaussianBlur(gray, (21,21), 0)

	if first_frame is None:
		first_frame = gray
		continue

	# Calcul de la différence d'intensité entre les la first_frame et la frame actuelle
	delta_frame = cv2.absdiff(first_frame,gray)

	thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
	thresh_frame = cv2.dilate(thresh_frame, None, iterations = 5)

	(_,cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	# Création d'un cadre vert qui va contenir la différence entre l'image initiale
	# et les images insérés
	for contour in cnts:
		if cv2.contourArea(contour) < 10000:
			continue
		# le cadre vert apparait, donc il y a un changement % à la frame d'origine
		status = 1

		(x,y,w,h) = cv2.boundingRect(contour)
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

	# on ajoute à la liste tous les états de changement
	status_list.append(status)

	# on ne garde pas les 2 derniers éléments
	status_list = status_list[-2:]

	# si la liste des status à la forme [0,1], on ajoute à la liste times l'heure de ce changement
	if status_list[-1] == 1 and status_list[-2] == 0:
		times.append(datetime.now())

	# si la liste des status à la forme [1,0], on ajoute ce retour à la frame initiale dans la liste temps
	if status_list[-1] == 0 and status_list[-2] == 1:
		times.append(datetime.now())

	# Affichage de toutes les fenêtres
	#cv2.imshow("Gray_frame", gray)
	#cv2.imshow("Delta Frame", delta_frame)
	#cv2.imshow("Threshold frame", thresh_frame)
	cv2.imshow("Color Frame", frame)
	
	key = cv2.waitKey(1)

	if key==ord('q'):
		# si on quitte le programme pendant une détection, on l'ajoute à la liste times
		if status == 1:
			times.append(datetime.now())
		break

# On enregistre toutes les données dans une dataframe...
for i in range(0, len(times),2):
	df = df.append({"Start":times[i], "End":times[i+1]}, ignore_index = True)

# ... qu'on exporte en format csv
df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows
