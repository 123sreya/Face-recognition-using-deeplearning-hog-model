
#importing the required libraries
import cv2
import face_recognition

#loading the image to detect
original_image = cv2.imread('images/testing/sp1.jpg')

#load the sample images and get the 128 face embeddings from them
modi_image = face_recognition.load_image_file('images/samples/modi.jpg')
modi_face_encodings = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file('images/samples/trump.jpg')
trump_face_encodings = face_recognition.face_encodings(trump_image)[0]

sreya_image = face_recognition.load_image_file('images/samples/sreya.jpeg')
sreya_face_encodings = face_recognition.face_encodings(sreya_image)[0]

preethi_image = face_recognition.load_image_file('images/samples/preethi.jpeg')
preethi_face_encodings = face_recognition.face_encodings(preethi_image)[0]

vishal_image = face_recognition.load_image_file('images/samples/vishal.jpeg')
vishal_face_encodings = face_recognition.face_encodings(vishal_image)[0]

ashu_image = face_recognition.load_image_file('images/samples/ashu.jpeg')
ashu_face_encodings = face_recognition.face_encodings(ashu_image)[0]


veeru_image = face_recognition.load_image_file('images/samples/veeru.jpeg')
veeru_face_encodings = face_recognition.face_encodings(veeru_image)[0]


#save the encodings and the corresponding labels in seperate arrays in the same order
known_face_encodings = [modi_face_encodings, trump_face_encodings , sreya_face_encodings , preethi_face_encodings, vishal_face_encodings , ashu_face_encodings, veeru_face_encodings]
known_face_names = ["Narendra Modi", "Donald Trump" , "Sreya Sree" , "Preethi" , "vishal", "Ashwini" , "Veeraraju"]

#load the unknown image to recognize faces in it
image_to_recognize = face_recognition.load_image_file('images/testing/sp1.jpg')

#detect all faces in the image
#arguments are image,no_of_times_to_upsample, model
all_face_locations = face_recognition.face_locations(image_to_recognize,model='hog')
#detect face encodings for all the faces detected
all_face_encodings = face_recognition.face_encodings(image_to_recognize,all_face_locations)

#print the number of faces detected
print('There are {} no of faces in this image'.format(len(all_face_locations)))

#looping through the face locations and the face embeddings
for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
    #splitting the tuple to get the four position values of current face
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    
    
    #find all the matches and get the list of matches
    all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
   
    #string to hold the label
    name_of_person = 'Unknown face'
    
    #check if the all_matches have at least one item
    #if yes, get the index number of face that is located in the first index of all_matches
    #get the name corresponding to the index number and save it in name_of_person
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]
    
    #draw rectangle around the face    
    cv2.rectangle(original_image,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
    
    #display the name as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
    
    #display the image
    cv2.imshow("Faces Identified",original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


