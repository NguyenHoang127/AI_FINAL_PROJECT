# Đoàn Nguyễn Hoàng - 20145195

# Xử Lý Giao diện
import tkinter as tk
from tkinter.ttk import *
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename

# Xử lý ảnh
import cv2
import numpy as np
from matplotlib.pyplot import show
from PIL import ImageTk ,Image
import matplotlib.pyplot as plt

# Xử lý kí tự
from tensorflow import keras
from keras.utils import load_img
from keras.utils.image_utils import img_to_array

model = keras.models.load_model('character_model2.h5')
class_name = ['0','1','2','3','4','5','6','7','8','9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '4', 'b', 'd', 'e', 'f', '9', 'h', 'n', 'q','r','t']

#------------------------------------------------------------------------------

WinDow = Tk()

WinDow.title("FINAL AI PROJECT")
WinDow.geometry("1280x700")

#Load ảnh vào chương trình
Login_image = Image.open("Frame_Logins.png")
Resize_image = Login_image.resize((1280,700),Image.Resampling.LANCZOS)
Image_LG = ImageTk.PhotoImage(Resize_image)

Work_image = Image.open("Frame1_Workspace.png")
Resize_W_image = Work_image.resize((1280,700),Image.Resampling.LANCZOS)
Image_W = ImageTk.PhotoImage(Resize_W_image)


WinDow.rowconfigure(0,weight=1)
WinDow.columnconfigure(0,weight=1)

#Tạo các frame con
login = tk.Frame(WinDow)
frame1 = tk.Frame(WinDow)

frame1.grid(row=0,column=0,sticky='nsew')
login.grid(row=0,column=0,sticky='nsew')

#------------------------------------------------------------------------------

#Frame workspace

label2 = Label(frame1,image=Image_W)
label2.pack()

frame1_bt = Button(frame1,text = "Process Image",height=6,width=28,bd=9,bg="green",font=('Arial',11,'bold'),command=lambda:get_image())
frame1_bt.place(x = 420,y = 250)

frame1_bt_Logout = Button(frame1,text = "Logout",height=2,width=18,command=lambda:logout(),bg="red")
frame1_bt_Logout.place(x = 920,y = 80)

label3_frame1 = Label(frame1)
label3_frame1.place(x = 80,y = 400)

label4_frame1 = Label(frame1)
label4_frame1.place(x = 430,y = 480)

label5_frame1 = Label(frame1)
label5_frame1.place(x = 1050,y = 600)

label6_frame1 = Label(frame1)
label6_frame1.place(x = 1050,y = 500)

result_frame = Label(frame1,height=2,width=18,bg="green",fg="Yellow",font=('Arial',15,'bold'))
result_frame.place(x = 1000,y = 250)

#-------------------------------------------------------------------------------

def show_frame(frame):
    frame.tkraise()
    
def detect_plate(img):
    plate_cascade = cv2.CascadeClassifier('indian_license_plate.xml')
    
    plate_img = img.copy() 
    plate = img.copy()
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.2, minNeighbors = 7)
    
    for (x,y,w,h) in plate_rect:
        plate = plate[y:y+h, x:x+w, :]
        cv2.rectangle(plate_img, (x,y), (x+w, y+h), (51,181,155), 3)

    return plate_img, plate 

def find_contours(dimensions, img) :

    # Tìm tất cả các đường viền có trong ảnh
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Lọc ra 15 ký tự có diện tích lớn nhất
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    x_cntr_list = []
    img_res = []
    
    for cntr in cntrs :

        #Đóng gói các ký tự vào trong hình chữ nhật
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        #Kiểm tra kích thước của đường viền để lọc ra các ký tự theo kích thước của đường viền
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) 

            char_copy = np.zeros((28,28))

            # Trích xuất từng ký tự
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            
            char = cv2.resize(char, (24, 24))
            cv2.rectangle(img, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:26, 2:26] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[26:28, :] = 0
            char_copy[:, 26:28] = 0

            img_res.append(char_copy) 

    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []

    for idx in indices:
        img_res_copy.append(img_res[idx])
    
    img_res = np.array(img_res_copy)
    img_res = img_res.astype(np.float32)

    return img_res,img

def segment_characters(image) :

    # Xử lý ảnh đầu vào
    img = cv2.resize(image, (333, 75))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_erode = cv2.erode(img_binary, (3,3))
    img_dilate = cv2.dilate(img_erode, (3,3))

    LP_WIDTH = img_dilate.shape[0]
    LP_HEIGHT = img_dilate.shape[1]

    # Make borders white
    img_dilate[0:9,:] = 255
    img_dilate[:,0:5] = 255
    img_dilate[66:75,:] = 255
    img_dilate[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6, LP_WIDTH/2, LP_HEIGHT/10, 2*LP_HEIGHT/3]

    return img_dilate,dimensions

def predict(image_sorted):
    photo=image_sorted/255
    photo=np.expand_dims(photo,axis=0)
    
    # Model predict return a 2D matrix
    photo = model.predict(photo)[0] 

    #Get label with max accuracy
    name = np.argmax(photo)
    pred = class_name[name]

    return pred

def loginx():
    username = entry1.get()
    password = entry2.get()
    if(username == "1" and password =="1"):
         show_frame(frame1)
         entry1.delete(0,'end')
         entry2.delete(0,'end')
    else:
        messagebox.showerror("","Vui lòng nhập lại!")

def logout():
    show_frame(login)

#--------------------------------------------------------------------------------------

def get_image():
    global fpath, label3_frame1, label4_frame1,label5_frame1,label6_frame1,plate_Tk,car_plate_Tk,result_frame
    #Get link image
    fpath = askopenfilename(filetypes=[("Text Files","*.jpg"),("All files","*.*")])

    if len(fpath) > 0:

        image = cv2.imread(fpath,cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #Detect Plate
        car_plate, plate = detect_plate(image)

        #Character Detect
        img_plate,dimensions = segment_characters(plate)
        img_res,plate_sorted = find_contours(dimensions,img_plate)
        

        Respond = []
        result = ''
        for i in range(8):
            kq = predict(img_res[i])
            Respond.append(kq) 

        for s in Respond:
            result += s

        print(result)

        # convert the images from cv2 to PIL format
        image = Image.fromarray(image)
        car_plate_Tk = Image.fromarray(car_plate)
        plate_Tk = Image.fromarray(plate)
        plate_sorted_Tk = Image.fromarray(plate_sorted)

        #Resize
        image_show = image.resize((300,270),Image.Resampling.LANCZOS)
        car_plate_Tk = car_plate_Tk.resize((270,150))
        plate_Tk = plate_Tk.resize((120,80))
        plate_sorted_Tk = plate_sorted_Tk.resize((120,80))

		# And then to ImageTk format
        image_Tk = ImageTk.PhotoImage(image_show)
        car_plate_Tk = ImageTk.PhotoImage(car_plate_Tk)
        plate_Tk = ImageTk.PhotoImage(plate_Tk)
        plate_sorted_Tk = ImageTk.PhotoImage(plate_sorted_Tk)
        
        if label3_frame1 is None or label4_frame1 is None or label5_frame1 is None or label6_frame1 is None or result_frame is None:
			
            label3_frame1 = Label(image=image_Tk)
            label3_frame1.image = image_Tk
			
            label4_frame1 = Label(image=car_plate_Tk)
            label4_frame1.image = car_plate_Tk

            label5_frame1 = Label(image=plate_Tk)
            label5_frame1.image = plate_Tk

            label6_frame1 = Label(image=plate_sorted_Tk)
            label6_frame1.image = plate_sorted_Tk

            result_frame = Label(text=result)
		# otherwise, update the image panels
        else:
            # update the pannels
            label3_frame1.configure(image=image_Tk)
            label4_frame1.configure(image=car_plate_Tk)
            label5_frame1.configure(image=plate_Tk)
            label6_frame1.configure(image=plate_sorted_Tk)
            result_frame.configure(text=result)

            label3_frame1.image = image_Tk
            label4_frame1.image = car_plate_Tk
            label5_frame1.image = plate_Tk
            label6_frame1.image = plate_sorted_Tk
    
#-------------------------------------------------------------------------------


#Frame Login
label1 = Label(login,image=Image_LG)
label1.pack()

#Ô dữ liệu đăng nhập
global entry1,entry2
entry1 = Entry(login,bd=10)
entry1.place(x = 990, y= 360)
entry2 = Entry(login,bd=10,show="*")
entry2.place(x = 995, y=485)

button1 = Button(login,text="Login",height=1,width=14,bd=9,command=loginx,bg="blue",font=('Arial',11,'bold')).place(x=993, y = 550)

#-----------------------------------------------------------------------------------

show_frame(login)
WinDow.mainloop() 
