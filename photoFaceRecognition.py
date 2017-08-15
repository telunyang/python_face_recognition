import cv2, sys, os

def faceDetection():
    # 開啟視窗
    cv2.namedWindow("Image")
    
    # 圖片名稱
    file_name = 'example03.jpg'
    
    # 圖片路徑
    file_path = os.path.join( os.getcwd(), 'input/' + file_name )
    
    # 讀取圖片
    img = cv2.imread(file_path, 1)
    
    # 比例(值愈小，照片寬度愈小)
    ratio = 0.06
    
    # 縮放後的寬 (img.shape[1] 是原來的寬)
    w = int(img.shape[1] * ratio)
    
    # 縮放後的高 (img.shape[0] 是原來的高)
    h = int(img.shape[0] * ratio)

    print('w = ' + str(w) + ', h = ' + str(h))
    
    # 改變圖片大小
    img_resize = cv2.resize(img, (w, h))
    
    # resize window
    #cv2.resizeWindow('image_window', w, h)
    
    #
    #cv2.rectangle(img_resize, (20,60), (1050,400), (0,0,255), 2)
    
    #
    #cv2.putText(img_resize, "Keep in touch", (40,140), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
    
    # 取得臉部特徵檔
    cascade_faces = cv2.CascadeClassifier( 
        os.path.join( os.getcwd(), 'files/haarcascades/haarcascade_frontalface_default.xml' )
    )
    
    # 取得眼部特徵檔
    cascade_eyes = cv2.CascadeClassifier( 
        os.path.join( os.getcwd(), 'files/haarcascades/haarcascade_eye.xml' ) 
    )
    
    # 取得眼鏡特徵檔
    cascade_glasses = cv2.CascadeClassifier( 
        os.path.join( os.getcwd(), 'files/haarcascades/haarcascade_eye_tree_eyeglasses.xml' ) 
    )
    
    # 找到臉部偵測後的資料集合
    faces = cascade_faces.detectMultiScale(
        img_resize,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (10,10),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # 找到眼部偵測後的資料集合
    eyes = cascade_eyes.detectMultiScale(
        img_resize,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (10,10),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    # 找到眼鏡偵測後的資料集合
    glasses = cascade_glasses.detectMultiScale(
        img_resize,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (10,10),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    # 將分析出來的臉部範圍，用四方形框線畫出來
    for (x, y, w, h) in faces:
        cv2.rectangle(img_resize, (x, y), (x+w, y+h), (128,255,0), 2)
        
    # 將分析出來的眼部範圍，用四方形框線畫出來
    for (x, y, w, h) in eyes:
        cv2.rectangle(img_resize, (x, y), (x+w, y+h), (0,0,255), 2)
        
    # 將分析出來的眼鏡範圍，用四方形框線畫出來
#     for (x, y, w, h) in glasses:
#         cv2.rectangle(img_resize, (x, y), (x+w, y+h), (255,0,0), 2)
    
    #print(glasses)
    
    # 顯示 resize 後的圖片
    cv2.imshow("Image", img_resize)
    
    # 儲存圖片
    cv2.imwrite(os.path.join( os.getcwd(), 'output/' + file_name ), img_resize, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    # 幾毫秒後才自動往下執行 (0 代表無限時間，需要使用者自行點擊任意鍵)
    cv2.waitKey(0)
    
    # 關閉所有圖片視窗
    cv2.destroyAllWindows()

try:
    faceDetection()
except:
    print("Exception: \n", sys.exc_info())