import cv2, sys, os

try:
    # 開啟視窗
    cv2.namedWindow("frame")
    
    # 取得 camera
    cap = cv2.VideoCapture(0)
    
    # 取得臉部特徵檔
    cascade_faces = cv2.CascadeClassifier( 
        os.path.join( os.getcwd(), 'files/haarcascades/haarcascade_frontalface_default.xml' )
    )
    
    # 取得眼部特徵檔
    cascade_eyes = cv2.CascadeClassifier( 
        os.path.join( os.getcwd(), 'files/haarcascades/haarcascade_eye.xml' ) 
    )
    
    # 確認 camera 是否開啟
    while ( cap.isOpened() ):
        # 讀取 camera 影像
        ret, img = cap.read()
        
        # 重新定義畫面大小
        #img = cv2.resize(img, (1280, 960))
        
        # 取得眼部特徵檔
        eyes = cascade_eyes.detectMultiScale(
            img,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (10,10),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        # 取得臉部特徵檔
        faces = cascade_faces.detectMultiScale(
            img,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (10,10),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        # 如果有取得影像(frame by frame)，則進行下列操作
        if ret == True:
            # 將分析出來的眼部範圍，用四方形框線畫出來
            for (x, y, w, h) in eyes:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
            
            # 將分析出來的臉部範圍，用四方形框線畫出來
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (128,255,0), 2)
            
            # 顯示 resize 後的圖片
            cv2.imshow("frame", img)
            
            # 每隔 0.025 秒檢查是否按下 q 鍵，有按則跳出
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    
    # 釋放 camera 資源
    cap.release()
    
    # 關閉視窗
    cv2.destroyWindow("frame")
except:
    print("Exception: \n", sys.exc_info())