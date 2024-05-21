import numpy as np
import mediapipe as mp
import cv2

'''
디지털 영상처리 - Chomakey(크로마키)

마우스 왼쪽 버튼 클릭 - 해당 부분의 컬러값을 제거
마우스 오른쪽 버튼 드래그 - 사각형 범위 내의 컬러들의 최소, 최대값만큼 제거
키보드 'q' - 얼굴 추적 활성화 (얼굴 부분은 제거할 영역에서 제외)
키보드 'r' - 초기화
'''

# 트렉바 Red값을 조절하는 함수
def onChangeRed(value):
    global delete_color, trackbar_marge
    delete_color[2] = value
    trackbar_marge[2][:] = value
    updateTrackbar()

# 트렉바 Green값을 조절하는 함수
def onChangeGreen(value):
    global delete_color, trackbar_marge
    delete_color[1] = value
    trackbar_marge[1][:] = value
    updateTrackbar()

# 트렉바 Blue값을 조절하는 함수
def onChangeBlue(value):
    global delete_color, trackbar_marge
    delete_color[0] = value
    trackbar_marge[0][:] = value
    updateTrackbar()

# 트렉바 high_margin값을 조절하는 함수
def onChangeHigh(value):
    global high_margin
    high_margin = value

# 트렉바 low_margin값을 조절하는 함수
def onChangeLow(value):
    global low_margin
    low_margin = value

# RPG트랙바 화면 업데이트
def updateTrackbar():
    global trackbar_marge, face_preserving, range
    # 트랙바 윈도우 색 변경
    trackbar = cv2.merge(trackbar_marge)

    if not isLeftBt and (range.shape[0] >= 3 and range.shape[1] >= 3):
        h, w = range.shape[0], range.shape[1]
        trackbar[0:h, 0:w, :] = range

    cv2.putText(trackbar, "Press the 'q' key", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 20, 147))
    cv2.putText(trackbar, "reset is 'r' key", (250, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 20, 147))
    # 얼굴 보존?이 되어있는지 체크
    if face_preserving:
        cv2.putText(trackbar, "Face Preserving : [ O ]", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,20,147))
    else:
        cv2.putText(trackbar, "Face Preserving : [ X ]", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,20,147))
    cv2.imshow("RGB_window", trackbar)

# 밝기 조절 함수
def onChangeBrightness(value):
    global brightness
    brightness = value

# 타원 크기 조절 함수
def onChangeLongAxis(value):
    global long_axis
    # 8이하면 화면을 다 덮는 크기이며, 0가 되면 오류로 프로그램 종료돼서 8이하로 안가도록 막음
    if value <= 8:
        long_axis = 8 / 10
    else:
        long_axis = value / 10

# 타원 모양 조절 함수
def onChangeShortAxis(value):
    global short_axis
    short_axis = value / 10

# 마우스 이벤트 함수
def onMouse(event, x, y, flags, param):
    global main_scene, delete_color, pt, range, isLeftBt, frame

    # 클릭한 위치에 해당하는 RGB값을 제거할 배경 색으로 지정함
    if event == cv2.EVENT_LBUTTONDOWN:
        isLeftBt = True
        color = main_scene[y, x, :]
        delete_color = [int(color[0]), int(color[1]), int(color[2])]
        cv2.setTrackbarPos("Red", "RGB_window", delete_color[2])
        cv2.setTrackbarPos("Green", "RGB_window", delete_color[1])
        cv2.setTrackbarPos("Blue", "RGB_window", delete_color[0])
        updateTrackbar()

    # 우클릭으로 사각형을 크기면 해당 부분 각 채널에 최소값, 최대값을 자동으로 결정하고 크로마키를 함
    elif event == cv2.EVENT_RBUTTONDOWN and pt[0] < 0:
        pt[0], pt[1] = x, y
    elif event == cv2.EVENT_MOUSEMOVE and pt[0] >= 0:
        pt[2], pt[3] = x, y
    elif event == cv2.EVENT_RBUTTONUP and pt[0] >= 0:
        isLeftBt = False

        leftX = min(pt[0], x)
        leftY = min(pt[1], y)

        rightX = max(pt[2], x)
        rightY = max(pt[3], y)

        range = main_scene[leftY:rightY, leftX:rightX, :]
        pt[0], pt[1], pt[2], pt[3] = -1, -1, -1, -1

        updateTrackbar()

# 크로마키에 필요한 마스크를 계산 후 리턴하는 함수
def getForeAndBackMask(frame, delete_color, high_margin, low_margin, range, isLeftBt):
    # 비트 연산을 통해서 보여줄 부분, 제거할 부분에 대한 Mask를 얻는다.

    # 채널 분리
    masks = cv2.split(frame)

    # 좌, 우클릭에 따라 제거할 RGB값이 다름
    if not isLeftBt and (range.shape[0] >= 3 and range.shape[1] >= 3):
        rangeSplit = cv2.split(range)
        (min_red, max_red, _, _) = cv2.minMaxLoc(rangeSplit[2])
        (min_green, max_green, _, _) = cv2.minMaxLoc(rangeSplit[1])
        (min_blue, max_blue, _, _) = cv2.minMaxLoc(rangeSplit[0])
    else:
        min_red, max_red = delete_color[2] - low_margin, delete_color[2] + high_margin
        min_green, max_green = delete_color[1] - low_margin, delete_color[1] + high_margin
        min_blue, max_blue = delete_color[0] - low_margin, delete_color[0] + high_margin

    # 각 R, G, B가 지정한 RGB값과 일치하는 부분만 가져온다, 이진화 -> 비트연산
    R_down = cv2.threshold(masks[2], min_red, 255, cv2.THRESH_BINARY)[1]
    R_up = cv2.threshold(masks[2], max_red, 255, cv2.THRESH_BINARY_INV)[1]
    R = cv2.bitwise_and(R_down, R_up)

    G_down = cv2.threshold(masks[1], min_green, 255, cv2.THRESH_BINARY)[1]
    G_up = cv2.threshold(masks[1], max_green, 255, cv2.THRESH_BINARY_INV)[1]
    G = cv2.bitwise_and(G_down, G_up)

    B_down = cv2.threshold(masks[0], min_blue, 255, cv2.THRESH_BINARY)[1]
    B_up = cv2.threshold(masks[0], max_blue, 255, cv2.THRESH_BINARY_INV)[1]
    B = cv2.bitwise_and(B_down, B_up)

    # 설정한 RGB 구역 찾기 - 비트 연산
    green_mask = cv2.bitwise_and(R, G)
    green_mask = cv2.bitwise_and(B, green_mask)

    # 화면에 크로마키에 사용할 마스크를 작업
    bg_green_mask = green_mask
    green_mask = cv2.bitwise_not(green_mask)

    # 마스크로 각 이미지를 만들고 리턴
    foreground = cv2.bitwise_and(frame, frame, mask=green_mask)
    background = cv2.bitwise_and(back, back, mask=bg_green_mask)

    return foreground, background

# 크로마키 작업을 하고, 최종 이미지를 리턴
def chromakey(frame, delete_color, high_margin, low_margin, range, isLeftBt, mp_image, face_preserving):
    if len(frame) == 0:
        return
    
    # 마스크 얻기
    foreground, background = getForeAndBackMask(frame, delete_color, high_margin, low_margin, range, isLeftBt)
    cv2.imshow("foreground", foreground)
    cv2.imshow("background", background)

    # 크로마키 작업에 얼굴 부분을 제외할 지 선택
    if face_preserving:
        # 미디어 파이프라인 이미지를 이진화한다.
        media_background = cv2.absdiff(main_scene, mp_image)
        media_background = cv2.threshold(media_background[:, :, 0], 1, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("media_background", media_background)

        # 이진화된 얼굴부분 이미지를 크로마키 이미지에 합성한다.
        face = cv2.bitwise_and(frame, frame, mask=media_background)
        frame = cv2.add(background, foreground)
        face_back = cv2.bitwise_not(media_background)
        result = cv2.bitwise_and(frame, frame, mask=face_back)

        frame = cv2.add(face, result)
    else:
        frame = cv2.add(background, foreground)

    return frame

# RGB, HIGH, LOW 조절 윈도우 화면 생성
def create_RGB_window():
    global trackbar_marge, delete_color, low_margin, high_margin

    # 제거할 배경 색을 지정하는 트렉바 윈도우 생성
    trackbar_image = np.zeros((480, 640, 3), np.uint8)

    # 트렉바에 표시할 이미지를 위해 R, G, B값을 나눔
    blue_split, green_split, red_split = cv2.split(trackbar_image)
    trackbar_marge = [blue_split, green_split, red_split]

    # 제거할 배경 색을 지정하는 트렉바 윈도우 생성, 각 R, G, B, range값으로 제거할 색, 범위를 제어할 수 있다
    cv2.imshow("RGB_window", trackbar_image)
    cv2.createTrackbar("Red", "RGB_window", delete_color[2], 255, onChangeRed)
    cv2.createTrackbar("Green", "RGB_window", delete_color[1], 255, onChangeGreen)
    cv2.createTrackbar("Blue", "RGB_window", delete_color[0], 255, onChangeBlue)
    cv2.createTrackbar("high", "RGB_window", high_margin, 200, onChangeHigh)
    cv2.createTrackbar("low", "RGB_window", low_margin, 200, onChangeLow)

# 기타 편집창 화면 생성
def create_edit_window():
    global brightness, long_axis, short_axis, avg_brightness
    # 기타 편집 윈도우
    cv2.imshow("edit_image", np.zeros((60, 400), np.uint8))
    cv2.createTrackbar("brightness", "edit_image", brightness, 255, onChangeBrightness)
    cv2.createTrackbar("face size", "edit_image", long_axis, 255, onChangeLongAxis)
    cv2.createTrackbar("face shape", "edit_image", short_axis, 255, onChangeShortAxis)

# 미디어 파이프라인 얼굴 추적 및 타원을 그리는 함수 - [오픈 소스 활용]
def ovalDrawOnFaceLines(face_mesh):
    global mp_image
    mp_image.flags.writeable = False
    mp_image = cv2.cvtColor(mp_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(mp_image)

    mp_image.flags.writeable = True
    mp_image = cv2.cvtColor(mp_image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            continue

    # GPT 수정 부분 - 미디어파이프라인에 얼굴을 하얀 가면으로 바꿔주는 작업 ▽
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), -float('inf'), -float('inf')

    # 얼굴 라인을 그리는 좌표에서 가장 작은 좌표와 큰 좌표를 가져온다
    for lm in face_landmarks.landmark:
        x, y = int(lm.x * mp_image.shape[1]), int(lm.y * mp_image.shape[0])
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)
    # 타원으로 얼굴 영역 채우기 - 위에서 얻은 좌표를 통해서 타원을 생성
    center = ((x_min + x_max) // 2, (y_min + y_max) // 2)  # 얼굴 중심점 구하기
    long_axis_length = int(((y_max - y_min)) // long_axis)
    short_axis_length = int(long_axis_length * short_axis)
    axes_length = (short_axis_length, long_axis_length)
    cv2.ellipse(mp_image, center, axes_length, 0, 0, 360, (255, 255, 255), cv2.FILLED)
    # GPT 수정 부분 △

# 화면 초기화
def reset_value():
    global pt, range, isLeftBt, avg_brightness, face_preserving, trackbar_marge, delete_color, low_margin, high_margin, brightness, long_axis, short_axis
    # 마우스 이벤트 변수
    pt = [-1, -1, -1, -1]
    isLeftBt = True

    # 얼굴 크로마키 영역 적용 여부
    face_preserving = False

    # 트랙바 RPG 윈도우 변수
    delete_color = [80, 200, 80]
    low_margin = 10
    high_margin = 10
    cv2.setTrackbarPos("Red", "RGB_window", delete_color[2])
    cv2.setTrackbarPos("Green", "RGB_window", delete_color[1])
    cv2.setTrackbarPos("Blue", "RGB_window", delete_color[0])
    cv2.setTrackbarPos("high", "RGB_window", high_margin)
    cv2.setTrackbarPos("low", "RGB_window", low_margin)
    cv2.setTrackbarPos("Blue", "RGB_window", delete_color[0])

    # 트랙바 edit 윈도우 변수
    brightness = 128
    avg_brightness = 128
    long_axis = 16
    short_axis = 6
    cv2.setTrackbarPos("brightness", "edit_image", brightness)
    cv2.setTrackbarPos("avg brightness", "edit_image", avg_brightness)
    cv2.setTrackbarPos("face size", "edit_image", long_axis)
    cv2.setTrackbarPos("face shape", "edit_image", short_axis)


''' ----------------- ▽ 변수 설정 ▽ ----------------- '''
# 미디어 파이프라인 코드 -> 얼굴을 추적하여 하얀 선이 그려진 이미지를 얻기 위함 - [오픈 소스 활용]
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# 메인 윈도우 변수
main_scene = np.array([])
frame = None

# 마우스 이벤트 변수
pt = [-1, -1, -1, -1]
range = np.zeros((5, 5, 3))
isLeftBt = True

# 얼굴 크로마키 영역 적용 여부
face_preserving = False

# 트랙바 RPG 윈도우 변수
trackbar_marge = np.zeros((1, 1, 3))
delete_color = [80, 200, 80]
low_margin = 10
high_margin = 10

# 트랙바 edit 윈도우 변수
brightness = 128
avg_brightness = 128
long_axis = 16
short_axis = 6

if __name__ == "__main__":
    # 트랙바 조절 윈도우 화면 생성
    create_RGB_window()
    create_edit_window()

    # 크로마키 이미지 가져오기 -> 화면(카메라) 이미지 크기에 맞게 크기를 변경
    back = cv2.imread("../image/abs_test1.jpg", cv2.IMREAD_COLOR)
    height, width = 480, 640
    back = cv2.resize(back, (width, height))

    # 내 화면(카메라) 가져오기 및 설정
    fps = 29.97  # 초당 프레임 수
    delay = round(1000 / fps)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("카메라 연결 안됨")
    cv2.namedWindow("chromakey")  # 윈도우 생성 - 반드시 생성 해야함
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 카메라 프레임 너비
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 카메라 프레임 높이

    # 마우스 클릭 이벤트 연결
    cv2.setMouseCallback("chromakey", onMouse)

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, mp_image = cap.read()
            if not ret:
                break

            # 영상 원본을 main_scene에 복사해둠, mp_image는 아래 함수에서 편집됨
            main_scene = mp_image.copy()

            # 미디어 파이프라인와 타원 그리기 GPT 도움받는 코드를 모아둔 함수 - [오픈 소스 활용]
            ovalDrawOnFaceLines(face_mesh)

            # 크로마키 영상 얻고 보여주기
            frame = chromakey(main_scene, delete_color, high_margin, low_margin, range, isLeftBt, mp_image, face_preserving)

            # 최종 화면 밝기 조절
            frame = cv2.add(frame, brightness - 128)

            # 우클릭 드래스 사각형 그리기
            if pt[0] != -1:
                cv2.rectangle(frame, (pt[0], pt[1], pt[2]-pt[0], pt[3]-pt[1]), (0, 0, 255), 3, cv2.LINE_4)

            cv2.imshow("chromakey", frame)

            # delay만큼 키입력을 기다림 -> 이를 이용하여 영상 프레임을 조절
            key = cv2.waitKey(delay)
            if key == ord('q'): # 얼굴 보존할 지 여부
                face_preserving = not face_preserving
                updateTrackbar()
            elif key == ord('r'): # 설정값 리셋
                reset_value()

    cap.release()
