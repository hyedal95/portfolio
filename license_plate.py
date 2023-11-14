#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
plt.style.use('dark_background')

#이미지 읽기
img = cv2.imread('test5.jpg')
#이미지 높이, 너비, 채널 가져오기
height, width, channel = img.shape
#이미지 그릴때 사용
temp_result = np.zeros((height, width, channel), dtype=np.uint8)
#이미지 크기 설정(1200x1000)
plt.figure(figsize=(12,10))

#####################################################################

#이미지 회색으로
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#이미지 노이즈 제거
img_blur = cv2.GaussianBlur(gray, ksize=(5,5), sigmaX=0)
#이미지 흑백처리 (INV반전)
img_threshold = cv2.adaptiveThreshold(
    img_blur,
    maxValue = 255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
#    thresholdType=cv2.THRESH_BINARY,
    blockSize=19,
    C=9
)

#이미지 윤곽선
contours, _ = cv2.findContours(
    img_threshold, 
    mode=cv2.RETR_LIST, 
    method=cv2.CHAIN_APPROX_SIMPLE
)

"""
#윤곽선 그리기
cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
"""

#####################################################################

#이미지 윤곽선에 대한 사각형 만들기 위한 좌표 생성
rectangle_dict = []

for rectangle_contour in contours:
    x, y, w, h = cv2.boundingRect(rectangle_contour)
#   윤곽선에 대한 사각형 좌표들
    rectangle_dict.append({
        'contour': rectangle_contour,
        'x': x,     #x 좌표
        'y': y,     #y 좌표
        'w': w,     #가로 길이
        'h': h,     #세로 길이
        'cx': x + (w / 2),  #x 좌표에서 사각형 가운데
        'cy': y + (h / 2)   #y 좌표에서 사각형 가운데
    })
#   윤곽선에 대한 사각형 그리기
#    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)

#####################################################################

#번호판 안에 있는 숫자 사각형과 비슷한 비율과 크기 찾기
MIN_AREA = 80   #최소 넓이 80
MIN_WIDTH, MIN_HEIGHT = 2, 8    #최소 가로 2, 세로 8
MIN_RATIO, MAX_RATIO = 0.25, 1.25    #가로,세로 비율 최소 1:4 ~ 최대 1:1

possible_rectangle = []

cnt = 0
for d in rectangle_dict:
    area = d['w'] * d['h']  #넓이
    ratio = d['w'] / d['h'] #가로, 세로 비율
    
    #측정
    if area > MIN_AREA \
    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_rectangle.append(d)

"""
for d in possible_rectangle:
#   윤곽선
#    cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
#   사각형
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
"""

#####################################################################

#번호판 안의 숫자 사각형 찾기
MAX_DIAG_MULTIPLYER = 5     # 사각형 안 대각선 길이와 사각형과 사각형 사이의 거리
MAX_ANGLE_DIFF = 12.0   # 사각형과 사각형 사이의 각도
MAX_AREA_DIFF = 0.5     # 사각형과 사각형의 넓이 차이
MAX_WIDTH_DIFF = 0.8    #사각형과 사각형의 가로 길이 차이
MAX_HEIGHT_DIFF = 0.2   #사각형과 사각형의 세로 길이 차이
MIN_N_MATCHED = 3   #매치되는게 최소 3개 이상

def find_chars(contour_list):
    matched_result_idx = []
    
    #d1 과 d2 비교
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue
            
            #비교한 사각형 중심 사이의 가로, 세로 길이
            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            #d1 사긱형의 대각선 길이
            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            #d1 과 d2 중심 사이 거리
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            
            #각도 측정
            if dx == 0:
                #dx가 0이면 90도임
                angle_diff = 90
            else:
                #arctan(dy/dx)로 각도 측정
                angle_diff = np.degrees(np.arctan(dy / dx))

            #넓이 비율 측정
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            #가로 길이 비율 측정
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            #세로 길이 비율 측정
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            #측정
            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        #매치되는거 3개 이상인 것 찾기
        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        #매치 안되는거
        unmatched_contour_idx = []
        for d3 in contour_list:
            if d3['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d3['idx'])

        unmatched_contour = np.take(possible_rectangle, unmatched_contour_idx)
        
        #매치 안되는거 find_chars 재귀함수 ㄱ
        recursive_contour_list = find_chars(unmatched_contour)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    #결과 반환
    return matched_result_idx
    
#함수 사용해서 matched_result에 저장
result_idx = find_chars(possible_rectangle)

matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_rectangle, idx_list))

"""
#번호판 숫자 찾은거 그리기
for r in matched_result:
    for d in r:
#       윤곽선
        cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
#       사각형
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
"""

#####################################################################

#번호판 테두리 찾기
MIN_LICENSE_ANGLE = 9     #최소 각도 9도
MAX_LICENSE_ANGLE = 30    #최대 각도 30도

possible_license = []

cnt = 0
for d in rectangle_dict:
    if d['w'] == 0:
        #d가 0이면 90도임
        license_angle = 90
    else:
        #arctan(dy/dx)로 각도 측정
        license_angle = np.degrees(np.arctan(d['h'] / d['w']))
    
    #측정
    if MAX_LICENSE_ANGLE > license_angle and MIN_LICENSE_ANGLE < license_angle:
        d['idx'] = cnt
        cnt += 1
        possible_license.append(d)

"""
#번호판 테두리 찾은거 그리기
for d in possible_license:
#   윤곽선
#    cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
#   사각형
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
"""

#####################################################################

#번호판 테두리와 숫자 사각형 찾은거 조합 후 번호판 테두리 찾기
#번호판 테두리 찾기
license_plate_result_idx = []

for d1 in possible_license:
    matched_license = 0     #번호판 테두리 안에 사각형이 몇개인지
    for r in matched_result:
        for d2 in r:
            #번호판 테두리 안에 번호가 존재하는지 찾기
            if d2['cx'] > d1['x'] and d2['cx'] < (d1['x'] + d1['w']) \
            and d2['cy'] > d1['y'] and d2['cy'] < (d1['y'] + d1['h']):
                matched_license += 1

            if matched_license >= 3:
                license_plate_result_idx.append(d1)

#번호판 테두리 추려내기 
license_plate_result = []
#넓이를 비교해서 젤 작은 것 찾기
sorted_license_plate = sorted(license_plate_result_idx, key=lambda x: x['w']*x['h'])
license_plate_result.append(sorted_license_plate[0])

"""
#번호판 테두리 출력
for d in license_plate_result:
#   윤곽선
    cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
#   사각형
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
"""

#####################################################################

#번호판 숫자 찾기
license_number_result= []

for d1 in license_plate_result:
    for r in matched_result:
        for d2 in r:
            if d2['cx'] > d1['x'] and d2['cx'] < (d1['x'] + d1['w']) \
            and d2['cy'] > d1['y'] and d2['cy'] < (d1['y'] + d1['h']):
                license_number_result.append(d2)

"""
#번호판 숫자 출력
for d in license_number_result:
#   윤곽선
    cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
#   사각형
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
"""

#####################################################################

#번호판 이미지 평평하게 돌리기
#번호판 숫자 정렬
sorted_chars_number = sorted(license_number_result, key=lambda x: x['w']*x['h']*x['x'])
#번호판 이미지 정렬
sorted_chars_plate = sorted(license_plate_result, key=lambda x: x['x'])

#번호판 길이의 x 중간
plate_cx = sorted_chars_plate[0]['cx']
#번호판 높이의 y 중간
plate_cy = sorted_chars_plate[0]['cy']

#높이 (번호판 숫자  맨끝과 그 전 비교)
triangle_height = sorted_chars_number[-1]['cy'] - sorted_chars_number[-2]['cy']
#빗변 (번호판 숫자  맨끝과 그 전 비교)
triangle_hypotenus = np.linalg.norm(
    np.array([sorted_chars_number[-2]['cx'], sorted_chars_number[-2]['cy']]) - 
    np.array([sorted_chars_number[-1]['cx'], sorted_chars_number[-1]['cy']])
)

#번호판 각도 arcsin 으로 찾기
angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
#각도만큼 돌려 평평하게
rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

#번호판 가로 길이
plate_width = sorted_chars_plate[0]['w']
#번호판 세로 길이
plate_height = sorted_chars_plate[0]['h']

#이미지 돌린거 저장
img_rotated = cv2.warpAffine(img_threshold, M=rotation_matrix, dsize=(width, height))
img_cropped = cv2.getRectSubPix(
    img_rotated, 
    patchSize=(int(plate_width), int(plate_height)), 
    center=(int(plate_cx), int(plate_cy))
)

#이미지 돌린거 딕셔너리 정보
license_plate_imgs = []
license_plate_infos = []
    
license_plate_imgs.append(img_cropped)
license_plate_infos.append({
    'x': int(plate_cx - plate_width / 2),
    'y': int(plate_cy - plate_height / 2),
    'w': int(plate_width),
    'h': int(plate_height)
})

# 나중에~~
#plt.subplot(len(matched_result), 1, i+1)

#####################################################################

#번호판 이미지 다시 깔끔히 처리
longest_idx, longest_text = -1, 0
plate_chars = []

for i, plate_img in enumerate(license_plate_imgs):
    plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
    _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # find contours again (same as above)
    contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    
    plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
    plate_max_x, plate_max_y = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        area = w * h
        ratio = w / h

        if area > MIN_AREA \
        and w > MIN_WIDTH and h > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            if x < plate_min_x:
                plate_min_x = x
            if y < plate_min_y:
                plate_min_y = y
            if x + w > plate_max_x:
                plate_max_x = x + w
            if y + h > plate_max_y:
                plate_max_y = y + h
                
    img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
    
    img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
    _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

    
    chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0')
    result_chars = ''
    has_digit = False
    for c in chars:
        if ord('가') or ord('나') or ord('다') or ord('라') or ord('마') \
        or ord('거') or ord('너') or ord('더') or ord('러') or ord('머') or ord('버') or ord('서') or ord('어') or ord('저') \
        or ord('고') or ord('노') or ord('도') or ord('로') or ord('모') or ord('보') or ord('소') or ord('오') or ord('조') \
        or ord('구') or ord('누') or ord('두') or ord('루') or ord('무') or ord('부') or ord('수') or ord('우') or ord('주') \
        or ord('바') or ord('사') or ord('아') or ord('자') \
        or ord('하') or ord('허') or ord('호') \
        or ord('배') \
        or c.isdigit():
            if c.isdigit():
                has_digit = True
            result_chars += c
    
    print(result_chars)
    plate_chars.append(result_chars)

    if has_digit and len(result_chars) > longest_text:
        longest_idx = i

"""
chars = pytesseract.image_to_string(img_cropped, lang='kor', config='--psm 7 --oem 0')
result_chars = ''
has_digit = False
for c in chars:
    if ord('가') or ord('나') or ord('다') or ord('라') or ord('마') \
    or ord('거') or ord('너') or ord('더') or ord('러') or ord('머') or ord('버') or ord('서') or ord('어') or ord('저') \
    or ord('고') or ord('노') or ord('도') or ord('로') or ord('모') or ord('보') or ord('소') or ord('오') or ord('조') \
    or ord('구') or ord('누') or ord('두') or ord('루') or ord('무') or ord('부') or ord('수') or ord('우') or ord('주') \
    or ord('바') or ord('사') or ord('아') or ord('자') \
    or ord('하') or ord('허') or ord('호') \
    or ord('배') \
    or c.isdigit():
        if c.isdigit():
            has_digit = True
        result_chars += c

print(result_chars)
"""

#이미지 보여주기
plt.subplot(1,2,1)
plt.imshow(temp_result, cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(img_result, cmap = 'gray')
# %%