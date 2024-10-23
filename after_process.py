import json
import math
import cv2
import open3d as o3d
import numpy as np


def draw_arrow(conners, r, g, b):
    x_center = (conners[0][0] + conners[1][0] + conners[2][0] + conners[3][0]) / 4.
    y_center = (conners[0][1] + conners[1][1] + conners[2][1] + conners[3][1]) / 4.
    z_center = (conners[0][2] + conners[1][2] + conners[2][2] + conners[3][2]) / 4.
    x_center2 = (conners[2][0] + conners[3][0]) / 2. + .5*((conners[2][0] + conners[3][0]) / 2. - x_center)
    y_center2 = (conners[2][1] + conners[3][1]) / 2. + .5*((conners[2][1] + conners[3][1]) / 2. - y_center)
    z_center2 = (conners[2][2] + conners[3][2]) / 2.

    polygon_points = np.array([[x_center, y_center, z_center], [x_center2, y_center2, z_center2]])
    lines = [[0, 1], [1, 0]]
    color = [[r, g, b] for i in range(len(lines))]
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.utility.Vector3dVector(color) 
    lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
    return lines_pcd

def draw_cube(conners, r, g, b):
    polygon_points = conners
    lines = [[0, 1], [1, 2], [2, 3],[3, 0],[0, 4], [1, 5],[2, 6],[3, 7], [4, 5],[5, 6],[6, 7],[7, 4]]
    color = [[r, g, b] for i in range(len(lines))]
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.utility.Vector3dVector(color) 
    lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
    return lines_pcd

def get_conners(label):
    conners = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.],[0., 0. ,0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    conners[0][0] = label[0] - label[3]/2
    conners[0][1] = label[1] + label[4]/2
    conners[0][2] = label[2] + label[5]/2

    conners[1][0] = label[0] - label[3]/2
    conners[1][1] = label[1] - label[4]/2
    conners[1][2] = label[2] + label[5]/2

    conners[2][0] = label[0] + label[3]/2
    conners[2][1] = label[1] - label[4]/2
    conners[2][2] = label[2] + label[5]/2

    conners[3][0] = label[0] + label[3]/2
    conners[3][1] = label[1] + label[4]/2
    conners[3][2] = label[2] + label[5]/2

    conners[4][0] = label[0] - label[3]/2
    conners[4][1] = label[1] + label[4]/2
    conners[4][2] = label[2] - label[5]/2

    conners[5][0] = label[0] - label[3]/2
    conners[5][1] = label[1] - label[4]/2
    conners[5][2] = label[2] - label[5]/2

    conners[6][0] = label[0] + label[3]/2
    conners[6][1] = label[1] - label[4]/2
    conners[6][2] = label[2] - label[5]/2

    conners[7][0] = label[0] + label[3]/2
    conners[7][1] = label[1] + label[4]/2
    conners[7][2] = label[2] - label[5]/2

    i = 0
    for conner in conners:
        exchange_x = (conner[0] - label[0]) * math.cos(label[6]) - (conner[1] - label[1]) * math.sin(label[6]) + label[0]
        exchange_y = (conner[0] - label[0]) * math.sin(label[6]) + (conner[1] - label[1]) * math.cos(label[6]) + label[1]
        conners[i][0] = exchange_x
        conners[i][1] = exchange_y
        i += 1
    return conners

def get_json_AI_result(path):
    labels = []
    categray = []
    with open(path, 'r') as f:
        data = json.load(f)
        
        for boxes in data['boxes_lidar']:
            label = []
            label.append(boxes[0])
            label.append(boxes[1])
            label.append(boxes[2])
            label.append(boxes[3])
            label.append(boxes[4])
            label.append(boxes[5])
            label.append(boxes[6])
            labels.append(label)
        for cat in data['name']:
            categray.append(cat)
    return labels, categray

def expand_box(cloud, conners, addvalue):

    x_max = -8888.
    y_max = -8888.
    x_min = 8888.
    y_min = 8888.
    for conner in conners:
        if x_max < conner[0]:
            x_max = conner[0]
        if y_max < conner[1]:
            y_max = conner[1]
        if x_min > conner[0]:
            x_min = conner[0]
        if y_min > conner[1]:
            y_min = conner[1]

    x_max += addvalue
    y_max += addvalue
    x_min -= addvalue
    y_min -= addvalue

    contour = np.array([[conners[0][0], conners[0][1]], [conners[1][0], conners[1][1]], [conners[2][0], conners[2][1]], [conners[3][0], conners[3][1]]])
    contour = np.float32(contour)
    
    
    contour_hull = cv2.convexHull(contour)

    inner_pts = []
    for point in cloud:
        if point[0] < x_max and point[0] > x_min and point[1] < y_max and point[1] > y_min and point[2] <conners[0][2] and point[2] > conners[4][2]:
            if cv2.pointPolygonTest(contour_hull, (point[0], point[1]), False):
                inner_pts.append(point)
    inner_pts = np.array(inner_pts)
    #print("inner_pts: ", inner_pts)
    return inner_pts

######################################################################
def get_vertical_point(pta, ptb, ptp):
    if pta[0] == ptb[0]:
        return (0, ptp[1])
    k = 0.
    k = (ptb[1] - pta[1])/(ptb[0] - pta[0])
    a = k
    b = -1.0
    c = pta[1] - k * pta[0]
    x = (b * b * ptp[0] - a * b * ptp[1] - a * c) / (a * a + b * b)
    y = (a * a * ptp[1] - a * b * ptp[0] - b * c) / (a * a + b * b)
    return (x, y)

def get_dist_P2L(pointP, pointA, pointB):
    A = 0.
    B = 0.
    C = 0.
    A = pointA[1] - pointB[1]
    B = pointB[0] - pointA[0]
    C = pointA[0] * pointB[1] - pointA[1] * pointB[0]
    distance = 0.
    distance = abs(A * pointP[0] + B * pointP[1] + C) / math.sqrt(A * A + B * B)
    return distance

def get_dist_P2L_vec(pointP, pointA, pointB):
    A = 0.
    B = 0.
    C = 0.
    A = pointA[1] - pointB[1]
    B = pointB[0] - pointA[0]
    C = pointA[0] * pointB[1] - pointA[1] * pointB[0]
    distance = 0.
    distance = (A * pointP[0] + B * pointP[1] + C) / math.sqrt(A * A + B * B)
    return distance

def get_L_fitted(cloud):

    minz = 888.
    maxZ = -888.
    for point in cloud:
        if point[2] > maxZ:
            maxZ = point[2]
        if point[2] < minz:
            minz = point[2]


    pts_chosen = []
    choose_step = len(cloud) / 55
    choose_step = int(choose_step)
    if choose_step < 1:
        choose_step = 1
    i = 0
    for point in cloud:
        i += choose_step
        if i >= len(cloud):
            break
        pts_chosen.append((cloud[i][0], cloud[i][1]))
    print_pt4 = 0

    minDisall = .1
    mostInsidePoints = 0
    anchor_points = []

    for i in range(len(pts_chosen)):
        pta = pts_chosen[i]
        for j in range(len(pts_chosen)):
            if j <= i:
                continue
            ptb = pts_chosen[j]
            if abs(ptb[0] - pta[0]) < 1.3 and abs(ptb[1] - pta[1]) < 0.5:
                continue
            for k in range(len(pts_chosen)):
                ptc = pts_chosen[k]
                if k <= j:
                    continue
                if abs(ptb[0] - ptc[0]) < 0.5 and abs(ptb[1] - ptc[1]) < 0.5:
                    continue
                if abs(math.atan2(pta[0] - ptb[0], pta[1] - ptb[1]) - math.atan2(pta[0] - ptc[0], pta[1] - ptc[1])) < .1:
                    continue
                ptd = get_vertical_point(pta, ptb, ptc)
                sumDis = 0
                max_shadow_points = 200
                downSample = len(cloud) / max_shadow_points
                downSample = int(downSample)
                if downSample < 1:
                    downSample = 1
                s = 0
                for a in range(len(cloud)):
                    s += downSample
                    if s >= len(cloud):
                        break
                    distance1 = get_dist_P2L(cloud[s], pta, ptb)
                    distance2 = get_dist_P2L(cloud[s], ptc, ptd)
                    if min(distance1, distance2) < minDisall:
                        sumDis+=1
                if sumDis > mostInsidePoints:
                    mostInsidePoints = sumDis
                    anchor_points = []
                    anchor_points.append(pta)
                    anchor_points.append(ptb)
                    anchor_points.append(ptc)
                    anchor_points.append(ptd)

    if len(anchor_points) == 0:
        return anchor_points
    
    plusMax = -8888.
    maxPt = (0., 0.)
    minusMax = 8888.
    minPt = (0., 0.)
    plusMax2 = -8888.
    maxPt2 = (0., 0.)
    minusMax2 = 8888.
    minPt2 = (0., 0.)

    xdevide = anchor_points[0][0] - anchor_points[1][0]
    ydevide = anchor_points[0][1] - anchor_points[1][1]
    xdevide2 = anchor_points[2][0] - anchor_points[3][0]
    ydevide2 = anchor_points[2][1] - anchor_points[3][1]

    pt1 = (0, ydevide)
    pt2 = (-xdevide, 0)
    pt3 = (0, ydevide2)
    pt4 = (-xdevide2, 0)

    i = 0
    for pt in cloud:
        pt_2f = (pt[0], pt[1])
        dis1 = get_dist_P2L_vec(pt_2f, pt1, pt2)
        if dis1 > plusMax:
            plusMax = dis1
            maxPt = pt
        if dis1 < minusMax:
            minusMax = dis1
            minPt = pt
        dis2 = get_dist_P2L_vec(pt_2f, pt3, pt4)
        if dis2 > plusMax2:
            plusMax2 = dis2
            maxPt2 = pt
        if dis2 < minusMax2:
            minusMax2 = dis2
            minPt2 = pt

    x_squre = xdevide * xdevide
    y_squre = ydevide * ydevide

    conner1 = [0., 0.]
    conner1[1] = (xdevide * ydevide * (maxPt2[0] - maxPt[0]) + x_squre * maxPt[1] + y_squre * maxPt2[1]) / (x_squre + y_squre)
    if ydevide == 0:
        conner1[0] = maxPt2[0]
    else:
        conner1[0] = maxPt[0] + xdevide * (conner1[1] - maxPt[1]) / ydevide

    conner2 = [0., 0.]
    conner2[1] = (xdevide * ydevide * (minPt2[0] - maxPt[0]) + x_squre * maxPt[1] + y_squre * minPt2[1]) / (x_squre + y_squre)
    if ydevide == 0:
        conner2[0] = minPt2[0]
    else:
        conner2[0] = maxPt[0] + xdevide * (conner2[1] - maxPt[1]) / ydevide

    conner3 = [0., 0.]
    conner3[1] = (xdevide * ydevide * (maxPt2[0] - minPt[0]) + x_squre * minPt[1] + y_squre * maxPt2[1]) / (x_squre + y_squre)
    if ydevide == 0:
        conner3[0] = maxPt2[0]
    else:
        conner3[0] = minPt[0] + xdevide * (conner3[1] - minPt[1]) / ydevide
    
    conner4 = [0., 0.]
    conner4[1] = (xdevide * ydevide * (minPt2[0] - minPt[0]) + x_squre * minPt[1] + y_squre * minPt2[1]) / (x_squre + y_squre)
    if ydevide == 0:
        conner4[0] = minPt2[0]
    else:
        conner4[0] = minPt[0] + xdevide * (conner4[1] - minPt[1]) / ydevide
    
    contour = np.array([[conner1[0], conner1[1]], [conner2[0], conner2[1]], [conner3[0], conner3[1]], [conner4[0], conner4[1]]])
    contour = np.float32(contour)

    rect = cv2.minAreaRect(contour)
    box = np.float32(cv2.boxPoints(rect))

    result = []
    
    for i in range(4):
        point_one = [0., 0., 0.]
        point_one[0] = box[i][0]
        point_one[1] = box[i][1]
        point_one[2] = maxZ
        result.append(point_one)
    for i in range(4):
        point_one = [0., 0., 0.]
        point_one[0] = box[i][0]
        point_one[1] = box[i][1]
        point_one[2] = minz
        result.append(point_one)
    result = np.array(result)

    return result

def run_re_orientation(cloud, label):
    label_out = label
    if len(cloud) < 200:
        return label_out
    conner2 = get_L_fitted(cloud)
    if len(conner2) == 0:
        return label_out

    length1 = math.sqrt((conner2[0][0] - conner2[1][0])*(conner2[0][0] - conner2[1][0]) + (conner2[0][1] - conner2[1][1])*(conner2[0][1] - conner2[1][1]))
    length2 = math.sqrt((conner2[2][0] - conner2[1][0])*(conner2[2][0] - conner2[1][0]) + (conner2[2][1] - conner2[1][1])*(conner2[2][1] - conner2[1][1]))
    if length1 > length2:
        angle = math.atan2(conner2[0][1] - conner2[1][1], conner2[0][0] - conner2[1][0])
        if abs(angle - label[6]) < 3.14159/2:
            label_out[6] =  angle
        else:
            if label[6] > angle:
                label[6] = angle + 3.14159
            else:
                label[6] = angle - 3.14159
    else:
        angle = math.atan2(conner2[2][1] - conner2[1][1], conner2[2][0] - conner2[1][0])
        if abs(angle - label[6]) < 3.14159/2:
            label_out[6] =  angle
        else:
            if label[6] > angle:
                label[6] = angle + 3.14159
            else:
                label[6] = angle - 3.14159
    return label_out

def run_re_orientation_all(cloud, labels):
    result = []
    for label in labels:
        label_out = label
        conners = get_conners(label)
        points_in = expand_box(cloud, conners, 0)
        if len(points_in) < 200:
            continue
        conner2 = get_L_fitted(points_in)
        if len(conner2) == 0:
            continue

        length1 = math.sqrt((conner2[0][0] - conner2[1][0])*(conner2[0][0] - conner2[1][0]) + (conner2[0][1] - conner2[1][1])*(conner2[0][1] - conner2[1][1]))
        length2 = math.sqrt((conner2[2][0] - conner2[1][0])*(conner2[2][0] - conner2[1][0]) + (conner2[2][1] - conner2[1][1])*(conner2[2][1] - conner2[1][1]))
        if length1 > length2:
            angle = math.atan2(conner2[0][1] - conner2[1][1], conner2[0][0] - conner2[1][0])
            if abs(angle - label[6]) < 3.14159/2:
               label_out[6] =  angle
            else:
                if label[6] > angle:
                    label[6] = angle + 3.14159
                else:
                    label[6] = angle - 3.14159
        else:
            angle = math.atan2(conner2[2][1] - conner2[1][1], conner2[2][0] - conner2[1][0])
            if abs(angle - label[6]) < 3.14159/2:
               label_out[6] =  angle
            else:
                if label[6] > angle:
                    label[6] = angle + 3.14159
                else:
                    label[6] = angle - 3.14159
        result.append(label_out)
    return result

######################################################################

def count_in_number_x1(cloud_around, label, bbx, set_test_width):
    intersept_rate = set_test_width/label[3]
    conners2f = []
    conners2f.append((bbx[0][0], bbx[0][1]))
    conners2f.append((bbx[1][0], bbx[1][1]))
    conners2f.append((bbx[1][0]+(bbx[2][0]-bbx[1][0])*intersept_rate, bbx[1][1]+(bbx[2][1]-bbx[1][1])*intersept_rate))
    conners2f.append((bbx[0][0]+(bbx[3][0]-bbx[0][0])*intersept_rate, bbx[0][1]+(bbx[3][1]-bbx[0][1])*intersept_rate))

    contour = np.array([[conners2f[0][0], conners2f[0][1]], [conners2f[1][0], conners2f[1][1]], [conners2f[2][0], conners2f[2][1]], [conners2f[3][0], conners2f[3][1]]])
    contour = np.float32(contour)

    cloud_in_one = []
    for point in cloud_around:
        pt = (point[0], point[1])
        jud = cv2.pointPolygonTest(contour, pt, 0)
        if jud >= 0:
            cloud_in_one.append(point)
    return len(cloud_in_one)

def count_in_number_x2(cloud_around, label, bbx, set_test_width):
    intersept_rate = set_test_width/label[3]
    conners2f = []
    conners2f.append((bbx[2][0], bbx[2][1]))
    conners2f.append((bbx[3][0], bbx[3][1]))
    conners2f.append((bbx[3][0]+(bbx[0][0]-bbx[3][0])*intersept_rate, bbx[3][1]+(bbx[0][1]-bbx[3][1])*intersept_rate))
    conners2f.append((bbx[2][0]+(bbx[1][0]-bbx[2][0])*intersept_rate, bbx[2][1]+(bbx[1][1]-bbx[2][1])*intersept_rate))

    contour = np.array([[conners2f[0][0], conners2f[0][1]], [conners2f[1][0], conners2f[1][1]], [conners2f[2][0], conners2f[2][1]], [conners2f[3][0], conners2f[3][1]]])
    contour = np.float32(contour)

    cloud_in_one = []
    for point in cloud_around:
        pt = (point[0], point[1])
        jud = cv2.pointPolygonTest(contour, pt, 0)
        if jud >= 0:
            cloud_in_one.append(point)
    return len(cloud_in_one)

def count_in_number_y1(cloud_around, label, bbx, set_test_width):
    intersept_rate = set_test_width/label[4]
    conners2f = []
    conners2f.append((bbx[0][0], bbx[0][1]))
    conners2f.append((bbx[3][0], bbx[3][1]))
    conners2f.append((bbx[3][0]+(bbx[2][0]-bbx[3][0])*intersept_rate, bbx[3][1]+(bbx[2][1]-bbx[3][1])*intersept_rate))
    conners2f.append((bbx[0][0]+(bbx[1][0]-bbx[0][0])*intersept_rate, bbx[0][1]+(bbx[1][1]-bbx[0][1])*intersept_rate))

    contour = np.array([[conners2f[0][0], conners2f[0][1]], [conners2f[1][0], conners2f[1][1]], [conners2f[2][0], conners2f[2][1]], [conners2f[3][0], conners2f[3][1]]])
    contour = np.float32(contour)

    cloud_in_one = []
    for point in cloud_around:
        pt = (point[0], point[1])
        jud = cv2.pointPolygonTest(contour, pt, 0)
        if jud >= 0:
            cloud_in_one.append(point)
    return len(cloud_in_one)

def count_in_number_y2(cloud_around, label, bbx, set_test_width):
    intersept_rate = set_test_width/label[4]
    conners2f = []
    conners2f.append((bbx[2][0], bbx[2][1]))
    conners2f.append((bbx[1][0], bbx[1][1]))
    conners2f.append((bbx[1][0]+(bbx[0][0]-bbx[1][0])*intersept_rate, bbx[1][1]+(bbx[0][1]-bbx[1][1])*intersept_rate))
    conners2f.append((bbx[2][0]+(bbx[3][0]-bbx[2][0])*intersept_rate, bbx[2][1]+(bbx[3][1]-bbx[2][1])*intersept_rate))

    contour = np.array([[conners2f[0][0], conners2f[0][1]], [conners2f[1][0], conners2f[1][1]], [conners2f[2][0], conners2f[2][1]], [conners2f[3][0], conners2f[3][1]]])
    contour = np.float32(contour)

    cloud_in_one = []
    for point in cloud_around:
        pt = (point[0], point[1])
        jud = cv2.pointPolygonTest(contour, pt, 0)
        if jud >= 0:
            cloud_in_one.append(point)
    return len(cloud_in_one)

def count_in_number_z1(cloud_around, label, bbx, set_test_width):
    intersept_value = bbx[0][2] - set_test_width
    cloud_in_one = []
    for point in cloud_around:
        if point[2] > intersept_value:
            cloud_in_one.append(point)
    return len(cloud_in_one)

def count_in_number_z2(cloud_around, label, bbx, set_test_width):
    intersept_value = bbx[4][2] + set_test_width
    cloud_in_one = []
    for point in cloud_around:
        if point[2] < intersept_value:
            cloud_in_one.append(point)
    return len(cloud_in_one)

def recover_size(cloud_around, label, bbx, standurd_width, standurd_height, standurd_zvalue):
    label_out = label

    judge_rate = 0.05 * len(cloud_around)
    number_x1 = count_in_number_x1(cloud_around, label, bbx, .2)
    number_x2 = count_in_number_x2(cloud_around, label, bbx, .2)
    number_y1 = count_in_number_y1(cloud_around, label, bbx, .2)
    number_y2 = count_in_number_y2(cloud_around, label, bbx, .2)
    number_z1 = count_in_number_z1(cloud_around, label, bbx, .2)
    number_z2 = count_in_number_z2(cloud_around, label, bbx, .2)
    
    is_dense_x1 = 1
    is_dense_x2 = 1
    is_dense_y1 = 1
    is_dense_y2 = 1
    is_dense_z1 = 1

    if number_x1 < 20 and number_x1 < judge_rate:
        is_dense_x1 = 0
    if number_x2 < 20 and number_x2 < judge_rate:
        is_dense_x2 = 0
    if number_y1 < 20 and number_y1 < judge_rate:
        is_dense_y1 = 0
    if number_y2 < 20 and number_y2 < judge_rate:
        is_dense_y2 = 0
    if number_z1 < 20 and number_z1 < judge_rate:
        is_dense_z1 = 0

    if is_dense_x1 == 0 and is_dense_x2 == 0:
        label_out[3] = standurd_height
    else:
        if is_dense_x1 == 0 and is_dense_x2 == 1:
            change_length = abs(label_out[3] - standurd_height)/2

            label_out[3] = standurd_height
            label_out[0] = label_out[0] - change_length*math.cos(label_out[6])
            label_out[1] = label_out[1] - change_length*math.sin(label_out[6])
        if is_dense_x1 == 1 and is_dense_x2 == 0:
            change_length = abs(label_out[3] - standurd_height)/2

            label_out[3] = standurd_height
            label_out[0] = label_out[0] - change_length*math.cos(label_out[6] + 3.14159)
            label_out[1] = label_out[1] - change_length*math.sin(label_out[6] + 3.14159)

    if is_dense_y1 == 0 and is_dense_y2 == 0:
        label_out[4] = standurd_width
    else:
        if is_dense_y1 == 0 and is_dense_y2 == 1:
            change_length = abs(label_out[4] - standurd_width)/2

            label_out[4] = standurd_width
            label_out[0] = label_out[0] + change_length*math.cos(label_out[6] + 3.14159/2)
            label_out[1] = label_out[1] + change_length*math.sin(label_out[6] + 3.14159/2)
        if is_dense_y1 == 1 and is_dense_y2 == 0:
            change_length = abs(label_out[4] - standurd_width)/2
 
            label_out[4] = standurd_width
            label_out[0] = label_out[0] + change_length*math.cos(label_out[6] + 3.14159 + 3.14159/2)
            label_out[1] = label_out[1] + change_length*math.sin(label_out[6] + 3.14159 + 3.14159/2)

        if is_dense_z1 == 0:
            change_length = abs(label_out[5] - standurd_zvalue)/2
            label_out[5] = standurd_zvalue
            label_out[2] = label_out[2] + change_length
    return label_out

def recover_size_expand(label, standurd_width, standurd_height, standurd_zvalue):
    label_out = label
    label_out[3] = standurd_width
    label_out[4] = standurd_height
    label_out[5] = standurd_zvalue
    return label_out

######################################################################

def run(cloud, labels, categray):
    result = []
    i = 0
    for labes in labels:
        cat = categray[i]
        i += 1
        if cat == "小车":
            conners = get_conners(labes)

            points_in = expand_box(cloud, conners, 0.)
            if len(points_in) < 200:
                label_out = recover_size(points_in, labes, conners, 2.1, 5, 1.7)
                result.append(label_out)

            else:
                labes_new = run_re_orientation(points_in, labes)
                conners_new = get_conners(labes_new)
                label_out = recover_size(points_in, labes_new, conners_new, 2.1, 5, 1.7)
                result.append(label_out)

        if cat == "大车":
            conners = get_conners(labes)

            points_in = expand_box(cloud, conners, 0.)
            if len(points_in) < 200:
                label_out = recover_size(points_in, labes, conners, 2.6, 8, 3)
                result.append(label_out)

            else:
                labes_new = run_re_orientation(points_in, labes)
                conners_new = get_conners(labes_new)
                label_out = recover_size(points_in, labes_new, conners_new, 2.6, 8, 3)
                result.append(label_out)
        
        if cat == "超大车":
            conners = get_conners(labes)

            points_in = expand_box(cloud, conners, 0.)
            if len(points_in) < 200:
                label_out = recover_size(points_in, labes, conners, 2.6, 18, 3.0)
                result.append(label_out)

            else:
                labes_new = run_re_orientation(points_in, labes)
                conners_new = get_conners(labes_new)
                label_out = recover_size(points_in, labes_new, conners_new, 2.6, 18, 3.0)
                result.append(label_out)
        
        if cat == "三轮车":
            result.append(recover_size_expand(labes, 1.5, 3.5, 1.7))
        if cat == "摩托车":
            result.append(recover_size_expand(labes, 0.8, 1.8, 1.6))
        if cat == "自行车":
            result.append(recover_size_expand(labes, 0.8, 1.8, 1.6))
        if cat == "行人":
            result.append(recover_size_expand(labes, 0.7, 0.7, 1.7))
        if cat == "锥形交通路标":
            result.append(recover_size_expand(labes, 0.4, 0.4, 0.8))
        if cat == "电动自行车":
            result.append(recover_size_expand(labes, 0.8, 1.8, 1.6))
    return result
    
######################################################################

if __name__ == "__main__":
    cloud = o3d.io.read_point_cloud("/home/yss/下载/pcds/L401_urbanroad_frame_569_1691053479968000_1691062830100000.pcd")
    points = np.array(cloud.points)   
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='绘制多边形')
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0]) 
    opt.point_size = 1 

    #points_pcd = o3d.geometry.PointCloud()
    #points_pcd.points = o3d.utility.Vector3dVector(polygon_points)
    #points_pcd.paint_uniform_color([0, 0.3, 0])
    vis.add_geometry(cloud)

    point_cloud_incar = o3d.geometry.PointCloud()
    points_in_all = []

    labels, categray = get_json_AI_result("/home/yss/下载/v1.8.3_3971/L401_urbanroad_frame_569_1691053479968000_1691062830100000.json")

    #lables_new = run(points, labels, categray)

    #labels = run_re_orientation(points, labels)
    i = 0
    for labes in labels:
        #print("lanels: ", labes)
        cat = categray[i]
        i += 1
        
        conners = get_conners(labes)
        #print("conners: ", conners)
        lines = draw_cube(conners, 1., 0., .0)
        arrow = draw_arrow(conners, 0, 1., 1)
        vis.add_geometry(lines)
        vis.add_geometry(arrow)

        if cat != 1:
            continue

        points_in = expand_box(points, conners, 0.)
        if len(points_in) < 200:
            label_out = recover_size(points_in, labes, conners, 2.1, 5, 1.7)
            conners_out = get_conners(label_out)
            lines_out = draw_cube(conners_out, 0., 1., 0.)
            arrow_out = draw_arrow(conners_out, 1., 1., 0.)
            vis.add_geometry(lines_out)
            vis.add_geometry(arrow_out)
        else:
            labes_new = run_re_orientation(points_in, labes)
            conners_new = get_conners(labes_new)
            label_out = recover_size(points_in, labes_new, conners_new, 2.1, 5, 1.7)
            conners_out = get_conners(label_out)
            lines_out = draw_cube(conners_out, 0., 1., 0.)
            arrow_out = draw_arrow(conners_out, 1., 1., 0.)
            vis.add_geometry(lines_out)
            vis.add_geometry(arrow_out)

    vis.run()
    #o3d.visualization.draw_geometries([cloud])