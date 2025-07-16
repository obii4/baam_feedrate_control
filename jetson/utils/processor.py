import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def offset_line(p1, p2, N):
    """Return endpoints of a line parallel to p1-p2, offset by N pixels."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy)
    if L == 0:
        return p1, p2
    # unit normal
    nx, ny = -dy/L, dx/L
    return (int(x1 + N*nx), int(y1 + N*ny)), (int(x2 + N*nx), int(y2 + N*ny))

def mean_between_lines(thermal: np.ndarray, p1: tuple, p2: tuple, p1_off: tuple, p2_off: tuple) -> float:
    """
    Compute the mean of thermal pixel values between two parallel line segments.
    
    Parameters:
      thermal : 2D array of thermal values
      p1, p2  : endpoints of the first line (x1,y1), (x2,y2)
      p1_off, p2_off : endpoints of the offset line parallel to the first
    
    Returns:
      mean_val : mean of thermal values between the two lines
    """
    # Create a mask polygon connecting the four endpoints
    poly = np.array([p1, p2, p2_off, p1_off], dtype=np.int32)
    mask = np.zeros_like(thermal, dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)  # fill polygon with 1s
    
    # Extract and compute mean
    values = thermal[mask == 1]
    if values.size == 0:
        return float('nan')

    return float(values.mean())


def cluster_by_angle(segments, min_deg=1, max_deg=5):
    """
    Cluster line segments by orientation, but only group segments whose
    angular difference to the cluster mean is > min_deg and < max_deg.
    
    segments: iterable of (x1,y1,x2,y2)
    min_deg: lower bound on abs(angle difference) in degrees
    max_deg: upper bound on abs(angle difference) in degrees
    """
    tol_min = np.deg2rad(min_deg)
    tol_max = np.deg2rad(max_deg)
    clusters = []
    
    for x1, y1, x2, y2 in segments:
        ang = np.arctan2(y2 - y1, x2 - x1) % np.pi
        placed = False
        
        for cl in clusters:
            diff = abs(ang - cl['angle_mean'])
            if tol_min < diff < tol_max:
                cl['segments'].append((x1, y1, x2, y2))
                cl['angles'].append(ang)
                cl['angle_mean'] = np.mean(cl['angles'])
                placed = True
                break
        
        if not placed:
            clusters.append({
                'angle_mean': ang,
                'angles':    [ang],
                'segments':  [(x1, y1, x2, y2)]
            })
    
    return clusters

def sort_line_cluster(clusters, type = 'min'):
    cmids = [np.mean([(s[0]+s[2])/2 for s in cl['segments']]) for cl in clusters]

    if type != 'max':
        idx   = int(np.argmax(cmids))

    idx   = int(np.argmin(cmids))
    rail  = clusters[idx]['segments']
    x1, y1, x2, y2 = max(rail, key=lambda s: max(s[1], s[3]))

    return x1, y1, x2, y2



class HoughBundler:    

    '''
    https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp
    
    ''' 
    def __init__(self,min_distance=5,min_angle=2):
        self.min_distance = min_distance
        self.min_angle = min_angle
    
    def get_orientation(self, line):
        orientation = math.atan2(abs((line[3] - line[1])), abs((line[2] - line[0])))
        return math.degrees(orientation)

    def check_is_line_different(self, line_1, groups, min_distance_to_merge, min_angle_to_merge):
        for group in groups:
            for line_2 in group:
                if self.get_distance(line_2, line_1) < min_distance_to_merge:
                    orientation_1 = self.get_orientation(line_1)
                    orientation_2 = self.get_orientation(line_2)
                    if abs(orientation_1 - orientation_2) < min_angle_to_merge:
                        group.append(line_1)
                        return False
        return True

    def distance_point_to_line(self, point, line):
        px, py = point
        x1, y1, x2, y2 = line

        def line_magnitude(x1, y1, x2, y2):
            line_magnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return line_magnitude

        lmag = line_magnitude(x1, y1, x2, y2)
        if lmag < 0.00000001:
            distance_point_to_line = 9999
            return distance_point_to_line

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (lmag * lmag)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = line_magnitude(px, py, x1, y1)
            iy = line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance_point_to_line = iy
            else:
                distance_point_to_line = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance_point_to_line = line_magnitude(px, py, ix, iy)

        return distance_point_to_line

    def get_distance(self, a_line, b_line):
        dist1 = self.distance_point_to_line(a_line[:2], b_line)
        dist2 = self.distance_point_to_line(a_line[2:], b_line)
        dist3 = self.distance_point_to_line(b_line[:2], a_line)
        dist4 = self.distance_point_to_line(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_into_groups(self, lines):
        groups = []  # all lines groups are here
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.check_is_line_different(line_new, groups, self.min_distance, self.min_angle):
                groups.append([line_new])

        return groups

    def merge_line_segments(self, lines):
        orientation = self.get_orientation(lines[0])
      
        if(len(lines) == 1):
            return np.block([[lines[0][:2], lines[0][2:]]])

        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        if 45 < orientation <= 90:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        return np.block([[points[0],points[-1]]])

    def process_lines(self, lines):
        lines_horizontal  = []
        lines_vertical  = []
  
        for line_i in [l[0] for l in lines]:
            orientation = self.get_orientation(line_i)
            # if vertical
            if 45 < orientation <= 90:
                lines_vertical.append(line_i)
            else:
                lines_horizontal.append(line_i)

        lines_vertical  = sorted(lines_vertical , key=lambda line: line[1])
        lines_horizontal  = sorted(lines_horizontal , key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_horizontal, lines_vertical]:
            if len(i) > 0:
                groups = self.merge_lines_into_groups(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(self.merge_line_segments(group))
                merged_lines_all.extend(merged_lines)
                    
        return np.asarray(merged_lines_all)