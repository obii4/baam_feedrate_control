"""
processor.py - Computer vision processing utilities for thermal image analysis.

This module provides line detection, clustering, and temperature extraction
algorithms for analyzing thermal images of 3D printed layers. It implements
Hough transform-based line detection with custom clustering to identify
printed beads and extract temperature measurements.

Core Functionality:
    - Line segment clustering by orientation
    - Parallel line offset calculation
    - Temperature sampling between line boundaries
    - Hough line merging and grouping

Author: Chris O'Brien
Date Created: 09-23-25
Version: 1.0
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def offset_line(p1, p2, N):
    """
    Calculate endpoints of a line parallel to p1-p2, offset by N pixels.
    
    Creates a parallel line at a perpendicular distance of N pixels from
    the original line. Used to define sampling regions for temperature
    extraction.
    
    Args:
        p1: Starting point (x1, y1) of original line
        p2: Ending point (x2, y2) of original line  
        N: Perpendicular offset distance in pixels (positive or negative)
        
    Returns:
        tuple: ((x1_off, y1_off), (x2_off, y2_off)) offset line endpoints
        
    Algorithm:
        1. Calculate line direction vector (dx, dy)
        2. Compute perpendicular unit normal (-dy/L, dx/L)
        3. Offset points by N * normal vector
        
    Example:
        >>> # Create parallel line 5 pixels to the right
        >>> p1_off, p2_off = offset_line((10, 20), (10, 50), 5)
        
    Note:
        Positive N offsets to the right when traversing from p1 to p2.
        Returns original points if line has zero length.
    """

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
    Calculate mean temperature in region between two parallel line segments.
    
    Defines a quadrilateral region bounded by two parallel lines and
    computes the mean of all thermal values within that region. Used
    to extract layer temperature from specific bead regions.
    
    Args:
        thermal: 2D array of thermal values (temperature in Celsius)
        p1, p2: Endpoints of first line segment (x1,y1), (x2,y2)
        p1_off, p2_off: Endpoints of parallel offset line
        
    Returns:
        float: Mean temperature in the region (NaN if empty)
        
    Implementation:
        Creates polygon mask using cv2.fillPoly()
        Extracts values where mask == 1
        Returns mean of extracted values
        

    The mean_between_lines function defines a quadrilateral sampling region:

        p1 ─────────── p2      <- Original line (detected edge)
        │              │
        │   SAMPLING   │       <- Temperature pixels averaged here  
        │    REGION    │
        │              │
        p1_off ──────── p2_off <- Offset line (parallel)

    Width of region (offset N) determines spatial averaging:
        - Small N (1-3 pixels): Precise but noisy
        - Large N (5-10 pixels): Smooth but may include background

        
    Note:
        Polygon vertices must be in correct order to avoid self-intersection.
        Returns NaN for empty regions (no pixels in mask).
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
    Group line segments by similar orientation within angle tolerance.
    
    Clusters line segments whose angles differ by more than min_deg but
    less than max_deg from the cluster mean. This unusual constraint helps
    separate nearly-parallel lines that represent different print layers.
    
    Args:
        segments: Iterable of line segments as (x1, y1, x2, y2) tuples
        min_deg: Minimum angle difference to cluster (degrees)
        max_deg: Maximum angle difference to cluster (degrees)
        
    Returns:
        list: Clusters, each containing:
            - 'angle_mean': Mean orientation of cluster (radians)
            - 'angles': List of individual segment angles
            - 'segments': List of (x1, y1, x2, y2) tuples
            
    Algorithm:
        1. Calculate segment angle using arctan2 (modulo π for undirected)
        2. For each segment, find cluster with angle diff in (min_deg, max_deg)
        3. Update cluster mean dynamically as segments are added
        4. Create new cluster if no match found
        
    Example:
        >>> segments = [(10,10,50,12), (15,20,55,22), (10,40,50,80)]
        >>> clusters = cluster_by_angle(segments, min_deg=1, max_deg=5)
        >>> # Returns clusters of nearly-horizontal and vertical lines
        
    Note:
        Angles are undirected (0 to π range) since line direction is arbitrary.
        Dynamic mean update allows clusters to drift slightly.
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

    """
    Extract the most significant line from clustered segments.
    
    Selects a representative line from clusters based on position and extent.
    Typically used to find the primary layer boundary from multiple detections.
    
    Args:
        clusters: List of clusters from cluster_by_angle()
        type: Selection criteria ('min' for leftmost, 'max' for rightmost)
        
    Returns:
        tuple: (x1, y1, x2, y2) endpoints of selected line segment
        
    Selection Process:
        1. Calculate mean x-coordinate for each cluster
        2. Select cluster based on type (leftmost/rightmost)
        3. Within cluster, choose segment with maximum y-extent
    """

    cmids = [np.mean([(s[0]+s[2])/2 for s in cl['segments']]) for cl in clusters]

    if type != 'max':
        idx   = int(np.argmax(cmids))

    idx   = int(np.argmin(cmids))
    rail  = clusters[idx]['segments']
    x1, y1, x2, y2 = max(rail, key=lambda s: max(s[1], s[3]))

    return x1, y1, x2, y2



class HoughBundler:    

    """
    Merge and group Hough-detected line segments by proximity and orientation.
    
    Implements line segment clustering algorithm to merge fragmented Hough
    transform detections into coherent line groups. Useful for combining
    multiple short segments that represent a single physical edge.
    
    Based on: https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp
    
    Attributes:
        min_distance: Maximum distance for segments to be considered same line (pixels)
        min_angle: Maximum angle difference for merging (degrees)
        
    Algorithm Overview:
        1. Group segments by proximity and orientation similarity
        2. Merge segments within each group into single line
        3. Return one representative line per group
        
    Example:
        >>> bundler = HoughBundler(min_distance=5, min_angle=2)
        >>> lines = cv2.HoughLinesP(edges, ...)
        >>> merged = bundler.process_lines(lines)
        >>> # Returns consolidated line segments
    """

    def __init__(self,min_distance=5,min_angle=2):
        self.min_distance = min_distance
        self.min_angle = min_angle
    
    def get_orientation(self, line):
        """
        Calculate line orientation in degrees (0-90 range).
        
        Args:
            line: Line segment as (x1, y1, x2, y2)
            
        Returns:
            float: Orientation angle in degrees
            
        Note:
            Uses absolute differences for undirected angle.
            Range is 0-90 degrees (first quadrant only).
        """

        orientation = math.atan2(abs((line[3] - line[1])), abs((line[2] - line[0])))
        return math.degrees(orientation)

    def check_is_line_different(self, line_1, groups, min_distance_to_merge, min_angle_to_merge):

        """
        Check if line should join existing group or create new one.
        
        Tests line against all existing groups. If close enough in both
        distance and angle to any group member, adds to that group.
        
        Args:
            line_1: Line segment to test (x1, y1, x2, y2)
            groups: List of existing line groups
            min_distance_to_merge: Distance threshold (pixels)
            min_angle_to_merge: Angle threshold (degrees)
            
        Returns:
            bool: True if line is different (should create new group)
            
        Side Effects:
            Appends line_1 to matching group if found
        """

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

        """
        Calculate perpendicular distance from point to line segment.
        
        Computes minimum distance from point to line segment, considering
        whether perpendicular projection falls within segment bounds.
        
        Args:
            point: Point coordinates (px, py)
            line: Line segment (x1, y1, x2, y2)
            
        Returns:
            float: Minimum distance in pixels
            
        Algorithm:
            1. Project point onto infinite line
            2. If projection is within segment, use perpendicular distance
            3. Otherwise, use distance to nearest endpoint
            
        Note:
            Returns 9999 for degenerate (zero-length) lines.
        """

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

        """
        Calculate minimum distance between two line segments.
        
        Tests all four point-to-line distances and returns minimum.
        Symmetric measure of line segment proximity.
        
        Args:
            a_line: First line segment (x1, y1, x2, y2)
            b_line: Second line segment (x1, y1, x2, y2)
            
        Returns:
            float: Minimum distance between segments (pixels)
        """

        dist1 = self.distance_point_to_line(a_line[:2], b_line)
        dist2 = self.distance_point_to_line(a_line[2:], b_line)
        dist3 = self.distance_point_to_line(b_line[:2], a_line)
        dist4 = self.distance_point_to_line(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_into_groups(self, lines):

        """
        Merge grouped line segments into single representative line.
        
        Combines multiple segments by finding extreme points along
        primary axis (x for horizontal, y for vertical lines).
        
        Args:
            lines: List of line segments in same group
            
        Returns:
            np.ndarray: Single merged line segment [[x1,y1,x2,y2]]
            
        Strategy:
            - Extract all endpoints
            - Sort by primary axis based on orientation
            - Connect extreme points
        """

        groups = []  # all lines groups are here
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.check_is_line_different(line_new, groups, self.min_distance, self.min_angle):
                groups.append([line_new])

        return groups

    def merge_line_segments(self, lines):

        """
        Main entry point for line merging pipeline.
        
        Separates lines into horizontal and vertical groups, merges
        similar segments, and returns consolidated line set.
        
        Args:
            lines: Raw Hough line output from cv2.HoughLinesP()
            
        Returns:
            np.ndarray: Merged line segments
            
        Process:
            1. Classify lines as horizontal (0-45°) or vertical (45-90°)
            2. Sort each category by position
            3. Group similar lines
            4. Merge groups into single lines
            5. Return all merged lines
            
        Note:
            45° threshold for horizontal/vertical classification.
            Sorting improves grouping efficiency.
        """

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