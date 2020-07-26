#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

class MarkerArrPublisher:
    """
    Publish all linked drive predicted pose locations
    """
    def __init__(self, frame_id):
        self.pub = rospy.Publisher(name="visualization_marker_array", data_class=MarkerArray, queue_size=1)
        self.frame_id = frame_id

    def _get_marker(self):
        """
        Subroutine for self.pub_from_locs
        """
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "/"
        marker.id = 0
        marker.action = marker.ADD
        marker.type = marker.POINTS
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02
        marker.scale.y = 0.02
        marker.color.g = 1.0
        marker.color.a = 1.0
        return marker

    def pub_from_locs(self, locs):
        """
        Use locs to create marker.points, then publish to visualization_marker_array topic.

        Args:
            locs (list of tuple): [(x, y), ...]
        """
        marker_array = MarkerArray()
        marker = self._get_marker()
        for loc in locs:
            p = Point()
            p.x = loc[0]
            p.y = loc[1]
            p.z = 0.0
            marker.points.append(p)
        marker_array.markers.append(marker)
        self.pub.publish(marker_array)
