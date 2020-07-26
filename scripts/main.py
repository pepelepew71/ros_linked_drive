#! /usr/bin/env python

import threading

import numpy as np

import rospy
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import tf
import tf2_ros

from _marker import MarkerArrPublisher


class LinkedDrive:
    """
    Attributes:
        weight_dist (float): weight coeff. for distance in optimization obj
        dt (float):
        L (float): distance between two solamr when connected with shelft
        dist_error (float):
        angle_resolution (float):
        rot_vel_max (float):
        lin_vel_max (float):
        rot_acc (float):
        lin_acc (float):
    """
    def __init__(self):
        self.weight_dist = 1.0
        self.dt = 0.3  # sec
        self.L = 1.0  # m
        self.dist_error = 0.05
        self.angle_resolution = 0.01
        self.rot_vel_max = 3.5718  # rad/s
        self.lin_vel_max = 5.0  # m/s
        self.rot_acc = 3.0  # rad/s^2
        self.lin_acc = 3.0  # m/s^2

    def get_predict_pose(self, pose, vel, th0):
        '''
        Prediction of trajectory using arc (combination of v and w).

        Args:
            pose (list): x, y in map coordinate
            vel (list): v, w in car coordinate
            th0 (float): theta in map coordinate
        Return:
            (tuple): x, y, theta in map coordinate
        '''
        x0 = pose[0]
        y0 = pose[1]
        v = vel[0]
        w = vel[1]

        # -- get dx, dy in car coordidate
        if abs(w) < 1e-5:
            dx = v * self.dt
            dy = 0.0
            dth = 0.0
        else:
            r = v / w
            dth = w * self.dt
            dx = r * np.sin(dth)
            dy = r * (1.0 - np.cos(dth))

        # -- get predicted x, y in map coordinate
        x1 = x0 + dx*np.cos(th0) - dy*np.sin(th0)
        y1 = y0 + dx*np.sin(th0) + dy*np.cos(th0)
        th1 = th0 + dth

        return x1, y1, th1

    def get_potential_poses(self):
        '''
        Get potential locations for follower to be at.

        Return:
            (list):
        '''
        x, y, _ = self.get_predict_pose(FRONT_POSE, FRONT_VEL, FRONT_TH)

        angles = np.arange(0, 2.0*np.pi, self.angle_resolution)  # rad, potential poses every _ rad
        xs = np.expand_dims(x + self.L*np.cos(angles), axis=1)  # (n, 1)
        ys = np.expand_dims(y + self.L*np.sin(angles), axis=1)  # (n, 1)
        poses = np.append(arr=xs, values=ys, axis=1)  # (n, 2)

        list_poses = poses.tolist()
        PUB_MARKER_ARRAY.pub_from_locs(locs=list_poses)

        return list_poses

    def vels_from_poses(self, poses):
        '''
        Get lin/ang velocities from given pose.

        Args:
            poses (list):
        Return:
            (dict): {pose: cmd_vel, ...}
        '''
        dict_reachable = dict()
        mat_rot_inv = np.array(
            [[ np.cos(CUR_TH), np.sin(CUR_TH)],
             [-np.sin(CUR_TH), np.cos(CUR_TH)]]
        )

        for target in poses:
            dx = target[0] - CUR_POSE[0]
            dy = target[1] - CUR_POSE[1]
            [[dx], [dy]] = np.dot(mat_rot_inv, [[dx], [dy]])  # dx, dy in car coordinate

            if abs(dy) < 1e-5:  # no rotation
                w = 0.0
                v = dx / self.dt
            else:
                r = (dx**2 + dy**2) / (2.0*dy)
                w = np.arcsin(dx / r) / self.dt
                v = r * w

            x, y, _ = self.get_predict_pose(CUR_POSE, (v, w), CUR_TH)  # x, y predicted location in map coordinate

            is_cmd_vel_in_range = self.check_vels_range((v, w))
            is_closed_target_and_xy = get_dist_from_two_poses((x, y), target) < 0.01

            if is_cmd_vel_in_range and is_closed_target_and_xy:
                dict_reachable[tuple(target)] = (v, w)

        return dict_reachable

    def check_vels_range(self, vels):
        '''
        Check if the (v, w) is within the bounds.

        Args:
            vels (tuple): v, w
        Return:
            (bool):
        '''
        v1, w1 = vels
        v0, w0 = CUR_VEL
        av, aw = (np.array(vels) - np.array(CUR_VEL)) / self.dt

        if (abs(v1) > self.lin_vel_max or
            abs(w1) > self.rot_vel_max or
            abs(av) > self.lin_acc or
            abs(aw) > self.rot_acc):
            return False
        else:
            return True

    def get_diff_theta(self, goal):
        '''
        Args:
            goal (list):
        Return:
            (float)
        '''
        x_diff = goal[0] - CUR_POSE[0]
        y_diff = goal[1] - CUR_POSE[1]
        theta_goal = np.arctan2(y_diff, x_diff)
        return theta_goal - CUR_TH

    def get_angular_vel(self, point, rate_ang=0.5, backward=True):
        '''
        Get angluar velocity for facing toward to the front car.

        Args:
            point (list):
            rate_ang (float):
            backward (bool):
        Return:
            (float):
        '''
        theta_tol = 0.0175  # in radian ~= 2 deg
        max_omega = self.rot_vel_max
        min_omega = 0.1
        theta_diff = self.get_diff_theta(point)

        # -- TEST: see if this can follow better
        # theta_diff -= np.pi/8
        ang_temp = 1e-10

        # -- prevent oscilliation
        if abs(theta_diff) < theta_tol*2:
            rate_ang *= 0.3
        elif abs(theta_diff) < theta_tol*11:
            rate_ang*=0.5
        else:
            pass

        # -- turn CW or CCW
        if theta_diff > 0:
            if theta_diff > np.pi:
                ang_temp =  - rate_ang * np.exp(2*np.pi - theta_diff)
            else :
                ang_temp =  rate_ang * np.exp(theta_diff)

        if theta_diff < 0:
            if abs(theta_diff) > np.pi:
                ang_temp = rate_ang * np.exp(2*np.pi + theta_diff)
            else :
                ang_temp = - rate_ang * np.exp(- theta_diff)

        if abs(ang_temp) >= max_omega:
            ang_temp = max_omega * abs(ang_temp)/ang_temp
        # elif abs(ang_temp) <= min_omega:
        #     ang_temp = min_omega * abs(ang_temp)/ang_temp
        # else:
        #     pass

        return ang_temp

    def check_face_same_dir(self, goal):
        '''
        Decide to drive forward or backward.

        Args:
            goal (list):
        Return:
            (bool):
        '''
        if abs(self.get_diff_theta(goal)) < np.pi/2.0 or abs(self.get_diff_theta(goal)) > np.pi*3.0/2.0 :
            return True  # same dir, drive forward
        else:
            return False  # opposite dir, drive reverse

    def get_linear_vel(self, goal, rate_lin=0.5):
        '''
        Args:
            goal (list):
            rate_lin (float):
        Return:
            (float):
        '''
        dist = get_dist_from_two_poses(CUR_POSE, goal)

        if self.check_face_same_dir(goal) :
            vel_temp = rate_lin * np.log(dist+1)
        else:
            vel_temp = - rate_lin * np.log(dist+1)
        # -- MIN and MAX
        if abs(vel_temp) >= self.lin_vel_max:
            vel_temp = self.lin_vel_max * abs(vel_temp)/vel_temp

        return vel_temp

    def get_rated_dist(self, target_pose):
        '''
        get the rate of distance between front and follower (closer the lower).

        Args:
            target_pose (list):
        Return:
            (float):
        '''
        follower_pose = CUR_POSE
        return self.weight_dist * get_dist_from_two_poses(follower_pose, target_pose)

    def get_rated_ori(self, target_pose, mode="shelft"):
        '''
        get the rate of orientation, standatd: 1.shelft orientation, 2.front car orientation.

        Args:
            target_pose (?):
            mode (str):
        Return:
            (float)
        '''
        # -- 1.rate by shelft orientation

        # -- 2.rate by front car orientation
        pass

    def get_opt_rear_vels(self):
        '''
        Get optimized linear and rotation velocity from reachable poses.

        Return:
            (tuple):
        '''
        potential_poses = self.get_potential_poses()
        dict_reachable = self.vels_from_poses(potential_poses)
        reachable_poses = dict_reachable.keys()
        print(dict_reachable)

        if len(reachable_poses) > 0:
            dict_cost = dict()
            # -- optimization according to : dist to target, face same direction as front
            for p, v in dict_reachable.items():
                # -- 1. dist to target
                dict_cost[str(v)] = self.get_rated_dist(p)

            vels, cost = sorted(dict_cost.items(), key=lambda item: item[1], reverse=False)[0]  # select min. cost
            return eval(vels)  # convert str to tuple

        else:
            # -- facing toward front car
            ang_vel = self.get_angular_vel(FRONT_POSE)

            # -- go straight to 1m away from front vehicle if there is no available vels
            lin_vel = self.get_linear_vel(FRONT_POSE)

            dist = get_dist_from_two_poses(FRONT_POSE, CUR_POSE)

            if abs(dist - self.L) < self.dist_error:
                lin_vel = 0.0
            elif dist < self.L:
                lin_vel = -lin_vel
            else:
                pass

            return (lin_vel, ang_vel)

    def start(self):
        '''
        Start linked drive main loop.
        It will keep publish cmd_vel to PUB_R_VEL.
        '''
        rate = rospy.Rate(hz=10)
        try:
            while not rospy.is_shutdown():
                ros_t0 = rospy.get_time()

                # -- update follower pose
                rear_vels = self.get_opt_rear_vels()

                _twist = Twist()
                _twist.linear.x = rear_vels[0]
                _twist.angular.z = rear_vels[1]

                PUB_R_VEL.publish(_twist)

                ros_td = rospy.get_time() - ros_t0
                if ros_td > 0 :
                    print("pub Hz = {0}".format(1.0/ros_td))

                rate.sleep()

        except rospy.ROSInterruptException:
            pass


class TfListener:
    """
    Use threading to get listener tf map1 -> car1 and map2 -> car2,
    and save them to global vars.

    Args:
        frames (dict):
    Attributes:
        frames (dict):
        _tf_buffer (tf2_ros.Buffer):
    """
    def __init__(self, frames):
        self.frames = frames
        self._tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self._tf_buffer)  # for buffer

    def _update_front_car(self):
        """
        Update car1 (front) pose and orientation.
        """
        try:
            t = self._tf_buffer.lookup_transform(
                target_frame=frames["map1_frame_id"],
                source_frame=frames["car1_frame_id"],
                time=rospy.Time())
        except Exception as err:
            rospy.loginfo(err)
        else:
            global FRONT_POSE, FRONT_ORI, FRONT_TH
            FRONT_POSE[0] = t.transform.translation.x
            FRONT_POSE[1] = t.transform.translation.y
            FRONT_ORI[0] = t.transform.rotation.x
            FRONT_ORI[1] = t.transform.rotation.y
            FRONT_ORI[2] = t.transform.rotation.z
            FRONT_ORI[3] = t.transform.rotation.w
            _, _, FRONT_TH = tf.transformations.euler_from_quaternion(FRONT_ORI)

    def _update_rear_car(self):
        """
        Update car2 (rear) pose and orientation.
        """
        try:
            t = self._tf_buffer.lookup_transform(
                target_frame=frames["map2_frame_id"],
                source_frame=frames["car2_frame_id"],
                time=rospy.Time())
        except Exception as err:
            rospy.loginfo(err)
        else:
            global CUR_POSE, CUR_ORI, CUR_TH
            CUR_POSE[0] = t.transform.translation.x
            CUR_POSE[1] = t.transform.translation.y
            CUR_ORI[0] = t.transform.rotation.x
            CUR_ORI[1] = t.transform.rotation.y
            CUR_ORI[2] = t.transform.rotation.z
            CUR_ORI[3] = t.transform.rotation.w
            _, _, CUR_TH = tf.transformations.euler_from_quaternion(CUR_ORI)

    def _job(self):
        '''
        Update pose and orientation of car1 and car2 in loop.
        '''
        rate = rospy.Rate(hz=10.0)
        while not rospy.is_shutdown():
            self._update_front_car()
            self._update_rear_car()
            rate.sleep()

    def start_thread(self):
        '''
        Start threading
        '''
        thread = threading.Thread(target=self._job, name='job')
        thread.start()


# --

def get_dist_from_two_poses(pose1, pose2):
    '''
    Get distance between two pose.

    Args:
        pose1 (list):
        pose2 (list):
    Return:
        (float):
    '''
    return np.sqrt(sum((np.array(pose1) - np.array(pose2))**2))

def _cb_car1_odom(data):
    """
    Callback for rospy.Subscriber car1 odom.
    """
    global FRONT_VEL
    FRONT_VEL[0] = data.twist.twist.linear.x
    FRONT_VEL[1] = data.twist.twist.angular.z

def _cb_car2_odom(data):
    """
    Callback for rospy.Subscriber car2 odom.
    """
    global CUR_VEL
    CUR_VEL[0] = data.twist.twist.linear.x
    CUR_VEL[1] = data.twist.twist.angular.z

if __name__ == '__main__':


    # -- global vars
    ## -- FRONT: car leader
    FRONT_POSE = [0.0, 0.0]  # [x, y]
    FRONT_ORI = [0.0, 0.0, 0.0, 0.0] # [x, y, z, w]
    FRONT_TH = 0.0
    FRONT_VEL = [0.0, 0.0]  # [v_linear, w_angular]

    ## -- CUR: car follower
    CUR_POSE = [0.0, 0.0]  # [x, y]
    CUR_ORI = [0.0, 0.0, 0.0, 0.0]  # [x, y, z, w]
    CUR_TH = 0.0
    CUR_VEL = [0.0, 0.0]  # [v_linear, w_angular]

    # -- ros node function
    ## -- parameters
    rospy.init_node('linked_drive')

    map1_frame_id = rospy.get_param(param_name="~map1_frame_id", default="map")
    map2_frame_id = rospy.get_param(param_name="~map2_frame_id", default="map")
    car1_frame_id = rospy.get_param(param_name="~car1_frame_id", default="solamr_1/base_footprint")
    car2_frame_id = rospy.get_param(param_name="~car2_frame_id", default="solamr_2/base_footprint")

    car1_odom = rospy.get_param(param_name="~car1_odom", default="solamr_1/odom")
    car2_odom = rospy.get_param(param_name="~car2_odom", default="solamr_2/odom")

    car1_cmd_vel = rospy.get_param(param_name="~car1_cmd_vel", default="solamr_1/cmd_vel")
    car2_cmd_vel = rospy.get_param(param_name="~car2_cmd_vel", default="solamr_2/cmd_vel")

    ## -- pub and sub
    rospy.Subscriber(name=car1_odom, data_class=Odometry, callback=_cb_car1_odom)  # for FRONT_VEL
    rospy.Subscriber(name=car2_odom, data_class=Odometry, callback=_cb_car2_odom)  # for CUR_VEL
    PUB_R_VEL = rospy.Publisher(name=car2_cmd_vel, data_class=Twist, queue_size=10)
    PUB_MARKER_ARRAY = MarkerArrPublisher(frame_id=map1_frame_id)

    ## -- tf listener for POSE, ORI and TH
    frames = {
        "map1_frame_id": map1_frame_id,
        "map2_frame_id": map2_frame_id,
        "car1_frame_id": car1_frame_id,
        "car2_frame_id": car2_frame_id,
    }
    tf_listener = TfListener(frames=frames)
    tf_listener.start_thread()

    # -- linked drive planner
    linked_drive = LinkedDrive()
    linked_drive.start()
