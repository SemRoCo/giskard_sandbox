#!/usr/bin/env python

import rospy
from giskard_msgs.srv import *
from giskard_msgs.msg import *

if __name__ == '__main__':
    rospy.init_node('dummy_db_query')
    rospy.loginfo('waiting for service')
    rospy.wait_for_service('/task_database/query_motion')
    rospy.loginfo('calling service')
    query_motion = rospy.ServiceProxy('/task_database/query_motion', QueryMotion)
    cup = TaskObject(name='cup', role=giskard_msgs.msg.TaskObject.TARGET_ROLE)
    bottle = TaskObject(name='bottle', role=giskard_msgs.msg.TaskObject.SOURCE_ROLE)
    req = giskard_msgs.srv.QueryMotionRequest()
    req.task.type = 'pouring'
    req.task.objects.append(cup)
    req.task.objects.append(bottle)
    resp = query_motion(req)
    rospy.loginfo(resp)
