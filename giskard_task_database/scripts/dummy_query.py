#!/usr/bin/env python

# Copyright (c) 2016, Georg Bartels (georg.bartels@cs.uni-bremen.de)
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Institute for Artificial Intelligence nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
    cup.pose.header.frame_id = "base_link"
    cup.pose.pose.position = geometry_msgs.msg.Point(0.1, 0.2, 0.3)
    bottle = TaskObject(name='bottle', role=giskard_msgs.msg.TaskObject.SOURCE_ROLE)
    bottle.pose.header.frame_id = "base_link"
    bottle.pose.pose.position = geometry_msgs.msg.Point(1.0, 1.0, 1.0)
    req = giskard_msgs.srv.QueryMotionRequest()
    req.task.type = 'pouring'
    req.task.objects.append(cup)
    req.task.objects.append(bottle)
    resp = query_motion(req)
    rospy.loginfo(resp)
