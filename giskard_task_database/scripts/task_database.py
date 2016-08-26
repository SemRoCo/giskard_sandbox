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

import yaml
import rospy
import rospkg
import cPickle as pickle
import PyKDL as kdl
from giskard_msgs.srv import *
from giskard_msgs.msg import *

dummy_mode = False
dummy_db = {}
learned_db = {}
name_dict = {
        'moveabovepan': 'move_above',
        'tiltbottle': 'tilt_down',
        'tiltback' : 'tilt_back',
        'mug-top-behind-maker': 'source_top_behind_goal_target_top',
        'mug-top-left-maker': 'source_top_left_goal_target_top',
        'mug-top-above-maker': 'source_top_above_goal_target_top',
        'mug-behind-itself': 'source_behind_itself',
        'mug-left-itself': 'source_left_itself',
        'mug-above-itself': 'source_above_itself'}

def dummy_learned_phases():
    result = {}
    phase_names = ["moveabovepan", "tiltbottle", "tiltback"]
    constraint_names = ["mug-top-behind-maker", "mug-top-left-maker", "mug-top-above-maker", "mug-behind-itself", "mug-left-itself", "mug-above-itself"]
    for phase_name in phase_names:
        result[phase_name] = {}
        counter = 0
        for constraint_name in constraint_names:
            counter += 0.1
            result[phase_name][constraint_name] = [-counter, counter]
    return result

def poseStamped_to_kdlFrame(ps):
    frame = kdl.Frame()
    frame.p = kdl.Vector(ps.pose.position.x, ps.pose.position.y, ps.pose.position.z)
    frame.M = kdl.Rotation.Quaternion(ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w)
             
    return([ps.header.frame_id, frame])

def get_ros_package_path():
    r = rospkg.RosPack()
    return r.get_path('giskard_task_database')

def read_yaml_file(postfix):
    filename = get_ros_package_path() + postfix
    with open(filename, 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def read_pickle_file(postfix):
    filename = get_ros_package_path() + postfix
    with open(filename, 'rb') as infile:
        amodel = pickle.load(infile)
        return amodel

def unpickle_file(postfix):
    rospy.loginfo("opening file '%s'", postfix)

def read_dummy_db():
    global dummy_db
    dummy_db = read_yaml_file("/config/dummy_db.yaml")

def read_learned_db():
    learned_db_config = read_yaml_file("/config/learned_db.yaml")
    global learned_db
    for task_name in learned_db_config:
        learned_db[task_name] = {}
        for source_name in learned_db_config[task_name]:
            learned_db[task_name][source_name] = {}
            for target_name in learned_db_config[task_name][source_name]:
                pickle_filename = learned_db_config[task_name][source_name][target_name]
                learned_db[task_name][source_name][target_name] = unpickle_file(pickle_filename);

def get_object_by_role(objs, role):
    for obj in objs:
        if obj.role == role:
            return obj

def get_source_object(objs):
    return get_object_by_role(objs, giskard_msgs.msg.TaskObject.SOURCE_ROLE)

def get_target_object(objs):
    return get_object_by_role(objs, giskard_msgs.msg.TaskObject.TARGET_ROLE)

def query_amodel(amodel, rel_pose, pour_vol, fill_vol):
    rospy.loginfo("rel_pose: %s, pour_vol: %s, fill_vol: %s", rel_pose, pour_vol, fill_vol)
    # TODO: Yuen please implement me, and replace the return statement below
    return dummy_learned_phases()

def dummy_phases_to_ros(phases):
    result = []
    for phase in phases:
        mp = MotionPhase(name=phase['name'])
        for constraint in phase['constraints']:
            c = Constraint(name=constraint['name'], lower=constraint['lower'], upper=constraint['upper'])
            mp.constraints.append(c)
        result.append(mp)
    return result

def learned_phases_to_ros(phases):
    result = []
    for phase in phases:
        mp = MotionPhase()
        mp.name = name_dict[phase]
        for constraint in phases[phase]:
            c = Constraint()
            c.name = name_dict[constraint]
            c.lower = phases[phase][constraint][0]
            c.upper = phases[phase][constraint][1]
            mp.constraints.append(c)
        result.append(mp)
    return result

def calc_rel_pose(source_pose_msg, target_pose_msg):
    (source_frame_id, source_frame) = poseStamped_to_kdlFrame(source_pose_msg)
    (target_frame_id, target_frame) = poseStamped_to_kdlFrame(target_pose_msg)
    if target_frame_id != source_frame_id:
        return None

    t = kdl.diff(target_frame, source_frame)
    return (t[0], t[1], t[2])

def query_dummy_db(req):
    resp = QueryMotionResponse()
    target_obj = get_target_object(req.task.objects)
    source_obj = get_source_object(req.task.objects)

    global dummy_db
    phases = dummy_db[req.task.type][source_obj.name][target_obj.name]
    resp.phases = dummy_phases_to_ros(phases)
    return resp

def query_models(req):
    resp = QueryMotionResponse()
    target_obj = get_target_object(req.task.objects)
    source_obj = get_source_object(req.task.objects)
    rel_pose = calc_rel_pose(source_obj.pose, target_obj.pose)

    global learned_db
    phases = query_amodel(learned_db[req.task.type][source_obj.name][target_obj.name], rel_pose, req.task.pour_volume, source_obj.liquid_volume)
    resp.phases = learned_phases_to_ros(phases)

    return resp

def handle_query_motion(req):
    if dummy_mode:
        return query_dummy_db(req)
    else:
        return query_models(req)

def task_db_server():
    rospy.init_node('task_database')
    s = rospy.Service('~query_motion', QueryMotion, handle_query_motion)
    global dummy_mode
    dummy_mode = rospy.get_param('~dummy_mode')
    if dummy_mode:
        read_dummy_db()
    else:
        read_learned_db()
    rospy.loginfo("dummy_mode: %s", dummy_mode)
    rospy.spin()

if __name__ == '__main__':
    task_db_server()
